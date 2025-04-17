from collections import defaultdict
from functools import cached_property
from typing import Any, Iterable, Optional

from joblib import delayed

from fastfusion.pareto import Pareto

from fastfusion.joining.mappinginfo import *
from fastfusion.util import parallel

class SIM:
    def __init__(self, compatibility: Mapping, mapping: Pareto):
        self.compatibility: Mapping = compatibility
        self.mappings: Pareto = mapping
        self.storage: dict[str, TensorStorage] = {
            t.name: t for t in self.compatibility.storage
        }
        self.n_pre_prune_mappings = 0

    def compatibility_str(self):
        compatibility = ",".join(str(l) for l in self.compatibility.loops)
        compatibility += " || " + ", ".join(str(t) for t in self.storage.values())
        return compatibility

    def mapping_str(self):
        return str(self.mappings.einsum_ids())

    @cached_property
    def tensor_names(self) -> set[str]:
        return set(self.storage)

    def copy(self) -> "SIM":
        return SIM(self.compatibility, self.mappings.copy())

    def merge_next(
        self, right: "SIM", live_tensors: set[str], delay: bool = False
    ) -> "SIM":
        shared_loop_index = self.compatibility.shared_loop_index(
            right.compatibility.tensor_names | live_tensors
        )
        compatibility = self.compatibility.merge_next(right.compatibility, live_tensors)
        next_shared_loop_index = compatibility.shared_loop_index(live_tensors)
        shared_storage = self.compatibility.storage & right.compatibility.storage
        mapping = delayed(
            self.mappings.merge_next
        )(right.mappings, shared_loop_index, live_tensors, shared_storage)

        if not delay:
            mapping = mapping[0](*mapping[1], **mapping[2])
        
        s = SIM(compatibility, mapping)
        assert (
            len(compatibility.loops) == next_shared_loop_index + 1
        ), f"{self.compatibility} {right.compatibility} {next_shared_loop_index + 1} -> {compatibility} {len(compatibility.loops)}"
        s.storage.update(right.storage)
        s.storage.update(self.storage)
        s.n_pre_prune_mappings = len(self.mappings.data) * len(right.mappings.data)
        return s

    def get_shared_loop_index(self, live_tensors: set[str]) -> int:
        live_tensors = list(self.compatibility.tensor_names) + [live_tensors]
        return self.compatibility.shared_loop_index(live_tensors)

    def free_squish(
        self,
        index: Optional[int],
        resource2capacity: dict[str, int] = None,
    ):
        needs_pareto = False
        needs_pareto = (
            self.mappings.free_to_loop_index(index) or needs_pareto
        )
        needs_pareto = self.mappings.squish_left_right(index) or needs_pareto
        if needs_pareto:
            self.mappings.make_pareto()

    def _right_consolidate(
        self,
        live_tensors: set[str] = None,
        resource2capacity: dict[str, int] = None,
        shared_tensors: set[str] = None,
    ):
        dead_tensors = set(self.storage) - (live_tensors or set())
        check_tensors = (shared_tensors or set()) | (live_tensors or set())
        shared_loop_index = self.compatibility.shared_loop_index(check_tensors)
        for t in dead_tensors:
            t = self.storage.pop(t)
            # self.mappings.alloc(
            #     t.resource_name, t.size, t.above_loop_index, resource2capacity
            # )

        if live_tensors is None:
            self.free_squish(0, resource2capacity)
        else:
            self.free_squish(shared_loop_index + 1, resource2capacity)
        return self

    def _left_consolidate(
        self,
        live_tensors: set[str] = None,
        resource2capacity: dict[str, int] = None,
        shared_tensors: set[str] = None,
    ):
        check_tensors = (shared_tensors or set()) | (live_tensors or set())
        shared_loop_index = self.compatibility.shared_loop_index(check_tensors)
        tensors_to_add = []
        for t in self.storage.values():
            if (
                t.above_loop_index > shared_loop_index
                or t.name not in check_tensors
            ):
                # self.mappings.alloc(
                #     t.resource_name, t.size, t.above_loop_index, resource2capacity
                # )
                tensors_to_add.append(t)
        if live_tensors is None:
            self.free_squish(-1, resource2capacity)
        else:
            self.free_squish(shared_loop_index + 1, resource2capacity)
        for t in tensors_to_add:
            self.mappings.add_tensor(t)
        return self

    @staticmethod
    def right_consolidate(
        sims: list["SIM"],
        live_tensors: set[str],
        resource2capacity: dict[str, int] = None,
        shared_tensors: set[str] = None,
        pbar: str = None,
    ) -> list["SIM"]:
        def job(s):
            return s._right_consolidate(live_tensors, resource2capacity, shared_tensors)

        return parallel([delayed(job)(s) for s in sims], pbar=pbar)

    @staticmethod
    def left_consolidate(
        sims: list["SIM"],
        live_tensors: set[str],
        resource2capacity: dict[str, int] = None,
        shared_tensors: set[str] = None,
        pbar: str = None,
    ) -> list["SIM"]:
        def job(s):
            return s._left_consolidate(live_tensors, resource2capacity, shared_tensors)

        return parallel([delayed(job)(s) for s in sims], pbar=pbar)

    def _hashable_attrs(self):
        return self.mappings, fzs(self.storage.items())

    @staticmethod
    def concat(sims: Iterable["SIM"], allow_different_compatibilitys: bool = False) -> "SIM":
        sims = list(sims)
        assert len(sims) > 0, "Cannot concat empty list of SIMs"
        if not allow_different_compatibilitys:
            s = set(fzs([(k, v) for k, v in s.storage.items()]) for s in sims)
            assert (
                len(s) == 1
            ), f"Cannot concat SIMs with different tensors:\n\t" + "\n\t".join(
                str(s2) for s2 in s
            )
        return SIM(sims[0].compatibility, Pareto.concat([s.mappings for s in sims]))

    @staticmethod
    def _group(
        sims: list["SIM"],
        live_tensors: set[str],
        keep_loops: bool = False,
        every_possible_n_loops: bool = False,
        keep_tensors: set[str] = None,
        drop_tags: bool = False,
    ) -> dict[tuple[Mapping, ...], list["SIM"]]:
        grouped = defaultdict(list)
        for s in sims:
            compatibility = s.compatibility.clear_dead_tensors(
                live_tensors,
                keep_loops=keep_loops or every_possible_n_loops,
                keep_tensors=keep_tensors,
                drop_tags=drop_tags,
            )
            if every_possible_n_loops:
                compatibility = compatibility.all_n_loops()
            else:
                compatibility = [compatibility]
            for t in compatibility:
                grouped[t].append(s)

        return grouped

    @staticmethod
    def combine_combineable(
        sims: list["SIM"],
        live_tensors: set[str],
        allow_different_compatibilitys: bool = False,
        drop_tags: bool = False,
        combine_reservations: bool = True,
    ) -> list["SIM"]:
        no_combine = []
        if not combine_reservations:
            has_reservations = [s.mappings.has_reservations() for s in sims]
            no_combine = [s for s, h in zip(sims, has_reservations) if h]
            sims = [s for s, h in zip(sims, has_reservations) if not h]
        groups = list(SIM._group(sims, live_tensors, drop_tags=drop_tags).values())
        groups_with_one = [g[0] for g in groups if len(g) == 1]
        if len(groups_with_one) == len(groups):
            return groups_with_one + no_combine
        others = parallel(
            [
                delayed(SIM.concat)(g, allow_different_compatibilitys)
                for g in groups
                if len(g) > 1
            ],
            pbar="Combining SIMs",
        )
        return groups_with_one + others + no_combine

    @staticmethod
    def filter_by_tensor_storage(
        sims: list["SIM"] | dict[Mapping, Any], tensors: set[str]
    ) -> list["SIM"]:
        def check(tensors_to_check):
            for t in tensors_to_check:
                for t2 in tensors:
                    if (
                        t2.name == "*" or t.name == t2.name
                    ) and t != t2:
                        return False
            return True

        tensors = set(tensors)
        if isinstance(sims, list):
            return [s for s in sims if check(s.compatibility.storage)]
        if isinstance(sims, dict):
            return {k: v for k, v in sims.items() if check(k.storage)}
        raise ValueError(f"Invalid type {type(sims)}")

    def filter_by_mapping_hashes(self, hashes: set[str]) -> Optional["SIM"]:
        self = SIM(self.compatibility, self.mappings.filter_by_mapping_hashes(hashes))
        return self if len(self.mappings.data) > 0 else None

    @staticmethod
    def group_left(
        sims: list["SIM"], live_tensors: set[str], drop_tags: bool = False
    ) -> dict[tuple[Mapping, ...], list["SIM"]]:
        return SIM._group(sims, live_tensors, keep_loops=True, drop_tags=drop_tags)

    @staticmethod
    def group_right(
        sims: list["SIM"], live_tensors: set[str], drop_tags: bool = False
    ) -> dict[tuple[Mapping, ...], list["SIM"]]:
        return SIM._group(
            sims, live_tensors, drop_tags=drop_tags, every_possible_n_loops=True
        )

    @staticmethod
    def remove_dead_tensors(sims: list["SIM"], live_tensors: set[str]) -> list["SIM"]:
        for s in sims:
            for t in list(s.storage):
                if t not in live_tensors:
                    del s.storage[t]

    def set_tags(self, *tags: Any) -> "SIM":
        self.compatibility = self.compatibility.set_tags(*tags)

    @property
    def tags(self) -> fzs[Any]:
        return self.compatibility.tags

    @staticmethod
    def get_possibly_compatible(
        left: list["SIM"],
        right: list["SIM"],
        left_live_tensors: set[str],
        right_live_tensors: set[str],
    ):
        assert left and right, "Cannot check for compatibility with empty list"
        shared_tensors = left[0].names & right[0].names
        left = SIM._group(left, right_live_tensors, keep_tensors=shared_tensors)
        right = SIM._group(right, left_live_tensors, keep_tensors=shared_tensors)
        left_keys = set().union(*(l.all_n_loops() for l in left))
        right_keys = set(right)
        left_list = [s for k in left for s in left[k] if k in right_keys]
        right_list = [s for k in right for s in right[k] if k in left_keys]
        return left_list, right_list
