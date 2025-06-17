from collections import defaultdict
from functools import cached_property
from typing import Any, Iterable, Optional

from joblib import delayed

from fastfusion.mapper.FFM.pareto import PartialMappings

from fastfusion.mapper.FFM.joining.mappinginfo import *
from fastfusion.util import parallel


class SIM:
    def __init__(self, compatibility: Compatibility, mappings: PartialMappings):
        self.compatibility: Compatibility = compatibility
        self.mappings: PartialMappings = mappings
        self.storage: dict[str, TensorStorage] = {
            t.name: t for t in self.compatibility.storage
        }
        self.n_pre_prune_mappings = 0

    def compatibility_str(self):
        compatibility = ",".join(str(l) for l in self.compatibility.storage)
        compatibility += " || " + ", ".join(str(t) for t in self.storage.values())
        return compatibility

    @cached_property
    def tensor_names(self) -> set[str]:
        return set(self.storage)

    def copy(self) -> "SIM":
        return SIM(self.compatibility, self.mappings.copy())

    def merge_next(
        self,
        right: "SIM",
        live_tensors: set[str],
        live_tensors_with_right: set[str],
        aliased_tensors: dict[str, str],
        resource2capacity: dict[str, int] = None,
        drop_valid_reservations: bool = True,
        delay: bool = False,
    ) -> "SIM":
        shared_loop_index = self.compatibility.shared_loop_index(
            right.compatibility.tensor_names | live_tensors
        )
        compatibility = self.compatibility.merge_next(right.compatibility, live_tensors)
        next_shared_loop_index = compatibility.shared_loop_index(live_tensors)
        shared_storage = self.compatibility.storage & right.compatibility.storage

        still_live_reservations = [
            r
            for r in self.compatibility.storage
            if r.name in live_tensors and r.name not in right.compatibility.tensor_names
        ]

        duplicated_aliased_tensors = set()
        for name, my_tensor in self.storage.items():
            aliased_tensor = right.storage.get(aliased_tensors.get(name, None), None)
            if aliased_tensor is None:
                continue
            if my_tensor.resource_name == aliased_tensor.resource_name:
                duplicated_aliased_tensors.add(aliased_tensor)

        mapping = delayed(self.mappings.merge_next)(
            right.mappings,
            shared_loop_index,
            next_shared_loop_index,
            live_tensors_with_right,
            shared_storage,
            still_live_reservations,
            duplicated_aliased_tensors,
            resource2capacity,
            drop_valid_reservations=drop_valid_reservations
        )

        if not delay:
            mapping = mapping[0](*mapping[1], **mapping[2])

        s = SIM(compatibility, mapping)
        assert (
            compatibility.max_above_loop_index == next_shared_loop_index + 1
        ), f"{self.compatibility} {right.compatibility} {next_shared_loop_index + 1} -> {compatibility} {len(compatibility.loops)}"
        s.storage.update(right.storage)
        s.storage.update(self.storage)
        s.n_pre_prune_mappings = len(self.mappings.data) * len(right.mappings.data)
        return s

    def get_shared_loop_index(self, live_tensors: set[str]) -> int:
        live_tensors = list(self.compatibility.tensor_names) + [live_tensors]
        return self.compatibility.shared_loop_index(live_tensors)

    def _right_consolidate(
        self,
        live_tensors: set[str] = None,
        shared_tensors: set[str] = None,
    ):
        dead_tensors = set(self.storage) - (live_tensors or set())
        check_tensors = (shared_tensors or set()) | (live_tensors or set())
        shared_loop_index = self.compatibility.shared_loop_index(check_tensors)
        for t in dead_tensors:
            t = self.storage.pop(t)
        if self.mappings.free_to_loop_index(
            shared_loop_index, live_tensors=live_tensors
        ):
            self.mappings.make_pareto()
        return self

    def _left_consolidate(self, live_tensors: set[str] = None):
        check_tensors = live_tensors or set()
        shared_loop_index = self.compatibility.shared_loop_index(check_tensors)
        self.mappings.free_to_loop_index(shared_loop_index, live_tensors=live_tensors)
        return self

    @staticmethod
    def right_consolidate(
        sims: list["SIM"],
        live_tensors: set[str],
        shared_tensors: set[str] = None,
        pbar: str = None,
    ) -> list["SIM"]:
        def job(s):
            return s._right_consolidate(live_tensors, shared_tensors)

        return parallel([delayed(job)(s) for s in sims], pbar=pbar)

    @staticmethod
    def left_consolidate(
        sims: list["SIM"],
        live_tensors: set[str],
        pbar: str = None,
    ) -> list["SIM"]:
        def job(s):
            return s._left_consolidate(live_tensors)

        return parallel([delayed(job)(s) for s in sims], pbar=pbar)

    def _hashable_attrs(self):
        return self.mappings, fzs(self.storage.items())

    @staticmethod
    def concat(
        sims: Iterable["SIM"], allow_different_compatibilitys: bool = False
    ) -> "SIM":
        sims = list(sims)
        assert len(sims) > 0, "Cannot concat empty list of SIMs"
        if not allow_different_compatibilitys:
            s = set(fzs([(k, v) for k, v in s.storage.items()]) for s in sims)
            assert (
                len(s) == 1
            ), f"Cannot concat SIMs with different tensors:\n\t" + "\n\t".join(
                str(s2) for s2 in s
            )
        return SIM(sims[0].compatibility, PartialMappings.concat([s.mappings for s in sims]))

    @staticmethod
    def _group(
        sims: list["SIM"],
        live_tensors: set[str],
        keep_loops: bool = False,
        every_possible_n_loops: bool = False,
        drop_tags: bool = False,
    ) -> dict[tuple[Compatibility, ...], list["SIM"]]:
        grouped = defaultdict(list)
        for s in sims:
            compatibility = s.compatibility.clear_dead_tensors(
                live_tensors,
                keep_loops=keep_loops or every_possible_n_loops,
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
        pbar_postfix: str = "",
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
            pbar=f"Grouping Partial Mappings{pbar_postfix}",
        )
        return groups_with_one + others + no_combine

    @staticmethod
    def filter_by_tensor_storage(
        sims: list["SIM"] | dict[Compatibility, Any], tensors: set[str]
    ) -> list["SIM"]:
        def check(tensors_to_check):
            for t in tensors_to_check:
                for t2 in tensors:
                    if (t2.name == "*" or t.name == t2.name) and t != t2:
                        return False
            return True

        tensors = set(tensors)
        if isinstance(sims, list):
            return [s for s in sims if check(s.compatibility.storage)]
        if isinstance(sims, dict):
            return {k: v for k, v in sims.items() if check(k.storage)}
        raise ValueError(f"Invalid type {type(sims)}")

    @staticmethod
    def group_left(
        sims: list["SIM"], live_tensors: set[str], drop_tags: bool = False
    ) -> dict[tuple[Compatibility, ...], list["SIM"]]:
        return SIM._group(sims, live_tensors, keep_loops=True, drop_tags=drop_tags)

    @staticmethod
    def group_right(
        sims: list["SIM"],
        live_tensors: set[str],
        drop_tags: bool = False,
    ) -> dict[tuple[Compatibility, ...], list["SIM"]]:
        return SIM._group(
            sims,
            live_tensors,
            every_possible_n_loops=True,
            drop_tags=drop_tags,
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
