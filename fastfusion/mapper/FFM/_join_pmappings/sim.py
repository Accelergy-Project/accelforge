from collections import defaultdict
from functools import cached_property
from typing import Any, Callable, Iterable
import pandas as pd
from joblib import delayed

from fastfusion.mapper.FFM._pmapping_group import PmappingGroup

from fastfusion.mapper.FFM._join_pmappings.compatibility import *
from fastfusion.util import parallel


class SIM:
    def __init__(self, compatibility: Compatibility, mappings: PmappingGroup):
        self.compatibility: Compatibility = compatibility
        self.mappings: PmappingGroup = mappings
        self.tensors: dict[str, TensorReservation] = {
            t.name: t for t in self.compatibility.tensors
        }
        self.n_pre_prune_mappings = 0

    def compatibility_str(self):
        compatibility = ",".join(str(l) for l in self.compatibility.tensors)
        compatibility += " || " + ", ".join(str(t) for t in self.tensors.values())
        return compatibility

    @cached_property
    def tensor_names(self) -> set[str]:
        return set(self.tensors)

    def copy(self) -> "SIM":
        return SIM(self.compatibility, self.mappings.copy())

    def __len__(self) -> int:
        return len(self.mappings)

    def merge_next(
        self,
        right: "SIM",
        live_tensors: set[str],
        live_tensors_with_right: set[str],
        aliased_tensors: dict[str, set[str]],
        compatibility_joined: Compatibility,
        resource2capacity: dict[str, int] = None,
        drop_valid_reservations: bool = True,
        ignore_reservations: set[str] = set(),
        delay: bool = False,
        pmapping_row_filter_function: Callable[[pd.Series], bool] | None = None,
    ) -> "SIM":
        shared_loop_index = self.compatibility.shared_loop_index(
            right.compatibility.tensor_names | live_tensors
        )
        next_shared_loop_index = compatibility_joined.shared_loop_index(live_tensors)

        still_live_reservations = [
            r
            for r in self.compatibility.tensors
            if r.name in live_tensors and r.name not in right.compatibility.tensor_names
        ]

        duplicated_aliased_tensors = set()
        for name, my_tensor in self.tensors.items():
            for aliased_tensor in aliased_tensors.get(name, set()):
                if (aliased_tensor := right.tensors.get(aliased_tensor, None)) is None:
                    continue
                if my_tensor.resource_name == aliased_tensor.resource_name:
                    duplicated_aliased_tensors.add(aliased_tensor.name)

        mapping = delayed(self.mappings.merge_next)(
            right.mappings,
            shared_loop_index,
            next_shared_loop_index,
            live_tensors_with_right,
            still_live_reservations,
            duplicated_aliased_tensors,
            compatibility_left=self.compatibility,
            compatibility_right=right.compatibility,
            compatibility_joined=compatibility_joined,
            resource2capacity=resource2capacity,
            drop_valid_reservations=drop_valid_reservations,
            ignore_reservations=ignore_reservations,
            pmapping_row_filter_function=pmapping_row_filter_function,
        )

        if not delay:
            mapping = mapping[0](*mapping[1], **mapping[2])

        s = SIM(compatibility_joined, mapping)
        assert (
            compatibility_joined.max_above_loop_index == next_shared_loop_index + 1
        ), f"{self.compatibility} {right.compatibility} {next_shared_loop_index + 1} -> {compatibility_joined} {len(compatibility_joined.loops)}"
        s.tensors.update(right.tensors)
        s.tensors.update(self.tensors)
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
        dead_tensors = set(self.tensors) - (live_tensors or set())
        check_tensors = (shared_tensors or set()) | (live_tensors or set())
        shared_loop_index = self.compatibility.shared_loop_index(check_tensors)
        for t in dead_tensors:
            t = self.tensors.pop(t)
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
        parallelize: bool = True,
    ) -> list["SIM"]:
        def job(s):
            return s._right_consolidate(live_tensors, shared_tensors)

        if not parallelize:
            return [s._right_consolidate(live_tensors, shared_tensors) for s in sims]

        return parallel([delayed(job)(s) for s in sims], pbar=pbar)

    @staticmethod
    def left_consolidate(
        sims: list["SIM"],
        live_tensors: set[str],
        pbar: str = None,
        parallelize: bool = True,
    ) -> list["SIM"]:
        def job(s):
            return s._left_consolidate(live_tensors)

        if not parallelize:
            return [s._left_consolidate(live_tensors) for s in sims]

        return parallel([delayed(job)(s) for s in sims], pbar=pbar)

    def _hashable_attrs(self):
        return self.mappings, fzs(self.tensors.items())

    @staticmethod
    def concat(
        sims: Iterable["SIM"], allow_different_compatibilies: bool = False
    ) -> "SIM":
        sims = list(sims)
        assert len(sims) > 0, "Cannot concat empty list of SIMs"
        if not allow_different_compatibilies:
            s = set(s.compatibility.clear_symbolic_tile_patterns() for s in sims)
            if len(s) > 1:
                a = sims[0]
                for b in sims[1:]:
                    if a.compatibility != b.compatibility:
                        break
                SIM.combine_combineable((a, b), "All")
                assert (
                    a == b
                ), f"Cannot concat SIMs with different compatibilies:\n\t{a}\n\t{b}"
                assert len(s) == 1, (
                    f"Cannot concat SIMs with different compatibilies:\n\t"
                    + "\n\t".join(str(s2) for s2 in s)
                )

        c0 = sims[0].compatibility
        to_concat = [sims[0]] + [s.rename_compatibility(c0) for s in sims[1:]]
        return SIM(c0, PmappingGroup.concat([s.mappings for s in to_concat]))

    def rename_compatibility(self, new_c: Compatibility) -> Compatibility:
        c, renamed = self.compatibility.rename_to_match(new_c)
        return SIM(c, self.mappings.rename(renamed))

    @staticmethod
    def _group(
        sims: list["SIM"],
        live_tensors: set[str] | Literal["All"],
        clear_tile_patterns_and_reservation_indices: bool = False,
        include_permutations: bool = False,
        clear_symbolic_tile_patterns: bool = False,
        try_permute_into_equivalent: bool = False,
    ) -> (
        dict[Compatibility, list["SIM"]]
        | dict[Compatibility, list[tuple["SIM", list[int]]]]
    ):
        """
        Clears dead tensors (may keep loops), then group SIMs based on
        compatibility.
        """
        grouped = defaultdict(list)

        def clear(c: Compatibility):
            if clear_symbolic_tile_patterns:
                c = c.clear_symbolic_tile_patterns()
            if clear_tile_patterns_and_reservation_indices:
                return c.clear_tile_patterns_and_reservation_indices()
            return c

        for s in sims:
            compatibility = s.compatibility.clear_dead_tensors(live_tensors)

            if include_permutations or try_permute_into_equivalent:
                keys = compatibility.make_equivalent_permutations()
                for t, loop_changes in keys:
                    # Line below DOES NOT MUTATE. It's check that the permutation works.
                    s.compatibility.permute(loop_changes)
                    grouped[clear(t)].append((s, loop_changes))
            else:
                grouped[clear(compatibility)].append(s)

        if clear_tile_patterns_and_reservation_indices:
            for k in grouped:
                assert (
                    len(k.reservation_indices) == 0
                ), f"Extra reservation indices are not empty: {k.reservation_indices}"

        if try_permute_into_equivalent:
            assert not include_permutations
            new_grouped = {}
            sims_remaining = {id(s) for s in sims}
            for c, g in sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True):
                if not sims_remaining:
                    break
                g = [
                    (s, loop_changes)
                    for s, loop_changes in g
                    if id(s) in sims_remaining
                ]
                if g:
                    sims_remaining -= {id(s) for s, _ in g}
                    permuted = [
                        SIM(s.compatibility.permute(lc), s.mappings) for s, lc in g
                    ]
                    new_grouped[c] = permuted
            grouped = new_grouped

        return grouped

    @staticmethod
    def combine_combineable(
        sims: list["SIM"],
        live_tensors: set[str] | Literal["All"],
        allow_different_compatibilies: bool = False,
        combine_reservations: bool = True,
        pbar_postfix: str = "",
    ) -> list["SIM"]:
        sims = [s for s in sims if len(s.mappings.data) > 0]
        no_combine = []
        if not combine_reservations:
            has_reservations = [s.mappings.has_reservations() for s in sims]
            no_combine = [s for s, h in zip(sims, has_reservations) if h]
            sims = [s for s, h in zip(sims, has_reservations) if not h]
        groups = list(
            SIM._group(
                sims,
                live_tensors,
                clear_symbolic_tile_patterns=True,
                try_permute_into_equivalent=True,
            ).values()
        )
        groups_with_one = [g[0] for g in groups if len(g) == 1]
        if len(groups_with_one) == len(groups):
            return groups_with_one + no_combine

        others = parallel(
            [
                delayed(SIM.concat)(g, allow_different_compatibilies)
                for g in groups
                if len(g) > 1
            ],
            pbar=f"Grouping pmappings{pbar_postfix}",
        )
        return groups_with_one + others + no_combine

    @staticmethod
    def filter_by_tensors(
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
            return [s for s in sims if check(s.compatibility.tensors)]
        if isinstance(sims, dict):
            return {k: v for k, v in sims.items() if check(k.tensors)}
        raise ValueError(f"Invalid type {type(sims)}")

    @staticmethod
    def group(
        sims: list["SIM"], live_tensors: set[str]
    ) -> dict[tuple[Compatibility, ...], list[tuple["SIM", list[int]]]]:
        x = SIM._group(
            sims,
            live_tensors,
            clear_tile_patterns_and_reservation_indices=True,
            include_permutations=True,
        )
        return x

    @staticmethod
    def remove_dead_tensors(sims: list["SIM"], live_tensors: set[str]):
        for s in sims:
            for t in list(s.tensors):
                if t not in live_tensors:
                    del s.tensors[t]
