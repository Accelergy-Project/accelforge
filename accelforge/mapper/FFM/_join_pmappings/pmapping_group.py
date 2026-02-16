from collections import defaultdict
from functools import cached_property
from typing import Any, Callable, Iterable
import pandas as pd
from joblib import delayed

from accelforge.mapper.FFM._join_pmappings.pmapping_dataframe import PmappingDataframe

from accelforge.mapper.FFM._join_pmappings.compatibility import *
from accelforge.mapper.FFM._pareto_df.df_convention import (
    is_fused_loop_col,
    make_fused_loop_col,
)
from accelforge.util import parallel


class PmappingGroup:
    def __init__(self, compatibility: Compatibility, mappings: PmappingDataframe):
        self.compatibility: Compatibility = compatibility
        self.mappings: PmappingDataframe = mappings
        self.tensors: dict[str, TensorReservation] = {
            t.name: t for t in self.compatibility.tensors
        }
        self.n_pre_prune_mappings = 0

        if isinstance(self.mappings, PmappingDataframe):
            checked = set()
            for s in self.compatibility.symbols():
                checked.add(s)
                assert (
                    s in self.mappings.data.columns
                ), f"Column {s} not found in mappings"

            for col_name in self.mappings.data.columns:
                if col_name not in checked and is_fused_loop_col(col_name):
                    raise ValueError(f"Column {col_name} not found in compatibility")

    def compatibility_str(self):
        compatibility = ",".join(str(l) for l in self.compatibility.tensors)
        compatibility += " || " + ", ".join(str(t) for t in self.tensors.values())
        return compatibility

    @cached_property
    def tensor_names(self) -> set[str]:
        return set(self.tensors)

    def copy(self) -> "PmappingGroup":
        return PmappingGroup(self.compatibility, self.mappings.copy())

    def __len__(self) -> int:
        return len(self.mappings)

    def merge_next(
        self,
        right: "PmappingGroup",
        live_tensors: set[str],
        live_tensors_with_right: set[str],
        aliased_tensors: dict[str, set[str]],
        compatibility_joined: Compatibility,
        ignored_resources: set[str],
        permuted_compatibility_left: Compatibility,
        permuted_compatibility_right: Compatibility,
        drop_valid_reservations: bool = True,
        delay: bool = False,
        _pmapping_row_filter_function: Callable[[pd.Series], bool] | None = None,
    ) -> "PmappingGroup":
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
            compatibility_left=permuted_compatibility_left,
            compatibility_right=permuted_compatibility_right,
            compatibility_joined=compatibility_joined,
            drop_valid_reservations=drop_valid_reservations,
            _pmapping_row_filter_function=_pmapping_row_filter_function,
            ignored_resources=ignored_resources,
        )

        if not delay:
            mapping = mapping[0](*mapping[1], **mapping[2])

        s = PmappingGroup(compatibility_joined, mapping)
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
        if live_tensors is None:
            self.mappings.clear_fused_loop_symbols()
        return self

    @staticmethod
    def right_consolidate(
        pmapping_groups: list["PmappingGroup"],
        live_tensors: set[str],
        shared_tensors: set[str] = None,
        pbar: str = None,
        parallelize: bool = True,
    ) -> list["PmappingGroup"]:
        def job(s):
            return s._right_consolidate(live_tensors, shared_tensors)

        if not parallelize:
            return [
                s._right_consolidate(live_tensors, shared_tensors)
                for s in pmapping_groups
            ]

        return parallel([delayed(job)(s) for s in pmapping_groups], pbar=pbar)

    @staticmethod
    def left_consolidate(
        pmapping_groups: list["PmappingGroup"],
        live_tensors: set[str],
        pbar: str = None,
        parallelize: bool = True,
    ) -> list["PmappingGroup"]:
        def job(s):
            return s._left_consolidate(live_tensors)

        if not parallelize:
            return [s._left_consolidate(live_tensors) for s in pmapping_groups]

        return parallel([delayed(job)(s) for s in pmapping_groups], pbar=pbar)

    def _hashable_attrs(self):
        return self.mappings, fzs(self.tensors.items())

    @staticmethod
    def concat(
        pmapping_groups: Iterable["PmappingGroup"],
        allow_different_compatibilies: bool = False,
    ) -> "PmappingGroup":
        pmapping_groups = list(pmapping_groups)
        assert len(pmapping_groups) > 0, "Cannot concat empty list of PmappingGroups"
        if not allow_different_compatibilies:
            s = set(
                s.compatibility.clear_symbolic_tile_patterns() for s in pmapping_groups
            )
            if len(s) > 1:
                a = pmapping_groups[0]
                for b in pmapping_groups[1:]:
                    if a.compatibility != b.compatibility:
                        break
                PmappingGroup.combine_combineable((a, b), "All")
                assert (
                    a == b
                ), f"Cannot concat PmappingGroups with different compatibilies:\n\t{a}\n\t{b}"
                assert len(s) == 1, (
                    f"Cannot concat PmappingGroups with different compatibilies:\n\t"
                    + "\n\t".join(str(s2) for s2 in s)
                )

        c0 = pmapping_groups[0].compatibility
        to_concat = [pmapping_groups[0]] + [
            s.rename_compatibility(c0) for s in pmapping_groups[1:]
        ]
        return PmappingGroup(
            c0, PmappingDataframe.concat([s.mappings for s in to_concat])
        )

    def rename_compatibility(self, new_c: Compatibility) -> Compatibility:
        c, renamed = self.compatibility._rename_to_match(new_c)
        return PmappingGroup(c, self.mappings.rename(renamed))

    @staticmethod
    def _group(
        pmapping_groups: list["PmappingGroup"],
        live_tensors: set[str] | Literal["All"],
        clear_tile_patterns_and_reservation_indices: bool = False,
        include_permutations: bool = False,
        clear_symbolic_tile_patterns: bool = False,
        try_permute_into_equivalent: bool = False,
        # mixable_ranks: dict[Rank, set[Rank]] = None,
    ) -> (
        dict[Compatibility, list["PmappingGroup"]]
        | dict[Compatibility, list[tuple["PmappingGroup", list[int]]]]
    ):
        """
        Clears dead tensors (may keep loops), then group PmappingGroups based on
        compatibility.
        """
        grouped = defaultdict(list)

        def clear(c: Compatibility):
            if clear_symbolic_tile_patterns:
                c = c.clear_symbolic_tile_patterns()
            if clear_tile_patterns_and_reservation_indices:
                return c.clear_tile_patterns_and_reservation_indices()
            return c

        for s in pmapping_groups:
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

        # if mixable_ranks is not None:
        #     new_grouped = {}
        #     for c, g in grouped.items():
        #         for c2 in c.get_equivalent_compatibilities(mixable_ranks):
        #             new_grouped.setdefault(c2, []).extend(g)
        #     grouped = new_grouped

        if try_permute_into_equivalent:
            assert not include_permutations
            new_grouped = {}
            pmgroups_remaining = {id(s) for s in pmapping_groups}
            for c, g in sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True):
                if not pmgroups_remaining:
                    break
                g = [
                    (s, loop_changes)
                    for s, loop_changes in g
                    if id(s) in pmgroups_remaining
                ]
                if g:
                    pmgroups_remaining -= {id(s) for s, _ in g}
                    permuted = [
                        PmappingGroup(
                            s.compatibility.permute(lc),
                            s.mappings.clear_irrelevant_columns(s.compatibility),
                        )
                        for s, lc in g
                    ]
                    new_grouped[c] = permuted
            grouped = new_grouped

        return grouped

    @staticmethod
    def combine_combineable(
        pmapping_groups: list["PmappingGroup"],
        live_tensors: set[str] | Literal["All"],
        allow_different_compatibilies: bool = False,
        combine_reservations: bool = True,
        print_progress: bool = True,
        pbar_postfix: str = "",
    ) -> list["PmappingGroup"]:
        pmapping_groups = [s for s in pmapping_groups if len(s.mappings.data) > 0]
        no_combine = []
        if not combine_reservations:
            has_reservations = [s.mappings.has_reservations() for s in pmapping_groups]
            no_combine = [s for s, h in zip(pmapping_groups, has_reservations) if h]
            pmapping_groups = [
                s for s, h in zip(pmapping_groups, has_reservations) if not h
            ]
        groups = list(
            PmappingGroup._group(
                pmapping_groups,
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
                delayed(PmappingGroup.concat)(g, allow_different_compatibilies)
                for g in groups
                if len(g) > 1
            ],
            pbar=f"Grouping pmappings{pbar_postfix}" if print_progress else None,
        )
        return groups_with_one + others + no_combine

    @staticmethod
    def filter_by_tensors(
        pmapping_groups: list["PmappingGroup"] | dict[Compatibility, Any],
        tensors: set[str],
    ) -> list["PmappingGroup"]:
        def check(tensors_to_check):
            for t in tensors_to_check:
                for t2 in tensors:
                    if (t2.name == "*" or t.name == t2.name) and t != t2:
                        return False
            return True

        tensors = set(tensors)
        if isinstance(pmapping_groups, list):
            return [s for s in pmapping_groups if check(s.compatibility.tensors)]
        if isinstance(pmapping_groups, dict):
            return {k: v for k, v in pmapping_groups.items() if check(k.tensors)}
        raise ValueError(f"Invalid type {type(pmapping_groups)}")

    @staticmethod
    def group(
        pmapping_groups: list["PmappingGroup"], live_tensors: set[str]#, mixable_ranks: dict[Rank, set[Rank]] = None
    ) -> dict[tuple[Compatibility, ...], list[tuple["PmappingGroup", list[int]]]]:
        x = PmappingGroup._group(
            pmapping_groups,
            live_tensors,
            clear_tile_patterns_and_reservation_indices=True,
            include_permutations=True,
            # mixable_ranks=mixable_ranks,
        )
        return x

    @staticmethod
    def remove_dead_tensors(
        pmapping_groups: list["PmappingGroup"], live_tensors: set[str]
    ):
        for s in pmapping_groups:
            for t in list(s.tensors):
                if t not in live_tensors:
                    del s.tensors[t]
