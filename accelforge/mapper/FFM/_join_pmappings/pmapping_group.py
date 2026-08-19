from collections import defaultdict
from functools import cached_property
from typing import Any, Callable, Iterable
from accelforge._accelerated_imports import pandas as pd
from joblib import delayed

from accelforge.mapper.FFM._join_pmappings.pmapping_dataframe import PmappingDataframe

from accelforge.mapper.FFM._join_pmappings.compatibility import *
from accelforge.mapper.FFM._join_pmappings.compatibility import (
    _multi_tensor_shared_loop_structure,
)
from accelforge.mapper.FFM._pareto_df.df_convention import (
    is_fused_loop_col,
    make_fused_loop_col,
)
from accelforge.util import parallel, oset


class PmappingGroup:
    def __init__(self, compatibility: Compatibility, mappings: PmappingDataframe):
        self.compatibility: Compatibility = compatibility
        self.mappings: PmappingDataframe = mappings
        self.tensors: dict[str, TensorReservation] = {
            t.name: t for t in self.compatibility.tensors
        }
        self.n_pre_prune_mappings = 0

        if isinstance(self.mappings, PmappingDataframe):
            checked = oset()
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
        return oset(self.tensors)

    def copy(self) -> "PmappingGroup":
        return PmappingGroup(self.compatibility, self.mappings.copy())

    def split_in_half(self) -> tuple["PmappingGroup", "PmappingGroup"]:
        first_mappings, second_mappings = self.mappings.split_in_half()
        return (
            PmappingGroup(self.compatibility, first_mappings),
            PmappingGroup(self.compatibility, second_mappings),
        )

    def __len__(self) -> int:
        return len(self.mappings)

    def merge_next(
        self,
        right: "PmappingGroup",
        live_tensors_post_join: set[str],
        live_tensors_with_right: set[str],
        aliased_tensors: dict[str, set[str]],
        compatibility_joined: Compatibility,
        ignored_resources: set[str],
        left_loop_to_right_loop: list[tuple],
        tensor_pair_constraints: list | None = None,
        delay: bool = False,
        _pmapping_row_filter_function: Callable[[pd.Series], bool] | None = None,
    ) -> "PmappingGroup":
        shared_loop_index = self.compatibility.shared_loop_index(
            right.compatibility.tensor_names | live_tensors_post_join
        )
        assert (
            shared_loop_index == self.compatibility.n_loops - 1
        ), "shared loop index not equal to left pmapping n_loops - 1"
        next_shared_loop_index = compatibility_joined.shared_loop_index(
            live_tensors_post_join
        )
        assert (
            next_shared_loop_index == compatibility_joined.n_loops - 1
        ), "next shared loop index not equal to joined pmapping n_loops - 1"
        assert compatibility_joined.tensor_names.issubset(
            live_tensors_post_join
        ), "joined compatibility includes tensors not live after joining"

        # Tensors on the right that are copies of a left tensor living in the same
        # memory level
        duplicated_aliased_tensors = oset()
        for name, my_tensor in self.tensors.items():
            for aliased_tensor in aliased_tensors.get(name, oset()):
                if (aliased_tensor := right.tensors.get(aliased_tensor, None)) is None:
                    continue
                if my_tensor.resource_name == aliased_tensor.resource_name:
                    duplicated_aliased_tensors.add(aliased_tensor)

        # Cannot free aliased tensors that will still be alive in this level later. Copy
        # Einsums will only make one reservation for all of the aliases, and we cannot
        # free that until all of them have been freed.
        all_tensors = {**right.tensors, **self.tensors}
        do_not_free = oset()
        for name, res in all_tensors.items():
            if name in live_tensors_post_join:
                continue
            for aliased in aliased_tensors.get(name, oset()):
                other = all_tensors.get(aliased)
                if (
                    other is not None
                    and other.resource_name == res.resource_name
                    and aliased in live_tensors_post_join
                ):
                    do_not_free.add(name)
                    break

        mapping = delayed(self.mappings.merge_next)(
            right.mappings,
            duplicated_aliased_tensors,
            compatibility_left=self.compatibility,
            compatibility_right=right.compatibility,
            compatibility_joined=compatibility_joined,
            left_loop_to_right_loop=left_loop_to_right_loop,
            tensor_pair_constraints=tensor_pair_constraints,
            do_not_free=do_not_free,
            _pmapping_row_filter_function=_pmapping_row_filter_function,
            ignored_resources=ignored_resources,
        )

        if not delay:
            mapping = mapping[0](*mapping[1], **mapping[2])

        s = PmappingGroup(compatibility_joined, mapping)
        assert (
            compatibility_joined.max_above_loop_index == next_shared_loop_index + 1
        ), f"{self.compatibility} {right.compatibility} {next_shared_loop_index + 1} -> {compatibility_joined} {compatibility_joined.n_loops}"
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
        dead_tensors = oset(self.tensors) - (live_tensors or oset())
        check_tensors = (shared_tensors or oset()) | (live_tensors or oset())
        for t in dead_tensors:
            t = self.tensors.pop(t)
        cleared = self.compatibility.clear_dead_tensors(check_tensors)
        if self.mappings.free_to_reservations(cleared, shift_bottom_left=False):
            self.mappings.make_pareto()
        return self

    def _left_consolidate(self, live_tensors: set[str] = None):
        check_tensors = live_tensors or oset()
        cleared = self.compatibility.clear_dead_tensors(check_tensors)
        if self.mappings.free_to_reservations(cleared, shift_bottom_left=False):
            self.mappings.make_pareto()
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
            s = oset(
                s.compatibility.clear_symbolic_tile_patterns() for s in pmapping_groups
            )
            if len(s) > 1:
                a = pmapping_groups[0]
                for b in pmapping_groups[1:]:
                    if a.compatibility != b.compatibility:
                        break
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
        catted = PmappingDataframe.concat([s.mappings for s in to_concat])
        return PmappingGroup(c0, catted)

    def rename_compatibility(self, new_c: Compatibility) -> Compatibility:
        c, renamed = self.compatibility._rename_to_match(new_c)
        return PmappingGroup(c, self.mappings.rename(renamed))

    @staticmethod
    def _group_equivalent(
        pmapping_groups: list["PmappingGroup"],
        live_tensors: set[str] | Literal["All"],
    ) -> list[list["PmappingGroup"]]:
        """
        Clears dead tensors (may keep loops), then group PmappingGroups based on
        compatibility.
        """
        grouped = defaultdict(list)
        for pg in pmapping_groups:
            key = (
                pg.compatibility.clear_dead_tensors(
                    live_tensors
                ).clear_symbolic_tile_patterns(),
                _multi_tensor_shared_loop_structure(pg.compatibility),
            )
            grouped[key].append(pg)
        return list(grouped.values())

    @staticmethod
    def combine_combineable(
        pmapping_groups: list["PmappingGroup"],
        live_tensors: set[str] | Literal["All"],
        allow_different_compatibilies: bool = False,
        _combine_reservations: bool = True,
        print_progress: bool = True,
        pbar_postfix: str = "",
    ) -> list["PmappingGroup"]:
        pmapping_groups = [s for s in pmapping_groups if len(s.mappings.data) > 0]
        no_combine = []
        if not _combine_reservations:
            has_reservations = [s.mappings.has_reservations() for s in pmapping_groups]
            no_combine = [s for s, h in zip(pmapping_groups, has_reservations) if h]
            pmapping_groups = [
                s for s, h in zip(pmapping_groups, has_reservations) if not h
            ]
        groups = PmappingGroup._group_equivalent(pmapping_groups, live_tensors)
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

        tensors = oset(tensors)
        if isinstance(pmapping_groups, list):
            return [s for s in pmapping_groups if check(s.compatibility.tensors)]
        if isinstance(pmapping_groups, dict):
            return {k: v for k, v in pmapping_groups.items() if check(k.tensors)}
        raise ValueError(f"Invalid type {type(pmapping_groups)}")

    @staticmethod
    def remove_dead_tensors(
        pmapping_groups: list["PmappingGroup"], live_tensors: set[str]
    ):
        for s in pmapping_groups:
            for t in list(s.tensors):
                if t not in live_tensors:
                    del s.tensors[t]
