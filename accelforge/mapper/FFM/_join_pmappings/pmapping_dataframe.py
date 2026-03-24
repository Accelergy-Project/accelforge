from collections import defaultdict
import copy
import functools
import itertools
from operator import or_

from typing import Any, Callable, Iterable

import sympy

from accelforge.frontend.mapping import Nested, TilePattern
from accelforge.frontend.mapping import Loop as MappingLoop
from accelforge.mapper.FFM._join_pmappings.compatibility import (
    Compatibility,
    Loop,
    TensorReservation,
)
from accelforge.util import _fillna_and__numeric_cast, _numeric_cast
from accelforge.util._frozenset import fzs, oset

from accelforge._accelerated_imports import pd

from accelforge.mapper.FFM._pareto_df.df_convention import *
from accelforge.mapper.FFM._pareto_df.pareto import makepareto

CHECK_CORRECTNESS = False
DEBUG_PRINT_NO_VALID = False


def error_check_wrapper(func):
    if not CHECK_CORRECTNESS:
        return func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            prev_args, prev_kwargs = copy.deepcopy(args), copy.deepcopy(kwargs)
            return func(*args, **kwargs)
        except Exception as e:
            print(f"EXCEPTION: {e}")
            live_tensors = oset()
            if "live_tensors" in kwargs:
                live_tensors = kwargs["live_tensors"]
            else:
                argnames = func.__code__.co_varnames[: func.__code__.co_argcount]
                if "live_tensors" in argnames:
                    idx = argnames.index("live_tensors")
                    if idx < len(args):
                        live_tensors = args[idx]
            for prev_arg in itertools.chain(prev_args, prev_kwargs.values()):
                if isinstance(prev_arg, PmappingDataframe):
                    prev_arg.fail(0, live_tensors)
                break
            func(*args, **kwargs)  # For debugging

    return wrapper


def reduce_precision(data: pd.DataFrame) -> pd.DataFrame:
    data = _numeric_cast(data)

    # def _reduce_precision(c: str, s: pd.Series) -> pd.Series:
    #     # If it's an int type, check the range. If within range of 8b change to 8b. If
    #     # within the range of 16b change to 16b...

    #     # If it's a float, cast to NUMPY_FLOAT_TYPE
    #     if pd.api.types.is_float_dtype(s) and s.dtype != NUMPY_FLOAT_TYPE:
    #         return s.astype(NUMPY_FLOAT_TYPE)

    #     if not is_fused_loop_col(c):
    #         return s

    #     # Get the range of the column
    #     min_val = s.min()
    #     if min_val < 0:
    #         return s

    #     max_val = s.max()
    #     if max_val <= 2**8 - 1 and s.dtype != np.uint8:
    #         return s.astype(np.uint8)
    #     elif max_val <= 2**16 - 1 and s.dtype != np.uint16:
    #         return s.astype(np.uint16)
    #     elif max_val <= 2**32 - 1 and s.dtype != np.uint32:
    #         return s.astype(np.uint32)
    #     return s

    # for c in data.columns:
    #     data.loc[:, c] = _reduce_precision(c, data.loc[:, c])

    return data


def get_reservation_or_parent(
    name: str,
    level: int,
    l_reservations: dict[str, set[int]],
    r_reservations: dict[str, set[int]],
    left: bool = False,
    return_name_level_left: bool = False,
) -> str | tuple[str, int, bool] | None:
    reservations = l_reservations if left else r_reservations
    if (reservations := reservations.get(name, None)) is not None:
        while level >= -1:
            if level in reservations:
                if return_name_level_left:
                    return name, level, left
                return reservation2col(name, level, left)
            # The parent of left nodes are right nodes, so if we don't find a
            # left node immediately then we're back on the right nodes
            reservations = r_reservations.get(name, oset())
            left = False
            level -= 1
    return None


class PmappingDataframe:
    def __init__(
        self,
        data: pd.DataFrame,
        n_total_pmappings: float,
        n_valid_pmappings: float,
        ignored_resources: set[str],
        drop_valid_reservations: bool,
        n_concurrent_threads: int,
        create_live_reservation_from_compatibility: Compatibility = None,
        skip_pareto: bool = False,
        fill_reservation_cols: set | str = fzs(),
        check_above_subset_below: bool = CHECK_CORRECTNESS,
        max_right_to_left: bool = False,
        next_shared_loop_index: int = None,
        excess_resource_tolerance: float = 0,
        track_binding_sequence: bool = False
    ):
        self._data: pd.DataFrame = reduce_precision(data)
        if track_binding_sequence and "binding_order" not in self._data:
            self._data.loc[:,"binding_order"] = [BindingOrder([]) for _ in range(self._data.shape[0])]

        self.n_total_pmappings: float = n_total_pmappings
        self.n_valid_pmappings: float = n_valid_pmappings
        self.drop_valid_reservations: bool = drop_valid_reservations
        self.excess_resource_tolerance: float = excess_resource_tolerance
        self.n_concurrent_threads = n_concurrent_threads
        self.track_binding_sequence = track_binding_sequence

        self._assert_no_duplicate_cols()

        if next_shared_loop_index is not None:
            assert (
                ignored_resources is not None
            ), "ignored_resources must be set if next_shared_loop_index is set"
            self.free_to_loop_index(next_shared_loop_index)
            self.limit_capacity(
                next_shared_loop_index=next_shared_loop_index,
                ignored_resources=ignored_resources,
            )

        if fill_reservation_cols:  # Affects PmappingDataframe so must go before
            self.fill_reservation_cols(fill_reservation_cols)
        if check_above_subset_below:
            self.check_above_subset_below()
        if max_right_to_left:  # Affects PmappingDataframe so must go before
            self.max_right_to_left()
        if check_above_subset_below:
            self.check_above_subset_below()

        if not skip_pareto:
            self.make_pareto()

        if check_above_subset_below:
            self.check_above_subset_below()

        self.ignored_resources = ignored_resources

        if create_live_reservation_from_compatibility is not None:
            self._create_live_reservation_from_compatibility(
                create_live_reservation_from_compatibility
            )

        assert len(self.data.columns) == len(
            oset(self.data.columns)
        ), f"Duplicate columns: {self.data.columns}"

    def rename(self, renames: dict[str, str]) -> "PmappingDataframe":
        new = self.copy()
        new.data.rename(columns=renames, inplace=True)
        return new

    @error_check_wrapper
    def fill_reservation_cols(self, columns: set | str):
        l_reservations, r_reservations = self._make_reservations()
        targets = []
        if columns == "auto":
            for left, reservations_dict in [
                (True, l_reservations),
                (False, r_reservations),
            ]:
                for resource, reservations in reservations_dict.items():
                    for r in sorted(reservations):
                        above = get_reservation_or_parent(
                            resource, r - 1, l_reservations, r_reservations
                        )
                        if above is not None:
                            below = reservation2col(resource, r, left=left)
                            targets.append((r, above, below))
        else:
            for below in columns:
                if (name_nloops := col2reservation(below)) is None:
                    raise ValueError(f"{below} is not a valid reservation column")
                name, nloops = name_nloops.name, name_nloops.nloops
                above = get_reservation_or_parent(
                    name, nloops - 1, l_reservations, r_reservations
                )
                if above is not None:
                    targets.append((nloops, above, below))

        # Sort so we go from top to bottom. Needed in case we have to max 0->1
        # then 1->2
        for _, above, below in sorted(targets, key=lambda x: x[0]):
            assert (
                above in self.data.columns
            ), f"Missing column {above}. Have columns:\n\t" + "\n\t".join(
                list(self.data.columns)
            )
            assert (
                below in self.data.columns
            ), f"Missing column {below}. Have columns:\n\t" + "\n\t".join(
                list(self.data.columns)
            )
            max_to_col(self.data, below, above)

    @error_check_wrapper
    def max_right_to_left(self):
        l_reservations, r_reservations = self._make_reservations()
        for resource, reservations in l_reservations.items():
            for r in reservations:
                if r in r_reservations.get(resource, oset()):
                    source = reservation2col(resource, r)
                    target = reservation2col(resource, r, left=True)
                    max_to_col(self.data, target, source)

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @error_check_wrapper
    def _make_reservations(self) -> tuple[dict[str, set[int]], dict[str, set[int]]]:
        """
        Create a dictionary of reservations for each resource.
        The dictionary keys are the resource names and the values are lists
        of column names for each loop index.
        """
        l_reservations, r_reservations = {}, {}
        for c in self.data.columns:
            if (key := col2reservation(c)) is not None:
                assert key.nloops >= -1
                if key.is_left:
                    l_reservations.setdefault(key.name, oset()).add(key.nloops)
                else:
                    r_reservations.setdefault(key.name, oset()).add(key.nloops)
        return l_reservations, r_reservations

    def clear_fused_loop_symbols(self):
        dropcols = [c for c in self.data.columns if is_fused_loop_col(c)]
        if not dropcols:
            return
        self.data.drop(columns=dropcols, inplace=True)
        self.make_pareto()

    @error_check_wrapper
    def free_to_loop_index(self, loop_index: int) -> bool:
        """
           A  B
            / | --- 0
           C  D
            / | --- 1  < Shared Loop Index
           E  F
            / | --- 2
           G  H
        ->
           A  B
            / | --- 0
           C  D
              | --- 1  < Shared Loop Index
          max(E,G,H)
        We skip incorporating E into the max because its reservations are
        already incorporated into F and G.
        """
        if loop_index < -1:
            raise ValueError("loop_index must be >= -1")

        if self.get_max_loop_index() <= loop_index:
            return False

        if not self._has_left_reservations():
            self.move_reservations_to_index(loop_index)
            return True

        updated = False
        while self.get_max_loop_index() > loop_index:
            updated = True
            if self._has_bottom_right():
                self.shift_bottom_reservation_left()
            self.consolidate_bottom_split()

        assert self._has_bottom_right()
        return updated


    def move_reservations_to_index(self, loop_index):
        for i in sorted(range(loop_index, self.get_max_loop_index())):
            for col in get_reservation_cols_with(self.data, nloops=i, is_left=False):
                key = col2reservation(col)
                target = reservation2col(key.name, loop_index)
                max_to_col(self.data, target, col)


    @error_check_wrapper
    def get_reservation_or_parent(
        self,
        name: str,
        level: int,
        l_reservations: dict[str, set[int]],
        r_reservations: dict[str, set[int]],
        left: bool = False,
        return_name_level_left: bool = False,
    ) -> str | tuple[str, int, bool] | None:
        reservations = l_reservations if left else r_reservations
        if (reservations := reservations.get(name, None)) is not None:
            while level >= -1:
                if level in reservations:
                    if return_name_level_left:
                        return name, level, left
                    return reservation2col(name, level, left)
                # The parent of left nodes are right nodes, so if we don't find a
                # left node immediately then we're back on the right nodes
                reservations = r_reservations.get(name, set())
                left = False
                level -= 1
        return None


    @error_check_wrapper
    def consolidate_bottom_split(self):
        """
        Consolidate bottom split.
        Example:
            Before:                After:
            A  B                   A B
             / | --- 0             / | --- 0
            C  D                  C  E+F-D
             / | --- 1
            F  E
        """
        assert not self._has_bottom_right()
        bottom_index = self.get_max_loop_index()
        target_index = bottom_index-1
        l_reservations, r_reservations = self._make_reservations()
        for resource in set(l_reservations) | set(r_reservations):
            target = reservation2col(resource, target_index, left=False)
            if resource not in l_reservations or bottom_index not in l_reservations[resource]:
                continue
            left_reservation_cols = set(get_reservation_cols_with(
                self.data,
                resource,
                bottom_index,
                is_left=True
            ))
            for left_reservation_col in left_reservation_cols:
                add_to_col(self.data, target, left_reservation_col)
            # Reservation above is included in the left, so we deduplicate
            reservation_above = self.get_reservation_or_parent(
                resource,
                bottom_index,
                l_reservations,
                r_reservations
            )
            if reservation_above:
                assert target_index >= col2reservation(reservation_above).nloops
                self.data.loc[:,target] -= self.data[reservation_above]*len(left_reservation_cols)
        drop_columns = [
            c for c in get_reservation_cols_with(self.data)
            if col2reservation(c).nloops > target_index
        ]

        self.data["Total<SEP>latency"] = 0
        for thread_i in range(self.n_concurrent_threads):
            max_to_col(self.data, "Total<SEP>latency", f"Total<SEP>latency<SEP>{bottom_index}<SEP>{thread_i}")
            drop_columns.append(f"Total<SEP>latency<SEP>{bottom_index}<SEP>{thread_i}")
        self.data.drop(columns=drop_columns, inplace=True)

        assert self._has_bottom_right()

    @error_check_wrapper
    def shift_bottom_reservation_left(self):
        """
        Shifts the bottom reservation from right to left.
        Example:
            Before:                After:
            A  B                   A  B
             / | --- 0             / | --- 0
            C  D                  C  D
               | --- 1             /   --- 1
               E                  E
        """
        if not self._has_bottom_right():
            return

        bottom_loop_index = self.get_max_loop_index()
        _l_reservations, r_reservations = self._make_reservations()

        # Explore placing right branch in any of the left threads
        left_concurrent_threads = list(range(self.n_concurrent_threads))
        assert len(left_concurrent_threads) > 0
        all_data = []
        for thread_i in left_concurrent_threads:
            df = self.data.copy()
            for resource in r_reservations:
                if bottom_loop_index not in r_reservations[resource]:
                    continue
                right_reservation = reservation2col(resource, bottom_loop_index)
                left_reservation = reservation2col(resource, bottom_loop_index, True, thread_i)

                for live_tensor in get_live_reservation_cols_with(
                    df,
                    resource=resource,
                    nloops=bottom_loop_index,
                    thread=thread_i
                ):
                    add_to_col(df, right_reservation, live_tensor)

                for live_tensor_in_right in get_live_reservation_cols_with(
                    df,
                    resource=resource,
                    nloops=bottom_loop_index,
                    thread=DEFAULT_THREAD,
                ):
                    right_key = col2live_reservation(live_tensor_in_right)
                    new_live_tensor = live_reservation2col(
                        resource,
                        right_key.tensor,
                        bottom_loop_index,
                        thread_i
                    )
                    add_to_col(df, new_live_tensor, live_tensor_in_right)
                    df.drop(columns=[live_tensor_in_right])

                for thread_j in left_concurrent_threads:
                    key = reservation2col(resource,bottom_loop_index,True,thread_j)
                    if key not in df:
                        df.loc[:,key] = 0

                if left_reservation in df:
                    max_to_col(df, left_reservation, right_reservation)
                    df.drop(columns=[right_reservation], inplace=True)
                else:
                    df.rename(columns={right_reservation: left_reservation}, inplace=True)

            for thread_j in left_concurrent_threads:
                key = f"Total<SEP>latency<SEP>{bottom_loop_index}<SEP>{thread_j}"
                if key not in df:
                    df.loc[:,key] = 0
            add_to_col(df, f"Total<SEP>latency<SEP>{bottom_loop_index}<SEP>{thread_i}", "Total<SEP>latency")
            df.drop(columns=["Total<SEP>latency"], inplace=True)

            if self.track_binding_sequence:
                df["binding_order"] += BindingOrder([thread_i])
            assert not set(get_reservation_cols_with(
                df,
                name=None,
                nloops=bottom_loop_index,
                is_left=False
            ))
            assert len(list(_get_duplicates(df.columns))) == 0
            all_data.append(df)

        assert len(all_data) > 0
        if len(all_data) == 1:
            self._data = all_data[0]
        else:
            self._data = pd.concat(all_data, ignore_index=True)

        assert not self._has_bottom_right()


    @staticmethod
    def _get_target_path(suffix: str = None) -> str:
        import os

        f = "./images"
        os.makedirs(f, exist_ok=True)
        suffix = "" if suffix is None else f".{suffix}"
        i = 0
        while os.path.exists(os.path.join(f, f"test_{i}{suffix}.png")):
            i += 1
        return os.path.join(f, f"test_{i}{suffix}.png")

    def get_max_loop_index(self):
        l_reservations, r_reservations = self._make_reservations()
        return max(
            max(
                (max(r, default=-1) for r in r_reservations.values()),
                default=-1,
            ),
            max(
                (max(r, default=-1) for r in l_reservations.values()),
                default=-1,
            ),
        )

    def get_min_loop_index(self):
        l_reservations, r_reservations = self._make_reservations()
        return min(
            min(
                (min(r, default=1000000) for r in r_reservations.values()),
                default=1000000,
            ),
            min(
                (min(r, default=1000000) for r in l_reservations.values()),
                default=1000000,
            ),
        )

    @error_check_wrapper
    def merge_next(
        self,
        right: "PmappingDataframe",
        duplicated_aliased_tensors: set[TensorReservation],
        compatibility_left: Compatibility,
        compatibility_right: Compatibility,
        compatibility_joined: Compatibility,
        ignored_resources: set[str],
        _pmapping_row_filter_function: Callable[[pd.Series], bool] | None = None,
        _force_allow_invalid_only_for_runtime_test: bool = False,
        _is_invalid: bool = False,
    ) -> "PmappingDataframe":
        """
           A  B            A2
            / | --- 0      |
           C  D            C2
              | --- 1      |     < Shared Loop Index
              E            E2
                           |
                           F2
           ->
           A  A+A2
            / | --- 0
        C+A2  C+C2
            / | --- 1  < Shared Loop Index
        E+C2  E2+D
              |
              F2+D
        """
        live_tensors = compatibility_joined.tensor_names
        shared_loop_index = compatibility_left.n_loops - 1
        next_shared_loop_index = compatibility_joined.n_loops - 1

        self.check_live_reservations(compatibility_left)
        self._remove_dead_reservations(compatibility_joined)

        assert compatibility_left.n_loops <= compatibility_right.n_loops
        if self._has_bottom_right():
            assert "Total<SEP>latency" in self.data
            self.shift_bottom_reservation_left()

        shared_tensor_names = (
            compatibility_left.tensor_names & compatibility_right.tensor_names
        )
        left_match, right_match = [], []
        make_empty_result = False

        def check_match(la: Loop, lb: Loop, param: str):
            a, b = getattr(la.tile_pattern, param), getattr(lb.tile_pattern, param)
            if isinstance(a, str) or isinstance(b, str):
                left_match.append(a)
                right_match.append(b)
            elif a != b:
                raise ValueError(f"Mismatch in {param}: {a} != {b}")

        try:
            for s in shared_tensor_names:
                ta = compatibility_left.get_reservation_of_tensor(s)
                tb = compatibility_right.get_reservation_of_tensor(s)
                for la, lb in zip(ta.loops, tb.loops):
                    check_match(la, lb, "initial_tile_shape")
                    check_match(la, lb, "tile_shape")

            for la, lb in zip(compatibility_left.loops, compatibility_right.loops):
                check_match(la, lb, "calculated_n_iterations")

        except ValueError as e:
            make_empty_result = True

        right_df_l_reservations, right_df_r_reservations = right._make_reservations()
        assert not right_df_l_reservations, f"{right_df_l_reservations} is not None"

        l_reservations, r_reservations = self._make_reservations()

        for resource, reservations in r_reservations.items():
            n_reservations = max(reservations, default=-1)
            assert (
                n_reservations <= shared_loop_index
            ), f"{resource}: {reservations} > {shared_loop_index}"

        for resource, reservations in l_reservations.items():
            n_reservations = max(reservations, default=-1)
            assert (
                n_reservations <= shared_loop_index + 1
            ), f"{resource}: {reservations} > {shared_loop_index}"

        max_nloops = max(
            shared_loop_index, self.get_max_loop_index(), right.get_max_loop_index()
        )
        min_nloops = min(self.get_min_loop_index(), right.get_min_loop_index())

        sd, rd = self.data, right.data
        if make_empty_result:
            sd = sd.iloc[0:0]
            rd = rd.iloc[0:0]

        # _force_allow_invalid_only_for_runtime_test -> only merge matched ones
        if left_match and not _force_allow_invalid_only_for_runtime_test:
            df = pd.merge(
                sd,
                rd,
                how="inner",
                left_on=left_match,
                right_on=right_match,
                suffixes=["", "_RIGHT_MERGE"],
            )
        else:
            df = pd.merge(sd, rd, how="cross", suffixes=["", "_RIGHT_MERGE"])

        df = reduce_precision(df)

        # Drop all fused loop columns that are not used anymore
        remaining_symbols = compatibility_joined.symbols()
        dropcols = [
            c for c in df.columns if is_fused_loop_col(c) and c not in remaining_symbols
        ]
        df = df.drop(columns=dropcols)

        # Number of combinations
        n_total_pmappings = self.n_total_pmappings * right.n_total_pmappings
        n_valid_pmappings = self.n_valid_pmappings * right.n_valid_pmappings
        scale_by = len(df) / max(1, len(self.data) * len(right.data))
        n_total_pmappings *= scale_by
        n_valid_pmappings *= scale_by

        # Calculate the merged reservations of both trees.
        # Reservations of shared tensors are going to be doubly-counted.
        # We free these reservations later.

        # Make sure everything is done in increasing loop order so we don't have
        # read-after-write hazards
        for nloops in range(max_nloops, min_nloops - 1, -1):

            def iter_reservations(reservations_dict):
                for resource in reservations_dict:
                    if nloops in reservations_dict[resource]:
                        yield resource

            # For the RIGHT tree, RIGHT reservations: If there is no matching node in
            # the left tree, add the above-this-level reservation from the left tree. If
            # there is a matching node in the left tree, then we'll add this node to it
            # in the next step.
            for resource in iter_reservations(right_df_r_reservations):
                if (
                    source := get_reservation_or_parent(
                        resource, nloops - 1, l_reservations, r_reservations
                    )
                ) is None:
                    continue
                target = reservation2col(resource, nloops)
                # If there's a merged version column, then it's in both trees
                if target + "_RIGHT_MERGE" in df:
                    continue
                add_to_col(df, target, source)
            # For LEFT tree, LEFT reservations: Add the immediately-above
            # reservation from the right tree.
            for resource in iter_reservations(l_reservations):
                if (
                    source := get_reservation_or_parent(
                        resource,
                        nloops - 1,
                        right_df_l_reservations,
                        right_df_r_reservations,
                    )
                ) is None:
                    continue
                if source not in df:
                    source += "_RIGHT_MERGE"
                    assert source in df
                for target in get_reservation_cols_with(
                    df,
                    name=resource,
                    nloops=nloops,
                    is_left=True,
                    filter=lambda col: "RIGHT_MERGE" not in col,
                ):
                    add_to_col(df, target, source)
            # For LEFT tree, RIGHT reservations: Add the same-level reservation from the
            # right tree.
            for resource in iter_reservations(r_reservations):
                if (
                    source := get_reservation_or_parent(
                        resource,
                        nloops,
                        right_df_l_reservations,
                        right_df_r_reservations,
                    )
                ) is None:
                    continue
                if source not in df:
                    source += "_RIGHT_MERGE"
                    assert source in df
                target = reservation2col(resource, nloops)
                add_to_col(df, target, source)

        # For everything else: Simple add
        dropcols = [c for c in df.columns if c.endswith("_RIGHT_MERGE")]
        for source in dropcols:
            target = source[: -len("_RIGHT_MERGE")]
            if is_tensor_col(target):
                continue
            if "Total<SEP>latency" in target:
                continue
            if not col_used_in_pareto(target):
                raise ValueError(f"{target} is not used in pareto")
            if col2reservation(target) is None:
                add_to_col(df, target, source)

        df = df.drop(columns=dropcols)
        result = PmappingDataframe(
            df,
            skip_pareto=True,
            check_above_subset_below=False,
            n_total_pmappings=n_total_pmappings,
            n_valid_pmappings=n_valid_pmappings,
            ignored_resources=self.ignored_resources,
            drop_valid_reservations=self.drop_valid_reservations,
            n_concurrent_threads=self.n_concurrent_threads,
        )

        result._remove_dead_reservations(compatibility_joined)

        doubly_counted_reservations = _get_doubly_counted_reservations(
            compatibility_left.tensors,
            compatibility_right.tensors,
            duplicated_aliased_tensors,
            shared_loop_index,
        )
        for resource, tensor_index_set in doubly_counted_reservations.items():
            result.free_reservations_of_resource(resource, tensor_index_set)

        if CHECK_CORRECTNESS:
            result.check_above_subset_below(live_tensors)
            result.check_reservations(live_tensors)

        assert result._has_bottom_right()
        result.free_to_loop_index(next_shared_loop_index)

        if not CHECK_CORRECTNESS:
            result.limit_capacity(
                next_shared_loop_index, ignored_resources=ignored_resources
            )
        # result.max_right_to_left()
        if _pmapping_row_filter_function is not None:
            result = result.filter_rows(_pmapping_row_filter_function)
        result.make_pareto()

        result.check_live_reservations(compatibility_joined)

        # This join was invalid (we only ran it for runtime measurement). Clear all the
        # mappings.
        if _is_invalid:
            result._data = result.data.iloc[0:0]

        # If we're running _force_allow_invalid_only_for_runtime_test, we don't want to
        # have all those invalid combinations, so return the actual merge result.
        elif _force_allow_invalid_only_for_runtime_test:
            return self.merge_next(
                right,
                duplicated_aliased_tensors,
                compatibility_left,
                compatibility_right,
                compatibility_joined,
                ignored_resources,
                _pmapping_row_filter_function,
                _force_allow_invalid_only_for_runtime_test=False,
            )

        return result

    @error_check_wrapper
    def free_reservations_of_resource(
        self,
        resource: str,
        tensor_indices_to_free: Iterable[tuple[str, int]],
    ):
        """
        For every `(tensor, nloops)` in `tensor_indices_to_free`, reduces all 
        right reservations of `resource` at `index >= nloops` and left reservations
        at `index > nloops` by the size of the tensor as recorded in
        `self.data[tensor2col(tensor)]`.
        """
        targets = defaultdict(int)
        for tensor_name, to_free_nloops in tensor_indices_to_free:
            size = self.data[tensor2col(tensor_name)]
            for col in get_reservation_cols_with(self.data, name=resource):
                key = col2reservation(col)
                if (
                    (key.is_left and key.nloops > to_free_nloops)
                    or
                    (key.is_right and key.nloops >= to_free_nloops)
                ):
                    targets[key.nloops, col] -= size

        # Now apply the allocations. Sort so we go from top to bottom in case
        # there are maxes that propagate down.
        for (_, target), size in sorted(
            targets.items(), key=lambda x: x[0], reverse=True
        ):
            assert target in self.data
            add_to_col(self.data, target, size)
            # Assert all reservations are >= 0
            assert (self.data[target] >= 0).all(), f"Negative reservation: {target}"

    @staticmethod
    def concat(
        paretos: list["PmappingDataframe"], skip_pareto: bool = False
    ) -> "PmappingDataframe":
        if len(paretos) == 0:
            raise ValueError("No paretos to concatenate")
        if len(paretos) == 1:
            return paretos[0]

        required_cols = oset.union(*[oset(p.data.columns) for p in paretos])
        shared_cols = oset.intersection(*[oset(p.data.columns) for p in paretos])
        reservation_cols_to_fill = [
            c for c in required_cols - shared_cols
            if col_used_in_pareto(c) and col2reservation(c)
        ]

        concatenated = pd.concat([p.data for p in paretos]).reset_index(drop=True)

        p = PmappingDataframe(
            _fillna_and__numeric_cast(concatenated, 0),
            skip_pareto=len(paretos) == 1 or skip_pareto,
            fill_reservation_cols=reservation_cols_to_fill,
            n_total_pmappings=sum(p.n_total_pmappings for p in paretos),
            n_valid_pmappings=sum(p.n_valid_pmappings for p in paretos),
            ignored_resources=next(iter(paretos)).ignored_resources,
            drop_valid_reservations=next(iter(paretos)).drop_valid_reservations,
            n_concurrent_threads=next(iter(paretos)).n_concurrent_threads,
        )
        return p

    def update(
        self,
        skip_pareto: bool,
        **kwargs,
    ) -> "PmappingDataframe":
        args = dict(
            data=self.data,
            skip_pareto=skip_pareto,
            check_above_subset_below=False,
            n_total_pmappings=self.n_total_pmappings,
            n_valid_pmappings=self.n_valid_pmappings,
            ignored_resources=self.ignored_resources,
            drop_valid_reservations=self.drop_valid_reservations,
            n_concurrent_threads=self.n_concurrent_threads,
        )
        args.update(kwargs)
        return PmappingDataframe(**args)

    def copy(self, copy_df: bool = True) -> "PmappingDataframe":
        return self.update(
            data=self.data.copy() if copy_df else self.data,
            skip_pareto=True,
            check_above_subset_below=False,
        )

    def limit_capacity(
        self,
        next_shared_loop_index: int = None,
        ignored_resources: set[str] = oset(),
        finished: bool = False,
    ):
        dropcols = []
        l_reservations, r_reservations = self._make_reservations()
        tolerance = self.excess_resource_tolerance
        for resource in sorted(oset(r_reservations) | oset(l_reservations)):
            # Right reservations: Only check the greatest-index level. If a loop
            # is 0 and the next shared loop index is -1, then we can drop the
            # column.
            right_loops = r_reservations.get(resource, oset())
            for l in list(right_loops):
                col = reservation2col(resource, l)
                if (
                    DEBUG_PRINT_NO_VALID
                    and sum(self.data[col] <= 1 + tolerance) == 0
                    and len(self.data) == 1
                    and tolerance == 0
                ):
                    print(
                        f"Resource {resource} has no valid reservations. Failed for {col}: {next(iter(self.data[col]))} <= {1 + tolerance}: {next(iter(self.data[col])) <= 1 + tolerance}"
                    )
                    for col2 in self.data.columns:
                        print(f"{col2}: {list[Any](self.data[col2])}")
                self._data = self.data[self.data[col] <= 1 + tolerance]
                if (
                    l == 0
                    and next_shared_loop_index == -1
                    # CAN'T DROP RESERVATIONS UNTIL WE'RE FINISHED JOINING. Persistent
                    # tensors may get saved later and would live at the same time as
                    # these reservations.
                    and finished
                    and self.drop_valid_reservations
                    and resource not in ignored_resources
                    and (tolerance == 0 or not any(self.data[col] > 1))
                ):
                    right_loops.discard(l)
                    dropcols.append(col)

            # Left reservations: Check all levels. If a loop is 0,
            # then we can drop the column.
            left_loops = l_reservations.get(resource, oset())
            for l in list(left_loops):
                for col in get_reservation_cols_with(self.data, name=resource, nloops=l, is_left=True):
                    if (
                        DEBUG_PRINT_NO_VALID
                        and sum(self.data[col] <= 1 + tolerance) == 0
                        and len(self.data) == 1
                        and tolerance == 0
                    ):
                        print(
                            f"Resource {resource} has no valid reservations. Failed for {col}: {next(iter(self.data[col]))} <= {1 + tolerance}: {next(iter(self.data[col])) <= 1 + tolerance}"
                        )
                        for col2 in self.data.columns:
                            print(f"{col2}: {list[Any](self.data[col2])}")
                    self._data = self.data[self.data[col] <= 1 + tolerance]
                    if (
                        l == 0
                        # CAN'T DROP RESERVATIONS UNTIL WE'RE FINISHED JOINING. Persistent
                        # tensors may get saved later.
                        and finished
                        and self.drop_valid_reservations
                        and resource not in ignored_resources
                        and (tolerance == 0 or not any(self.data[col] > 1))
                    ):
                        left_loops.discard(l)
                        dropcols.append(col)

        self._data = self.data.drop(columns=dropcols)

    def make_pareto(
        self,
        columns: list[str] = None,
        objective_tolerance: float = 0,
        resource_usage_tolerance: float = 0,
        absolute_resource_usage_tolerance: float = 0,
    ):
        # The error for absolute_resource_usage_tolerance sums each time we modify the
        # df and prune, so if we use it more, we need to use a lower threshold. The
        # max_n_einsums value assumes that absolute_resource_usage_tolerance is only
        # used for joining.
        if self.drop_valid_reservations:
            resource_usage_tolerance = objective_tolerance

        self._data = makepareto(
            self.data,
            columns,
            resource_usage_tolerance=resource_usage_tolerance,
            absolute_resource_usage_tolerance=absolute_resource_usage_tolerance,
            objective_tolerance=objective_tolerance,
        )

    def has_reservations(self):
        return any(col2reservation(c) is not None for c in self.data.columns)

    # ============================================================================
    # Helper functions
    # ============================================================================
    def _remove_dead_reservations(self, compatibility: Compatibility):
        live_tensors = oset(tensor.name for tensor in compatibility.tensors)
        dropcols = []
        for col in get_live_reservation_cols_with(self.data):
            key = col2live_reservation(col)
            if key.tensor not in live_tensors:
                dropcols.append(col)
        self.data.drop(columns=dropcols)

    def _create_live_reservation_from_compatibility(self, compatibility: Compatibility):
        for tensor in compatibility.tensors:
            col = live_reservation2col(
                tensor.resource_name,
                tensor.name,
                tensor.above_loop_index
            )
            add_to_col(self.data, col, tensor2col(tensor.name))

    def _get_split_above(self, idx: int):
        """
        Return the first index < `idx` which has both a left and right reservation.
        """
        return max([i for i in self._indices_with_split() if i < idx], default=-1)

    def _indices_with_split(self) -> set[int]:
        return {
            col2reservation(c).nloops
            for c in (
                set(get_reservation_cols_with(self.data, is_left=False))
                &
                set(get_reservation_cols_with(self.data, is_left=True))
            )
        }

    def _has_bottom_right(self):
        """Whether the pmapping has a RIGHT branch at the bottom."""
        if self._has_right_latency():
            assert self._has_bottom_right_reservations() or not self._has_right_reservations()
            return True
        assert not self._has_bottom_right_reservations()
        return False

    def _has_bottom_left(self):
        bottom_index = self.get_max_loop_index()
        return (
            len(list(get_reservation_cols_with(self.data, nloops=bottom_index, is_left=True)))
            >
            0
        )

    def _has_left_reservations(self):
        return len(list(get_reservation_cols_with(self.data, is_left=True))) > 0

    def _has_right_reservations(self):
        return (
            len(list(get_reservation_cols_with(self.data, is_left=False)))
            >
            0
        )

    def _has_bottom_right_reservations(self):
        bottom_index = self.get_max_loop_index()
        return (
            len(list(get_reservation_cols_with(self.data, nloops=bottom_index, is_left=False)))
            >
            0
        )

    def _has_right_latency(self):
        return "Total<SEP>latency" in self.data

    # ============================================================================
    # Checking functions
    # ============================================================================
    def check_above_subset_below(self, live_tensors: set[str] = fzs()):
        assert not self.data.isnull().values.any(), f"NaN in {self.data}"
        targets = []
        l_reservations, r_reservations = self._make_reservations()
        for left, reservations_dict in [
            (True, l_reservations),
            (False, r_reservations),
        ]:
            for resource, reservations in reservations_dict.items():
                for r in reservations:
                    above = get_reservation_or_parent(
                        resource, r - 1, l_reservations, r_reservations
                    )
                    if above is not None:
                        below = reservation2col(resource, r, left=left)
                        targets.append((above, below))

        for above, below in targets:
            if (self.data[below] < self.data[above]).any():
                first_failing_index = (self.data[below] < self.data[above]).idxmax()
                fail_row = self.data.iloc[first_failing_index]
                error = f"""
                {below} column is less than {above} column. A reservation at
                a level should include all reservations above it. There were {len(fail_row)} rows
                with this error. One example: {fail_row}
                """
                self.fail(first_failing_index, live_tensors)
                raise ValueError(error)

    def filter_rows(
        self, _pmapping_row_filter_function: Callable[[pd.Series], bool] | None = None
    ) -> "PmappingDataframe":
        if _pmapping_row_filter_function is None:
            return self.copy()

        # s = _pmapping_row_filter_function(self._data)
        # if s.sum() > 0:
        #     print(f"Filter rate: {s.sum() / len(s):.2%}")
        return self.update(
            data=self._data[_pmapping_row_filter_function(self._data)].copy(),
            skip_pareto=True,
        )

    def __len__(self) -> int:
        return len(self._data)

    def _assert_no_duplicate_cols(self):
        if len(list(_get_duplicates(self._data.columns))) > 0:
            raise ValueError(
                "Some columns in PmappingDataframe are duplicated:\n"
                +
                " " + list(_get_duplicates(self._data.columns))
            )

    def check_live_reservations(self, compatibility: Compatibility):
        for tensor in compatibility.tensors:
            if not contains_live_reservation(
                self.data,
                resource=tensor.resource_name,
                tensor=tensor.name,
                nloops=tensor.above_loop_index,
            ):
                colnames = ""
                for c in self.data.columns:
                    colnames += f"  {c}\n"
                raise RuntimeError(f"missing live reservation for {tensor}. columns:\n" + colnames)

    # @error_check_wrapper
    # def check_reservations(self, live_tensors: set[int]):
    #     from accelforge.visualization.reservationtree import mappings2reservationtree
    #     assert not self.data.isnull().values.any(), f"NaN in {self.data}"

    #     self = self.copy()

    #     self.free_to_loop_index(-1)
    #     self.shift_bottom_reservation_left()

    #     for i, r in self.data.iterrows():
    #         looptree = mappings2reservationtree(
    #             r[MAPPING_COLUMN],
    #             r.get(STATS, None),
    #             still_live_tensors=live_tensors
    #         )
    #         reservations = dict(looptree.get_reservations())

    #         # If r doesn't have any columns, continue. It's a copy Einsum so it has no
    #         # stats.
    #         if r.empty:
    #             continue

    #         for k, v in reservations.items():
    #             col = get_reservation_or_parent(k, 0, left=True)
    #             if str(k) == "0":
    #                 continue
    #             if col not in self.data.columns:
    #                 got = r[[c for c in self.data.columns if col2reservation(c) is not None]]
    #                 self.fail(i, live_tensors)
    #                 raise ValueError(f"Missing {k}: Expected {reservations}. Got: {got}")
    #             if r[col] != v:
    #                 got = r[[c for c in self.data.columns if col2reservation(c) is not None]]
    #                 self.fail(i, live_tensors)
    #                 looptree = mappings2reservationtree(
    #                     r[MAPPING_COLUMN],
    #                     r.get(STATS, None),
    #                     # skip_backing_tensors_in_right_branch=live_tensors,
    #                     still_live_tensors=live_tensors,
    #                 )
    #                 raise ValueError(
    #                     f"Mismatched {k}: {v} != {r[col]}. Expected {reservations}. Got: {got}"
    #                 )

    # def fail(self, index, live_tensors):
    #     from accelforge.mapper.FFM._join_pmappings.pmapping_group import TensorReservation
    #     r = self.data.iloc[index]
    #     assert not self.data.isnull().values.any(), f"NaN in {self.data}"
    #     self = self.copy()
    #     self._draw_index(index, live_tensors, self._get_target_path(suffix="fail"))
    #     all_tensors = oset(t for tn in r[MAPPING_COLUMN].values() for t in tn.tensors)
    #     all_tensors = TensorReservation.get_backing_tensors(all_tensors)
    #     for t in sorted(all_tensors):
    #         print(f"{t.__repr__()},")

    # def _draw_index(self, index: int, live_tensors, to_file: str = "test.png"):
    #     from accelforge.visualization.reservationtree import mappings2reservationtree
    #     import pydot
    #     looptree = mappings2reservationtree(
    #         self.data.iloc[index][MAPPING_COLUMN],
    #         self.data.iloc[index].get(STATS, None),
    #         still_live_tensors=live_tensors,
    #     )
    #     graph = pydot.Dot(graph_type="digraph", ranksep="0.2", nodesep="0.2")
    #     looptree.to_pydot(graph)
    #     row = self.data.iloc[index]
    #     all_data = sorted(f"{k}: {v}" for k, v in row.items() if k not in DICT_COLUMNS)
    #     data_str = "\n".join(all_data)
    #     graph.add_node(pydot.Node("data", label=data_str, shape="plaintext"))
    #     with open(to_file, "wb") as f:
    #         f.write(graph.create_png())

    def clear_irrelevant_columns(
        self, compatibility: Compatibility
    ) -> "PmappingDataframe":
        return self.update(
            data=compatibility.clear_unrelated_columns(self._data),
            skip_pareto=True,
        )


def row2pmappings(
    row: pd.Series,
    einsum_names: list[str],
    rank_variable_bounds: dict[str, dict[str, int]],
) -> list[Nested]:
    pmappings: list[Nested] = []
    for einsum_name in einsum_names:
        pmapping: Nested = copy.deepcopy(row[f"{einsum_name}<SEP>{MAPPING_COLUMN}"])
        for node in pmapping.nodes:

            def acc(s: str | None | int):
                s = s.name if isinstance(s, sympy.Symbol) else s
                return row[f"{einsum_name}<SEP>{s}"] if isinstance(s, str) else s

            if isinstance(node, MappingLoop):
                tp: TilePattern = node.tile_pattern
                node.tile_pattern = tp.update(
                    initial_tile_shape=acc(tp.initial_tile_shape),
                    tile_shape=acc(tp.tile_shape),
                )
        pmappings.append(pmapping)
        pmapping._beautify_loops(rank_variable_bounds)
    return pmappings


def _get_duplicates(items: Iterable):
    count = defaultdict(lambda: 0)
    for i in items:
        count[i] += 1
    for key, value in count.items():
        if value > 1:
            yield key


class BindingOrder:
    """Lightweight class to hold the binding choices during merging."""
    def __init__(self, choices: list[int]):
        self._sequence = choices

    def __add__(self, new_sequence: "BindingOrder"):
        assert isinstance(new_sequence, BindingOrder)
        return BindingOrder(self._sequence + new_sequence._sequence)

    def __iadd__(self, new_sequence: "BindingOrder"):
        # copy-on-write: create a new copy of _sequence, which may be shared
        # originally
        assert isinstance(new_sequence, BindingOrder)
        self._sequence = list(self._sequence)
        self._sequence.extend(new_sequence._sequence)


def _get_doubly_counted_reservations(
    left_tensor_reservations: oset[TensorReservation],
    right_tensor_reservations: oset[TensorReservation],
    duplicated_aliased_tensors: oset[TensorReservation],
    shared_loop_index,
):
    doubly_counted_reservations = {}
    shared_reservations = left_tensor_reservations & right_tensor_reservations
    for s in shared_reservations | duplicated_aliased_tensors:
        if s.above_loop_index > shared_loop_index:
            continue
        tensor_index_set = doubly_counted_reservations.get(s.resource_name, oset())
        tensor_index_set.add((s.name, s.above_loop_index))
        doubly_counted_reservations[s.resource_name] = tensor_index_set
    return doubly_counted_reservations