from collections import defaultdict
import copy
import functools
import itertools

from typing import Callable, Iterable, Optional

import sympy

from accelforge.frontend.mapping import Nested, TilePattern
from accelforge.frontend.mapping import Loop as MappingLoop
from accelforge.mapper.FFM._join_pmappings.compatibility import (
    Compatibility,
    Loop,
    TensorReservation,
)
from accelforge.util._frozenset import fzs

from accelforge._accelerated_imports import pd

from accelforge.mapper.FFM._pareto_df.df_convention import *
from accelforge.mapper.FFM._pareto_df.pareto import makepareto


CHECK_CORRECTNESS = False


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
            live_tensors = set()
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


class PmappingDataframe:
    def __init__(
        self,
        data: pd.DataFrame,
        n_total_pmappings: float,
        n_valid_pmappings: float,
        skip_pareto: bool = False,
        fill_reservation_cols: set | str = fzs(),
        check_above_subset_below: bool = CHECK_CORRECTNESS,
        max_right_to_left: bool = False,
        next_shared_loop_index: int = None,
        parallelize_pareto: bool = False,
        limit_capacity_drop_valid_reservations: bool = True,
        ignored_resources: set[str] = None,
    ):
        self._data: pd.DataFrame = data
        self.right_reservations: dict[set] = None
        self.left_reservations: dict[set] = None
        self._prev_free_to_loop_index = None
        self._parallelize_pareto = parallelize_pareto
        self._make_reservations()
        self.n_total_pmappings: float = n_total_pmappings
        self.n_valid_pmappings: float = n_valid_pmappings

        if next_shared_loop_index is not None:
            assert (
                ignored_resources is not None
            ), "ignored_resources must be set if next_shared_loop_index is set"
            self.free_to_loop_index(loop_index=next_shared_loop_index)
            self.limit_capacity(
                next_shared_loop_index=next_shared_loop_index,
                drop_valid_reservations=limit_capacity_drop_valid_reservations,
                ignored_resources=ignored_resources,
            )
            self._check_reservations()

        if fill_reservation_cols:  # Affects PmappingDataframe so must go before
            self.fill_reservation_cols(fill_reservation_cols)
        if check_above_subset_below:
            self.check_above_subset_below()
        if max_right_to_left:  # Affects PmappingDataframe so must go before
            self.max_right_to_left()
        if check_above_subset_below:
            self.check_above_subset_below()

        if not skip_pareto:
            self.make_pareto(parallelize=parallelize_pareto)

        if check_above_subset_below:
            self.check_above_subset_below()

        self._check_reservations()

        assert len(self.data.columns) == len(
            set(self.data.columns)
        ), f"Duplicate columns: {self.data.columns}"

    def all_reservation_levels(self):
        return set().union(
            set(),
            *self.left_reservations.values(),
            *self.right_reservations.values(),
        )

    def rename(self, renames: dict[str, str]) -> "PmappingDataframe":
        new = self.copy()
        new.data.rename(columns=renames, inplace=True)
        return new

    @error_check_wrapper
    def fill_reservation_cols(self, columns: set | str):
        self._check_reservations()
        targets = []
        if columns == "auto":
            for left, reservations_dict in [
                (True, self.left_reservations),
                (False, self.right_reservations),
            ]:
                for resource, reservations in reservations_dict.items():
                    for r in sorted(reservations):
                        above = self.get_reservation_or_parent(resource, r - 1)
                        if above is not None:
                            below = nameloop2col(resource, r, left=left)
                            targets.append((r, above, below))
        else:
            for below in columns:
                if (name_nloops := col2nameloop(below)) is None:
                    raise ValueError(f"{below} is not a valid reservation column")
                name, nloops = name_nloops
                above = self.get_reservation_or_parent(name, nloops - 1)
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

        self._check_reservations()

    @error_check_wrapper
    def max_right_to_left(self):
        for resource, reservations in self.left_reservations.items():
            for r in reservations:
                if r in self.right_reservations.get(resource, set()):
                    source = nameloop2col(resource, r)
                    target = nameloop2col(resource, r, left=True)
                    max_to_col(self.data, target, source)
        self._make_reservations()
        self._check_reservations()

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @error_check_wrapper
    def _make_reservations(self):
        """
        Create a dictionary of reservations for each resource.
        The dictionary keys are the resource names and the values are lists
        of column names for each loop index.
        """
        self.left_reservations, self.right_reservations = {}, {}
        for c in self.data.columns:
            if (name_nloops := col2nameloop(c)) is not None:
                name, nloops = name_nloops
                target = (
                    self.left_reservations
                    if is_left_col(c)
                    else self.right_reservations
                )
                target.setdefault(name, set()).add(nloops)
                assert nloops >= -1

    def _check_reservations(self):
        prev_left, prev_right = self.left_reservations, self.right_reservations
        self._make_reservations()
        assert (
            self.left_reservations == prev_left
        ), f"Left reservations changed: {self.left_reservations} != {prev_left}"
        assert (
            self.right_reservations == prev_right
        ), f"Right reservations changed: {self.right_reservations} != {prev_right}"

    def clear_fused_loop_symbols(self):
        dropcols = [c for c in self.data.columns if is_fused_loop_col(c)]
        if not dropcols:
            return
        self.data.drop(columns=dropcols, inplace=True)
        self._make_reservations()
        self._check_reservations()
        self.make_pareto()

    @error_check_wrapper
    def free_to_loop_index(
        self,
        loop_index: int,
        live_tensors: set[int] = None,
        check_correctness: bool = CHECK_CORRECTNESS,
    ) -> bool:
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
        if loop_index == self._prev_free_to_loop_index:
            return False
        self._prev_free_to_loop_index = loop_index

        drop_columns = []
        for resource in set(self.left_reservations) | set(self.right_reservations):
            max_columns = []
            left_reservations = self.left_reservations.get(resource, set())
            right_reservations = self.right_reservations.get(resource, set())
            left_big_enough = [l for l in left_reservations if l >= loop_index + 1]
            right_big_enough = [
                r for r in right_reservations if r >= loop_index + 2
            ]  # + 1 is target

            if len(right_big_enough) > 1:  # All ones above the last are subsets
                right_biggest = max(right_big_enough)
                right_big_enough.remove(right_biggest)
                drop_columns += [nameloop2col(resource, r) for r in right_big_enough]
                right_big_enough = [right_biggest]

            max_columns = [nameloop2col(resource, r) for r in right_big_enough] + [
                nameloop2col(resource, l, left=True) for l in left_big_enough
            ]

            if not max_columns:
                continue

            target = nameloop2col(resource, loop_index + 1)
            if target in self.data:
                max_columns.append(target)

            if len(max_columns) == 1:
                self.data.rename(columns={max_columns[0]: target}, inplace=True)
            else:
                for c in max_columns:
                    max_to_col(self.data, target, c)
                drop_columns += [m for m in max_columns if m != target]
        self.data.drop(columns=drop_columns, inplace=True)
        self._make_reservations()

        if check_correctness and live_tensors is not None:
            self.copy().check_reservations(live_tensors=live_tensors)
        self._check_reservations()
        return len(drop_columns) != 0

    @error_check_wrapper
    def get_reservation_or_parent(
        self,
        name: str,
        level: int,
        left: bool = False,
        return_name_level_left: bool = False,
    ) -> str | tuple[str, int, bool] | None:
        reservations = self.left_reservations if left else self.right_reservations
        if (reservations := reservations.get(name, None)) is not None:
            while level >= -1:
                if level in reservations:
                    if return_name_level_left:
                        return name, level, left
                    return nameloop2col(name, level, left)
                # The parent of left nodes are right nodes, so if we don't find a
                # left node immediately then we're back on the right nodes
                reservations = self.right_reservations.get(name, set())
                left = False
                level -= 1
        return None

    @error_check_wrapper
    def shift_bottom_reservation_left(self, shared_loop_index: int):
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
        for resource in self.right_reservations:
            if shared_loop_index + 1 not in self.right_reservations[resource]:
                continue
            self.left_reservations.setdefault(resource, set())
            self.right_reservations[resource].remove(shared_loop_index + 1)
            self.left_reservations[resource].add(shared_loop_index + 1)
            source = nameloop2col(resource, shared_loop_index + 1)
            target = nameloop2col(resource, shared_loop_index + 1, left=True)
            if target in self.data:
                max_to_col(self.data, target, source)
                self.data.drop(columns=[source], inplace=True)
            else:
                self.data.rename(columns={source: target}, inplace=True)
        self._make_reservations()
        self._check_reservations()

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
        return max(
            max(
                (max(r, default=-1) for r in self.right_reservations.values()),
                default=-1,
            ),
            max(
                (max(r, default=-1) for r in self.left_reservations.values()),
                default=-1,
            ),
        )

    def get_min_loop_index(self):
        return min(
            min(
                (min(r, default=1000000) for r in self.right_reservations.values()),
                default=1000000,
            ),
            min(
                (min(r, default=1000000) for r in self.left_reservations.values()),
                default=1000000,
            ),
        )

    @error_check_wrapper
    def merge_next(
        self,
        right: "PmappingDataframe",
        shared_loop_index: int,
        next_shared_loop_index: int,
        live_tensors: set[int],
        still_live_reservations: set[TensorReservation],
        duplicated_aliased_tensors: set[TensorReservation],
        compatibility_left: Compatibility,
        compatibility_right: Compatibility,
        compatibility_joined: Compatibility,
        ignored_resources: set[str],
        drop_valid_reservations: bool = True,
        _pmapping_row_filter_function: Callable[[pd.Series], bool] | None = None,
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
        self._check_reservations()
        right._check_reservations()
        self.free_to_loop_index(shared_loop_index, live_tensors=live_tensors)
        self.shift_bottom_reservation_left(shared_loop_index)

        shared_tensor_names = (
            compatibility_left.tensor_names & compatibility_right.tensor_names
        )
        shared_tensors = [
            compatibility_left.get_tensor_by_name(s) for s in shared_tensor_names
        ]
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
                ta = compatibility_left.get_tensor_by_name(s)
                tb = compatibility_right.get_tensor_by_name(s)
                for la, lb in zip(ta.loops, tb.loops):
                    check_match(la, lb, "initial_tile_shape")
                    check_match(la, lb, "tile_shape")

            for la, lb in zip(compatibility_left.loops, compatibility_right.loops):
                check_match(la, lb, "calculated_n_iterations")

        except ValueError as e:
            make_empty_result = True

        assert not right.left_reservations, f"{right.left_reservations} is not None"

        for resource, reservations in self.right_reservations.items():
            n_reservations = max(reservations, default=-1)
            assert (
                n_reservations <= shared_loop_index
            ), f"{resource}: {reservations} > {shared_loop_index}"

        for resource, reservations in self.left_reservations.items():
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

        if left_match:
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

        # Make sure everything is done in increasing loop order so we don't have
        # read-after-write hazards
        for nloops in range(max_nloops, min_nloops - 1, -1):

            def iter_reservations(reservations_dict):
                for resource in reservations_dict:
                    if nloops in reservations_dict[resource]:
                        yield resource

            # For the RIGHT tree, RIGHT reservations: If there is no matching node in the left
            # tree, add the above-this-level reservation from the left tree. If there is a matching
            # node in the left tree, then we'll add this node to it in the next step.
            for resource in iter_reservations(right.right_reservations):
                if (
                    source := self.get_reservation_or_parent(resource, nloops - 1)
                ) is None:
                    continue
                target = nameloop2col(resource, nloops)
                # If there's a merged version column, then it's in both trees
                if target + "_RIGHT_MERGE" in df:
                    continue
                add_to_col(df, target, source)
            # For LEFT tree, LEFT reservations: Add the immediately-above
            # reservation from the right tree.
            for resource in iter_reservations(self.left_reservations):
                if (
                    source := right.get_reservation_or_parent(resource, nloops - 1)
                ) is None:
                    continue
                right_merge_source = source + "_RIGHT_MERGE"
                target = nameloop2col(resource, nloops, left=True)
                if source is not None:
                    add_to_col(
                        df,
                        target,
                        right_merge_source if right_merge_source in df else source,
                    )
            # For LEFT tree, RIGHT reservations: Add the same-level reservation from
            # the right tree. This will double-count reservations that are in both branches,
            # so we remove them later.
            for resource in iter_reservations(self.right_reservations):
                if (
                    source := right.get_reservation_or_parent(resource, nloops)
                ) is None:
                    continue
                right_merge_source = source + "_RIGHT_MERGE"
                target = nameloop2col(resource, nloops)
                if source is not None:
                    add_to_col(
                        df,
                        target,
                        right_merge_source if right_merge_source in df else source,
                    )

        # For everything else: Simple add
        dropcols = [c for c in df.columns if c.endswith("_RIGHT_MERGE")]
        for source in dropcols:
            target = source[: -len("_RIGHT_MERGE")]
            if is_tensor_col(target):
                continue
            if not col_used_in_pareto(target):
                raise ValueError(f"{target} is not used in pareto")
            if col2nameloop(target) is None:
                add_to_col(df, target, source)

        df = df.drop(columns=dropcols)
        result = PmappingDataframe(
            df,
            skip_pareto=True,
            check_above_subset_below=False,
            n_total_pmappings=n_total_pmappings,
            n_valid_pmappings=n_valid_pmappings,
        )
        # Remove tensors that were allocated in both branches and got added
        # together.
        shared_to_free = [
            s for s in shared_tensors if s.above_loop_index <= shared_loop_index
        ]
        live_to_alloc = [
            s for s in still_live_reservations if s.above_loop_index > shared_loop_index
        ]
        result.adjust_reservations(
            alloc=live_to_alloc,
            free=list(itertools.chain(shared_to_free, duplicated_aliased_tensors)),
            ignored_resources=ignored_resources,
        )

        if CHECK_CORRECTNESS:
            result.check_above_subset_below(live_tensors)
            result.check_reservations(live_tensors)

        result.free_to_loop_index(next_shared_loop_index, live_tensors=live_tensors)
        if not CHECK_CORRECTNESS:
            result.limit_capacity(
                next_shared_loop_index,
                drop_valid_reservations,
                ignored_resources=ignored_resources,
            )
        result.max_right_to_left()
        if _pmapping_row_filter_function is not None:
            result = result.filter_rows(_pmapping_row_filter_function)
        result.make_pareto()
        result._check_reservations()

        return result

    @error_check_wrapper
    def _adjust_reservations_one_resource(
        self,
        resource: str,
        alloc: Iterable[TensorReservation],
        free: Iterable[TensorReservation],
    ):
        alloc, free = list(alloc), list(free)
        # Iterate through each reservation and level
        targets = defaultdict(int)

        # Must allocate at the above_loop_index level
        for t in itertools.chain(alloc, free):
            self.right_reservations.setdefault(resource, set()).add(t.above_loop_index)

        for t, negate in [(t, False) for t in alloc] + [(t, True) for t in free]:
            size = self.data[tensor2col(t.name)]
            size = -size if negate else size
            targets[t.above_loop_index, False] += size
            # Allocate at any levels below the above_loop_index level
            for level in self.right_reservations[resource]:
                if level > t.above_loop_index:
                    targets[level, False] += size
            for level in self.left_reservations.get(resource, set()):
                if level > t.above_loop_index:
                    targets[level, True] += size

        # Now apply the allocations. Sort so we go from top to bottom in case
        # there are maxes that propagate down.
        for (level, left), size in sorted(
            targets.items(), key=lambda x: x[0], reverse=True
        ):
            target = nameloop2col(resource, level, left=left)
            if target in self.data:
                add_to_col(self.data, target, size)
                continue

            # We're creating a new column, so copy allocations from any parents
            source = self.get_reservation_or_parent(resource, level - 1)
            if source:
                add_to_col(self.data, target, source)
            else:
                self.data[target] = size

            # Assert all reservations are >= 0
            assert (self.data[target] >= 0).all(), f"Negative reservation: {target}"

    @error_check_wrapper
    def adjust_reservations(
        self,
        alloc: Iterable[TensorReservation],
        free: Iterable[TensorReservation],
        ignored_resources: set[str] = set(),
    ):
        alloc, free = list(alloc), list(free)
        all_resources = {t.resource_name for t in alloc} | {
            t.resource_name for t in free
        }
        # Handle each resource separately
        for resource in all_resources:
            if resource in ignored_resources:
                continue
            cur_alloc = [t for t in alloc if t.resource_name == resource]
            cur_free = [t for t in free if t.resource_name == resource]
            if cur_alloc or cur_free:
                self._adjust_reservations_one_resource(resource, cur_alloc, cur_free)

    @staticmethod
    def concat(
        paretos: list["PmappingDataframe"], skip_pareto: bool = False
    ) -> "PmappingDataframe":
        if len(paretos) == 0:
            raise ValueError("No paretos to concatenate")
        if len(paretos) == 1:
            return paretos[0]

        required_cols = set.union(*[set(p.data.columns) for p in paretos])
        shared_cols = set.intersection(*[set(p.data.columns) for p in paretos])
        fill_cols = required_cols - shared_cols
        fill_cols = [c for c in fill_cols if col_used_in_pareto(c)]

        concatenated = pd.concat([p.data for p in paretos]).reset_index(drop=True)

        p = PmappingDataframe(
            concatenated.fillna(0),
            skip_pareto=len(paretos) == 1 or skip_pareto,
            fill_reservation_cols=fill_cols,
            n_total_pmappings=sum(p.n_total_pmappings for p in paretos),
            n_valid_pmappings=sum(p.n_valid_pmappings for p in paretos),
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
        )
        args.update(kwargs)
        return PmappingDataframe(**args)

    def copy(self) -> "PmappingDataframe":
        return self.update(
            data=self.data.copy(),
            skip_pareto=True,
            check_above_subset_below=False,
        )
        return p

    def limit_capacity(
        self,
        next_shared_loop_index: int = None,
        drop_valid_reservations: bool = True,
        ignored_resources: set[str] = set(),
    ):
        dropcols = []
        for resource in sorted(
            set(self.right_reservations) | set(self.left_reservations)
        ):
            # Right reservations: Only check the greatest-index level. If a loop
            # is 0 and the next shared loop index is -1, then we can drop the
            # column.
            right_loops = self.right_reservations.get(resource, set())
            if right_loops:
                n = max(right_loops)
                col = nameloop2col(resource, n)
                self._data = self.data[self.data[col] <= 1]
            for l in list(right_loops):
                if (
                    l == 0
                    and next_shared_loop_index == -1
                    and drop_valid_reservations
                    and resource not in ignored_resources
                ):
                    right_loops.discard(l)
                    dropcols.append(col)

            # Left reservations: Check all levels. If a loop is 0,
            # then we can drop the column.
            left_loops = self.left_reservations.get(resource, set())
            for l in list(left_loops):
                col = nameloop2col(resource, l, left=True)
                self._data = self.data[self.data[col] <= 1]
                if (
                    l == 0
                    and drop_valid_reservations
                    and resource not in ignored_resources
                ):
                    left_loops.discard(l)
                    dropcols.append(col)

        self._data = self.data.drop(columns=dropcols)
        self._make_reservations()

    def make_pareto(self, columns: list[str] = None, parallelize: bool = False):
        self._check_reservations()
        self._data = makepareto(self.data, columns, parallelize=parallelize)
        self._check_reservations()

    def has_reservations(self):
        return any(col2nameloop(c) is not None for c in self.data.columns)

    # ============================================================================
    # Checking functions
    # ============================================================================
    def check_above_subset_below(self, live_tensors: set[str] = fzs()):
        assert not self.data.isnull().values.any(), f"NaN in {self.data}"
        targets = []
        for left, reservations_dict in [
            (True, self.left_reservations),
            (False, self.right_reservations),
        ]:
            for resource, reservations in reservations_dict.items():
                for r in reservations:
                    above = self.get_reservation_or_parent(resource, r - 1)
                    if above is not None:
                        below = nameloop2col(resource, r, left=left)
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

    # @error_check_wrapper
    # def check_reservations(self, live_tensors: set[int]):
    #     from accelforge.visualization.reservationtree import mappings2reservationtree
    #     assert not self.data.isnull().values.any(), f"NaN in {self.data}"

    #     self = self.copy()

    #     self.free_to_loop_index(-1, check_correctness=False)
    #     self.shift_bottom_reservation_left(-1)

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
    #             col = self.get_reservation_or_parent(k, 0, left=True)
    #             if str(k) == "0":
    #                 continue
    #             if col not in self.data.columns:
    #                 got = r[[c for c in self.data.columns if col2nameloop(c) is not None]]
    #                 self.fail(i, live_tensors)
    #                 raise ValueError(f"Missing {k}: Expected {reservations}. Got: {got}")
    #             if r[col] != v:
    #                 got = r[[c for c in self.data.columns if col2nameloop(c) is not None]]
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
    #     all_tensors = set(t for tn in r[MAPPING_COLUMN].values() for t in tn.tensors)
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
