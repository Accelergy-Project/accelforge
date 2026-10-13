from collections import defaultdict
import copy
import functools
import itertools

from typing import Any, Callable, Iterable

import sympy

from accelforge.frontend.arch.arch import _is_multi_einsum_parallel_resource
from accelforge.frontend.mapping import Nested, TilePattern
from accelforge.frontend.mapping import Loop as MappingLoop
from accelforge.mapper.FFM._join_pmappings.compatibility import (
    Compatibility,
    Loop,
    TensorReservation,
)
from accelforge.util import _fillna_and__numeric_cast, _numeric_cast
from accelforge.util._mathfuncs import RESERVATION_PRECISION_TOLERANCE
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
    # Previously we had the below here, but I think that's bugged because it would
    # return None if we checked for left, it didn't exist on the left, and did exist on
    # the right.
    # if (levels := reservations.get(name, None)) is not None:

    levels = reservations.get(name, oset())
    while level >= -1:
        if level in levels:
            if return_name_level_left:
                return name, level, left
            return reservation2col(name, level, left)
        # The parent of left nodes are right nodes, so if we don't find a
        # left node immediately then we're back on the right nodes
        levels = r_reservations.get(name, oset())
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
        skip_pareto: bool = False,
        fill_reservation_cols: set | str = fzs(),
        check_above_subset_below: bool = CHECK_CORRECTNESS,
        max_right_to_left: bool = False,
        next_shared_loop_index: int = None,
        excess_resource_tolerance: float = 0,
    ):
        self._data: pd.DataFrame = reduce_precision(data)
        self._prev_free_to_loop_index = None
        self.n_total_pmappings: float = n_total_pmappings
        self.n_valid_pmappings: float = n_valid_pmappings
        self.drop_valid_reservations: bool = drop_valid_reservations
        self.excess_resource_tolerance: float = excess_resource_tolerance

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
                if (name_nloops := col2reservation(below)) is not None:
                    name, nloops = name_nloops.name, name_nloops.nloops
                    above = get_reservation_or_parent(
                        name, nloops - 1, l_reservations, r_reservations
                    )
                    if above is not None:
                        targets.append((nloops, above, below))
                elif not col_used_in_pareto(below):
                    raise ValueError(f"{below} is not a valid reservation column")

        # Sort so maxes propagate from top to bottom (e.g. 0->1 then 1->2)
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
        Create a dictionary of reservations for each resource. The dictionary keys are
        the resource names and the values are lists of above_loop_index values for that
        resource.
        """
        l_reservations, r_reservations = {}, {}
        for c in self.data.columns:
            if (key := col2reservation(c)) is not None:
                target = l_reservations if is_left_col(c) else r_reservations
                target.setdefault(key.name, oset()).add(key.nloops)
                assert key.nloops >= -1

        return l_reservations, r_reservations

    def _make_per_component_latencies(self) -> dict[str, oset]:
        """
        Per-component latency at a level is the latency that must be spent on that
        component during all Einsums that share that level. Upward-inclusive: a
        level's component latency includes the component latencies of all levels below
        it.

        There is no left/right. Only right, because when a component latency is not
        shared with future Einsums, we can immediately max it into the total latency.

        Returns {component name: set of above_loop_index}
        """
        latencies = {}
        for c in self.data.columns:
            if (key := col2complatency(c)) is not None:
                latencies.setdefault(key.name, oset()).add(key.nloops)
                assert key.nloops >= -1
        return latencies

    def _make_commlatencies(self) -> dict[str, oset]:
        """
        Ascent and descent latencies at a level. Ascent latency is the latency that is
        spent moving UP FROM a level to the next-above level. Descent latency is the
        latency that is spent moving DOWN TO from a level to the next-above level. This
        specific up from / down to is chosen so that when we free a level, we can free
        both the descent and ascent latencies at that level & add them to total latency.

        NOT inclusive.

        No left/right.

        Returns {"ascent" or "descent": set of above_loop_index}
        """
        levels = {"descent": oset(), "ascent": oset()}
        for c in self.data.columns:
            if (key := col2commlatency(c)) is not None:
                levels[key.direction].add(key.nloops)
                assert key.nloops >= -1
        return levels

    def _make_total_latency(self) -> tuple[oset, oset]:
        """
        Total latency at a level is the current branch's latency. May increase with
        ascent/descent latency or per-component latency.

        NOT inclusive.

        Left is the latency of fenced sequential branches at this level. Right is the
        latency of the current branch and all branches parallelized with it.

        Returns (above_loop_index of left, above_loop_index of right)
        """
        l_levels, r_levels = oset(), oset()
        for c in self.data.columns:
            if (key := col2total_latency(c)) is not None:
                (l_levels if key.left else r_levels).add(key.nloops)
                assert key.nloops >= -1
        return l_levels, r_levels

    def _n_loops(self) -> int:
        l_total, r_total = self._make_total_latency()
        l_reservations, r_reservations = self._make_reservations()
        levels = itertools.chain(
            l_total,
            r_total,
            *l_reservations.values(),
            *r_reservations.values(),
            *self._make_per_component_latencies().values(),
            *self._make_commlatencies().values(),
        )
        return max(levels, default=-1)

    def clear_fused_loop_symbols(self):
        dropcols = [c for c in self.data.columns if is_fused_loop_col(c)]
        if not dropcols:
            return
        self.data.drop(columns=dropcols, inplace=True)
        self.make_pareto()

    def free_tree(
        self,
        loop_index: int,
        level2col: Callable[..., str],
        l_levels: oset,
        r_levels: oset,
        lefts_rights_sum_before_moving_up: bool = False,
        upward_inclusive: bool = False,
        downward_inclusive: bool = False,
        cols_to_max_into_right: dict[int, list[str]] | None = None,
        cols_to_add_into_right: dict[int, list[tuple[str, Any]]] | None = None,
    ) -> list[str]:
        """
         Free one type of column. Every level below loop_index goes into the target, by
         default the right leaf at the bottom (loop_index + 1). Returns the columns to
         drop.

         Follows:
            A  B
             / | --- 0
            C  D
             / | --- 1  <- Shared Loop Index
            E  F
             / | --- 2
            G  H
         ->
            A  B
             / | --- 0
            C  D
             / | --- 1  <- Shared Loop Index
            E  accumulated(E,G,H)

         Order is always below right -> cur right cols_to_max_into_right -> cur right,
         cols_to_add_into_right -> cur right, cur left -> cur right

         Parameters
         ----------

         lefts_rights_sum_before_moving_up: If True, left and right branches sum before
         being moved to the next-above right.

         upward_inclusive: High in the tree (low above_loop_index) includes low in the
             tree (high above_loop_index).

         downward_inclusive: Low in the tree (high above_loop_index) includes high in the
             tree (low above_loop_index).

         cols_to_max_into_right: {level: columns maxed into the level}. Used to get
             component latency into total latency. Maxed into right BEFORE right and left
             are accumulated.

         cols_to_add_into_right: {level: [(column, factor)]}. Each column times its
             factor is added into the level AFTER the max. Used to pay wind-up latency
             into the branch's steady state, times every loop above it.
        """
        bottom = loop_index + 1
        target = level2col(bottom)
        cols_to_max_into_right = {
            m: cols for m, cols in (cols_to_max_into_right or {}).items() if m > bottom
        }
        cols_to_add_into_right = {
            m: cols for m, cols in (cols_to_add_into_right or {}).items() if m > bottom
        }
        drop_cols = []

        lefts = oset(l for l in l_levels if l > bottom)
        rights = sorted((r for r in r_levels if r > bottom), reverse=True)

        # If we've already got something that's inclusive of the others, can just drop
        # removed columns instead of accumulating them.
        assert not (upward_inclusive and downward_inclusive)
        if rights and (upward_inclusive or downward_inclusive):
            kept = rights[0] if downward_inclusive else rights[-1]
            drop_cols += [level2col(r) for r in rights if r != kept]
            rights = [kept]

        accumulator = None
        accumulated_cols = []

        def combine(accumulator, src_col, combine_is_add, factor=1):
            src = self.data[src_col] * factor
            if accumulator is None:
                return src
            if combine_is_add:
                return accumulator + src
            return pd.concat([accumulator, src], axis=1).max(axis=1)

        levels = oset(rights) | lefts | oset(cols_to_max_into_right)
        levels |= oset(cols_to_add_into_right)
        for m in sorted(levels, reverse=True):
            # This order is important. See docstring.
            if m in rights:
                accumulator = combine(accumulator, level2col(m), False)
                accumulated_cols.append(level2col(m))
            for col in cols_to_max_into_right.get(m, ()):
                accumulator = combine(accumulator, col, False)
            for col, factor in cols_to_add_into_right.get(m, ()):
                accumulator = combine(accumulator, col, True, factor)
                accumulated_cols.append(col)
            if m in lefts:
                col_name = level2col(m, left=True)
                accumulator = combine(
                    accumulator, col_name, lefts_rights_sum_before_moving_up
                )
                accumulated_cols.append(col_name)

        if accumulator is not None:
            if target in self.data:
                accumulator = combine(accumulator, target, False)
            self.data[target] = accumulator

        return drop_cols + accumulated_cols

    @error_check_wrapper
    def free_to_loop_index(
        self,
        loop_index: int,
        per_loop_calculated_n_iterations: tuple = (),
        levels_with_sequential_splits: fzs = fzs(),
    ) -> bool:
        """
        In the following, "timestep" as a loop index means that the timestep is shared
        between all Einsums that are tiled fused at that loop index (i.e., these Einsums
        interleave execution during the timestep) or are parallelized at that loop index
        (i.e., these Einsums execute simultaneously during the timestep).

        The order is always: - make pmappings -> free - (fence if join is sequential) ->
        join -> free - (fence if join is sequential) -> join -> free - ... - -> final
        free -> final fence always happens

        RESERVATIONS:

        - How much of a resource is reserved during a particular timestep?
        - DOWNWARD INCLUSIVE. Low in tree (high above_loop_index) includes high
        - Includes left (not shared with future Einsums at this level) and right (yes
          shared with future Einsums at this level)

        - Joining

            - Don't touch left
            - For reservations above and equal to the _multi_einsum_parallel spatial,
              sum reservations on the right (shared between Einsums) with the next. For
              reservations below, max the right with the next.
            - The next better not have any left reservations since it's just one Einsum

        - Freeing

            - Need max resource usage in the timestep
            - Take max of every left and right reservation at this level and below to
              grab the max resource usage in the timestep, then max upward
            - Freeing always goes into the right leaf at the new bottom

        - Fencing. At this level (which must be the bottommost):

            - Max the right leaf to the left (not shared with future Einsums at this
              level) & drop the right leaf

        PER-COMPONENT LATENCIES:

        - How much time must a resource spend during a particular timestep?
        - UPWARD INCLUSIVE. High in tree (low above_loop_index) includes low in the tree
          (high above_loop_index)
        - Only right. If it's not shared with future Einsums (the case that would make
          it left), we just max it into the total latency
        - NOTE: We're not allowed to have any shared component latencies below the
          _multi_einsum_parallel_spatial in the arch (forcing latencies to be MAXED) and
          above the split point in any mapping (forcing latencies to be SUMMED) because
          I'm not making the logic to handle that.

        - Joining

            - Sum with the next. Busy time is total busy time for both Einsums

        - Freeing

            - Need total busy time of the resource in the timestep
            - Max all latencies at this level and below. Inclusive, so this'll have the
              total resource busy time.

        - Fencing. At this level (which must be the bottommost):

            - Max into the total latency
            - Delete per-component latencies if we have latencies above, otherwise move
              these up

        WINDUP LATENCY:

        - How much windup time is spend each time this timestep is repeated temporally?
        - NOT inclusive because we'll be multiplying each level by some # iterations
        - Only has right

        - Joining

            - Sum with the next (assuming windup latency is serialized)

        - Freeing

            - If this split is sequential, then multiply by the #iterations at every
              level above & add to the steady state latency at this level (must repeat
              this latency each time the split occurs).
            - If this split is parallel, then add to the ascent/descent latency at the
              level above, and repeat going up. This ascent/descent latency is NOT
              repeated when this split is repeated, but may be repeated when the
              next-above repeats.

        - Fencing. At this level (which must be the bottommost):

            - Do nothing.

        TOTAL LATENCY:

        - How much time must the accelerator spend during a particular timestep?
        - NOT INCLUSIVE.
        - Left is sum latency of all previously-completed sequential latencies at this
          level
        - Right is max latency of all currently-parallelized branches at this level
        - Total latency for parallelized Einsums will be (sum ascent/descent latencies)
          + max(per-branch steady state, per-branch steady state latency, per-resource
          cumulative latency).
        - Total latency for sequentilized Einsums will be (sum ascent/descent latencies)
          + max(sum(per-branch steady state latency), per-resource cumulative latency).

        - Joining

            - Max the right with next

        - Freeing

            - Max this level (completed parallel branches) and below (the current
              parallel branch that's being worked on)

        - Fencing. At this level (which must be the bottommost):

            - Set left = left + right then delete right
        """
        if loop_index == self._prev_free_to_loop_index:
            return False
        assert (
            self._prev_free_to_loop_index is None
            or loop_index < self._prev_free_to_loop_index
        ), f"free to {loop_index} below the last free to {self._prev_free_to_loop_index}"
        self._prev_free_to_loop_index = loop_index

        drop_cols = []
        component_latencies = self._make_per_component_latencies()

        component_latency_to_max_to_total_latency = defaultdict(list)
        for component, levels in component_latencies.items():
            for m in levels:
                component_latency_to_max_to_total_latency[m].append(
                    complatency2col(component, m)
                )

        windups_to_add = defaultdict(list)
        for direction, levels in self._make_commlatencies().items():
            for m in range(max(levels, default=-1), loop_index + 1, -1):
                col = commlatency2col(direction, m)
                if col not in self.data:
                    continue

                # Sequential -> repeat each loop iteration -> multiply by number of
                # tiimes it'll repeat and then add it to the steady-state latency of
                # this loop level
                # Ctrl-F for WINDUP_LATENCY_PERFECT_ASSUMPTION: use the product of the
                # per-loop iteration counts, ignoring imperfect
                if m in levels_with_sequential_splits or m == 0:
                    factor = 1
                    for n in per_loop_calculated_n_iterations[:m]:
                        factor = factor * (self.data[n] if isinstance(n, str) else n)
                    windups_to_add[m].append((col, factor))

                # Parallel -> do not repeat each loop iteration -> added to
                # communication latency of immediately-above level so that next level
                # can decide how many times it's repeated.
                else:
                    add_to_col(self.data, commlatency2col(direction, m - 1), col)
                    drop_cols.append(col)

        l_total_latency, r_total_latency = self._make_total_latency()
        drop_cols += self.free_tree(
            loop_index,
            total_latency2col,
            l_total_latency,
            r_total_latency,
            lefts_rights_sum_before_moving_up=True,
            cols_to_max_into_right=component_latency_to_max_to_total_latency,
            cols_to_add_into_right=windups_to_add,
        )

        l_reservations, r_reservations = self._make_reservations()
        for resource in oset(l_reservations) | oset(r_reservations):
            drop_cols += self.free_tree(
                loop_index,
                functools.partial(reservation2col, resource),
                l_reservations.get(resource, oset()),
                r_reservations.get(resource, oset()),
                downward_inclusive=True,
            )

        for component, levels in component_latencies.items():
            drop_cols += self.free_tree(
                loop_index,
                functools.partial(complatency2col, component),
                oset(),
                levels,
                upward_inclusive=True,
            )

        self._data = self.data.drop(columns=drop_cols)
        assert self._n_loops() <= loop_index + 1
        return len(drop_cols) != 0

    @error_check_wrapper
    def sequential_fence_to_level(self, loop_index: int):
        """
        Put a fence at level loop_index. Everything with above_loop_index >= loop_index
        + 1 stops being shared with future Einsums.

        Note: This MUST be the bottom level. There should already have been a
        free_to_loop_index with a loop_index == this loop index.

        - Per-component latencies: runtime is >= latency of any component. Max each
          component's latency into total latency. If the component's latency exists
          above this point, we can delete because upward-inclusiveness means that above
          levels still track this component's latency.
        - Reservations: Bottom-right leaf (live branch) is maxed into the left leaf (max
          of all dead branches).
        - Total latency: Bottom-right leaf (max of all currently-parallelized branches)
          is added to the left leaf (sum of all branches that are not parallelized with
          the right).
        - Wind-ups: No change. We assume that they are the same for parallel and
          sequential.
        """
        bottom = loop_index + 1
        assert self._prev_free_to_loop_index == loop_index, "free_to_loop_index first"
        assert self._n_loops() <= bottom, "columns below the bottom"
        total = total_latency2col(bottom)
        drop_cols = []

        for component, levels in self._make_per_component_latencies().items():
            if bottom in levels:
                col = complatency2col(component, bottom)
                assert total in self.data, col
                max_to_col(self.data, total, col)
                drop_cols.append(col)

        drop_cols += self._free_dead_reservations(bottom)
        if total in self.data:
            add_to_col(self.data, total_latency2col(bottom, left=True), total)
            drop_cols.append(total)

        self._data = self.data.drop(columns=drop_cols)

    def finish(self):
        # Next shared = -1 -> no shared loops in the future
        self.limit_capacity(next_shared_loop_index=-1, finished=True)

        # We do an additional bump to -2 because globally-shared tensors (persistent
        # tensors) are reserved at above_loop_index -1
        self.free_to_loop_index(-2)
        self.sequential_fence_to_level(-2)

        # Make total latency column from the windups and steady states
        assert not self._make_per_component_latencies(), "unfenced component latencies"
        assert not any(self._make_commlatencies().values()), "unfreed wind-ups"
        l_total, r_total = self._make_total_latency()
        assert set(l_total) <= {-1} and not r_total, (l_total, r_total)

        total = total_latency2col(-1, left=True)
        if total in self.data:
            self.data.rename(columns={total: "Total<SEP>latency"}, inplace=True)

        # Add leak power in to energy
        if "Total<SEP>leak_power" in self.data:
            leak = self.data.pop("Total<SEP>leak_power")
            for c in ["Total<SEP>energy", "Total<SEP>leak_energy"]:
                if c in self.data:
                    self.data[c] += leak * self.data["Total<SEP>latency"]

        self.make_pareto()

    def _free_dead_reservations(
        self, n_loops: int, parallel: bool = False
    ) -> list[str]:
        """
        If next join is parallel & the resource is shared across parallel branches, do
        nothing.

        If next join is sequential or the resource is not shared across parallel
        branches, max to the left leaf (not shared with future Einsums at this level)
        and drop the right leaf.

        Done ONLY AT THE BOTTOMMOST LEVEL. There should already have been a
        free_to_loop_index with a loop_index == this loop index.
        """
        assert self._prev_free_to_loop_index == n_loops - 1, "free_to_loop_index first"
        assert self._n_loops() <= n_loops, "columns below the bottom"
        drop_cols = []
        _, r_reservations = self._make_reservations()
        for resource, levels in r_reservations.items():
            if n_loops not in levels:
                continue
            if parallel and _is_multi_einsum_parallel_resource(resource):
                continue
            col = reservation2col(resource, n_loops)
            max_to_col(self.data, reservation2col(resource, n_loops, left=True), col)
            drop_cols.append(col)
        return drop_cols

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
        sequential: bool = True,
        do_not_free: set[str] = fzs(),
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

        The figs show just a sum, but there's a variety of maxes and such going on. See
        free_to_loop_index docstring for rules on each column type.
        """
        live_tensors = compatibility_joined.tensor_names
        shared_loop_index = compatibility_left.n_loops - 1
        next_shared_loop_index = compatibility_joined.n_loops - 1
        iterations_above = compatibility_right.per_loop_calculated_n_iterations
        levels_with_sequential_splits = compatibility_left.levels_with_sequential_splits
        if sequential:
            levels_with_sequential_splits |= fzs([shared_loop_index + 1])
        joined = oset(compatibility_joined.symbols())
        iters_cols = oset(n for n in iterations_above if isinstance(n, str)) - joined

        self = self.copy()
        right = right.copy()
        right.free_to_loop_index(compatibility_right.n_loops - 1)
        self.free_to_loop_index(
            shared_loop_index,
            compatibility_left.per_loop_calculated_n_iterations,
            compatibility_left.levels_with_sequential_splits,
        )
        if sequential:
            self.sequential_fence_to_level(shared_loop_index)
        else:
            drop_cols = self._free_dead_reservations(
                shared_loop_index + 1, parallel=True
            )
            self._data = self.data.drop(columns=drop_cols)

        shared_tensor_names = (
            compatibility_left.tensor_names & compatibility_right.tensor_names
        )
        shared_tensors = [
            compatibility_left.get_reservation_of_tensor(s) for s in shared_tensor_names
        ]
        left_match, right_match = [], []
        make_empty_result = False

        def check_match(la: Loop, lb: Loop, param: str):
            a, b = getattr(la.tile_pattern, param), getattr(lb.tile_pattern, param)
            if isinstance(a, str) and isinstance(b, str):
                if a in left_match and b in left_match:
                    return
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
                for la, lb in zip(ta.physical_spatial_loops, tb.physical_spatial_loops):
                    check_match(la, lb, "initial_tile_shape")
                    check_match(la, lb, "tile_shape")
                    check_match(la, lb, "calculated_n_iterations")

            for la, lb in zip(compatibility_left.loops, compatibility_right.loops):
                check_match(la, lb, "calculated_n_iterations")

        except ValueError as e:
            make_empty_result = True

        right_df_l_reservations, right_df_r_reservations = right._make_reservations()
        assert not right_df_l_reservations, f"{right_df_l_reservations} is not None"
        # Assert that the right doesn't have any left latency columns. It's one Einsum
        # so there can't be branches in its tree.
        assert not right._make_total_latency()[0], right.data.columns

        l_reservations, r_reservations = self._make_reservations()

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
        remaining_symbols = joined | iters_cols
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

        def from_right(c: str) -> str:
            # If a column is in both DFs, do the appropriate rename to grab the one from
            # the right side
            right_merge = c + "_RIGHT_MERGE"
            return right_merge if right_merge in df else c

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
                target = reservation2col(resource, nloops, left=True)
                if source is not None:
                    add_to_col(df, target, from_right(source))
            # For LEFT tree, RIGHT reservations: Add the same-level reservation from the
            # right tree. This will double-count reservations that are in both branches,
            # so we remove them later.
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
                target = reservation2col(resource, nloops)
                if source is not None:
                    add_to_col(df, target, from_right(source))

        # Most other things: Simple add. Component latencies and descent/ascent wind-up
        # latencies are summed. Total latency rights (right is parallelized with next
        # Einsum) are maxed.
        dropcols = [c for c in df.columns if c.endswith("_RIGHT_MERGE")]
        for source in dropcols:
            target = source[: -len("_RIGHT_MERGE")]
            if is_tensor_col(target):
                continue
            # Ctrl-F for LEAK_POWER_MAX
            if target == "Total<SEP>leak_power":
                max_to_col(df, target, source)
                continue
            if not col_used_in_pareto(target):
                raise ValueError(f"{target} is not used in pareto")
            if col2reservation(target) is not None:
                continue
            if (key := col2total_latency(target)) is not None and not key.left:
                max_to_col(df, target, source)
            else:
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
        )
        # Remove tensors that were allocated in both branches and got added
        # together.
        shared_to_free = [
            s
            for s in shared_tensors
            if s.above_loop_index <= shared_loop_index and s.name not in do_not_free
        ]
        reservations_of_live_tensor_not_in_right = [
            compatibility_joined.get_reservation_of_tensor(t)
            for t in compatibility_joined.tensor_names
            - compatibility_right.tensor_names
        ]
        live_to_alloc = [
            r
            for r in reservations_of_live_tensor_not_in_right
            if r.above_loop_index > shared_loop_index
        ]

        # Assert duplicated aliased tensors have the same reservation sizes
        shared_keys = {(s.resource_name, s.above_loop_index) for s in shared_to_free}
        free = list(shared_to_free)
        for d in duplicated_aliased_tensors:
            key = (d.resource_name, d.above_loop_index)
            if key in shared_keys:
                for s in shared_to_free:
                    if (s.resource_name, s.above_loop_index) == key:
                        assert (
                            result.data[tensor2col(s.name)]
                            == result.data[tensor2col(d.name)]
                        ).all(), (
                            f"Duplicated tensors {s.name} and {d.name} in "
                            f"{d.resource_name} have different reservation sizes"
                        )
                continue
            free.append(d)

        result.adjust_reservations(
            alloc=live_to_alloc,
            free=free,
            ignored_resources=ignored_resources,
        )

        dead_tensor_cols = [
            c
            for c in result.data.columns
            if is_tensor_col(c) and col2nametensor(c) not in live_tensors
        ]
        result.data.drop(columns=dead_tensor_cols, inplace=True)

        if CHECK_CORRECTNESS:
            result.check_above_subset_below(live_tensors)
            result.check_reservations(live_tensors)

        result.free_to_loop_index(
            next_shared_loop_index, iterations_above, levels_with_sequential_splits
        )
        result.data.drop(columns=list(iters_cols), inplace=True)
        if not CHECK_CORRECTNESS:
            result.limit_capacity(
                next_shared_loop_index, ignored_resources=ignored_resources
            )
        result.max_right_to_left()
        if _pmapping_row_filter_function is not None:
            result = result.filter_rows(_pmapping_row_filter_function)
        result.make_pareto()

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
                sequential,
                _pmapping_row_filter_function,
                _force_allow_invalid_only_for_runtime_test=False,
            )

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

        l_reservations, r_reservations = self._make_reservations()

        # Must allocate at the above_loop_index level
        for t in itertools.chain(alloc, free):
            r_reservations.setdefault(resource, oset()).add(t.above_loop_index)

        for t, negate in [(t, False) for t in alloc] + [(t, True) for t in free]:
            size = self.data[tensor2col(t.name)]
            size = -size if negate else size
            targets[t.above_loop_index, False] += size
            # Allocate at any levels below the above_loop_index level
            for level in r_reservations[resource]:
                if level > t.above_loop_index:
                    targets[level, False] += size
            for level in l_reservations.get(resource, oset()):
                if level > t.above_loop_index:
                    targets[level, True] += size

        # Now apply the allocations. Sort so we go from top to bottom in case
        # there are maxes that propagate down.
        for (level, left), size in sorted(
            targets.items(), key=lambda x: x[0], reverse=True
        ):
            target = reservation2col(resource, level, left=left)
            if target in self.data:
                add_to_col(self.data, target, size)
                continue

            # We're creating a new column, so copy allocations from any parents
            source = get_reservation_or_parent(
                resource, level - 1, l_reservations, r_reservations
            )
            if source:
                assert (
                    source in self.data.columns
                ), f"{source} not in {sorted(self.data.columns)}"
                add_to_col(self.data, target, source)
                add_to_col(self.data, target, size)
            else:
                self.data[target] = size

            # We made a new column! Update our reservations so future iterations
            # know about it.
            l_reservations, r_reservations = self._make_reservations()

            # Assert all reservations are >= 0
            assert (self.data[target] >= 0).all(), f"Negative reservation: {target}"

    @error_check_wrapper
    def adjust_reservations(
        self,
        alloc: Iterable[TensorReservation],
        free: Iterable[TensorReservation],
        ignored_resources: set[str] = oset(),
    ):
        alloc, free = list(alloc), list(free)
        all_resources = oset(t.resource_name for t in alloc) | oset(
            t.resource_name for t in free
        )
        ignored_resources = ignored_resources | self.ignored_resources
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

        required_cols = oset.union(*[oset(p.data.columns) for p in paretos])
        shared_cols = oset.intersection(*[oset(p.data.columns) for p in paretos])
        fill_cols = required_cols - shared_cols
        fill_cols = [c for c in fill_cols if col_used_in_pareto(c)]

        concatenated = pd.concat([p.data for p in paretos]).reset_index(drop=True)

        p = PmappingDataframe(
            _fillna_and__numeric_cast(concatenated, 0),
            skip_pareto=len(paretos) == 1 or skip_pareto,
            fill_reservation_cols=fill_cols,
            n_total_pmappings=sum(p.n_total_pmappings for p in paretos),
            n_valid_pmappings=sum(p.n_valid_pmappings for p in paretos),
            ignored_resources=next(iter(paretos)).ignored_resources,
            drop_valid_reservations=next(iter(paretos)).drop_valid_reservations,
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
        )
        args.update(kwargs)
        return PmappingDataframe(**args)

    def copy(self, copy_df: bool = True) -> "PmappingDataframe":
        return self.update(
            data=self.data.copy() if copy_df else self.data,
            skip_pareto=True,
            check_above_subset_below=False,
        )

    def split_in_half(self) -> tuple["PmappingDataframe", "PmappingDataframe"]:
        mid = len(self.data) // 2
        half_total = self.n_total_pmappings / 2
        half_valid = self.n_valid_pmappings / 2
        first = self.update(
            data=self.data.iloc[:mid].copy(),
            skip_pareto=True,
            n_total_pmappings=half_total,
            n_valid_pmappings=half_valid,
        )
        second = self.update(
            data=self.data.iloc[mid:].copy(),
            skip_pareto=True,
            n_total_pmappings=half_total,
            n_valid_pmappings=half_valid,
        )
        return first, second

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
                self._data = self.data[
                    self.data[col] <= 1 + tolerance + RESERVATION_PRECISION_TOLERANCE
                ]
                if (
                    l <= 0
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
                col = reservation2col(resource, l, left=True)
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
                self._data = self.data[
                    self.data[col] <= 1 + tolerance + RESERVATION_PRECISION_TOLERANCE
                ]
                if (
                    l <= 0
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
        inplace: bool = True,
    ) -> "PmappingDataframe":
        # The error for absolute_resource_usage_tolerance sums each time we modify the
        # df and prune, so if we use it more, we need to use a lower threshold. The
        # max_n_einsums value assumes that absolute_resource_usage_tolerance is only
        # used for joining.
        if self.drop_valid_reservations:
            resource_usage_tolerance = objective_tolerance

        new_data = makepareto(
            self.data,
            columns,
            resource_usage_tolerance=resource_usage_tolerance,
            absolute_resource_usage_tolerance=absolute_resource_usage_tolerance,
            objective_tolerance=objective_tolerance,
        )
        if inplace:
            self._data = new_data
            return self
        else:
            return self.update(data=new_data, skip_pareto=True)

    def has_reservations(self):
        return any(col2reservation(c) is not None for c in self.data.columns)

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

    # @error_check_wrapper
    # def check_reservations(self, live_tensors: set[int]):
    #     from accelforge.visualization.reservationtree import mappings2reservationtree
    #     assert not self.data.isnull().values.any(), f"NaN in {self.data}"

    #     self = self.copy()

    #     self.free_to_loop_index(-1)

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
