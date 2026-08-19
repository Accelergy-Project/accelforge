from collections import defaultdict, namedtuple
import copy
import functools
import itertools

from numbers import Number
from typing import Any, Callable, Iterable, NamedTuple

from pandas import DataFrame, Series
import sympy

from accelforge.frontend.mapping import Nested, TilePattern
from accelforge.frontend.mapping import Loop as MappingLoop
from accelforge.mapper.FFM._join_pmappings.compatibility import (
    Compatibility,
    TensorReservation,
    _symbol_values,
)
from accelforge.mapper.FFM._join_pmappings.reservation_dataframe import (
    ReservationData,
    ReservationDataFrame,
    _reservation_size_above,
    _reservation_size_at_or_above,
)
from accelforge.mapper.FFM._pareto_df.df_convention import _RIGHT_RESERVATION_OFFSET
from accelforge.util import _fillna_and_numeric_cast, _numeric_cast
from accelforge.util._frozenset import fzs, oset

from accelforge._accelerated_imports import pd, np

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


_MATCH_COL = "__match_col<SEP>"
_MATCH_GROUP = "__match_group<SEP>"
_LEFT_GROUP = _MATCH_GROUP + "left"
_RIGHT_GROUP = _MATCH_GROUP + "right"


TensorPairConstraint = namedtuple(
    "TensorPairConstraint",
    [
        "valid_combinations",
        "shape_symbols_a",
        "shape_symbols_b",
        "is_left_a",
        "is_left_b",
    ],
)


def _match_dtypes(frame: DataFrame, ref: DataFrame) -> DataFrame:
    for c in frame.columns:
        if c not in ref.columns or frame[c].dtype == ref[c].dtype:
            continue
        # None slots can't cast to integers; such rows never match anyway.
        if ref[c].dtype.kind in "iu" and frame[c].isna().any():
            frame = frame[frame[c].notna()]
        frame = frame.assign(**{c: frame[c].astype(ref[c].dtype)})
    return frame


def _dfify_group(values: dict[str, Any], index: pd.Series, group_id_col: str) -> tuple[np.ndarray, DataFrame]:
    keys = DataFrame(values, index=index)
    if values:
        cols = list(keys.columns)
        gid = keys.groupby(cols, sort=False, dropna=False).ngroup().values
    else:
        gid = np.zeros(len(index), dtype=np.int64)
    keys[group_id_col] = gid
    return gid, keys.drop_duplicates(group_id_col)


def _constrained_match(
    ld: DataFrame,
    rd: DataFrame,
    match_columns: list[tuple[str, str]],
    tensor_pair_constraints: list[TensorPairConstraint],
) -> tuple[DataFrame, DataFrame, TensorPairConstraint]:
    """
    For two DataFrames, return each side's per-row group ids and the (left group,
    right group) id pairs satisfying the match columns and the constraints.
    """
    # One key column per match column pair and per constrained shape. Constraints
    # reading the same shape share a key column, so merges intersect on it.
    l_match_cols, r_match_cols = {}, {}

    # Match match_columns. These are matches between the left shapes & right shapes
    for i, (a, b) in enumerate(match_columns):
        key = f"{_MATCH_COL}eq{i}"
        l_match_cols[key] = _symbol_values(a, ld).values
        r_match_cols[key] = _symbol_values(b, rd).values

    created_cols_cache = {}
    def get_match_col(is_left: bool, symbol) -> str:
        col_default = f"{_MATCH_COL}{len(created_cols_cache)}"
        match_col = created_cols_cache.setdefault((is_left, symbol), col_default)
        columns = l_match_cols if is_left else r_match_cols
        if match_col not in columns:
            df = ld if is_left else rd
            columns[match_col] = _symbol_values(symbol, df).values
        return match_col

    # Match with tensor_pair_constraints. These record all valid combinations of two
    # tensors that appear in these Einsums
    valid_shape_combos = []
    for c in tensor_pair_constraints:
        columns = [
            get_match_col(is_left, s)
            for is_left, symbols in (
                (c.is_left_a, c.shape_symbols_a),
                (c.is_left_b, c.shape_symbols_b),
            )
            for s in symbols
        ]
        combos = c.valid_combinations.set_axis(columns, axis="columns")
        assert not combos.columns.duplicated().any(), f"Duplicate columns in {c}"
        valid_shape_combos.append(combos)

    l_match_df, l_unique = _dfify_group(l_match_cols, ld.index, _LEFT_GROUP)
    r_match_df, r_unique = _dfify_group(r_match_cols, rd.index, _RIGHT_GROUP)
    
    # Algorithm below:
    #
    # First, we'll only work with unique shapes, not all rows, to reduce combinatorial
    # explosion.
    #
    # Next, one equivalent algorithm we could do is to join left x right then, for each
    # of our constraints, filter out rows that don't match. 
    # 
    # This is equivalent to:
    #
    #   left x right x constraints[0] x constraints[1] x ... 
    #
    # where the cross with constraints is just a filter since there may only be one
    # valid combo.
    #
    # We'll rearrange this to do the cross products in order of (most shared columns ->
    # least shared columns) to maximize pruning potential as we go.

    to_cross = []
    for combos in valid_shape_combos:
        l_cols = [c for c in combos.columns if c in l_unique.columns]
        r_cols = [c for c in combos.columns if c in r_unique.columns]
        # Earlier code should have filtered only constratins that cross sides.
        # One-Einsum is already done in an initial filter, and every left combination
        # should be handled in previous steps. Only two-sided constraints should be
        # passed here. 
        assert l_cols and r_cols, f"No columns on one side"

        # Filter only combinations that occur on both sides        
        combos = _match_dtypes(_match_dtypes(combos, l_unique), r_unique)
        combos = combos.merge(l_unique[l_cols].drop_duplicates(), on=l_cols)
        combos = combos.merge(r_unique[r_cols].drop_duplicates(), on=r_cols)
        to_cross.append(combos)

    to_cross.extend([l_unique, r_unique])
    
    def _cross(a: DataFrame, b: DataFrame) -> DataFrame:
        cols = [c for c in a.columns if c in b.columns]
        return a.merge(b, on=cols) if cols else a.merge(b, how="cross")
    
    def _priority(a: DataFrame, b: DataFrame) -> tuple[int, int]:
        return len(set(a.columns) & set(b.columns)), -len(a) * len(b)

    while len(to_cross) > 1:
        i, j = max(
            itertools.combinations(range(len(to_cross)), 2),
            key=lambda ij: _priority(to_cross[ij[0]], to_cross[ij[1]]),
        )
        to_cross.append(_cross(to_cross.pop(j), to_cross.pop(i)))

    return (
        l_match_df,
        r_match_df,
        to_cross[0][[_LEFT_GROUP, _RIGHT_GROUP]].drop_duplicates(),
    )


class PmappingDataframe:
    def __init__(
        self,
        data: ReservationDataFrame,
        n_total_pmappings: float,
        n_valid_pmappings: float,
        ignored_resources: set[str],
        drop_valid_reservations: bool,
        skip_pareto: bool = False,
        check_above_subset_below: bool = CHECK_CORRECTNESS,
        sort_and_merge_reservations: bool = False,
        excess_resource_tolerance: float = 0,
    ):
        self._data: ReservationDataFrame = _numeric_cast(data)
        self._prev_freed_to = None
        self.n_total_pmappings: float = n_total_pmappings
        self.n_valid_pmappings: float = n_valid_pmappings
        self.drop_valid_reservations: bool = drop_valid_reservations
        self.excess_resource_tolerance: float = excess_resource_tolerance

        if sort_and_merge_reservations:
            assert (
                ignored_resources is not None
            ), "ignored_resources must be set if sort_and_merge_reservations is set"
            self.drop_redundant_reservations()
            self.limit_capacity(ignored_resources=ignored_resources)

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

    @property
    def data(self) -> DataFrame:
        return self._data

    def drop_redundant_reservations(self) -> bool:
        if len(self.data) == 0:
            return False
        self._data = (df := ReservationDataFrame(self._data))
        df.sort_reservations()
        if (changed := df.drop_redundant_data()):
            self.make_pareto()
        return changed

    def clear_fused_loop_symbols(self):
        dropcols = [c for c in self.data.columns if is_fused_loop_col(c)]
        if not dropcols:
            return
        self.data.drop(columns=dropcols, inplace=True)
        self.make_pareto()

    @error_check_wrapper
    def free_to_reservations(
        self, compatibility: Compatibility, shift_bottom_left: bool
    ) -> bool:
        """
           A  B
            / | --- 0
           C  D
            / | --- 1  < Deepest live backing
           E  F
            / | --- 2
           G  H
        ->
           A  B
            / | --- 0
           C  D
              | --- 1  < Deepest live backing
          max(E,G,H)
        
        Returns True if new pruning opportunities may be found after freeing.
        """
        if compatibility is self._prev_freed_to and not shift_bottom_left:
            return False
        self._prev_freed_to = compatibility
        if len(self.data) == 0 or not self.has_reservations():
            return False

        free_to = self._compatibility_iters_above(compatibility)
        return self._free_more_than_iters_above(free_to, shift_bottom_left)

    def free_all_reservations(self):
        self._prev_freed_to = None
        self._free_more_than_iters_above(0, shift_bottom_left=False)

    def _free_more_than_iters_above(self, threshold, shift_bottom_left: bool) -> bool:
        df = ReservationDataFrame(self.data)
        n_columns = len(df.columns)
        df.sort_reservations()
        changed = df.free_to_iters_above(threshold, shift_bottom_left)
        self._data = df
        return changed or len(self.data.columns) != n_columns

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

    @error_check_wrapper
    def merge_next(
        self,
        right: "PmappingDataframe",
        duplicated_aliased_tensors: set[TensorReservation],
        compatibility_left: Compatibility,
        compatibility_right: Compatibility,
        compatibility_joined: Compatibility,
        ignored_resources: set[str],
        left_loop_to_right_loop: list[tuple],
        tensor_pair_constraints: list[TensorPairConstraint] | None = None,
        do_not_free: set[str] = fzs(),
        _pmapping_row_filter_function: Callable[[Series], bool] | None = None,
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

        self.free_to_reservations(compatibility_left, shift_bottom_left=True)

        shared_tensor_names = (
            compatibility_left.tensor_names & compatibility_right.tensor_names
        )

        from accelforge.mapper.FFM._join_pmappings.compatibility import Loop

        match_columns = []
        make_empty_result = False

        def check_match(a: str | Number, b: str | Number):
            if isinstance(a, str) and isinstance(b, str):
                if (a, b) not in match_columns:
                    match_columns.append((a, b))
            elif a != b:
                raise ValueError(f"Mismatch {a} != {b}")

        try:
            for s in shared_tensor_names:
                ta = compatibility_left.get_reservation_of_tensor(s)
                tb = compatibility_right.get_reservation_of_tensor(s)
                symbols_l = ta.compatibility_shape_symbols()
                symbols_r = tb.compatibility_shape_symbols()
                assert len(symbols_l) == len(symbols_r)
                for sl, sb in zip(symbols_l, symbols_r):
                    check_match(sl, sb)
        except ValueError as e:
            make_empty_result = True

        paired_columns = []
        for a, b in left_loop_to_right_loop:
            if isinstance(a, str) and isinstance(b, str):
                paired_columns.append((a, b))
            elif a != b:
                raise ValueError(f"Mismatch in {a}, {b}")

        for datas in ReservationDataFrame(right.data)._get_all_reservations().values():
            for r in datas:
                assert r.left_col is None, f"Right side has left reservations: {r}"

        ld, rd = self.data, right.data
        if make_empty_result:
            ld = ld.iloc[0:0]
            rd = rd.iloc[0:0]

        # Renumber right side's reservations so they don't collide with the left side's
        renames = {}
        offs = lambda x: x._replace(index=x.index + _RIGHT_RESERVATION_OFFSET)
        for c in rd.columns:
            renames[c] = update_reservation_col(c, offs)
        renames = {k: v for k, v in renames.items() if k != v}
        if renames:
            rd = rd.rename(columns=renames)

        left_columns = oset(ld.columns)
        if tensor_pair_constraints:
            left_group_id, right_group_id, match = _constrained_match(
                ld, rd, match_columns, tensor_pair_constraints
            )
            df = (
                ld.assign(**{_LEFT_GROUP: left_group_id})
                .merge(match, on=_LEFT_GROUP)
                .merge(
                    rd.assign(**{_RIGHT_GROUP: right_group_id}),
                    on=_RIGHT_GROUP,
                    suffixes=["", "_RIGHT_MERGE"],
                )
            )
        elif match_columns:
            df = pd.merge(
                ld,
                rd,
                how="inner",
                left_on=[a for a, _ in match_columns],
                right_on=[b for _, b in match_columns],
                suffixes=["", "_RIGHT_MERGE"],
            )
        else:
            df = pd.merge(ld, rd, how="cross", suffixes=["", "_RIGHT_MERGE"])

        dropcols = [c for c in df.columns if c.startswith(_MATCH_GROUP)]
        if dropcols:
            df = df.drop(columns=dropcols)

        # TODO: We're going to be replacing this with the loops-match algo
        if paired_columns and len(df):
            mask = np.ones(len(df), dtype=bool)
            for a, b in paired_columns:
                if b in left_columns:
                    b = f"{b}_RIGHT_MERGE"
                mask &= df[a].values == df[b].values
            if not mask.all():
                df = df[mask].reset_index(drop=True)

        df = _numeric_cast(df)

        # Pre-calculate bc we're going to be dropping some of the iterations columns
        # before we use this
        iters_above = {
            t: self._tensor_reservation_iters_above(t, df)
            for t in itertools.chain(
                compatibility_left.tensors,
                compatibility_right.tensors,
                compatibility_joined.tensors,
            )
        }

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

        # Merge the two sides' reservations. To keep things inclusive, each side must
        # include reservations from the other side.
        df = ReservationDataFrame(df)
        for datas in df._get_all_reservations().values():
            from_left = [r for r in datas if r.index < _RIGHT_RESERVATION_OFFSET]
            from_right = [r for r in datas if r.index >= _RIGHT_RESERVATION_OFFSET]
            writes = {}
            # Rights live at the same time --> at-or-above
            for r in from_right:
                parent = _reservation_size_at_or_above(from_left, r.iters_above)
                writes[r.right_col] = r.right + parent
                assert r.left_col is None, "Right side has left reservations"
            for r in from_left:
                parent = _reservation_size_at_or_above(from_right, r.iters_above)
                writes[r.right_col] = r.right + parent

                # Left-of-left is dead --> not alive at current level, but still alive
                # at the same time as things above its level (which were alive in the
                # past when left was still around)
                if r.left_col is not None:
                    parent = _reservation_size_above(from_right, r.iters_above)
                    writes[r.left_col] = r.left + parent
            for col, values in writes.items():
                df.loc[:, col] = values

        df = ReservationDataFrame(df)
        df.set_reservation_column_numbers()

        # For everything else: Simple add
        dropcols = [c for c in df.columns if c.endswith("_RIGHT_MERGE")]
        for source in dropcols:
            target = source[: -len("_RIGHT_MERGE")]
            if is_tensor_col(target):
                continue
            if not col_used_in_pareto(target):
                raise ValueError(f"{target} is not used in pareto")
            if not is_reservation_col(target):
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
        # Remove tensors that were allocated in both branches and got added together.
        shared_to_free = []
        for name in compatibility_left.tensor_names & compatibility_right.tensor_names:
            s = compatibility_left.get_reservation_of_tensor(name)
            if s.above_loop_index <= shared_loop_index and s.name not in do_not_free:
                shared_to_free.append(s)
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
            iters_above=iters_above,
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

        result.free_to_reservations(compatibility_joined, shift_bottom_left=False)
        if not CHECK_CORRECTNESS:
            result.limit_capacity(ignored_resources=ignored_resources)
        result.drop_redundant_reservations()
        if _pmapping_row_filter_function is not None:
            result = result.filter_rows(_pmapping_row_filter_function)
        result.make_pareto()

        return result

    def _tensor_reservation_iters_above(
        self, t: TensorReservation, df: DataFrame = None
    ):
        df = self.data if df is None else df
        if t.persistent:
            return 0
        result = 1
        for l in t.iter_fused_loops():
            result = result * df[l.tile_pattern.calculated_n_iterations]
        return result

    def _compatibility_iters_above(self, compatibility: Compatibility):
        result = 1
        for t in compatibility.tensors:
            result = np.maximum(result, self._tensor_reservation_iters_above(t))
        return result

    @error_check_wrapper
    def _adjust_reservations_one_resource(
        self,
        resource: str,
        alloc: Iterable[TensorReservation],
        free: Iterable[TensorReservation],
        iters_above: dict[TensorReservation, Any],
    ):
        alloc, free = list(alloc), list(free)
        if len(self.data) == 0:
            return

        df = ReservationDataFrame(self.data)

        for t, negate in [(t, False) for t in alloc] + [(t, True) for t in free]:
            df.alloc_resource(
                resource,
                size=self.data[tensor2col(t.name)] * (-1 if negate else 1),
                n_iters_above=iters_above[t],
            )

        for r in df._get_all_reservations().get(resource, []):
            assert (df[r.right_col] >= 0).all(), f"Negative reservation: {r.right_col}"

    @error_check_wrapper
    def adjust_reservations(
        self,
        alloc: Iterable[TensorReservation],
        free: Iterable[TensorReservation],
        ignored_resources: set[str],
        iters_above: dict[TensorReservation, Any],
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
                self._adjust_reservations_one_resource(
                    resource, cur_alloc, cur_free, iters_above
                )

    @staticmethod
    def concat(
        paretos: list["PmappingDataframe"], skip_pareto: bool = False
    ) -> "PmappingDataframe":
        if len(paretos) == 0:
            raise ValueError("No paretos to concatenate")
        if len(paretos) == 1:
            return paretos[0]

        concatenated = pd.concat([p.data for p in paretos]).reset_index(drop=True)
        concatenated = ReservationDataFrame(concatenated)
        concatenated.fill_missing_rows()

        p = PmappingDataframe(
            _fillna_and_numeric_cast(concatenated, 0),
            skip_pareto=True,
            n_total_pmappings=sum(p.n_total_pmappings for p in paretos),
            n_valid_pmappings=sum(p.n_valid_pmappings for p in paretos),
            ignored_resources=next(iter(paretos)).ignored_resources,
            drop_valid_reservations=next(iter(paretos)).drop_valid_reservations,
        )
        if not (p.drop_redundant_reservations() or skip_pareto):
            p.make_pareto()
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
        ignored_resources: set[str] = oset(),
        finished: bool = False,
    ):
        dropcols = []
        reservations = ReservationDataFrame(self.data)._get_all_reservations()
        tolerance = self.excess_resource_tolerance
        for resource, datas in sorted(reservations.items()):
            r = datas[-1]  # Only check the greatest-index level
            size_cols = [r.right_col]
            if r.left_col is not None:
                size_cols.append(r.left_col)
            for col in size_cols:
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
                # CAN'T DROP RESERVATIONS UNTIL WE'RE FINISHED JOINING. Persistent
                # tensors may get saved later and would live at the same time as these
                # reservations.
                finished
                and self.drop_valid_reservations
                and resource not in ignored_resources
                and (
                    tolerance == 0
                    or not any(self.data[col].max() > 1 for col in size_cols)
                )
            ):
                # If we're finished, there should only be 1 level
                assert len(datas) == 1
                dropcols += size_cols + [r.iters_above_col]

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
        return any(col2reservationsize(c) is not None for c in self.data.columns)

    # ============================================================================
    # Checking functions
    # ============================================================================
    def check_above_subset_below(self, live_tensors: set[str] = fzs()):
        assert not self.data.isnull().values.any(), f"NaN in {self.data}"
        if len(self.data) != 0:
            ReservationDataFrame(self.data).assert_reservations_ordered()

    def filter_rows(
        self, _pmapping_row_filter_function: Callable[[Series], bool] | None = None
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
    #             col = get_reservation_or_parent(k, 0, left=True)
    #             if str(k) == "0":
    #                 continue
    #             if col not in self.data.columns:
    #                 got = r[[c for c in self.data.columns if col2reservationsize(c) is not None]]
    #                 self.fail(i, live_tensors)
    #                 raise ValueError(f"Missing {k}: Expected {reservations}. Got: {got}")
    #             if r[col] != v:
    #                 got = r[[c for c in self.data.columns if col2reservationsize(c) is not None]]
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
    row: Series,
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
