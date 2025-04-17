from collections import defaultdict
import copy
import itertools
import re

# Disable numba. We need user_has_package("numba") to be False
import sys
from typing import Optional, Tuple, Union

from joblib import delayed

from fastfusion.joining.mappinginfo import TensorStorage
from fastfusion.util import fzs

sys.modules["numba"] = None


from paretoset import paretoset

import pandas as pd
import functools

LOGSTRING = "__Mappings"
MAPPING = "__LOOPNEST"
STATS = "__STATS"
OCCUPANCY = "__Occupancy"
TENSORS = "__TENSORS"
IN_PROGRESS_STATS = "__IN_PROGRESS_STATS"
MAPPING_HASH = "__MAPPING_HASH"
TAGS = "__TAGS"
VALID = "__VALID"
PER_COMPONENT_ACCESSES_ENERGY = "Per-Component Energy"

DICT_COLUMNS = set(
    [
        LOGSTRING,
        MAPPING,
        STATS,
        TENSORS,
        IN_PROGRESS_STATS,
        MAPPING_HASH,
        TAGS,
        PER_COMPONENT_ACCESSES_ENERGY,
    ]
)
RESERVED_COLUMNS = set([VALID]) | DICT_COLUMNS

TUPLABE_COLUMNS = set([MAPPING, TENSORS])

CHECK_CORRECTNESS = True

_resource_name_nloops_reg = re.compile(r"RESOURCE_(.+?)(?:_LEFT)?_LEVEL_(-?\d+)")


def dict_cached(func):
    cache = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = (args, fzs(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return wrapper


# TODO: Make these tuples?


@dict_cached
def col2nameloop(x):
    m = _resource_name_nloops_reg.match(x)
    return (m.group(1), int(m.group(2))) if m is not None else None


@dict_cached
def nameloop2col(name, nloops, left: bool = False):
    if left:
        return f"RESOURCE_{name}_LEFT_LEVEL_{nloops}"
    return f"RESOURCE_{name}_LEVEL_{nloops}"


@dict_cached
def is_left_col(x):
    return "_LEFT_LEVEL_" in x


MERGE_SUFFIXES = ["", "_RIGHT_MERGE"]


def is_merge_col(c):
    return any(c.endswith(s) for s in MERGE_SUFFIXES)


def add_to_col(df, target, source):
    if target in df:
        df.loc[:, target] = df[target] + df[source]
    else:
        df.loc[:, target] = df[source]


def max_to_col(df, target, source):
    if target in df:
        df.loc[:, target] = df[[target, source]].max(axis=1)
    else:
        df.loc[:, target] = df[source]


def is_special_col(c):
    return c in RESERVED_COLUMNS or col2nameloop(c) is not None


# Above index 0: Freed when Einsum fully terminates
# Above index 1: Freed after each iteration of the outermost loop

# -1 -> global resource
# 0 -> einsum only

# Shared index -1: Sum -1 resources, max everyone below
# Shared index 0: Sum 0 resources, max everyone below


def quick_pareto(df):
    df["chosen"] = False
    i = 0
    while i != len(df):
        i += 1
        # Pick the entry which is not chosen & has the lowest first columns
        idx = df.loc[~df["chosen"], df.columns[0]].idxmin()
        df.loc[idx, "chosen"] = True
        dominated = (df.iloc[:, :-1] >= df.loc[idx, df.columns[:-1]]).all(axis=1)
        df = df[~dominated]
    return df.drop(columns=["chosen"])


def makepareto(
    data: pd.DataFrame,
    reverse_free: bool = True,
) -> pd.DataFrame:
    # Drop any columns that are all zeros or all equal
    columns = [
        c for c in data.columns if c not in RESERVED_COLUMNS and not is_merge_col(c)
    ]
    for c in list(columns):
        if not data[c].any():
            data = data.drop(columns=[c])
            columns.remove(c)
        elif data[c].nunique() == 1:
            columns.remove(c)

    if len(data) == 1:
        return data

    # if reverse_free:
    #     df2 = _reverse_free(data[columns].copy())
    #     return data[paretoset(df2)].reset_index(drop=True)

    return data[paretoset(data[columns])].reset_index(drop=True)


# def _reverse_free(data: pd.DataFrame) -> pd.DataFrame:
#     resource_name_to_max_level = defaultdict(int)
#     resource_name_to_min_level = defaultdict(int)
#     keep_columns = [
#         c for c in data.columns if col2nameloop(c) is None and c not in RESERVED_COLUMNS
#     ]
#     for c in data.columns:
#         if (name_nloops := col2nameloop(c)) is not None:
#             if is_left_col(c):
#                 keep_columns.append(c)
#             else:
#                 name, n = name_nloops
#                 for f, target in (
#                     (max, resource_name_to_max_level),
#                     (min, resource_name_to_min_level),
#                 ):
#                     target.setdefault(name, n)
#                     target[name] = f(target[name], n)

#     for name in resource_name_to_max_level:
#         min_level, max_level = (
#             resource_name_to_min_level[name],
#             resource_name_to_max_level[name],
#         )
#         for i in range(min_level, max_level):
#             target = nameloop2col(name, i)
#             next_target = nameloop2col(name, i + 1)
#             next_target_left = nameloop2col(name, i + 1, left=True)

#             if next_target_left in data:
#                 add_to_col(data, next_target_left, target)

#             if next_target in data:
#                 add_to_col(data, next_target, target)
#                 keep_columns.append(target)
#             else:
#                 data.rename(columns={target: next_target}, inplace=True)

#             if next_target_left in data:
#                 max_to_col(data, next_target_left, next_target)

#         keep_columns.append(nameloop2col(name, max_level))

#     return data[keep_columns]


def squish_left_right(
    data: pd.DataFrame, shared_loop_index: int = None, return_needs_pareto: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, bool]]:
    nloops2left = defaultdict(set)
    dropcols = []
    needs_pareto = False
    for c in data.columns:
        if (name_nloops := col2nameloop(c)) is not None:
            if is_left_col(c):
                name, nloops = name_nloops
                if shared_loop_index is None or nloops == shared_loop_index:
                    nloops2left[nloops].add((c, name))
                    dropcols.append(c)
                    needs_pareto = True

    for n in nloops2left.keys():
        for c, name in nloops2left[n]:
            target = nameloop2col(name, n)
            max_to_col(data, target, c)
    if return_needs_pareto:
        return data[[c for c in data.columns if c not in dropcols]], needs_pareto
    return data[[c for c in data.columns if c not in dropcols]]


def paretofy_by(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return data[paretoset(data[columns])].reset_index(drop=True)


def draw_looptree(row: pd.DataFrame, live_tensors: set[int]):
    from fastfusion.visualization.reservationtree import mappings2reservationtree

    looptree = mappings2reservationtree(
        row[MAPPING],
        row.get(STATS, None),
        skip_backing_tensors_in_right_branch=live_tensors,
        still_live_tensors=live_tensors,
    )
    import pydot

    graph = pydot.Dot(graph_type="digraph", ranksep="0.2", nodesep="0.2")
    looptree.to_pydot(graph)
    with open(f"test.png", "wb") as f:
        f.write(graph.create_png())


# def check_correctness(data: pd.DataFrame, live_tensors: set[int]):
#     from fastfusion.visualization.reservationtree import mappings2reservationtree
#     from fastfusion.joining.sim import TensorStorage

#     def fail(index):
#         draw_looptree(data.iloc[index], live_tensors)
#         all_tensors = set(t for tn in r[MAPPING].values() for t in tn.storage)
#         all_tensors = TensorStorage.get_backing_stores(all_tensors)
#         for t in sorted(all_tensors):
#             print(f"{t.__repr__()},")

#     df_check = _free_to_loop_index(data.copy(), -1)
#     for i, r in df_check.iterrows():
#         looptree = mappings2reservationtree(
#             r[MAPPING],
#             r.get(STATS, None),
#             skip_backing_tensors_in_right_branch=live_tensors,
#             still_live_tensors=live_tensors,
#         )
#         reservations = dict(looptree.get_reservations())
#         for k, v in reservations.items():
#             col = nameloop2col(k, -1)
#             if str(k) == "0":
#                 continue
#             if col not in df_check.columns:
#                 got = r[[c for c in df_check.columns if col2nameloop(c) is not None]]
#                 fail(i)
#                 raise ValueError(f"Missing {k}: Expected {reservations}. Got: {got}")
#             if r[col] != v:
#                 got = r[[c for c in df_check.columns if col2nameloop(c) is not None]]
#                 fail(i)
#                 looptree = mappings2reservationtree(
#                     r[MAPPING],
#                     r.get(STATS, None),
#                     # skip_backing_tensors_in_right_branch=live_tensors,
#                     still_live_tensors=live_tensors,
#                 )
#                 raise ValueError(
#                     f"Mismatched {k}: {v} != {r[col]}. Expected {reservations}. Got: {got}"
#                 )


class Pareto:
    def __init__(
            self, 
            data: pd.DataFrame, 
            skip_pareto: bool = False, 
            fill_reservations: bool = False,
            check_format: bool = True,
            max_right_to_left: bool = False,
        ):
        self._data: pd.DataFrame = data
        self.right_reservations: dict[set] = None
        self.left_reservations: dict[set] = None
        self._make_reservations()

        if fill_reservations: # Affects Pareto so must go before
            self.fill_reservations()

        if max_right_to_left: # Affects Pareto so must go before
            self.max_right_to_left()

        if not skip_pareto:
            self._data = makepareto(self.data)

            
        if check_format:
            self.check_format()
            
    def fill_reservations(self):
        for resource, reservations in self.left_reservations.items():
            prev = None
            for r in sorted(reservations):
                if prev is not None:
                    source = nameloop2col(resource, prev, left=True)
                    target = nameloop2col(resource, r, left=True)
                    max_to_col(self.data, target, source)
                prev = r
        for resource, reservations in self.right_reservations.items():
            prev = None
            for r in sorted(reservations):
                if prev is not None:
                    source = nameloop2col(resource, prev)
                    target = nameloop2col(resource, r)
                    max_to_col(self.data, target, source)
                prev = r
                
    def check_format(self):
        targets = []
        
        
        for left, reservations_dict in [
            (True, self.left_reservations),
            (False, self.right_reservations),
        ]:
            for resource, reservations in reservations_dict.items():
                for r in reservations:
                    above = self.get_reservation_at_level(resource, r - 1)
                    if above is not None:
                        below = nameloop2col(resource, r, left=left)
                        targets.append((above, below))
                
        for above, below in targets:
            if (self.data[below] < self.data[above]).any():
                first_failing_index = (self.data[below] < self.data[above]).idxmax()
                fail_row = self.data.iloc[first_failing_index]
                error = f"""
                {above} column is less than {below} column. A reservation at
                a level should include all reservations above it. There were {len(fail_row)} rows
                with this error. One example: {fail_row}
                """
                raise ValueError(error)
            
    def max_right_to_left(self):
        for resource, reservations in self.left_reservations.items():
            for r in reservations:
                source = self.get_reservation_at_level(resource, r)
                if source is None:
                    continue
                target = nameloop2col(resource, r, left=True)
                max_to_col(self.data, target, source)

    @property
    def data(self) -> pd.DataFrame:
        return self._data
        
    def _make_reservations(self) -> Tuple[dict[str, list], dict[str, list]]:
        """
        Create a dictionary of reservations for each resource.
        The dictionary keys are the resource names and the values are lists
        of column names for each loop index.
        """
        self.left_reservations, self.right_reservations = {}, {}
        for c in self.data.columns:
            if (name_nloops := col2nameloop(c)) is not None:
                name, nloops = name_nloops
                target = self.left_reservations if is_left_col(c) else self.right_reservations
                target.setdefault(name, set()).add(nloops)
                assert nloops >= 0
                
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
        drop_columns = []
        for resource in set(self.left_reservations) | set(self.right_reservations):
            max_columns = []
            
            left_reservations = self.left_reservations.get(resource, set())
            right_reservations = self.right_reservations.get(resource, set())
            left_big_enough = [l for l in left_reservations if l >= loop_index + 1]
            right_big_enough = [r for r in right_reservations if r >= loop_index + 2] # + 1 is target

            if len(right_big_enough) > 1: # All ones above the last are subsets
                right_biggest = max(right_big_enough)
                right_big_enough.remove(right_biggest)
                drop_columns += [nameloop2col(resource, r) for r in right_big_enough]
                right_big_enough = [right_biggest]

            max_columns = [nameloop2col(resource, r) for r in right_big_enough] + [nameloop2col(resource, l, left=True) for l in left_big_enough]

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
        return len(drop_columns) != 0
    
    def get_reservation_at_level(self, name: str, level: int, left: bool = False) -> Optional[str]:
        reservations = self.left_reservations if left else self.right_reservations
        if name not in reservations:
            return None
        reservations = reservations[name]
        while level >= 0:
            if level in reservations:
                return nameloop2col(name, level, left)
            level -= 1
        return None

    def shift_bottom_reservation_left(self, shared_loop_index: int):
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
    
    def merge_next(
        self,
        right: "Pareto",
        shared_loop_index: int,
        live_tensors: set[int],
        shared_storage: set[TensorStorage],
        _raise_exceptions: bool = False,
    ) -> "Pareto":
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
        self.free_to_loop_index(shared_loop_index)
        self.shift_bottom_reservation_left(shared_loop_index)
        self._make_reservations()
        right._make_reservations()
        

        assert not right.left_reservations, f"{right.left_reservations} is not None"

        for resource, reservations in self.right_reservations.items():
            n_reservations = max(reservations, default=-1)
            assert n_reservations <= shared_loop_index, f"{resource}: {reservations} > {shared_loop_index}"

        for resource, reservations in self.left_reservations.items():
            n_reservations = max(reservations, default=-1)
            assert n_reservations <= shared_loop_index + 1, f"{resource}: {reservations} > {shared_loop_index}"
            
        max_nloops = max(
            shared_loop_index,
            max((max(r, default=-1) for r in self.right_reservations.values()), default=-1),
            max((max(r, default=-1) for r in self.left_reservations.values()), default=-1),
            max((max(r, default=-1) for r in right.right_reservations.values()), default=-1),
            max((max(r, default=-1) for r in right.left_reservations.values()), default=-1),
        )

        df = pd.merge(self.data, right.data, how="cross", suffixes=MERGE_SUFFIXES)

        # Make sure everything is done in increasing loop order so we don't have
        # read-after-write hazards
        for nloops in range(max_nloops, -1, -1):
            def iter_reservations(reservations_dict):
                for resource in reservations_dict:
                    if nloops in reservations_dict[resource]:
                        yield resource
            
            # For the RIGHT tree, RIGHT reservations: If there is no matching node in the left
            # tree, add the above-this-level reservation from the left tree. If there is a matching
            # node in the left tree, then we'll add this node to it in the next step.
            for resource in iter_reservations(right.right_reservations):
                if (source := self.get_reservation_at_level(resource, nloops - 1)) is None:
                    continue
                target = nameloop2col(resource, nloops)
                # If there's a merged version column, then it's in both trees
                if target + "_RIGHT_MERGE" in df:
                    continue
                df.loc[:, target] += df[source]
            # For LEFT tree, LEFT reservations: Add the immediately-above
            # reservation from the right tree.
            for resource in iter_reservations(self.left_reservations):
                if (source := right.get_reservation_at_level(resource, nloops - 1)) is None:
                    continue
                right_merge_source = source + "_RIGHT_MERGE"
                target = nameloop2col(resource, nloops, left=True)
                if source is not None:
                    df.loc[:, target] += df[right_merge_source if right_merge_source in df else source]
            # For LEFT tree, RIGHT reservations: Add the same-level reservation from
            # the right tree. This will double-count reservations that are in both branches,
            # so we remove them later.
            for resource in iter_reservations(self.right_reservations):
                if (source := right.get_reservation_at_level(resource, nloops)) is None:
                    continue
                right_merge_source = source + "_RIGHT_MERGE"
                target = nameloop2col(resource, nloops)
                if source is not None:
                    df.loc[:, target] += df[right_merge_source if right_merge_source in df else source]

        # For everything else: Simple add
        dropcols = [c for c in df.columns if c.endswith("_RIGHT_MERGE")]
        for source in dropcols:
            target = source[:-len("_RIGHT_MERGE")]
            if col2nameloop(target) is not None:
                continue
            if target in DICT_COLUMNS:
                df[target] = (
                    df.apply(lambda row: {**row[target], **row[source]}, axis=1)
                    if len(df) > 0
                    else []
                )
            else:
                df.loc[:, target] += df[source]
        df = df.drop(columns=dropcols)
        try:
            result = Pareto(df, skip_pareto=CHECK_CORRECTNESS, max_right_to_left=True)
        except Exception as e:
            if _raise_exceptions:
                raise e
            print(e)
            self.merge_next(
                right,
                shared_loop_index,
                live_tensors,
                shared_storage=shared_storage,
                _raise_exceptions=True,
            )

        # Remove tensors that were allocated in both branches and got added
        # together. We can do this after pareto calculation because it affects
        # all mappings equally.
        for t in shared_storage:
            if t.above_loop_index > shared_loop_index:
                continue
            result.subtract_duplicated(t)

        if CHECK_CORRECTNESS:
            try:
                result.check_correctness(live_tensors)
            except Exception as e:
                if _raise_exceptions:
                    raise e
                # raise e
                print(e)
                self.merge_next(
                    right,
                    shared_loop_index,
                    live_tensors,
                    shared_storage=shared_storage,
                    _raise_exceptions=True,
                )
            result.make_pareto()
        
        return result
    
    def subtract_duplicated(self, tensor: TensorStorage):
        for resource, reservations in self.right_reservations.items():
            for r in reservations:
                if r >= tensor.above_loop_index:
                    target = nameloop2col(resource, r)
                    self.data.loc[:, target] -= tensor.storage[resource]
        for resource, reservations in self.left_reservations.items():
            for r in reservations:
                if r > tensor.above_loop_index:
                    target = nameloop2col(resource, r, left=True)
                    self.data.loc[:, target] -= tensor.storage[resource]

    def check_correctness(self, live_tensors: set[int]):
        from fastfusion.visualization.reservationtree import mappings2reservationtree
        from fastfusion.joining.sim import TensorStorage
        
        self = self.copy()

        def fail(index):
            draw_looptree(self.data.iloc[index], live_tensors)
            all_tensors = set(t for tn in r[MAPPING].values() for t in tn.storage)
            all_tensors = TensorStorage.get_backing_stores(all_tensors)
            for t in sorted(all_tensors):
                print(f"{t.__repr__()},")
                
        self.free_to_loop_index(-1)
        self.squish_left_right(-1)

        for i, r in self.data.iterrows():
            looptree = mappings2reservationtree(
                r[MAPPING],
                r.get(STATS, None),
                # skip_backing_tensors_in_right_branch=live_tensors,
                still_live_tensors=live_tensors,
            )
            reservations = dict(looptree.get_reservations())
            for k, v in reservations.items():
                col = self.get_reservation_at_level(k, 0)
                if str(k) == "0":
                    continue
                if col not in self.data.columns:
                    got = r[[c for c in self.data.columns if col2nameloop(c) is not None]]
                    fail(i)
                    self.free_to_loop_index(-1)
                    raise ValueError(f"Missing {k}: Expected {reservations}. Got: {got}")
                if r[col] != v:
                    got = r[[c for c in self.data.columns if col2nameloop(c) is not None]]
                    fail(i)
                    looptree = mappings2reservationtree(
                        r[MAPPING],
                        r.get(STATS, None),
                        # skip_backing_tensors_in_right_branch=live_tensors,
                        still_live_tensors=live_tensors,
                    )
                    raise ValueError(
                        f"Mismatched {k}: {v} != {r[col]}. Expected {reservations}. Got: {got}"
                    )
        return


        # df = pd.merge(left, right, how="cross", suffixes=MERGE_SUFFIXES)
        # shared_columns = set(left.columns) & set(right.columns) - RESERVED_COLUMNS
        # dropcols = []
        
        # prev_left = None
        # prev_right = None
        # for resource_name in resources:
        #     for loop_index in range(shared_loop_index):
        #         left = nameloop2col(resource_name, loop_index, left=True) + MERGE_SUFFIXES[0]
        #         right = nameloop2col(resource_name, loop_index) + MERGE_SUFFIXES[1]
        #         df.iloc[:, right] = df[left] + df[right]
                
        
        # resource2column_left, resource2column_right = {}, {}
        # for source, target in [
        #     (left, resource2column_left),
        #     (right, resource2column_right),
        # ]:
        #     for c in source.columns:
        #         if (name_nloops := col2nameloop(c)) is not None:
        #             name, nloops = name_nloops
        #             target.setdefault(name, {})[nloops] = c

        # Pipeline:
        # - Need to share temporal loops up to the spatial loop index
        #   Resources:
        #   - Energy
        #   - PE utilization
        #   - Buf utilization
        #   - Buf accesses (for BW calculation later)

        # - Options:
        #   - Non-pipelined: Sum resources above shared loops, max below.
        #   - Pipelined: Sum resources above shared loops, max below. Sum
        #     PE utilization. Latency is pipeline latency summed.
        #
        #  *  Can't bake into compatiblity unless we have a notion of left vs.
        #     right pipelined.

        # PIPELINE CHANGES REQUIRED:
        # - Latency above above loop index (first tile), below (all subsequent tiles)
        # - Mapping includes information for how may be fused:
        #   - Pipelined: Max below latencies,
        #   - Non-pipelined:
        # Shared resources:
        # -
        # SEQUENTIAL:
        # - In parallel: Fetch all above-shared-loop resources for all operations
        # - Sequentially: Fetch any below-shared-loop resources for all operations
        # PIPELINE:
        # - In parallel: Fetch all above-shared-loop resources for all operations
        # - Sequentially: Fetch any below-shared-loop resources for the first iteration of all operations
        # - In parallel: Fetch all below-shared-loop resources for all operations in all subsequent iterations

        # df = free_to_loop_index(df, next_shared_loop_index + 1)
        # for resource, capacity in resource2capacity.items():
        #     colname = nameloop2col(resource, 0)
        #     if colname in df:
        #         if capacity is not None:
        #             df = df[df[colname] <= capacity]
        #         del df[colname]

        df.drop(columns=dropcols, inplace=True)
        if not CHECK_CORRECTNESS:
            cols = [c for c in df.columns if c in RESERVED_COLUMNS or not is_merge_col(c)]
            if pareto_prune:
                df = makepareto(df)

        for k in DICT_COLUMNS:
            if k not in left.columns:
                continue
            c0, c1 = k + MERGE_SUFFIXES[0], k + MERGE_SUFFIXES[1]
            df[k] = (
                df.apply(lambda row: {**row[c0], **row[c1]}, axis=1) if len(df) > 0 else []
            )
        df = df[[c for c in df.columns if not is_merge_col(c)]]

        cols = [c for c in df.columns if c not in DICT_COLUMNS]

        if IN_PROGRESS_STATS in df.columns:
            first_row = df.iloc[0]
            einsums = list(first_row[IN_PROGRESS_STATS].keys())
            last = einsums[-1]
            for i, r in df[cols].iterrows():
                df.at[i, IN_PROGRESS_STATS][last] = r.to_dict()

        if CHECK_CORRECTNESS:
            check_correctness(df, live_tensors)
            if pareto_prune:
                df = makepareto(df)

        # Assert no NaNs
        assert not df.isnull().values.any()

        return Pareto(df, skip_pareto=True) if as_pareto else df

    def add_tensor(self, tensor):
        if len(self.data) == 0:
            return
        if TENSORS in self.data:
            last_einsum = list(self.data.iloc[0][TENSORS].keys())[-1]
            if tensor in self.data[TENSORS].iloc[0][last_einsum]:
                return
            for t in self.data[TENSORS]:
                t[last_einsum].append(tensor)

    def einsum_ids(self):
        return fzs(self.data[LOGSTRING].iloc[0].keys())

    @staticmethod
    def concat(paretos: list["Pareto"], skip_pareto: bool = False) -> "Pareto":
        return Pareto(
            pd.concat([p.data for p in paretos]).fillna(0),
            skip_pareto=len(paretos) == 1 or skip_pareto,
            fill_reservations=True,
        )

    def copy(self) -> "Pareto":
        return Pareto(self.data.copy())

    def limit_capacity(
        self, n: int, resource2capacity: dict[str, Optional[int]]
    ) -> bool:
        resource2capacity = resource2capacity or {}
        if resource2capacity:
            assert all(isinstance(v, str) for v in resource2capacity.keys())
        for c in self.data.columns:
            if (name_nloops := col2nameloop(c)) is not None:
                name, nloops = name_nloops
                if name in resource2capacity:
                    capacity = resource2capacity.get(name)
                    if capacity is not None:
                        self.data = self.data[self.data[c] <= capacity]
                        if nloops == n:
                            del self.data[c]
                else:
                    del self.data[c]
        self._make_reservations()

    def squish_left_right(self, shared_loop_index: int = None) -> bool:
        needs_pareto = False
        for resource, reservations in self.left_reservations.items():
            if shared_loop_index + 1 not in reservations:
                continue
            self.right_reservations.setdefault(resource, set())
            source = nameloop2col(resource, shared_loop_index + 1, left=True)
            target = nameloop2col(resource, shared_loop_index + 1)
            self.right_reservations[resource].add(shared_loop_index + 1)
            self.left_reservations[resource].remove(shared_loop_index + 1)
            
            if shared_loop_index + 1 in self.right_reservations[resource]:
                max_to_col(self.data, target, source)
                needs_pareto = True
            else:
                self.data.rename(columns={source: target}, inplace=True)
            
        return needs_pareto

    def filter_by_mapping_hashes(self, hashes: set[int]):
        self.data = self.data[
            self.data[MAPPING_HASH].apply(
                lambda x: all(i in hashes for i in x.values())
            )
        ]
        return self

    def make_pareto(self):
        self._data = makepareto(self.data)
        self._make_reservations()

    def has_reservations(self):
        return any(col2nameloop(c) is not None for c in self.data.columns)

    def get_reservations(self):
        return tuple(
            sorted(c for c in self.data.columns if col2nameloop(c) is not None)
        )


import unittest


class ParetoTest(unittest.TestCase):
    def test_pareto(self):
        occ_key = nameloop2col("GLB", 5)
        data = pd.DataFrame({"A": [1, 2], occ_key: [2, 1], LOGSTRING: [{"A": "A"}] * 2})
        Pareto(data)

    def test_vertical_combine(self):
        occ_key = nameloop2col("GLB", 5)
        data1 = pd.DataFrame(
            {
                "A": [1, 3, 3],
                "B": [3, 1, 3],
                occ_key: [3, 3, 3],
                LOGSTRING: [{"A": "A"}] * 3,
            }
        )
        data2 = pd.DataFrame(
            {
                "A": [3, 3, 3],
                "B": [3, 3, 3],
                occ_key: [3, 3, 1],
                LOGSTRING: [{"A": "A"}] * 3,
            }
        )

        p1 = Pareto(data1)
        self.assertEqual(len(p1.data), 2)
        p2 = Pareto(data2)
        self.assertEqual(len(p2.data), 1)
        pd12 = Pareto.concat([p1, p2])
        self.assertEqual(len(pd12.data), 3)

    def test_merge(self):
        data1 = pd.DataFrame(
            {"A": [1, 3, 3], "B": [3, 1, 3], LOGSTRING: [{"A": "A"}] * 3}
        )
        data2 = pd.DataFrame(
            {"A": [3, 3, 3], "B": [3, 3, 3], LOGSTRING: [{"A": "A"}] * 3}
        )
        p = Pareto(data1).merge_next(Pareto(data2), 0)
        d = p.data
        self.assertEqual(d["A"].tolist(), [4, 6])
        self.assertEqual(d["B"].tolist(), [6, 4])

    def test_merge_shared_resources(self):
        occ_key = nameloop2col("GLB", 4)
        data1 = pd.DataFrame(
            {
                "A": [1, 3, 3],
                "B": [3, 1, 3],
                LOGSTRING: [{"A": "A"}] * 3,
                occ_key: [3, 3, 3],
            }
        )
        data2 = pd.DataFrame(
            {
                "A": [3, 3, 3],
                "B": [3, 3, 3],
                LOGSTRING: [{"A": "A"}] * 3,
                occ_key: [2, 2, 2],
            }
        )
        p = Pareto(data1).merge_next(Pareto(data2), 5)
        d = p.data
        self.assertEqual(d["A"].tolist(), [4, 6])
        self.assertEqual(d["B"].tolist(), [6, 4])
        self.assertEqual(d[occ_key].tolist(), [5, 5])

        p2 = Pareto(data1).merge_next(Pareto(data2), 3)
        d = squish_left_right(p2.data)
        self.assertEqual(d["A"].tolist(), [4, 6])
        self.assertEqual(d["B"].tolist(), [6, 4])
        self.assertEqual(d[occ_key].tolist(), [3, 3])

    def test_merge_shared_resources_2nloops(self):
        occ_key_1 = nameloop2col("GLB", 0)
        occ_key_2 = nameloop2col("GLB", 1)
        data1 = pd.DataFrame(
            {
                "A": [1, 3, 3],
                "B": [3, 1, 3],
                LOGSTRING: [{"A": "A"}] * 3,
                occ_key_1: [3, 3, 3],
                occ_key_2: [8, 8, 8],
            }
        )
        data2 = pd.DataFrame(
            {
                "A": [3, 3, 3],
                "B": [3, 3, 3],
                LOGSTRING: [{"A": "A"}] * 3,
                occ_key_1: [4, 4, 4],
                occ_key_2: [6, 6, 6],
            }
        )

        # 0 --> GLOBAL RESOURCE
        # 1 --> Shared with all who share ONE loop

        # occ_key_1       occ_key_1    Level 0 shared
        # for             for          Co-tiled with nloops 1 merge
        # occ_key_2       occ_key_2    Level 1 shared
        # for             for          Co-tiled with nloops 2 merge

        # Untiled fused --> Max everything
        d = Pareto(data1).merge_next(Pareto(data2), -1).data
        d = squish_left_right(d)
        self.assertEqual(d[occ_key_1].tolist(), [11, 11])

        # Tiled at nloops 1 --> Sum everything stored at 0
        d = Pareto(data1).merge_next(Pareto(data2), 0).data
        d = squish_left_right(d)
        self.assertEqual(d[occ_key_1].tolist(), [7, 7])
        self.assertEqual(d[occ_key_2].tolist(), [8, 8])

        # Tiled at nloops 2 --> Sum everything stored at 0 and 1
        d = Pareto(data1).merge_next(Pareto(data2), 1).data
        d = squish_left_right(d)
        self.assertEqual(d[occ_key_1].tolist(), [7, 7])
        self.assertEqual(d[occ_key_2].tolist(), [14, 14])

    def test_free_to_loop_index(self):
        # 0 --> Untiled fused
        occ_key_1 = nameloop2col("GLB", 0)
        occ_key_2 = nameloop2col("GLB", 1)
        data1 = pd.DataFrame(
            {
                "A": [1, 3, 3],
                "B": [3, 1, 3],
                LOGSTRING: [{"A": "A"}] * 3,
                occ_key_1: [3, 3, 3],
                occ_key_2: [8, 8, 8],
            }
        )

        p = Pareto(data1)
        d = p.data
        p.free_to_loop_index(2)
        self.assertEqual(
            d.columns.tolist(), ["A", "B", LOGSTRING, occ_key_1, occ_key_2]
        )

        p.free_to_loop_index(0)
        d = p.data
        self.assertEqual(d.columns.tolist(), ["A", "B", LOGSTRING, occ_key_1])
        self.assertEqual(d[occ_key_1].tolist(), [11, 11])


if __name__ == "__main__":
    unittest.main()
