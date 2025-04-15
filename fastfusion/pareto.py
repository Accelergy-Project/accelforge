from collections import defaultdict
import copy
import itertools
import re

# Disable numba. We need user_has_package("numba") to be False
import sys
from typing import Optional, Tuple, Union

from joblib import delayed

from fastfusion.util.util import fzs

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

CHECK_CORRECTNESS = False


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


MERGE_SUFFIXES = ["_LEFT_MERGE", "_RIGHT_MERGE"]


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

    if reverse_free:
        df2 = _reverse_free(data[columns].copy())
        return data[paretoset(df2)].reset_index(drop=True)

    return data[paretoset(data[columns])].reset_index(drop=True)


def _reverse_free(data: pd.DataFrame) -> pd.DataFrame:
    resource_name_to_max_level = defaultdict(int)
    resource_name_to_min_level = defaultdict(int)
    keep_columns = [
        c for c in data.columns if col2nameloop(c) is None and c not in RESERVED_COLUMNS
    ]
    for c in data.columns:
        if (name_nloops := col2nameloop(c)) is not None:
            if is_left_col(c):
                keep_columns.append(c)
            else:
                name, n = name_nloops
                for f, target in (
                    (max, resource_name_to_max_level),
                    (min, resource_name_to_min_level),
                ):
                    target.setdefault(name, n)
                    target[name] = f(target[name], n)

    for name in resource_name_to_max_level:
        min_level, max_level = (
            resource_name_to_min_level[name],
            resource_name_to_max_level[name],
        )
        for i in range(min_level, max_level):
            target = nameloop2col(name, i)
            next_target = nameloop2col(name, i + 1)
            next_target_left = nameloop2col(name, i + 1, left=True)

            if next_target_left in data:
                add_to_col(data, next_target_left, target)

            if next_target in data:
                add_to_col(data, next_target, target)
                keep_columns.append(target)
            else:
                data.rename(columns={target: next_target}, inplace=True)

            if next_target_left in data:
                max_to_col(data, next_target_left, next_target)

        keep_columns.append(nameloop2col(name, max_level))

    return data[keep_columns]


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


def _free_to_loop_index(
    data: pd.DataFrame,
    shared_loop_index: int,
    return_needs_pareto: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, bool]]:
    nloops2left = defaultdict(set)
    nloops2right = defaultdict(set)
    for c in data.columns:
        if (name_nloops := col2nameloop(c)) is not None:
            name, nloops = name_nloops
            target = nloops2left if is_left_col(c) else nloops2right
            target[nloops].add((c, name))

    max_nloops = max(
        max(nloops2left.keys(), default=-1), max(nloops2right.keys(), default=-1)
    )
    needs_pareto = False
    for n in range(max_nloops, shared_loop_index, -1):
        # LEFT data: Max to the same level on the right
        for c, name in nloops2left[n]:
            target = nameloop2col(name, n)
            if target in data:
                max_to_col(data, target, c)
                needs_pareto = True
            else:
                # This does change things but we won't need another pareto
                data.rename(columns={c: target}, inplace=True)
            nloops2right[n].add((target, name))
        # RIGHT data: Sum to the level below on the right
        for c, name in nloops2right[n]:
            target = nameloop2col(name, n - 1)
            if target in data:
                add_to_col(data, target, c)
                needs_pareto = True
            else:
                # This does change things but we won't need another pareto
                data.rename(columns={c: target}, inplace=True)
            nloops2right[n - 1].add((target, name))

    keepcols = []
    for c in data.columns:
        if (name_nloops := col2nameloop(c)) is not None:
            name, nloops = name_nloops
            if nloops <= shared_loop_index:
                keepcols.append(c)
        else:
            keepcols.append(c)

    needs_pareto = needs_pareto or (len(keepcols) != len(data.columns))
    data = data[keepcols]

    return (data, needs_pareto) if return_needs_pareto else data


def paretofy_by(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return data[paretoset(data[columns])].reset_index(drop=True)


def draw_looptree(row: pd.DataFrame, live_tensors: set[int]):
    from fastfusion.visualization.reservationtree import tilings2looptree

    looptree = tilings2looptree(
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


def check_correctness(data: pd.DataFrame, live_tensors: set[int]):
    from fastfusion.visualization.reservationtree import tilings2looptree
    from fastfusion.sim import TensorStorage

    def fail(index):
        draw_looptree(data.iloc[index], live_tensors)
        all_tensors = set(t for tn in r[MAPPING].values() for t in tn.storage)
        all_tensors = TensorStorage.get_backing_stores(all_tensors)
        for t in sorted(all_tensors):
            print(f"{t.__repr__()},")

    df_check = _free_to_loop_index(data.copy(), -1)
    for i, r in df_check.iterrows():
        looptree = tilings2looptree(
            r[MAPPING],
            r.get(STATS, None),
            skip_backing_tensors_in_right_branch=live_tensors,
            still_live_tensors=live_tensors,
        )
        reservations = dict(looptree.get_reservations())
        for k, v in reservations.items():
            col = nameloop2col(k, -1)
            if str(k) == "0":
                continue
            if col not in df_check.columns:
                got = r[[c for c in df_check.columns if col2nameloop(c) is not None]]
                fail(i)
                raise ValueError(f"Missing {k}: Expected {reservations}. Got: {got}")
            if r[col] != v:
                got = r[[c for c in df_check.columns if col2nameloop(c) is not None]]
                fail(i)
                looptree = tilings2looptree(
                    r[MAPPING],
                    r.get(STATS, None),
                    skip_backing_tensors_in_right_branch=live_tensors,
                    still_live_tensors=live_tensors,
                )
                raise ValueError(
                    f"Mismatched {k}: {v} != {r[col]}. Expected {reservations}. Got: {got}"
                )


def merge_cross(
    left: pd.DataFrame,
    right: pd.DataFrame,
    shared_loop_index: int,
    live_tensors: set[int],
    as_pareto: bool = False,
    pareto_prune: bool = True,
) -> pd.DataFrame:
    for c in left.columns:
        if (name_nloops := col2nameloop(c)) is not None:
            if c not in right.columns:
                right[c] = 0

    df = pd.merge(left, right, how="cross", suffixes=MERGE_SUFFIXES)
    shared_columns = set(left.columns) & set(right.columns) - RESERVED_COLUMNS
    dropcols = []

    for c in shared_columns:
        # Unknown loop index -> add
        if (name_nloops := col2nameloop(c)) is None:
            callfunc = add_to_col
            for suffix in MERGE_SUFFIXES:
                callfunc(df, c, c + suffix)
                dropcols.append(c + suffix)
        else:
            name, nloops = name_nloops
            # <= shared_loop_index -> add
            if nloops <= shared_loop_index:
                callfunc = add_to_col
                for suffix in MERGE_SUFFIXES:
                    callfunc(df, c, c + suffix)
            # > shared_loop_index -> create new left column
            else:
                max_to_col(
                    df, nameloop2col(name, nloops, left=True), c + MERGE_SUFFIXES[0]
                )
                max_to_col(df, nameloop2col(name, nloops), c + MERGE_SUFFIXES[1])
                dropcols.extend([c + suffix for suffix in MERGE_SUFFIXES])

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
    # - Tiling includes information for how may be fused:
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


class Pareto:
    def __init__(self, data: pd.DataFrame, skip_pareto: bool = False):
        self.data: pd.DataFrame = data if skip_pareto else makepareto(data)

    def einsum_ids(self):
        return fzs(self.data[LOGSTRING].iloc[0].keys())

    @staticmethod
    def concat(paretos: list["Pareto"], skip_pareto: bool = False) -> "Pareto":
        return Pareto(
            pd.concat([p.data for p in paretos]).fillna(0),
            skip_pareto=len(paretos) == 1 or skip_pareto,
        )

    def merge_next(
        self,
        other: "Pareto",
        shared_loop_index: int,
        live_tensors: set[int],
        delay: bool = False,
    ) -> "Pareto":
        d = delayed(merge_cross)(
            self.data,
            other.data,
            shared_loop_index,
            live_tensors=live_tensors,
            as_pareto=True,
        )
        return d if delay else d[0](*d[1], **d[2])

    @staticmethod
    def get_dummy() -> "Pareto":
        df = pd.DataFrame({OCCUPANCY: [1, 2], LOGSTRING: [{"A": "A"}] * 2})
        return Pareto(df)

    def free_to_loop_index(
        self,
        n: int,
        resource2capacity: Optional[dict[str, Optional[int]]] = None,
    ) -> bool:
        self.data, needs_pareto = _free_to_loop_index(
            self.data, n, return_needs_pareto=True
        )
        if resource2capacity is not None:
            self.limit_capacity(n, resource2capacity)
        return needs_pareto

    def alloc(
        self,
        resource_name: str,
        size: int,
        above_loop_index: int,
        resource2capacity: dict[str, Optional[int]],
    ):
        resource2capacity = resource2capacity or {}
        resource_name = str(resource_name)

        if resource_name not in resource2capacity:
            return

        n = nameloop2col(resource_name, above_loop_index)
        if n in self.data:
            self.data[n] = self.data[n] + size
        else:
            self.data[n] = size

    def add_tensor(self, tensor):
        if len(self.data) == 0:
            return
        if TENSORS in self.data:
            last_einsum = list(self.data.iloc[0][TENSORS].keys())[-1]
            if tensor in self.data[TENSORS].iloc[0][last_einsum]:
                return
            for t in self.data[TENSORS]:
                t[last_einsum].append(tensor)

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

    def squish_left_right(self, shared_loop_index: int = None) -> bool:
        self.data, needs_pareto = squish_left_right(
            self.data, shared_loop_index, return_needs_pareto=True
        )
        return needs_pareto

    def filter_by_mapping_hashes(self, hashes: set[int]):
        self.data = self.data[
            self.data[MAPPING_HASH].apply(
                lambda x: all(i in hashes for i in x.values())
            )
        ]
        return self

    def make_pareto(self):
        self.data = makepareto(self.data)

    def has_reservations(self):
        return any(col2nameloop(c) is not None for c in self.data.columns)

    def get_reservations(self):
        return tuple(
            sorted(c for c in self.data.columns if col2nameloop(c) is not None)
        )

    def tuplefy_data(self):
        if self.data.empty:
            return
        from fastfusion.sim import Tiling

        for col in self.data.columns:
            if col in TUPLABE_COLUMNS:
                self.data[col] = self.data[col].apply(
                    lambda x: {k: v.to_tuple() for k, v in x.items()}
                )

    def detuplefy_data(self):
        if self.data.empty:
            return
        from fastfusion.sim import Tiling

        for col in self.data.columns:
            if col in TUPLABE_COLUMNS:
                self.data[col] = self.data[col].apply(
                    lambda x: {k: Tiling.from_dict_small(v) for k, v in x.items()}
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
