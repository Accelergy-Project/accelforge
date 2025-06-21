from collections import defaultdict
import copy
import itertools
import re

# Disable numba. We need user_has_package("numba") to be False
import sys
import time
from typing import Any, Callable, Iterable, Optional, Tuple, Union, NamedTuple

from joblib import delayed
import numpy as np

from fastfusion.frontend.workload.workload import EinsumName
from fastfusion.mapper.FFM.joining.mappinginfo import TensorStorage
from fastfusion.util import fzs
from fastfusion.util.util import parallel

sys.modules["numba"] = None


from paretoset import paretoset

import pandas as pd
import functools

LOGSTRING = "__Mappings"
MAPPING_COLUMN = "__MAPPING"
STATS = "__STATS"
OCCUPANCY = "__Occupancy"
TENSORS = "__TENSORS"
IN_PROGRESS_STATS = "__IN_PROGRESS_STATS"
MAPPING_HASH = "__MAPPING_HASH"
TAGS_COLUMN = "__TAGS"
PER_COMPONENT_ACCESSES_ENERGY = "Per-Component Energy"
COMPRESSED_INDEX = "__COMPRESSED_INDEX"

DICT_COLUMNS = set(
    [
        LOGSTRING,
        MAPPING_COLUMN,
        STATS,
        TENSORS,
        IN_PROGRESS_STATS,
        MAPPING_HASH,
        TAGS_COLUMN,
        PER_COMPONENT_ACCESSES_ENERGY,
    ]
)
RESERVED_COLUMNS = DICT_COLUMNS

TUPLABE_COLUMNS = set([MAPPING_COLUMN, TENSORS])

CHECK_CORRECTNESS = False

_resource_name_nloops_reg = re.compile(r"RESOURCE_(.+?)(?:_LEFT)?_LEVEL_(-?\d+)")

_resource_name_tensor_reg = re.compile(r"RESOURCE_(.+?)_LEVEL_(.+?)")

def dict_cached(func):
    cache = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = (args, fzs(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return wrapper

def error_check_wrapper(func):
    if not CHECK_CORRECTNESS:
        return func
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            prev_args, prev_kwargs = copy.deepcopy(args), copy.deepcopy(kwargs)
            return func(*args, **kwargs)
        except Exception as e:
            print(f'EXCEPTION: {e}')
            live_tensors = set()
            if 'live_tensors' in kwargs:
                live_tensors = kwargs['live_tensors']
            else:
                argnames = func.__code__.co_varnames[:func.__code__.co_argcount]
                if 'live_tensors' in argnames:
                    idx = argnames.index('live_tensors')
                    if idx < len(args):
                        live_tensors = args[idx]
            for prev_arg in itertools.chain(prev_args, prev_kwargs.values()):
                if isinstance(prev_arg, PartialMappings):
                    prev_arg.fail(0, live_tensors)
                break
            func(*args, **kwargs) # For debugging
    return wrapper

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
def tensor2col(tensor):
    return f"TENSOR_{tensor}"

@dict_cached
def col2nametensor(col):
    m = _resource_name_tensor_reg.match(col)
    return (m.group(1), m.group(2)) if m is not None else None

@dict_cached
def col2nameloopleft(x):
    m = _resource_name_nloops_reg.match(x)
    return (m.group(1), int(m.group(2)), is_left_col(x)) if m is not None else None

def is_reservation_col(x):
    return col2nameloop(x) is not None

@dict_cached
def is_left_col(x):
    return "_LEFT_LEVEL_" in x

def add_to_col(df, target, source):
    df.loc[:, target] = df[target] + df[source] if target in df else df[source]


def max_to_col(df, target, source):
    df.loc[:, target] = df[[target, source]].max(axis=1) if target in df else df[source]


def is_special_col(c):
    return c in RESERVED_COLUMNS or col2nameloop(c) is not None

def col_used_in_pareto(c):
    return col2nameloop(c) is not None or c.startswith("metric_")
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
# - Compatibility includes information for how may be fused:
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


# Above index 0: Freed when Einsum fully terminates
# Above index 1: Freed after each iteration of the outermost loop

# -1 -> global resource
# 0 -> einsum only

# Shared index -1: Sum -1 resources, max everyone below
# Shared index 0: Sum 0 resources, max everyone below

def dominates(a: pd.Series, b: pd.Series) -> bool:
    return all(a[i] <= b[i] for i in range(len(a)))

def check_dominance(df: pd.DataFrame, n_optimal: int):
    # mask = np.zeros(len(df), dtype=bool)
    # mask[:new_point] = True
    mask = np.zeros(len(df) - n_optimal, dtype=bool)
    for col in df.columns:
        compare = df.iloc[n_optimal - 1][col]
        mask = mask | (df[col].iloc[n_optimal:] < compare)
    return np.concatenate([np.ones(n_optimal, dtype=bool), mask])


def quickpareto(df: pd.DataFrame) -> pd.DataFrame:
    # Step 1: Sort by the column with the most unique values
    # Step 2: Extract the first row. Add it to the pareto set
    # Step 3: Remove all dominated points
    # Step 4: Repeat until no more points to add
    
    # Step 1: Sort by the column with the most unique values
    original_len = len(df)
    col_to_sort = max(df.columns, key=lambda c: df[c].nunique())
    df = df.sort_values(by=col_to_sort).drop(columns=[col_to_sort])
    
    new_point = 0
    while new_point < len(df):
        mask = check_dominance(df, new_point + 1)
        df = df[mask]
        new_point += 1
        
    # Turn the index into a mask
    mask = np.zeros(original_len, dtype=bool)
    mask[df.index] = True
    return mask

from fast_pareto import is_pareto_front
def makepareto_quick2(mappings: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    m2 = mappings[columns]
    m2 = m2[is_pareto_front(m2.to_numpy())].drop_duplicates()
    return mappings.loc[m2.index]
    
def makepareto_quick(mappings: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return mappings[quickpareto(mappings[columns])]

def paretofy_chunk(chunk):
    return paretoset(chunk)

def makepareto_merge(mappings: pd.DataFrame, columns: list[str], parallelize: bool = False) -> pd.DataFrame:
    chunk_size = 10000
    if len(mappings) <= 1:
        return mappings
    
    to_chunk = mappings[columns]
    chunks = parallel(
        [delayed(paretofy_chunk)(chunk)
        for chunk in [to_chunk[i:i+chunk_size] for i in range(0, len(to_chunk), chunk_size)]],
        n_jobs = 1 if parallelize else None
    )
    mappings = mappings[np.concatenate(chunks)]
    return mappings[paretoset(mappings[columns])]

def makepareto_time_compare(mappings: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    t0 = time.time()
    pareto = makepareto_merge(mappings, columns)
    t1 = time.time()
    merge_time = t1 - t0
    print(f"Time to make pareto with merge: {t1 - t0: .2f}. Number of pareto points: {len(pareto)}")
    
    t0 = time.time()
    pareto2 = makepareto_quick2(mappings, columns)
    t1 = time.time()
    print(f"Time to make pareto with quick: {t1 - t0: .2f}. Number of pareto points: {len(pareto2)}")
    quick_time = t1 - t0
    
    print(f'Quick is {quick_time / merge_time: .2f}x slower')
    
    if len(pareto) != len(pareto2):
        print(f"mismatch: {len(pareto)} != {len(pareto2)}")
        makepareto_quick2(mappings)
        
    return pareto2


def makepareto(mappings: pd.DataFrame, columns: list[str] = None, parallelize: bool = False) -> pd.DataFrame:
    # return makepareto_time_compare(mappings)
    if columns is None:
        columns = [c for c in mappings.columns if col_used_in_pareto(c)]
    return makepareto_merge(mappings, columns, parallelize=parallelize)
    if len(mappings) <= 1:
        return mappings
    columns = [c for c in mappings.columns if col_used_in_pareto(c)]
    sense = ["min"] * len(columns)
    columns += list(extra_columns)
    sense += ["diff"] * len(extra_columns)
    return mappings[paretoset(mappings[columns], sense=sense)]
    
class DecompressData(NamedTuple):
    prefix: EinsumName
    decompress_data: pd.DataFrame
    extra_data: dict[str, Any]
    
    def compressed_index_column(self):
        return f"{self.prefix}{COMPRESSED_INDEX}"
    
    
class GroupedDecompressData(NamedTuple):
    prefix2datalist: dict[EinsumName, dict[int, DecompressData]]
    
    def register_decompress_data(self, einsum_name: EinsumName, job_id: int, decompress_data: DecompressData, ):
        target = self.prefix2datalist.setdefault(einsum_name, {})
        assert job_id not in target
        target[job_id] = decompress_data

class PartialMappings:
    def __init__(
            self, 
            data: pd.DataFrame, 
            skip_pareto: bool = False, 
            fill_reservation_cols: set | str = fzs(),
            check_above_subset_below: bool = CHECK_CORRECTNESS,
            max_right_to_left: bool = False,
            next_shared_loop_index: int = None,
            parallelize_pareto: bool = False,
            n_pmappings: int = None,
        ):
        self._data: pd.DataFrame = data
        self.right_reservations: dict[set] = None
        self.left_reservations: dict[set] = None
        self.parents = []
        self._prev_free_to_loop_index = None
        self._parallelize_pareto = parallelize_pareto
        self._make_reservations()
        self.n_pmappings = n_pmappings if n_pmappings is not None else len(self.data)

        if next_shared_loop_index is not None:
            self.free_to_loop_index(loop_index=next_shared_loop_index)
            self.limit_capacity(resource2capacity={}, next_shared_loop_index=next_shared_loop_index)

        if fill_reservation_cols: # Affects PartialMappings so must go before
            self.fill_reservation_cols(fill_reservation_cols)
        if check_above_subset_below:
            self.check_above_subset_below()
        if max_right_to_left: # Affects PartialMappings so must go before
            self.max_right_to_left()
        if check_above_subset_below:
            self.check_above_subset_below()

        if not skip_pareto:
            self.make_pareto(parallelize=parallelize_pareto)
            
        if check_above_subset_below:
            self.check_above_subset_below()

    def all_reservation_levels(self):
        return set().union(
            set(),
            *self.left_reservations.values(),
            *self.right_reservations.values(),
        )

    @error_check_wrapper
    def fill_reservation_cols(self, columns: set | str):
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
            assert above in self.data.columns
            assert below in self.data.columns
            max_to_col(self.data, below, above)

    @error_check_wrapper
    def max_right_to_left(self):
        for resource, reservations in self.left_reservations.items():
            for r in reservations:
                if r in self.right_reservations.get(resource, set()):
                    source = nameloop2col(resource, r)
                    target = nameloop2col(resource, r, left=True)
                    max_to_col(self.data, target, source)

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @error_check_wrapper
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

        if check_correctness and live_tensors is not None:
            self.copy().check_reservations(live_tensors=live_tensors)

        return len(drop_columns) != 0
    
    @error_check_wrapper
    def get_reservation_or_parent(
            self, 
            name: str, 
            level: int, 
            left: bool = False,
            return_name_level_left: bool = False,
        ) -> Optional[Union[str, Tuple[str, int, bool]]]:
        reservations = self.left_reservations if left else self.right_reservations
        if (reservations := reservations.get(name, None)) is not None:
            while level >= 0:
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
            max((max(r, default=-1) for r in self.right_reservations.values()), default=-1),
            max((max(r, default=-1) for r in self.left_reservations.values()), default=-1),
        )

    @error_check_wrapper
    def merge_next(
        self,
        right: "PartialMappings",
        shared_loop_index: int,
        next_shared_loop_index: int,
        live_tensors: set[int],
        shared_storage: set[TensorStorage],
        still_live_reservations: set[TensorStorage],
        duplicated_aliased_tensors: set[TensorStorage],
        resource2capacity: dict[str, int] = None,
        drop_valid_reservations: bool = True,
        df_lambda: Callable[[pd.DataFrame], pd.DataFrame] = lambda df: df,
    ) -> "PartialMappings":
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
        self.free_to_loop_index(shared_loop_index, live_tensors=live_tensors)
        self.shift_bottom_reservation_left(shared_loop_index)

        assert not right.left_reservations, f"{right.left_reservations} is not None"

        for resource, reservations in self.right_reservations.items():
            n_reservations = max(reservations, default=-1)
            assert n_reservations <= shared_loop_index, f"{resource}: {reservations} > {shared_loop_index}"

        for resource, reservations in self.left_reservations.items():
            n_reservations = max(reservations, default=-1)
            assert n_reservations <= shared_loop_index + 1, f"{resource}: {reservations} > {shared_loop_index}"
            
        max_nloops = max(
            shared_loop_index,
            self.get_max_loop_index(),
            right.get_max_loop_index()
        )

        df = pd.merge(self.data, right.data, how="cross", suffixes=["", "_RIGHT_MERGE"])

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
                if (source := self.get_reservation_or_parent(resource, nloops - 1)) is None:
                    continue
                target = nameloop2col(resource, nloops)
                # If there's a merged version column, then it's in both trees
                if target + "_RIGHT_MERGE" in df:
                    continue
                df.loc[:, target] += df[source]
            # For LEFT tree, LEFT reservations: Add the immediately-above
            # reservation from the right tree.
            for resource in iter_reservations(self.left_reservations):
                if (source := right.get_reservation_or_parent(resource, nloops - 1)) is None:
                    continue
                right_merge_source = source + "_RIGHT_MERGE"
                target = nameloop2col(resource, nloops, left=True)
                if source is not None:
                    df.loc[:, target] += df[right_merge_source if right_merge_source in df else source]
            # For LEFT tree, RIGHT reservations: Add the same-level reservation from
            # the right tree. This will double-count reservations that are in both branches,
            # so we remove them later.
            for resource in iter_reservations(self.right_reservations):
                if (source := right.get_reservation_or_parent(resource, nloops)) is None:
                    continue
                right_merge_source = source + "_RIGHT_MERGE"
                target = nameloop2col(resource, nloops)
                if source is not None:
                    df.loc[:, target] += df[right_merge_source if right_merge_source in df else source]

        # For everything else: Simple add
        dropcols = [c for c in df.columns if c.endswith("_RIGHT_MERGE")]
        for source in dropcols:
            target = source[:-len("_RIGHT_MERGE")]
            if not col_used_in_pareto(target):
                raise ValueError(f"{target} is not used in pareto")
            if col2nameloop(target) is None:
                df.loc[:, target] += df[source]
        df = df_lambda(df)
                
        df = df.drop(columns=dropcols)
        n_pmappings = self.n_pmappings * right.n_pmappings
        result = PartialMappings(df, skip_pareto=True, check_above_subset_below=False, n_pmappings=n_pmappings)
        # Remove tensors that were allocated in both branches and got added
        # together.
        shared_to_free = [s for s in shared_storage if s.above_loop_index <= shared_loop_index]
        live_to_alloc = [s for s in still_live_reservations if s.above_loop_index > shared_loop_index]
        result.adjust_reservations(
            alloc=live_to_alloc,
            free=list(itertools.chain(shared_to_free, duplicated_aliased_tensors)),
        )

        if CHECK_CORRECTNESS:
            result.check_above_subset_below(live_tensors)
            result.check_reservations(live_tensors)

        result.free_to_loop_index(next_shared_loop_index, live_tensors=live_tensors)
        if not CHECK_CORRECTNESS:
            result.limit_capacity(resource2capacity, next_shared_loop_index, drop_valid_reservations)
        result.max_right_to_left()
        result.make_pareto()
        
        return result

    @error_check_wrapper
    def _adjust_reservations_one_resource(
        self,
        resource: str,
        alloc: Iterable[TensorStorage],
        free: Iterable[TensorStorage],
    ):
        alloc, free = list(alloc), list(free)
        # Iterate through each reservation and level
        targets = defaultdict(int)
        
        # Must allocate at the above_loop_index level
        for t in itertools.chain(alloc, free):
            self.right_reservations.setdefault(resource, set()).add(t.above_loop_index)

        for t, negate in [(t, False) for t in alloc] + [(t, True) for t in free]:
            size = -t.size if negate else t.size
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
        for (level, left), size in sorted(targets.items(), key=lambda x: x[0], reverse=True):
            target = nameloop2col(resource, level, left=left)
            if target in self.data:
                self.data.loc[:, target] += size
                continue

            # We're creating a new column, so copy allocations from any parents
            source = self.get_reservation_or_parent(resource, level-1)
            # source is None -> We're at the top level, no one to inherit from
            self.data[target] = size + (self.data[source] if source else 0)

    @error_check_wrapper
    def adjust_reservations(
            self,
            alloc: Iterable[TensorStorage],
            free: Iterable[TensorStorage],
        ):
        alloc, free = list(alloc), list(free)
        all_resources = {t.resource_name for t in alloc} | {t.resource_name for t in free}
        # Handle each resource separately
        for resource in all_resources:
            cur_alloc = [t for t in alloc if t.resource_name == resource]
            cur_free = [t for t in free if t.resource_name == resource]
            if cur_alloc or cur_free:
                self._adjust_reservations_one_resource(resource, cur_alloc, cur_free)

    @staticmethod
    def concat(paretos: list["PartialMappings"], skip_pareto: bool = False) -> "PartialMappings":
        if len(paretos) == 0:
            raise ValueError("No paretos to concatenate")
        if len(paretos) == 1:
            return paretos[0]
        
        required_cols = set.union(*[set(p.data.columns) for p in paretos])
        shared_cols = set.intersection(*[set(p.data.columns) for p in paretos])
        fill_cols = required_cols - shared_cols
        fill_cols = [c for c in fill_cols if col_used_in_pareto(c)]
        
        concatenated = pd.concat([p.data for p in paretos]).reset_index(drop=True)
        
        p = PartialMappings(
            concatenated.fillna(0),
            skip_pareto=len(paretos) == 1 or skip_pareto,
            fill_reservation_cols=fill_cols,
            n_pmappings=sum(p.n_pmappings for p in paretos),
        )

        p.parents = paretos[0].parents
        return p

    def copy(self) -> "PartialMappings":
        p = PartialMappings(self.data.copy(), skip_pareto=True, check_above_subset_below=False)
        p.parents = copy.deepcopy(self.parents)
        return p
    
    def limit_capacity(
        self, 
        resource2capacity: dict[str, Optional[int]],
        next_shared_loop_index: int=None,
        drop_valid_reservations: bool = True,
    ) -> bool:
        resource2capacity = resource2capacity or {}
        dropcols = []
        for resource in sorted(set(self.right_reservations) | set(self.left_reservations)):
            capacity = resource2capacity.get(resource, None)
            # Right reservations: Only check the greatest-index level. If a loop
            # is 0 and the next shared loop index is -1, then we can drop the
            # column.
            right_loops = self.right_reservations.get(resource, set())
            if right_loops:
                n = max(right_loops)
                col = nameloop2col(resource, n)
                self._data = self.data[self.data[col] <= capacity] if capacity is not None else self.data
            for l in list(right_loops):
                if l == 0 and next_shared_loop_index == -1 and drop_valid_reservations:
                    right_loops.discard(l)
                    dropcols.append(col)

            # Left reservations: Check all levels. If a loop is 0,
            # then we can drop the column.
            left_loops = self.left_reservations.get(resource, set())
            for l in list(left_loops):
                col = nameloop2col(resource, l, left=True)
                self._data = self.data[self.data[col] <= capacity] if capacity is not None else self.data
                if l == 0 and drop_valid_reservations:
                    left_loops.discard(l)
                    dropcols.append(col)
                    
        self._data = self.data.drop(columns=dropcols)

    def make_pareto(self, columns: list[str] = None, parallelize: bool = False):
        self._data = makepareto(self.data, columns, parallelize=parallelize)

    def has_reservations(self):
        return any(col2nameloop(c) is not None for c in self.data.columns)


    # ============================================================================
    # Checking functions
    # ============================================================================
    def check_above_subset_below(self, live_tensors: set[str]=fzs()):
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
            
    @error_check_wrapper
    def check_reservations(self, live_tensors: set[int]):
        from fastfusion.visualization.reservationtree import mappings2reservationtree
        assert not self.data.isnull().values.any(), f"NaN in {self.data}"

        self = self.copy()

        self.free_to_loop_index(-1, check_correctness=False)
        self.shift_bottom_reservation_left(-1)

        for i, r in self.data.iterrows():
            looptree = mappings2reservationtree(
                r[MAPPING_COLUMN],
                r.get(STATS, None),
                still_live_tensors=live_tensors
            )
            reservations = dict(looptree.get_reservations())
            
            # If r doesn't have any columns, continue. It's a copy Einsum so it has no
            # stats.
            if r.empty:
                continue

            for k, v in reservations.items():
                col = self.get_reservation_or_parent(k, 0, left=True)
                if str(k) == "0":
                    continue
                if col not in self.data.columns:
                    got = r[[c for c in self.data.columns if col2nameloop(c) is not None]]
                    self.fail(i, live_tensors)
                    raise ValueError(f"Missing {k}: Expected {reservations}. Got: {got}")
                if r[col] != v:
                    got = r[[c for c in self.data.columns if col2nameloop(c) is not None]]
                    self.fail(i, live_tensors)
                    looptree = mappings2reservationtree(
                        r[MAPPING_COLUMN],
                        r.get(STATS, None),
                        # skip_backing_tensors_in_right_branch=live_tensors,
                        still_live_tensors=live_tensors,
                    )
                    raise ValueError(
                        f"Mismatched {k}: {v} != {r[col]}. Expected {reservations}. Got: {got}"
                    )

    def fail(self, index, live_tensors):
        from fastfusion.mapper.FFM.joining.sim import TensorStorage
        r = self.data.iloc[index]
        assert not self.data.isnull().values.any(), f"NaN in {self.data}"
        self = self.copy()
        self._draw_index(index, live_tensors, self._get_target_path(suffix="fail"))
        all_tensors = set(t for tn in r[MAPPING_COLUMN].values() for t in tn.storage)
        all_tensors = TensorStorage.get_backing_stores(all_tensors)
        for t in sorted(all_tensors):
            print(f"{t.__repr__()},")
    
    def _draw_index(self, index: int, live_tensors, to_file: str = "test.png"):
        from fastfusion.visualization.reservationtree import mappings2reservationtree
        import pydot
        looptree = mappings2reservationtree(
            self.data.iloc[index][MAPPING_COLUMN],
            self.data.iloc[index].get(STATS, None),
            still_live_tensors=live_tensors,
        )
        graph = pydot.Dot(graph_type="digraph", ranksep="0.2", nodesep="0.2")
        looptree.to_pydot(graph)
        row = self.data.iloc[index]
        all_data = sorted(
            f"{k}: {v}" for k, v in row.items() if k not in DICT_COLUMNS and k != LOGSTRING
        )
        data_str = "\n".join(all_data)
        graph.add_node(pydot.Node("data", label=data_str, shape="plaintext"))
        with open(to_file, "wb") as f:
            f.write(graph.create_png())
            
    def prefix_data(self, prefix: str):
        rename = lambda col: f"{prefix}_{col}" if not col_used_in_pareto(col) else col
        self.data.rename(columns=rename, inplace=True)
        
    # def _compress_data(self, prefix: EinsumName = None, job_id: int = None) -> pd.DataFrame:
    #     compressed_index_col = pd.Series(self.data.index)
    #     keep_cols = [c for c in self.data.columns if col_used_in_pareto(c)]
    #     recovery = self.data[[c for c in self.data.columns if c not in keep_cols]].copy()
    #     self._data = self.data[keep_cols].copy()
    #     self.data[COMPRESSED_INDEX] = compressed_index_col.apply(lambda x: (job_id, x))
    #     recovery[COMPRESSED_INDEX] = compressed_index_col
    #     if prefix is not None:
    #         recovery.rename(columns={c: f"{prefix}{c}" for c in recovery.columns}, inplace=True)
    #         self.data.rename(columns={COMPRESSED_INDEX: f"{prefix}{COMPRESSED_INDEX}"}, inplace=True)
    #     assert len(recovery.columns) == len(set(recovery.columns)), f"Duplicate columns in {recovery.columns}"
    #     return recovery
    def _compress_data(self, prefix: EinsumName = None, job_id: int = None) -> pd.DataFrame:
        self.data[COMPRESSED_INDEX] = self.data.index
        keep_cols = [COMPRESSED_INDEX] + [c for c in self.data.columns if col_used_in_pareto(c)]
        # recovery = self.data[[c for c in self.data.columns if c not in keep_cols] + [COMPRESSED_INDEX]]
        # self._data = self.data[keep_cols]
        recovery = self.data # TODO: We may want to use the above if compressing uses too much memory
        self._data = self.data[keep_cols].copy()
        self._data[COMPRESSED_INDEX] = self._data[COMPRESSED_INDEX].apply(lambda x: (job_id, x))
        if prefix is not None:
            recovery.rename(columns={c: f"{prefix}{c}" for c in recovery.columns}, inplace=True)
            self.data.rename(columns={COMPRESSED_INDEX: f"{prefix}{COMPRESSED_INDEX}"}, inplace=True)
        return recovery

    def _decompress_data(self, prefix: EinsumName, decompress_data: list[DecompressData]):
        src_idx_col = f"{prefix}{COMPRESSED_INDEX}"
        rows = []
        extra_data = []
        for r in self.data[src_idx_col]:
            assert isinstance(r, tuple), \
                f"Expected tuple, got {type(r)}. Was register_decompress_data called with this Pareto?"
            assert len(r) == 2, f"Expected tuple of length 2, got {r}"
            data_index, row_index = r
            data = decompress_data[data_index]
            # Find the row in the data with src_idx_col == row_index
            row = data.decompress_data[data.decompress_data[src_idx_col] == row_index]
            assert len(row) == 1, f"Expected 1 row, got {len(row)}"
            rows.append(row)
            extra_data.append(data.extra_data)
            
        new_df = pd.concat(rows)
        extra_data_df = pd.DataFrame(extra_data)

        # Drop the src_idx_col
        self.data.drop(columns=[src_idx_col], inplace=True)
        # Reset index to avoid duplicate index values that cause InvalidIndexError
        a = new_df.reset_index(drop=True)
        b = extra_data_df.reset_index(drop=True)
        c = self.data.reset_index(drop=True)
        self._data = pd.concat([a, b, c], axis=1)
        self._data.drop(columns=[src_idx_col], inplace=True)
                
    def decompress(self, decompress_data: GroupedDecompressData):
        for einsum_name, decompress_data_list in decompress_data.prefix2datalist.items():
            assert decompress_data_list, f"No decompress data found for {einsum_name}"
            self._decompress_data(einsum_name, decompress_data_list)
        

    @classmethod
    def compress_paretos(cls, prefix: EinsumName, paretos: list["PartialMappings"], job_id: int, extra_data: dict[str, Any]) -> DecompressData:
        index = 0
        decompress_data = []
        for p in paretos:
            p.data.reset_index(drop=True, inplace=True)
            p.data.index += index
            index += len(p.data)
            decompress_data.append(p._compress_data(prefix, job_id))
            
        decompress_data = pd.concat(decompress_data) if decompress_data else pd.DataFrame()
        decompress_data.reset_index(drop=True, inplace=True)
        return DecompressData(
            prefix=prefix,
            decompress_data=decompress_data,
            extra_data={f"{prefix}{k}": v for k, v in extra_data.items()},
        )
        
    def _tuplefy_compress_data(self, index: int, decompress_data: DecompressData):
        col = decompress_data.compressed_index_column()
        self.data[col] = self.data[col].apply(lambda x: (index, x))
