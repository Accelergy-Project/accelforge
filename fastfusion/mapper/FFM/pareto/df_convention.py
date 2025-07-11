import functools
import re

from fastfusion.util import fzs


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

TILE_SHAPE_PREFIX = "__tile_shape"

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

