import functools
import re

from fastfusion.util import fzs


MAPPING_COLUMN = "mapping"
COMPRESSED_INDEX = "compressed_index"
TILE_SHAPE_PREFIX = "tile_shape"

DICT_COLUMNS = set([MAPPING_COLUMN])
RESERVED_COLUMNS = DICT_COLUMNS

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


def partition_col(col, prefix, expected_len=None):
    col = col.split("\0")
    if col[0] != prefix:
        return None
    if expected_len is not None and len(col) != expected_len:
        raise ValueError(
            f"Expected {expected_len} parts in \"{col}\" with prefix \"{prefix}\" "
            f"but got {len(col)}"
        )
    return col[1:]


@dict_cached
def col2nameloop(x):
    """ Format: reservation name level left """
    x = partition_col(x, "reservation", 4)
    if x is None:
        return None
    return x[0], int(x[1])


@dict_cached
def nameloop2col(name, nloops, left: bool = False):
    """ Format: reservation name level left """
    return f"reservation\0{name}\0{nloops}\0left" if left else "right"

@dict_cached
def tensor2col(tensor):
    """ Format: tensor tensor_name """
    return f"tensor\0{tensor}"


@dict_cached
def col2nametensor(col):
    """ Format: tensor tensor_name """
    x = partition_col(col, "tensor", 2)
    if x is None:
        return None
    return x[1]


@dict_cached
def col2nameloopleft(x):
    """ Format: reservation name level left """
    x = partition_col(x, "reservation", 4)
    if x is None:
        return None
    return x[0], x[1], x[2] == "left"


def is_reservation_col(x):
    return col2nameloop(x) is not None


@dict_cached
def is_left_col(x):
    """ Format: reservation name level left """
    x = partition_col(x, "reservation", 4)
    if x is None:
        return False
    return x[2] == "left"


def add_to_col(df, target, source):
    if target in df:
        target_type = df[target].dtype
        source_type = df[source].dtype
        if target_type != source_type:
            df[target] = df[target].astype("float64")
            df[source] = df[source].astype("float64")
    df.loc[:, target] = df[target] + df[source] if target in df else df[source]


def max_to_col(df, target, source):
    df.loc[:, target] = df[[target, source]].max(axis=1) if target in df else df[source]


def is_special_col(c):
    return c in RESERVED_COLUMNS or col2nameloop(c) is not None


def col_used_in_pareto(c):
    return col2nameloop(c) is not None or partition_col(c, "Total") is not None


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

