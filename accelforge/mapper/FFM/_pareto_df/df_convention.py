import functools
import re

from accelforge.util._frozenset import fzs
from accelforge.frontend.workload import Rank
from accelforge.util._base_analysis_types import ActionKey, VerboseActionKey


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


def partition_col(col, prefix, expected_len=None) -> list[str] | None:
    col = col.split("<SEP>")
    if col[0] != prefix:
        return None
    if expected_len is not None and len(col) != expected_len:
        raise ValueError(
            f'Expected {expected_len} parts in "{col}" with prefix "{prefix}" '
            f"but got {len(col)}"
        )
    return col[1:]


@dict_cached
def action2col(action: ActionKey | VerboseActionKey) -> str:
    if isinstance(action, VerboseActionKey):
        return f"action<SEP>{action.level}<SEP>{action.tensor}<SEP>{action.action}"
    elif isinstance(action, ActionKey):
        return f"action<SEP>{action.level}<SEP>{action.action}"


@dict_cached
def col2action(colname: str) -> ActionKey | VerboseActionKey:
    separated_names = colname.split("<SEP>")
    if len(separated_names) == 4:
        assert separated_names[0] == "action"
        return ActionKey(separated_names[1], separated_names[2])
    elif len(separated_names) == 5:
        assert separated_names[1] == "action"
        return VerboseActionKey(
            separated_names[2],
            separated_names[4],
            separated_names[3],
            separated_names[0],
        )
    else:
        raise ValueError(f"bad column name: {colname}")


@dict_cached
def energy2col(action: ActionKey | VerboseActionKey) -> str:
    if isinstance(action, VerboseActionKey):
        return f"energy<SEP>{action.level}<SEP>{action.tensor}<SEP>{action.action}"
    elif isinstance(action, ActionKey):
        return f"energy<SEP>{action.level}<SEP>{action.action}"


@dict_cached
def col2energy(colname: str) -> ActionKey | VerboseActionKey:
    separated_names = colname.split("<SEP>")
    if len(separated_names) == 4:
        assert separated_names[1] == "energy", colname
        return ActionKey(separated_names[2], separated_names[3])
    elif len(separated_names) == 5:
        assert separated_names[1] == "energy"
        return VerboseActionKey(
            separated_names[2],
            separated_names[4],
            separated_names[3],
            separated_names[0],
        )
    else:
        raise ValueError(f"bad column name: {colname}")


@dict_cached
def col2nameloop(x: str) -> tuple[str, int] | None:
    """Format: reservation name level left"""
    x = partition_col(x, "reservation", 4)
    if x is None:
        return None
    return x[0], int(x[1])


@dict_cached
def nameloop2col(name: str, nloops: int, left: bool = False) -> str:
    """Format: reservation name level left"""
    return f"reservation<SEP>{name}<SEP>{nloops}<SEP>" + ("left" if left else "right")


@dict_cached
def stride2col(rank_name: Rank, nloops: int) -> str:
    """Format: stride rank_name nloops"""
    return f"stride<SEP>{rank_name}<SEP>{nloops}"


@dict_cached
def col2stride(col: str) -> tuple[Rank, int] | None:
    """Format: stride rank_name nloops"""
    x = partition_col(col, "stride", 3)
    return x[0], int(x[1])


@dict_cached
def initial2col(rank_name: Rank, nloops: int) -> str:
    """Format: initial rank_name nloops"""
    return f"initial<SEP>{rank_name}<SEP>{nloops}"


@dict_cached
def col2initial(col: str) -> tuple[Rank, int] | None:
    """Format: initial rank_name nloops"""
    x = partition_col(col, "initial", 3)
    return x[0], int(x[1])


@dict_cached
def iterations2col(nloops: int) -> str:
    """Format: n_iterations nloops"""
    return f"n_iterations<SEP>{nloops}"


@dict_cached
def col2iterations(col: str) -> int | None:
    """Format: n_iterations nloops"""
    x = partition_col(col, "n_iterations", 2)
    return x[0]


@dict_cached
def firstlatency2col(name: str, nloops: int) -> str:
    """Format: first latency name level"""
    return f"first_latency<SEP>{name}<SEP>{nloops}"


@dict_cached
def tensor2col(tensor: str) -> str:
    """Format: tensor tensor_name"""
    return f"tensor<SEP>{tensor}"


@dict_cached
def col2nametensor(col: str) -> str | None:
    """Format: tensor tensor_name"""
    x = partition_col(col, "tensor", 2)
    if x is None:
        return None
    return x[1]


@dict_cached
def is_tensor_col(c: str) -> bool:
    return c.startswith("tensor<SEP>")


@dict_cached
def col2nameloopleft(x: str) -> tuple[str, int, bool] | None:
    """Format: reservation name level left"""
    x = partition_col(x, "reservation", 4)
    if x is None:
        return None
    return x[0], x[1], x[2] == "left"


def is_reservation_col(x: str) -> bool:
    return col2nameloop(x) is not None


@dict_cached
def is_left_col(x: str) -> bool:
    """Format: reservation name level left"""
    x = partition_col(x, "reservation", 4)
    if x is None:
        return False
    return x[2] == "left"


def make_fused_loop_col(s: str) -> str:
    return f"fused_loop<SEP>{s}"


def is_fused_loop_col(c: str) -> bool:
    return c.startswith("fused_loop<SEP>")


def is_n_iterations_col(c: str) -> bool:
    return c.startswith("fused_loop<SEP>n_iterations")


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


def is_objective_col(c):
    return partition_col(c, "Total") is not None


def col_used_in_pareto(c):
    return col2nameloop(c) is not None or is_objective_col(c)


def col_used_in_joining(c):
    assert not c.startswith("n_iterations"), "Improperly formatted n_iterations column"
    return (
        col_used_in_pareto(c)
        or is_fused_loop_col(c)
        or is_tensor_col(c)
        or is_n_iterations_col(c)
    )


# Pipeline:
# - Need to share temporal loops up to the spatial loop index
#   Resources:
#   - Energy
#   - PE usage
#   - Buf usage
#   - Buf accesses (for BW calculation later)

# - Options:
#   - Non-pipelined: Sum resources above shared loops, max below.
#   - Pipelined: Sum resources above shared loops, max below. Sum
#     PE usage. Latency is pipeline latency summed.
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
