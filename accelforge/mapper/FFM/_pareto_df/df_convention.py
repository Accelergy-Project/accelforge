from collections import namedtuple
import functools
import re
from typing import Callable
from accelforge._accelerated_imports import pandas as pd
from accelforge.util import NUMPY_FLOAT_TYPE
from accelforge.util._frozenset import fzs, oset
from accelforge.frontend.workload import Rank
from accelforge.util._base_analysis_types import ActionKey, VerboseActionKey


class ColName(str):
    def __truediv__(self, other: "ColName"):
        if not isinstance(other, ColName):
            raise ValueError(f"{other} must be a ColName")
        return ColName(f"{self}{SEP}{other}")


# Keywords
SEP = "<SEP>"
ACTION = ColName("action")
TOTAL = ColName("Total")
USAGE = ColName("usage")
MEMORY = ColName("memory")


MAPPING_COLUMN = "mapping"
COMPRESSED_INDEX = "compressed_index"
TILE_SHAPE_PREFIX = "tile_shape"

DICT_COLUMNS = oset([MAPPING_COLUMN])
RESERVED_COLUMNS = DICT_COLUMNS


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
    col = col.split(SEP)
    if col[0] != prefix:
        return None
    if expected_len is not None and len(col) != expected_len:
        raise ValueError(
            f'Expected {expected_len} parts in "{col}" with prefix "{prefix}" '
            f"but got {len(col)}"
        )
    return col[1:]


@dict_cached
def memory_usage2col(memory_level: str, tensor: str) -> str:
    return f"usage<SEP>memory<SEP>{memory_level}<SEP>{tensor}"


@dict_cached
def col2memory_usage(col: str) -> tuple[str, str, str]:
    """Returns a tuple (memory_level, tensor, einsum)."""
    separated_names = col.split(SEP)
    assert len(separated_names) == 5, f"invalid column {col}"
    einsum = separated_names[0]
    assert separated_names[1] == "usage"
    assert separated_names[2] == "memory"
    memory = separated_names[3]
    tensor = separated_names[4]
    return memory, tensor, einsum


@dict_cached
def action2col(action: ActionKey | VerboseActionKey) -> str:
    if isinstance(action, VerboseActionKey):
        return f"action<SEP>{action.level}<SEP>{action.tensor}<SEP>{action.action}"
    elif isinstance(action, ActionKey):
        return f"action<SEP>{action.level}<SEP>{action.action}"


@dict_cached
def col2action(colname: str) -> ActionKey | VerboseActionKey:
    separated_names = colname.split(SEP)
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
    separated_names = colname.split(SEP)
    if len(separated_names) == 4:
        assert separated_names[1] == "energy", colname
        return VerboseActionKey(
            separated_names[2],
            separated_names[3],
            "None",
            separated_names[0],
        )
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


ReservationKey = namedtuple(
    "ReservationKey", ["name", "index", "left"]
)

# Used for renumbering columns to not collide when joining two dataframes
_RIGHT_RESERVATION_OFFSET = 1_000_000_000

@dict_cached
def col2reservationsize(x: str) -> ReservationKey | None:
    """Format: reservation_size name index left"""
    p = partition_col(x, "reservation_size", 4)
    if p is None:
        return None
    return ReservationKey(p[0], int(p[1]), p[2] == "left")


@dict_cached
def col2reservationiters(x: str) -> ReservationKey | None:
    """Format: reservation_iters_above name index"""
    p = partition_col(x, "reservation_iters_above", 3)
    if p is None:
        return None
    return ReservationKey(p[0], int(p[1]), False)


@dict_cached
def reservationkey2sizecol(x: ReservationKey, left: bool = None) -> str:
    return reservation2sizecol(x.name, x.index, x.left if left is None else left)


@dict_cached
def reservationkey2iterscol(x: ReservationKey) -> str:
    return reservation2iterscol(x.name, x.index)

def assert_valid_index(index: int):
    # Reservations are ordered linearly, so if one is to exceed
    # _RIGHT_RESERVATION_OFFSET and cause problems aliasing right and left, one will
    # certainly exceed half of that. We can't just assert < _RIGHT_RESERVATION_OFFSET
    # because we'll be adding _RIGHT_RESERVATION_OFFSET to the right side
    half_offs = _RIGHT_RESERVATION_OFFSET // 2
    full_offs = _RIGHT_RESERVATION_OFFSET
    three_halves = half_offs + full_offs
    if index < full_offs:
        assert index < half_offs, f"index {index} exceeds max {half_offs}"
    if index >= full_offs:
        assert index < three_halves, f"index {index} exceeds max {three_halves}"

@dict_cached
def reservation2sizecol(name: str, index: int, left: bool = False) -> str:
    """Format: reservation_size name index left"""
    assert_valid_index(index)

    lr = "left" if left else "right"
    return f"reservation_size<SEP>{name}<SEP>{index}<SEP>{lr}"


@dict_cached
def reservation2iterscol(name: str, index: int) -> str:
    """Format: reservation_iters_above name index"""
    assert_valid_index(index)
    return f"reservation_iters_above<SEP>{name}<SEP>{index}"


def update_reservation_col(c: str, f: Callable) -> str:
    key = col2reservationsize(c)
    if key is not None:
        key = f(key)
        return reservation2sizecol(key.name, key.index, key.left)
    key = col2reservationiters(c)
    if key is not None:
        key = f(key)
        return reservation2iterscol(key.name, key.index)
    return c


@dict_cached
def stride2col(rank_name: Rank, loop_column: str) -> str:
    """Format: stride rank_name loop_column"""
    return f"stride<SEP>{rank_name}<SEP>{loop_column}"


@dict_cached
def col2stride(col: str) -> tuple[Rank, str] | None:
    """Format: stride rank_name loop_column"""
    x = partition_col(col, "stride", 3)
    return x[0], x[1]


@dict_cached
def initial2col(rank_name: Rank, loop_column: str) -> str:
    """Format: initial rank_name loop_column"""
    return f"initial<SEP>{rank_name}<SEP>{loop_column}"


@dict_cached
def col2initial(col: str) -> tuple[Rank, str] | None:
    """Format: initial rank_name loop_column"""
    x = partition_col(col, "initial", 3)
    return x[0], x[1]


@dict_cached
def iterations2col(loop_column: str) -> str:
    """Format: n_iterations loop_column"""
    return f"n_iterations<SEP>{loop_column}"


@dict_cached
def col2iterations(col: str) -> str | None:
    """Format: [prefix...] n_iterations loop_column"""
    parts = col.split(SEP)
    if "n_iterations" not in parts:
        return None
    return parts[parts.index("n_iterations") + 1]


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
    return x[0]


@dict_cached
def is_tensor_col(c: str) -> bool:
    return c.startswith("tensor<SEP>")


def is_reservation_col(x: str) -> bool:
    return col2reservationsize(x) is not None or col2reservationiters(x) is not None


@dict_cached
def is_left_col(x: str) -> bool:
    key = col2reservationsize(x)
    return key is not None and key.left


def make_fused_loop_col(s: str) -> str:
    return f"fused_loop<SEP>{s}"


def is_fused_loop_col(c: str) -> bool:
    return c.startswith("fused_loop<SEP>")


def make_binding_col(s: str) -> str:
    return f"binding<SEP>{s}"


def is_binding_col(c: str) -> bool:
    return c.startswith("binding<SEP>")


def is_n_iterations_col(c: str) -> bool:
    return c.startswith("fused_loop<SEP>n_iterations")


def ensure_float_type(df, target, source):
    if target in df:
        target_type = df[target].dtype
    else:
        target_type = NUMPY_FLOAT_TYPE

    if isinstance(source, pd.Series):
        if target in df and target_type != source.dtype:
            df[target] = df[target].astype(NUMPY_FLOAT_TYPE)
    elif source in df:
        if target_type != df[source].dtype:
            if target in df:
                df[target] = df[target].astype(NUMPY_FLOAT_TYPE)
            df[source] = df[source].astype(NUMPY_FLOAT_TYPE)


def add_to_col(df, target, source):
    ensure_float_type(df, target, source)
    df.loc[:, target] = df[target] + df[source] if target in df else df[source]


def max_to_col(df, target, source):
    ensure_float_type(df, target, source)
    df.loc[:, target] = df[[target, source]].max(axis=1) if target in df else df[source]


def add_to_col(df, target, source):
    ensure_float_type(df, target, source)
    if isinstance(source, pd.Series):
        df.loc[:, target] = df[target] + source
    else:
        df.loc[:, target] = df[target] + df[source] if target in df else df[source]


def is_objective_col(c):
    return partition_col(c, "Total") is not None


def col_used_in_pareto(c):
    return is_reservation_col(c) or is_objective_col(c)


def col_used_in_joining(c):
    assert not c.startswith("n_iterations"), "Improperly formatted n_iterations column"
    return (
        col_used_in_pareto(c)
        or is_fused_loop_col(c)
        or is_binding_col(c)
        or is_tensor_col(c)
        or is_n_iterations_col(c)
    )


# Pipeline:
# - Need to share temporal loops up to the spatial loop index
#   Resources:
#   - Energy
#   - ProcessingElement usage
#   - Buf usage
#   - Buf accesses (for BW calculation later)

# - Options:
#   - Non-pipelined: Sum resources above shared loops, max below.
#   - Pipelined: Sum resources above shared loops, max below. Sum
#     ProcessingElement usage. Latency is pipeline latency summed.
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
