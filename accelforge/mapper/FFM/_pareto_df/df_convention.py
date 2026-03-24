from collections.abc import Mapping as MappingABC
from typing import Any
from typing import NamedTuple
import functools
import pandas as pd
from accelforge.util import NUMPY_FLOAT_TYPE
from accelforge.util._frozenset import fzs, oset
from accelforge.frontend.workload import Rank
from accelforge.util._base_analysis_types import ActionKey, VerboseActionKey


class ColName(str):
    def __truediv__(self, other: "ColName"):
        if not isinstance(other, (ColName, str)):
            raise ValueError(f"{other} must be a ColName or str")
        return ColName(f"{self}{SEP}{other}")


# Defaults
DEFAULT_THREAD = -1
DEFAULT_SPATIAL_DIMENSION = "<DEFAULT>"


# Keywords
SEP = "<SEP>"
ACTION = ColName("action")
TOTAL = ColName("Total")
USAGE = ColName("usage")
MEMORY = ColName("memory")
LATENCY = ColName("latency")
LIVE = ColName("live")
SPATIAL = ColName("spatial")
RESERVATION = ColName("reservation")
LEFT = ColName("left")
RIGHT = ColName("right")


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


def partition_col(col: str, prefix, expected_len=None) -> list[str] | None:
    """
    Returns elements of `col` split at `SEP` if `col` starts with `prefix`;
    otherwise, returns None.

    If given, the number of elements compared to `expected_len`. If not equal,
    an exception is raised.
    """
    if not col.startswith(prefix + SEP):
        return None
    col = col.removeprefix(prefix + SEP)
    col = col.split(SEP)
    if expected_len is not None and len(col) != expected_len:
        raise ValueError(
            f'Expected {expected_len} parts in "{col}" with prefix "{prefix}" '
            f"but got {len(col)}"
        )
    return col


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
def memorylatency2col(memory_name: str):
    return str(LATENCY / MEMORY / memory_name)


@dict_cached
def col2memorylatency(col: str):
    split_col = partition_col(col, LATENCY)
    if split_col is None or len(split_col) != 2:
        return None
    return split_col[1]


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


class LiveReservationKey(NamedTuple):
    resource: str
    tensor: str
    nloops: int
    thread: int

@dict_cached
def live_reservation2col(resource: str, tensor: str, nloops: int, thread: int = DEFAULT_THREAD) -> str:
    return str(LIVE / resource / tensor / str(nloops) / str(thread))

def get_live_reservation_cols_with(df, **kwargs):
    yield from _filter(df, col2live_reservation, kwargs)

def is_live_reservation_col(col):
    return LIVE in col

@dict_cached
def col2live_reservation(col: str) -> LiveReservationKey:
    split_col = partition_col(col, LIVE, 4)
    return LiveReservationKey(split_col[0], split_col[1], int(split_col[2]), int(split_col[3]))

def contains_live_reservation(df, **kwargs) -> bool:
    return len(list(get_live_reservation_cols_with(df, **kwargs))) > 0


class ReservationKey(NamedTuple):
    name: str
    nloops: int
    is_left: bool
    thread: int

    @property
    def is_right(self):
        return not self.is_left

@dict_cached
def reservation2col(name: str, nloops: int, left: bool = False, thread: int = DEFAULT_THREAD) -> str:
    left = LEFT if left else RIGHT
    return RESERVATION / MEMORY / name / str(nloops) / left / str(thread)

@dict_cached
def col2reservation(x: str) -> ReservationKey | None:
    x = partition_col(x, RESERVATION / MEMORY, 4)
    if x is None:
        return None
    return ReservationKey(x[0], int(x[1]), x[2] == LEFT, int(x[3]))

def get_reservation_cols_with(
    df,
    name: str=None,
    nloops: int=None,
    is_left: bool=None,
    thread: int=None,
    filter=None,
):
    if filter is None:
        filter = lambda col: True
    for c in df.columns:
        if (not (RESERVATION / MEMORY) in c) or (not filter(c)):
            continue
        key = col2reservation(c)
        if name is not None and key.name != name:
            continue
        if nloops is not None and key.nloops != nloops:
            continue
        if is_left is not None and key.is_left != is_left:
            continue
        if thread is not None and key.thread != thread:
            continue
        yield c


class SpatialReservationKey(NamedTuple):
    name: str
    dimension: str
    is_left: bool
    thread: int

def spatial_reservation2col(
    name: str,
    dimension: str=DEFAULT_SPATIAL_DIMENSION,
    is_left: bool=False,
    thread: int=DEFAULT_THREAD
):
    left = LEFT if is_left else RIGHT
    return RESERVATION / SPATIAL / name / dimension / left / str(thread)

def col2spatial_reservation(col: str):
    split_col = partition_col(col, RESERVATION / SPATIAL, 4)
    return SpatialReservationKey(*split_col)


@dict_cached
def stride2col(rank_name: Rank, nloops: int) -> str:
    """Format: stride rank_name nloops"""
    return f"stride<SEP>{rank_name}<SEP>{nloops}"


@dict_cached
def col2stride(col: str) -> tuple[Rank, int] | None:
    """Format: stride rank_name nloops"""
    x = partition_col(col, "stride", 2)
    return x[0], int(x[1])


@dict_cached
def initial2col(rank_name: Rank, nloops: int) -> str:
    """Format: initial rank_name nloops"""
    return f"initial<SEP>{rank_name}<SEP>{nloops}"


@dict_cached
def col2initial(col: str) -> tuple[Rank, int] | None:
    """Format: initial rank_name nloops"""
    x = partition_col(col, "initial", 2)
    return x[0], int(x[1])


@dict_cached
def iterations2col(nloops: int) -> str:
    """Format: n_iterations nloops"""
    return f"n_iterations<SEP>{nloops}"


@dict_cached
def col2iterations(col: str) -> int | None:
    """Format: n_iterations nloops"""
    x = partition_col(col, "n_iterations", 1)
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
def is_tensor_col(c: str) -> bool:
    return c.startswith("tensor<SEP>")


def is_reservation_col(x: str) -> bool:
    return col2reservation(x) is not None


def make_fused_loop_col(s: str) -> str:
    return f"fused_loop<SEP>{s}"


def is_fused_loop_col(c: str) -> bool:
    return c.startswith("fused_loop<SEP>")


def is_n_iterations_col(c: str) -> bool:
    return c.startswith("fused_loop<SEP>n_iterations")


def ensure_float_type(df, target, source):
    try:
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
    except Exception as e:
        raise RuntimeError(
            "Failed to ensure matching types between target and source.\n"
            "target: " + target + "\nsource: " + source
        ) from e


def add_to_col(df, target, source):
    ensure_float_type(df, target, source)
    df.loc[:, target] = df[target] + df[source] if target in df else df[source]


def max_to_col(df, target, source):
    ensure_float_type(df, target, source)
    df.loc[:, target] = df[[target, source]].max(axis=1) if target in df else df[source]


def add_to_col(df, target, source):
    ensure_float_type(df, target, source)
    if isinstance(source, pd.Series):
        df[target] = df[target] + source
    else:
        df.loc[:, target] = df[target] + df[source] if target in df else df[source]


def is_objective_col(c):
    return partition_col(c, "Total") is not None


def col_used_in_pareto(c):
    return col2reservation(c) is not None or col2memorylatency(c) is not None or is_objective_col(c) or is_live_reservation_col(c)


def col_used_in_joining(c):
    assert not c.startswith("n_iterations"), "Improperly formatted n_iterations column"
    return (
        col_used_in_pareto(c)
        or is_fused_loop_col(c)
        or is_tensor_col(c)
        or is_n_iterations_col(c)
        or is_live_reservation_col(c)
    )


def _filter(df, from_col_f, ref_key: MappingABC[str, Any]):
    def should_include(col):
        try:
            key = from_col_f(col)
        except:
            return False
        for k, v in ref_key.items():
            if not hasattr(key, k) or getattr(key, k) != v:
                return False
        return True
    return filter(should_include, df.columns)
