import functools
from math import ceil, comb
import functools
import math
import numbers
from accelforge._accelerated_imports import pandas as pd
from accelforge._accelerated_imports import numpy as np
from accelforge.util._frozenset import oset

NUMPY_FLOAT_TYPE = np.float32


@functools.lru_cache(maxsize=None)
def _count_factorizations_imperfect(n, into_n_parts):
    # Factorize n into into_n_parts parts
    RUBY_STYLE_IMPERFECT = True
    # RUBY_STYLE_IMPERFECT = True
    if n <= 1:
        return 1
    if into_n_parts <= 0:
        return 1

    shapes = list(range(1, ceil(n**0.5) + 1))
    shapes = shapes + [ceil(n / s) for s in shapes]
    shapes = sorted(oset(shapes))

    if RUBY_STYLE_IMPERFECT:
        shapes = list(range(1, n + 1))

    total = 0
    for s in shapes:
        n = _count_factorizations_imperfect(ceil(n / s), into_n_parts - 1)
        total += _count_factorizations_imperfect(ceil(n / s), into_n_parts - 1)

    return total


def _prime_factorization(n):
    f = []
    i = 2
    while n > 1:
        if n % i == 0:
            f.append(i)
            n //= i
        else:
            i += 1
    return f


def _count_factorizations(n, into_n_parts, imperfect=False):
    if into_n_parts <= 1:
        return 1
    f = _prime_factorization(n)
    factors = {f2: f.count(f2) for f2 in oset(f)}
    total = 1
    for exp in factors.values():
        total *= comb(exp + into_n_parts - 1, into_n_parts - 1)  # n choose k

    if imperfect:
        n = _count_factorizations_imperfect(n, into_n_parts)
        assert n >= total, f"n: {n} < total: {total}"
        return n

    return total


def _fillna_and__numeric_cast(df: pd.DataFrame, value: float) -> pd.DataFrame:
    def _is_float(x) -> bool:
        return isinstance(x, numbers.Real)

    def _is_int(x) -> bool:
        return (
            isinstance(x, numbers.Integral)
            or isinstance(x, numbers.Real)
            and (math.isnan(x) or int(x) == x)
        )

    for col in [c for c in df.columns if df.dtypes[c] == object]:
        # If it's an object col and all of them are integers, convert to int. nans count
        # as True
        if all(_is_int(x) for x in df[col]):
            df[col] = df[col].replace(float("nan"), value).astype(int)
        elif all(_is_float(x) for x in df[col]):
            df[col] = df[col].replace(float("nan"), value).astype(float)

    cols = df.select_dtypes(include=[np.floating, float, np.integer, int]).columns
    df[cols] = df[cols].fillna(value)
    for col in df.columns:
        assert (
            not df[col].isna().any()
        ), f"df has nans in column {col} with dtype {df[col].dtype}. " + "\n".join(
            f"{x} {type(x)=} {_is_int(x)=} {_is_float(x)=} " for x in df[col]
        )
    return df


def _numeric_cast(df: pd.DataFrame) -> pd.DataFrame:
    def _is_float(x) -> bool:
        return isinstance(x, numbers.Real)

    def _is_int(x) -> bool:
        return (
            isinstance(x, numbers.Integral)
            or isinstance(x, numbers.Real)
            and int(x) == x
        )

    for col in [c for c in df.columns if df.dtypes[c] == object]:
        series = df[col]
        if all(_is_int(x) for x in series):
            df[col] = series.astype(int)
        elif all(_is_float(x) for x in series):
            df[col] = series.astype(NUMPY_FLOAT_TYPE)
    return df
