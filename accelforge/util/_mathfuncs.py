import functools
from math import ceil, comb
import functools
import math
import numbers
import symengine as se
from accelforge._accelerated_imports import pandas as pd
from accelforge._accelerated_imports import numpy as np
from accelforge.util._frozenset import oset


def _is_numeric(x) -> bool:
    # symengine scalars (RealDouble, Integer, ...) don't register as
    # numbers.Real even though they have a working __float__.
    return isinstance(x, (numbers.Real, se.Basic))


def _is_intlike(x) -> bool:
    if isinstance(x, numbers.Integral):
        return True
    if isinstance(x, (numbers.Real, se.Basic)):
        f = float(x)
        return math.isnan(f) or int(f) == f
    return False


NUMPY_FLOAT_TYPE = np.float32


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


@functools.lru_cache(maxsize=None)
def _divisors(n):
    return tuple(d for d in range(1, n + 1) if n % d == 0)


@functools.lru_cache(maxsize=None)
def _count_factorizations(n, imperfect_per_loop: tuple[bool, ...]):
    if len(imperfect_per_loop) <= 1:
        return 1

    others = imperfect_per_loop[1:]
    if imperfect_per_loop[0]:
        return sum(_count_factorizations(ceil(n / s), others) for s in range(1, n + 1))
    return sum(_count_factorizations(n // d, others) for d in _divisors(n))


def _fillna_and__numeric_cast(df: pd.DataFrame, value: float) -> pd.DataFrame:
    dtypes = df.dtypes
    for col in [c for c in df.columns if dtypes[c] == object]:
        vals = df[col]
        if all(_is_intlike(x) for x in vals):
            df[col] = vals.map(
                lambda x: (
                    value if (isinstance(x, float) and math.isnan(x)) else float(x)
                )
            ).astype(int)
        elif all(_is_numeric(x) for x in vals):
            df[col] = vals.map(
                lambda x: (
                    value if (isinstance(x, float) and math.isnan(x)) else float(x)
                )
            ).astype(float)

    cols = df.select_dtypes(include=[np.floating, float, np.integer, int]).columns
    df[cols] = df[cols].fillna(value)
    for col in df.columns:
        assert (
            not df[col].isna().any()
        ), f"df has nans in column {col} with dtype {df[col].dtype}"
    return df


def _numeric_cast(df: pd.DataFrame) -> pd.DataFrame:
    dtypes = df.dtypes
    for col in [c for c in df.columns if dtypes[c] == object]:
        series = df[col]
        if all(_is_intlike(x) for x in series):
            df[col] = series.map(float).astype(int)
        elif all(_is_numeric(x) for x in series):
            df[col] = series.map(float).astype(NUMPY_FLOAT_TYPE)
    return df
