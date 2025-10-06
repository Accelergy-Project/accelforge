import time

import pandas as pd

from paretoset import paretoset
from joblib import delayed

from fastfusion.accelerated_imports import np
from fastfusion.util.util import parallel

from .df_convention import (
    col_used_in_pareto,
    is_fused_loop_col,
    is_n_iterations_col,
    is_objective_col,
)


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


def makepareto_quick2(mappings: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    from fast_pareto import is_pareto_front

    m2 = mappings[columns]
    m2 = m2[is_pareto_front(m2.to_numpy())].drop_duplicates()
    return mappings.loc[m2.index]


def makepareto_quick(mappings: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return mappings[quickpareto(mappings[columns])]


def paretofy_chunk(chunk, sense: list[str]):
    return paretoset(chunk, sense=sense)


def makepareto_merge(
    mappings: pd.DataFrame,
    columns: list[str],
    parallelize: bool = False,
    split_by_cols: list[str] = (),
) -> pd.DataFrame:
    chunk_size = 10000
    if len(mappings) <= 1:
        return mappings

    sense = ["min"] * len(columns) + ["diff"] * len(split_by_cols)

    to_chunk = mappings[columns + list(split_by_cols)]
    chunks = parallel(
        [
            delayed(paretofy_chunk)(chunk, sense)
            for chunk in [
                to_chunk[i : i + chunk_size]
                for i in range(0, len(to_chunk), chunk_size)
            ]
        ],
        n_jobs=1 if parallelize else None,
    )
    mappings = mappings[np.concatenate(chunks)]
    return mappings[paretoset(mappings[columns + list(split_by_cols)], sense=sense)]


def makepareto_time_compare(mappings: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    t0 = time.time()
    pareto = makepareto_merge(mappings, columns)
    t1 = time.time()
    merge_time = t1 - t0
    print(
        f"Time to make pareto with merge: {t1 - t0: .2f}. Number of pareto points: {len(pareto)}"
    )

    t0 = time.time()
    pareto2 = makepareto_quick2(mappings, columns)
    t1 = time.time()
    print(
        f"Time to make pareto with quick: {t1 - t0: .2f}. Number of pareto points: {len(pareto2)}"
    )
    quick_time = t1 - t0

    print(f"Quick is {quick_time / merge_time: .2f}x slower")

    if len(pareto) != len(pareto2):
        print(f"mismatch: {len(pareto)} != {len(pareto2)}")
        makepareto_quick2(mappings)

    return pareto2


# 2d. Blockwise vectorized CuPy Pareto front with sorting by one objective (full check)
# 2c. Fully vectorized CuPy brute-force Pareto front
# (returns numpy mask for compatibility)
def pareto_front_cupy_vectorized(X):
    # if len(X) > 1000:
    #     return X[paretoset(X.get(), sense=["min"] * X.shape[1])]

    # Broadcast X_gpu to (n, n, m) for all-pairs comparison
    A = X[:, None, :]  # shape (n, 1, m)
    B = X[None, :, :]  # shape (1, n, m)
    less_equal = (B <= A).all(axis=2)  # shape (n, n)
    strictly_less = (B < A).any(axis=2)  # shape (n, n)
    dominated = less_equal & strictly_less  # shape (n, n)
    is_pareto = ~dominated.any(axis=1)
    return is_pareto


# 2d. Recursive blockwise merge CuPy Pareto front with sorting by one objective
def pareto_front_cupy_blockwise_sorted_recursive(X, block_size=2000):
    N = X.shape[0]
    if N <= block_size:
        # Base case: just compute Pareto front directly
        mask = pareto_front_cupy_vectorized(X)
        return mask
    # Split into two halves
    mid = N // 2
    a, b = X[:mid], X[mid:]
    mask_a = pareto_front_cupy_blockwise_sorted_recursive(a, block_size)
    mask_b = pareto_front_cupy_blockwise_sorted_recursive(b, block_size)
    # Get Pareto-optimal points from both halves
    pareto_points_a = a[mask_a]
    pareto_points_b = b[mask_b]
    merged_points = np.vstack([pareto_points_a, pareto_points_b])
    # Compute Pareto front of the merged set
    merged_mask = pareto_front_cupy_vectorized(merged_points)
    merged_indices = np.where(merged_mask)[0]
    # Map merged_indices back to the original indices in X
    # First, get the indices in X for the merged points
    indices_a = np.where(mask_a)[0]
    indices_b = np.where(mask_b)[0] + mid
    all_indices = np.concatenate([indices_a, indices_b])
    merged_indices_in_X = all_indices[merged_indices]
    # Build the final mask for X
    mask = np.zeros(N, dtype=bool)
    mask[merged_indices_in_X] = True
    return mask


# def makepareto(
#     mappings: pd.DataFrame,
#     columns: list[str] = None,
#     parallelize: bool = False,
#     split_by_cols: list[str] = (),
# ) -> pd.DataFrame:
#     # return makepareto_time_compare(mappings)
#     if columns is None:
#         columns = [c for c in mappings.columns if col_used_in_pareto(c)]
#     if accelerated_imports.ACCELERATED:
#         mask = pareto_front_cupy_blockwise_sorted_recursive(mappings[columns].to_cupy())
#         return mappings[mask]


TOLERANCE = 0.00


def logify(x: pd.Series) -> pd.Series:
    if 0 < TOLERANCE < 1:
        pass
    else:
        assert (
            TOLERANCE == 0
        ), f"Tolerance must be between 0 and 1. Tolerance {TOLERANCE} is invalid."
        return x

    if x.min() <= 0:
        return x

    logged = np.log(x)

    return np.round(logged / TOLERANCE) * TOLERANCE



def makepareto(mappings: pd.DataFrame, columns: list[str] = None, parallelize: bool = False, split_by_cols: list[str] = ()) -> pd.DataFrame:
    # return makepareto_time_compare(mappings)
    if columns is None:
        columns = [c for c in mappings.columns if col_used_in_pareto(c)]

    # Number of iterations is derived from the tile shapes, so we don't need to use it,
    # since any row with the same tile shapes will have the same number of iterations.
    split_by_cols = list(split_by_cols) + [c for c in mappings.columns if is_fused_loop_col(c) and not is_n_iterations_col(c)]

    goals = []
    to_pareto = []
    pareto_cols = []
    for c in mappings.columns:
        if mappings[c].nunique() <= 1:
            continue

        if c in columns and is_objective_col(c):# or col_used_in_pareto(c)):
            to_pareto.append(logify(mappings[c]))
            pareto_cols.append(c)
            goals += ["min"]
        elif c in split_by_cols:
            to_pareto.append(mappings[c])
            pareto_cols.append(c)
            goals.append("diff")
        elif c in columns:
            to_pareto.append(mappings[c])
            pareto_cols.append(c)
            goals.append("min")

    if not to_pareto:
        return mappings.iloc[0:1]

    return mappings[paretoset(pd.concat(to_pareto, axis=1), sense=goals)]

    f = pd.concat(to_pareto, axis=1)
    x = list(f.groupby([c for c, d in zip(pareto_cols, goals) if d == "diff"]))
    print(x)