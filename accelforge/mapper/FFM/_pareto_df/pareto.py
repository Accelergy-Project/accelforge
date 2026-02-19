import functools
from math import prod
import time

import pandas as pd

from paretoset import paretoset
from joblib import delayed
from sympy import factorint

from accelforge._accelerated_imports import np
from accelforge.util.parallel import parallel

from accelforge.mapper.FFM._pareto_df.df_convention import (
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
#     if _accelerated_imports.ACCELERATED:
#         mask = pareto_front_cupy_blockwise_sorted_recursive(mappings[columns].to_cupy())
#         return mappings[mask]


TOLERANCE = 0.0


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


def makepareto(
    mappings: pd.DataFrame,
    columns: list[str] = None,
    parallelize: bool = False,
    split_by_cols: list[str] = (),
) -> pd.DataFrame:
    # return makepareto_time_compare(mappings)
    if columns is None:
        columns = [c for c in mappings.columns if col_used_in_pareto(c)]

    # Number of iterations is derived from the tile shapes, so we don't need to use it,
    # since any row with the same tile shapes will have the same number of iterations.
    split_by_cols = list(split_by_cols) + [
        c
        for c in mappings.columns
        if is_fused_loop_col(c) and not is_n_iterations_col(c)
    ]

    goals = []
    to_pareto = []
    pareto_cols = []
    for c in mappings.columns:
        if mappings[c].nunique() <= 1:
            continue

        if c in columns and is_objective_col(c):  # or col_used_in_pareto(c)):
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


@functools.lru_cache(maxsize=10000)
def _factorint_cached(x: int):
    return factorint(x)


def prime_factor_counts(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=int)
    unique_vals = np.unique(arr)
    factorizations = {x: _factorint_cached(x) for x in unique_vals}

    # Gather all unique primes
    all_primes = sorted({p for f in factorizations.values() for p in f})

    # Build result matrix
    result = np.zeros((len(arr), len(all_primes)), dtype=int)
    prime_index = {p: j for j, p in enumerate(all_primes)}

    for i, x in enumerate(arr):
        for p, exp in factorizations[x].items():
            result[i, prime_index[p]] = exp

    return result


def paretoset_grouped_dirty(df: pd.DataFrame, sense: list[str]):
    # return paretoset(df, sense=sense)

    assert all(i == c for i, c in enumerate(df.columns))
    assert len(sense) == len(df.columns)

    from paretoset.algorithms_numba import paretoset_jit
    from paretoset.algorithms_numba import BNL

    for c in df.columns:
        if sense[c] == "max":
            df[c] = -df[c]
            sense[c] = "min"

    GROUP_SIZE = 128

    group_by = [c for c in df.columns if sense[c] == "diff"]
    n_groups = prod(len(df[c].unique()) for c in group_by)

    if len(df) / n_groups < GROUP_SIZE:
        return paretoset(df, sense=sense)

    c2unique = {c: len(df[c].unique()) for c in df.columns if c not in group_by}
    while c2unique:
        col, n = min(c2unique.items(), key=lambda x: x[1])
        c2unique.pop(col)
        n_groups *= n
        if len(df) / n_groups < GROUP_SIZE:
            break
        group_by.append(col)

    n_diffs = sum(x == "diff" for x in sense)
    if len(group_by) < 2 or len(group_by) == n_diffs:
        return paretoset(df, sense=sense)

    def _row_from_group(mins, group):
        per_col_mins = group.min(axis=0)
        per_col_maxs = group.max(axis=0)
        good_row = group.iloc[
            np.argmin((group ** (1 / len(group.columns))).prod(axis=1))
        ]
        return [mins, per_col_mins, per_col_maxs, good_row, group]

    groups = list(df.groupby(group_by))
    groups_by_diff = {}
    keepcols = [c for c in df.columns if c not in group_by]
    for x, group in groups:
        diffs, mins = x[:n_diffs], x[n_diffs:]
        group = group[keepcols]
        groups_by_diff.setdefault(diffs, []).append(_row_from_group(mins, group))

    # print(f'Grouped into {len(groups)} groups using {len(group_by)} columns')
    # orig_size = len(df)
    # n_groups = len(groups)
    # n_cols = len(keepcols)
    # new_size = sum(len(g2) for g in groups_by_diff.values() for _, _, _, g2 in g)
    # print(f'Grouped into {n_groups} groups, {orig_size} -> {new_size} rows, {n_cols} columns. Remaining {len(keepcols)} columns')

    for groups in groups_by_diff.values():
        for i, (
            mins_a,
            per_col_mins_a,
            per_col_maxs_a,
            good_row_a,
            group_a,
        ) in enumerate(groups):
            if group_a is None:
                continue

            for j, (
                mins_b,
                per_col_mins_b,
                per_col_maxs_b,
                good_row_b,
                group_b,
            ) in enumerate(groups):
                if group_b is None or i == j:
                    continue

                if all(a <= b for a, b in zip(good_row_a, per_col_mins_b)):
                    groups[j][-1] = None
                    continue

                if all(a <= b for a, b in zip(good_row_a, good_row_b)):
                    # The good row of a dominates the good row of b. It'll likely
                    # dominate many b!
                    group_b = group_b[(group_b < good_row_a).any(axis=1)]
                    if len(group_b) == 0:
                        groups[j][-1] = None
                        continue
                    groups[j].clear()
                    groups[j].extend(_row_from_group(mins_b, group_b))

                # # a can only dominate b if all of the min columns dominate
                # if not all(a <= b for a, b in zip(mins_a, mins_b)):
                #     continue

                # # Check if any b beats all a. If so, continue.
                # if any(a > b for a, b in zip(per_col_mins_a, per_col_maxs_b)):
                #     continue

                # # # Check if any a beats every b. If so, get rid of b.
                # # a_doms = all(a <= b for a, b in zip(per_col_maxs_a, per_col_mins_b))
                # # if a_doms:
                # #     groups[j][-1] = None
                # #     # print(f'Dropping dominated group {j}')
                # #     continue

                # row_a = group_a.iloc[np.random.randint(len(group_a))]
                # if all(a <= b for a, b in zip(row_w_min_first_obj_b, per_col_mins_b)):
                #     groups[j][-1] = None

                # Everything below just ended up making things slower

                # if any(a > b for a, b in zip(row_a, per_col_maxs_b)):
                #     continue

                # continue

                # # Grab a random a. Get rid of all b that are dominated by it.
                # a_lt_b_maxes = group_a.iloc[
                #     np.where(np.all(group_a <= per_col_maxs_b, axis=1))[0]
                # ]
                # if len(a_lt_b_maxes) == 0:
                #     continue

                # row_a = a_lt_b_maxes.iloc[np.random.randint(len(a_lt_b_maxes))]

                # b_idx = np.where(np.any(group_b < row_a, axis=1))[0]
                # if len(b_idx) == 0:
                #     groups[j][-1] = None
                # else:
                #     groups[j][-1] = group_b.iloc[b_idx]
                #     groups[j][1] = group_b.iloc[b_idx].min(axis=0)
                #     groups[j][2] = group_b.iloc[b_idx].max(axis=0)

                # # Now we're in a case where a may dominate b. Update b.
                # catted = pd.concat([group_a, group_b], axis=0)
                # mask = np.concatenate([
                #     np.zeros(len(group_a), dtype=bool),
                #     np.ones(len(group_b), dtype=bool)
                # ])
                # catted = catted[paretoset_jit(catted.to_numpy()) & mask]
                # groups[j][1] = catted.min(axis=0)
                # groups[j][2] = catted.max(axis=0)
                # groups[j][3] = catted

    result = np.zeros(len(df), dtype=bool)
    for group in groups_by_diff.values():
        for _, _, _, _, group in group:
            if group is not None:
                result[group[paretoset_jit(group.to_numpy())].index] = True

    return result


def makepareto_numpy(
    mappings: np.ndarray,
    goals: list[str],
    dirty: bool = False,
) -> pd.DataFrame:

    to_pareto = []
    new_goals = []
    assert len(goals) == mappings.shape[1]
    for c in range(mappings.shape[1]):
        if len(np.unique(mappings[:, c])) <= 1:
            continue

        goal = goals[c]
        # if goal != "diff" and dirty and len(np.unique(mappings[:, c])) < np.log2(mappings.shape[0]):
        #     # print(f"Changed {goal} to diff because there are {len(np.unique(mappings[:, c]))} unique values for {mappings.shape[0]} rows")
        #     goal = "diff"

        if goal in ["min", "max"]:
            l = logify(mappings[:, c].reshape((-1, 1)))
            to_pareto.append(l if goal == "min" else -l)
            new_goals.append("min")
        elif goal == "diff":
            to_pareto.append(mappings[:, c].reshape((-1, 1)))
            new_goals.append("diff")
            if dirty and len(np.unique(to_pareto[-1])) > mappings.shape[0] / 2:
                return np.ones(mappings.shape[0], dtype=bool)
        elif goal == "min_per_prime_factor":
            if not dirty:
                # Paretoset tends to be faster with these as diffs. Tanner tried for a
                # long time to get min_per_prime_factor to be faster, but it
                # didn't work. What it would do is say that if one choice for an inner
                # loop has used up fewer of every prime factor than another choice, then
                # the latter would give a superset of options for outer loops.
                # Intuitively, we could enable more pruning by doing this instead of
                # "diff", which is overconservative. Likewise, we could do "min" for
                # imperfect instead of "diff". However, this ultimately made things
                # slower because it didn't get much Pareto pruning, but caused many more
                # Pareto comparisons ("diff" partitioning into N partitions --> N^2
                # improvement). I hypothesize that the reason that it doesn't improve
                # pruning much is that when we've enumerated a loop but not the loop
                # above it, the given loop is almost always trading off tile shape for
                # accesses, leading to no point being dominated by another point.
                to_pareto.append(mappings[:, c].reshape((-1, 1)))
                new_goals.append("diff")
            else:
                counts = prime_factor_counts(mappings[:, c])
                for i in range(counts.shape[1]):
                    to_pareto.append(counts[:, i].reshape((-1, 1)))
                    new_goals.append("min")
        elif goal == "max_per_prime_factor":
            if not dirty:
                # See above big comment.
                to_pareto.append(mappings[:, c].reshape((-1, 1)))
                new_goals.append("diff")
            else:
                counts = prime_factor_counts(mappings[:, c])
                for i in range(counts.shape[1]):
                    to_pareto.append(counts[:, i].reshape((-1, 1)))
                    new_goals.append("max")
        else:
            raise ValueError(f"Unknown goal: {goal}")

    if not to_pareto:
        n = np.zeros(mappings.shape[0], dtype=bool)
        n[0] = True
        return n

    df = pd.DataFrame(np.concatenate(to_pareto, axis=1), columns=range(len(to_pareto)))

    if dirty:
        return paretoset_grouped_dirty(df, sense=new_goals)
    return paretoset(df, sense=new_goals)
