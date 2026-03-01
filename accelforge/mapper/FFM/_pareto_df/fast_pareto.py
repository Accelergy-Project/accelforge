"""
Fast Pareto front computation.
Handles min, max, diff, min_per_prime_factor, max_per_prime_factor goals.

SFS + Block-BNL (block size 16) with:
- _is_constant JIT for O(1) column filtering
- np.take for fast contiguous column extraction
- Float32 preservation for all-min goals
- Pair-packed int64 encoding for float32 diff columns
- Per-group varying-column detection, 1D/2D special paths
- Window-min quick-check + block-min pruning
"""

import numpy as np
import numba
import pandas as pd
from sympy import factorint
import functools

from accelforge.util import NUMPY_FLOAT_TYPE


# ============================================================================
# Numba helpers
# ============================================================================

@numba.jit(nopython=True)
def _is_constant(arr, n):
    """Check if a 1D array is constant. O(1) best case via early exit."""
    v0 = arr[0]
    for i in range(1, n):
        if arr[i] != v0:
            return False
    return True


@numba.jit(nopython=True)
def _counting_sort(codes, n_groups):
    """Counting sort: O(n + k)."""
    n = len(codes)
    counts = np.zeros(n_groups, dtype=np.int64)
    for i in range(n):
        counts[codes[i]] += 1
    offsets = np.empty(n_groups + 1, dtype=np.int64)
    offsets[0] = 0
    for i in range(n_groups):
        offsets[i + 1] = offsets[i] + counts[i]
    pos = offsets[:n_groups].copy()
    result = np.empty(n, dtype=np.int64)
    for i in range(n):
        c = codes[i]
        result[pos[c]] = i
        pos[c] += 1
    return result, offsets


# ============================================================================
# Prime factor utilities
# ============================================================================

@functools.lru_cache(maxsize=10000)
def _factorint_cached(x: int):
    return factorint(x)


def prime_factor_counts(arr):
    """Expand an integer column into columns of prime factor counts."""
    arr = np.asarray(arr, dtype=int).ravel()
    unique_vals = np.unique(arr)
    factorizations = {int(x): _factorint_cached(int(x)) for x in unique_vals}
    all_primes = sorted({p for f in factorizations.values() for p in f})
    if not all_primes:
        return np.zeros((len(arr), 1), dtype=NUMPY_FLOAT_TYPE)
    result = np.zeros((len(arr), len(all_primes)), dtype=NUMPY_FLOAT_TYPE)
    prime_index = {p: j for j, p in enumerate(all_primes)}
    for i, x in enumerate(arr):
        for p, exp in factorizations[int(x)].items():
            result[i, prime_index[p]] = exp
    return result


# ============================================================================
# Core SFS + Block-BNL (block size 16)
# ============================================================================

@numba.jit(nopython=True, fastmath=True)
def _sfs_bnl_core(data, sorted_idx, offsets, n_total_groups, result_mask):
    d = data.shape[1]
    max_n = numba.int64(0)
    for g in range(n_total_groups):
        gs = offsets[g + 1] - offsets[g]
        if gs > max_n:
            max_n = gs

    if max_n <= 1:
        for g in range(n_total_groups):
            if offsets[g + 1] > offsets[g]:
                result_mask[sorted_idx[offsets[g]]] = True
        return

    col_min = np.empty(d, dtype=data.dtype)
    col_max = np.empty(d, dtype=data.dtype)
    varying = np.empty(d, dtype=numba.int64)
    local = np.empty((max_n, d), dtype=data.dtype)
    window = np.empty((max_n, d), dtype=data.dtype)
    sums_buf = np.empty(max_n, dtype=NUMPY_FLOAT_TYPE)
    max_blocks = (max_n >> 4) + 1
    block_mins = np.empty((max_blocks, d), dtype=data.dtype)
    window_min = np.empty(d, dtype=data.dtype)

    for g in range(n_total_groups):
        start = offsets[g]
        end = offsets[g + 1]
        n = end - start
        if n == 0:
            continue
        if n == 1:
            result_mask[sorted_idx[start]] = True
            continue

        group_idx = sorted_idx[start:end]

        if d == 1:
            min_val = data[group_idx[0], 0]
            for i in range(1, n):
                v = data[group_idx[i], 0]
                if v < min_val:
                    min_val = v
            for i in range(n):
                if data[group_idx[i], 0] <= min_val:
                    result_mask[group_idx[i]] = True
            continue

        # Find varying columns
        for k in range(d):
            val0 = data[group_idx[0], k]
            col_min[k] = val0
            col_max[k] = val0
        for i in range(1, n):
            for k in range(d):
                v = data[group_idx[i], k]
                if v < col_min[k]:
                    col_min[k] = v
                if v > col_max[k]:
                    col_max[k] = v

        n_varying = 0
        for k in range(d):
            if col_min[k] != col_max[k]:
                varying[n_varying] = k
                n_varying += 1

        if n_varying == 0:
            for i in range(n):
                result_mask[group_idx[i]] = True
            continue

        dv = n_varying

        if dv == 1:
            vc = varying[0]
            min_val = col_min[vc]
            for i in range(n):
                if data[group_idx[i], vc] <= min_val:
                    result_mask[group_idx[i]] = True
            continue

        # Copy varying columns into contiguous local buffer
        for i in range(n):
            gi = group_idx[i]
            for kk in range(dv):
                local[i, kk] = data[gi, varying[kk]]

        if dv == 2:
            # 2D: sort by col0, group-aware sweep
            order = np.argsort(local[:n, 0])
            best_c1 = numba.float64(1e308)
            i_start = numba.int64(0)
            while i_start < n:
                c0_val = local[order[i_start], 0]
                i_end = i_start + 1
                g_min_c1 = local[order[i_start], 1]
                while i_end < n and local[order[i_end], 0] == c0_val:
                    v = local[order[i_end], 1]
                    if v < g_min_c1:
                        g_min_c1 = v
                    i_end += 1
                if g_min_c1 < best_c1:
                    for k in range(i_start, i_end):
                        if local[order[k], 1] == g_min_c1:
                            result_mask[group_idx[order[k]]] = True
                    best_c1 = g_min_c1
                i_start = i_end
            continue

        # General case: SFS + block BNL (block size 16)
        for i in range(n):
            s = 0.0
            for kk in range(dv):
                s += local[i, kk]
            sums_buf[i] = s

        order = np.argsort(sums_buf[:n])

        n_blk = (n >> 4) + 1
        for b in range(n_blk):
            for kk in range(dv):
                block_mins[b, kk] = 1e30

        idx0 = order[0]
        for kk in range(dv):
            v = local[idx0, kk]
            window[0, kk] = v
            window_min[kk] = v
            block_mins[0, kk] = v
        result_mask[group_idx[idx0]] = True
        w_size = numba.int64(1)

        for ii in range(1, n):
            i = order[ii]

            quick_safe = False
            for kk in range(dv):
                if window_min[kk] > local[i, kk]:
                    quick_safe = True
                    break

            dominated = False
            if not quick_safe:
                n_blocks = (w_size >> 4) + 1
                for b in range(n_blocks):
                    block_ok = True
                    for kk in range(dv):
                        if block_mins[b, kk] > local[i, kk]:
                            block_ok = False
                            break
                    if not block_ok:
                        continue

                    b_start = b << 4
                    b_end = b_start + 16
                    if b_end > w_size:
                        b_end = w_size
                    for w in range(b_start, b_end):
                        all_leq = True
                        any_less = False
                        for kk in range(dv):
                            wk = window[w, kk]
                            ck = local[i, kk]
                            if wk > ck:
                                all_leq = False
                                break
                            if wk < ck:
                                any_less = True
                        if all_leq and any_less:
                            dominated = True
                            break
                    if dominated:
                        break

            if not dominated:
                for kk in range(dv):
                    v = local[i, kk]
                    window[w_size, kk] = v
                    if v < window_min[kk]:
                        window_min[kk] = v
                b = w_size >> 4
                for kk in range(dv):
                    v = window[w_size, kk]
                    if v < block_mins[b, kk]:
                        block_mins[b, kk] = v
                w_size += 1
                result_mask[group_idx[i]] = True


# ============================================================================
# Group encoding
# ============================================================================

def _encode_groups(data, diff_cols):
    """Encode diff columns into integer group IDs using pd.factorize."""
    n = data.shape[0]
    codes = np.zeros(n, dtype=np.int64)
    multiplier = 1

    if data.dtype == np.float32 and len(diff_cols) >= 2:
        # np.take all diff cols at once, view pairs as int64
        diff_indices = np.array(diff_cols, dtype=np.intp)
        diff_data = np.take(data, diff_indices, axis=1)
        n_diff = len(diff_cols)
        if n_diff % 2 == 1:
            diff_data = np.column_stack([diff_data, np.zeros(n, dtype=np.float32)])
        packed = diff_data.view(np.int64)
        for j in range(packed.shape[1]):
            inv, uniques = pd.factorize(packed[:, j], sort=False)
            n_unique = len(uniques)
            if multiplier * n_unique > 2**62:
                _, codes = np.unique(codes, return_inverse=True)
                codes = codes.astype(np.int64)
                multiplier = int(len(_))
            codes += inv.astype(np.int64) * np.int64(multiplier)
            multiplier *= n_unique
    else:
        for c in diff_cols:
            inv, uniques = pd.factorize(data[:, c], sort=False)
            n_unique = len(uniques)
            if multiplier * n_unique > 2**62:
                _, codes = np.unique(codes, return_inverse=True)
                codes = codes.astype(np.int64)
                multiplier = int(len(_))
            codes += inv.astype(np.int64) * np.int64(multiplier)
            multiplier *= n_unique

    n_total = multiplier
    if n_total > n or n_total > 2**62:
        _, codes = np.unique(codes, return_inverse=True)
        codes = codes.astype(np.int64)
        n_total = int(len(_))
    return codes, n_total


# ============================================================================
# Deduplication
# ============================================================================

def _dedup_mask(data, n):
    """Return a boolean mask keeping only the first occurrence of each unique row."""
    if n == 0:
        return np.zeros(0, dtype=bool)
    if data.dtype.kind in ('f', 'i', 'u'):
        c = np.ascontiguousarray(data)
        dt = np.dtype([(f'f{i}', c.dtype) for i in range(c.shape[1])])
        _, idx = np.unique(c.view(dt).ravel(), return_index=True)
    else:
        codes = np.zeros(n, dtype=np.int64)
        multiplier = 1
        for c in range(data.shape[1]):
            inv, uniques = pd.factorize(data[:, c], sort=False)
            n_unique = len(uniques)
            if multiplier * n_unique > 2**62:
                _, codes = np.unique(codes, return_inverse=True)
                codes = codes.astype(np.int64)
                multiplier = int(len(_))
            codes += inv.astype(np.int64) * np.int64(multiplier)
            multiplier *= n_unique
        _, idx = np.unique(codes, return_index=True)
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask


# ============================================================================
# Main entry point
# ============================================================================

def fast_pareto_mask(df_values, goals, distinct=True):
    """
    Compute Pareto mask for a numpy array with given goals.

    Args:
        df_values: 2D numpy array (or DataFrame.values)
        goals: list of goal strings per column:
               'min', 'max', 'diff', 'min_per_prime_factor', 'max_per_prime_factor'
        distinct: if True, keep only first occurrence of duplicate rows

    Returns:
        boolean mask array
    """
    data = np.asarray(df_values)
    n = data.shape[0]
    if n <= 1:
        return np.ones(n, dtype=bool)

    # Parse goals
    diff_cols = []
    simple_opt_cols = []   # (col_index, sign)
    extra_opt_parts = []   # (array, sign) for prime factor expansions

    for i, g in enumerate(goals):
        if g == 'diff':
            diff_cols.append(i)
        elif g == 'min':
            simple_opt_cols.append((i, 1.0))
        elif g == 'max':
            simple_opt_cols.append((i, -1.0))
        elif g == 'min_per_prime_factor':
            counts = prime_factor_counts(data[:, i])
            for j in range(counts.shape[1]):
                extra_opt_parts.append((counts[:, j], 1.0))
        elif g == 'max_per_prime_factor':
            counts = prime_factor_counts(data[:, i])
            for j in range(counts.shape[1]):
                extra_opt_parts.append((counts[:, j], -1.0))
        else:
            raise ValueError(f"Unknown goal: {g}")

    if len(simple_opt_cols) == 0 and len(extra_opt_parts) == 0:
        if distinct:
            return _dedup_mask(data, n)
        return np.ones(n, dtype=bool)

    # Filter to non-constant columns (JIT early-exit for numeric dtypes)
    effective_cols = []
    use_jit = data.dtype.kind in ('f', 'i', 'u')
    for col_idx, sign in simple_opt_cols:
        col = data[:, col_idx]
        if use_jit:
            is_const = _is_constant(col, n)
        else:
            is_const = col.min() == col.max()
        if not is_const:
            effective_cols.append((col_idx, sign))

    effective_extra = []
    for col, sign in extra_opt_parts:
        if not _is_constant(col, n):
            effective_extra.append((col, sign))

    n_eff = len(effective_cols) + len(effective_extra)
    if n_eff == 0:
        if distinct:
            return _dedup_mask(data, n)
        return np.ones(n, dtype=bool)

    # Build eff_data
    all_simple = len(effective_extra) == 0
    all_min = all_simple and all(sign == 1.0 for _, sign in effective_cols)
    use_f32 = all_min and data.dtype == np.float32
    eff_dtype = np.float32 if use_f32 else NUMPY_FLOAT_TYPE

    if all_simple and all_min:
        col_indices = np.array([c for c, _ in effective_cols], dtype=np.intp)
        eff_data = np.take(data, col_indices, axis=1)
        if eff_data.dtype != eff_dtype:
            eff_data = eff_data.astype(eff_dtype)
    elif all_simple:
        col_indices = np.array([c for c, _ in effective_cols], dtype=np.intp)
        eff_data = np.take(data, col_indices, axis=1)
        if eff_data.dtype != eff_dtype:
            eff_data = eff_data.astype(eff_dtype)
        else:
            eff_data = eff_data.copy()
        for j, (_, sign) in enumerate(effective_cols):
            if sign != 1.0:
                np.negative(eff_data[:, j], out=eff_data[:, j])
    else:
        eff_data = np.empty((n, n_eff), dtype=eff_dtype)
        j = 0
        for col_idx, sign in effective_cols:
            if sign == 1.0:
                eff_data[:, j] = data[:, col_idx]
            else:
                eff_data[:, j] = -data[:, col_idx]
            j += 1
        for col, sign in effective_extra:
            if sign == 1.0:
                eff_data[:, j] = col
            else:
                eff_data[:, j] = -col
            j += 1

    # Groupby diff columns
    if len(diff_cols) == 0:
        sorted_idx = np.arange(n, dtype=np.int64)
        offsets = np.array([0, n], dtype=np.int64)
        n_total_groups = 1
    else:
        codes, n_total_groups = _encode_groups(data, diff_cols)
        sorted_idx, offsets = _counting_sort(codes, n_total_groups)

    # Run SFS + Block-BNL core
    mask = np.zeros(n, dtype=np.bool_)
    _sfs_bnl_core(eff_data, sorted_idx, offsets, n_total_groups, mask)

    # Deduplicate
    if distinct:
        pareto_idx = np.where(mask)[0]
        n_pareto = len(pareto_idx)
        if n_pareto > 1:
            pareto_rows = data[pareto_idx]
            dup_mask = pd.DataFrame(pareto_rows).duplicated(keep='first').values
            if dup_mask.any():
                new_mask = np.zeros(n, dtype=bool)
                new_mask[pareto_idx[~dup_mask]] = True
                mask = new_mask

    return mask


# ============================================================================
# Warmup
# ============================================================================

def warmup():
    """Pre-compile all numba functions."""
    d3 = np.random.rand(20, 5).astype(np.float32)
    idx = np.arange(20, dtype=np.int64)
    off = np.array([0, 10, 20], dtype=np.int64)
    mask = np.zeros(20, dtype=np.bool_)
    _sfs_bnl_core(d3, idx, off, 2, mask)

    d3_64 = d3.astype(NUMPY_FLOAT_TYPE)
    mask2 = np.zeros(20, dtype=np.bool_)
    _sfs_bnl_core(d3_64, idx, off, 2, mask2)

    _counting_sort(np.array([0, 1, 0], dtype=np.int64), 2)
    _is_constant(d3[:, 0], 20)
    _is_constant(d3_64[:, 0], 20)
