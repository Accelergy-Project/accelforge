# Sparsity-Support Branch: Code Review Summary

Review of all code changes between `sparsity-support` and `main` branches.
162 files changed, ~18K lines added/modified.

All 12 issues below have been fixed. 297 tests pass with no regressions.

---

## High Priority (Fixed)

### 1. Unconditional `break` in compute_latency_ratio
**File:** `accelforge/model/sparse_adjustment.py:1103-1108`
**Fix:** Indented `break` inside `if pre > 0:` so the loop finds the
first compute level with a valid pre-SAF count.

### 2. Double-divide by `s.fanout` in power gating
**File:** `accelforge/mapper/FFM/.../run_model.py:134-138`
**Fix:** Pass raw `used_fanout` to `_power_gating` instead of the
pre-divided `spatial_usage`. `_power_gating` divides by `s.fanout`
internally, so passing pre-divided values caused a double divide.

### 3. Bare `except Exception: pass` in sparse latency
**File:** `accelforge/mapper/FFM/.../run_model.py:96-103`
**Fix:** Narrowed to `except (TypeError, ValueError)` with
`logging.warning` (matching the dense path's error reporting style).

---

## Medium Priority (Fixed)

### 4. Binomial loop O(tile_size)
**File:** `accelforge/model/sparse_adjustment.py:452-458`
**Fix:** Replaced Python `for` loop + `math.comb` with vectorized
`scipy.stats.binom.pmf` + `numpy`. Imports are local to the function
to avoid loading numpy/scipy when position-space utilization isn't used.

### 5. `_run_format_cascade` — no length validation on zip
**File:** `accelforge/model/sparse_formats.py:273-277`
**Fix:** Added `len(rank_formats) != len(dimension_sizes)` check with
`ValueError` before the `zip` loop.

### 6. `position_skip_level` scalar — last-write-wins
**File:** `accelforge/model/sparse_adjustment.py:881-887`
**Fix:** Added validation that raises `ValueError` if self-conditioned
skipping is declared at multiple different levels.

### 7. `_emit()` / `_emit_if_declared()` set max_per_unit = total
**File:** `accelforge/model/sparse_adjustment.py:161-177, 180-199`
**Fix:** Added optional `max_per_unit` parameter to both `_emit()` and
`_emit_if_declared()`, forwarded through the call chain. Defaults to
`total` (correct for fanout=1). Callers with spatial context can pass
the per-unit value explicitly.

### 8. Python `max()` on potentially symbolic values
**File:** `accelforge/mapper/FFM/.../run_model.py:409-416`
**Fix:** Replaced `max(...)` with `Max(...)` (sympy) for
`max_tensor_read_actions` and `max_tensor_write_actions`.

---

## Low Priority (Fixed)

### 9. `has_format()` returns True for empty RepresentationFormat entries
**File:** `accelforge/frontend/sparse.py:279-288`
**Fix:** Changed to `any(rf.format is not None or rf.ranks is not None ...)`
so entries with neither `format` nor `ranks` are ignored.

### 10. Size-1 dimension filtering undocumented
**File:** `accelforge/model/sparse_adjustment.py:498-502`
**Fix:** Added comment explaining the intentional behavior: trivial
dimensions (size 1) are excluded because UOP on a size-1 dim produces
zero overhead, and format auto-expansion uses the count of non-trivial dims.

### 11. No validation on `kind` fields
**File:** `accelforge/frontend/sparse.py:165, 196`
**Fix:** Changed `kind: str` to `Literal["gating", "skipping"]` on
`ActionOptimization` and `ComputeOptimization`. Pydantic now rejects
invalid values. Self-conditioned skipping is expressed as
`kind: "skipping"` with `target in condition_on`.

### 12. Variable shadowing in spatial usage loop
**File:** `accelforge/mapper/FFM/.../run_model.py:131`
**Fix:** Renamed shadowed loop variable from `s = f"usage<SEP>..."` to
`usage_key = f"usage<SEP>..."`.

---

## Not Issues (Verified as Intentional)

- **`"parent" in attr` substring match** (symbolic.py:182-184): Intentional
  naming convention — lines 143-144 explicitly state attributes are named
  with "parent" so the substring match captures them all.

- **Output tensor drain reads suppressed** (symbolic.py:1349-1358):
  Intentional design — writeback drains don't incur separate read cost.

- **`conditioned()` uses `__new__`** (density_model.py:110): Intentional
  to avoid `ceil(ceil(x)/N * N)` drift from re-running `__init__`.

- **Negative density handling** (density_model.py:59): Properly handled
  with `if density <= 0: self.r = 0`.
