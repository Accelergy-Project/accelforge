#!/usr/bin/env bash
#
# run_profile.sh -- run one profiling leg (FC or torus) for one or more
# NCCL collectives on a provisioned 8x-H100 instance, and parse each raw
# log into the unified CSV schema consumed by the correlation notebook.
#
# This script runs ON the EC2 instance (not locally); it has no GPUs to
# talk to when checked out anywhere else, so it is verified here by
# `bash -n` (syntax check) only -- see the WP2 report for details.
#
# Usage:
#   ./run_profile.sh <results_dir> <topology fc|torus> <min_bytes> \
#       <max_bytes> <dims> <collective> [<collective>...]
#
# Env overrides:
#   NCCL_TESTS_DIR   Path to a built nccl-tests checkout. Default: $HOME/nccl-tests
#   TORUS_BENCH_BIN  Path to the built torus_bench binary. Default: $HOME/torus_bench/torus_bench
#   PARSE_NCCL       Path to parse_nccl.py. Default: the copy next to this script.
#   WARMUP           nccl-tests/torus_bench warmup iteration count. Default: 5
#   ITERS            nccl-tests/torus_bench measured iteration count. Default: 20
#
# Design: fail fast and loud (set -euo pipefail) rather than silently
# continuing past a failed collective run or a failed parse -- a partial,
# uncaught failure here would otherwise show up much later as a confusing
# gap in the correlation notebook's data rather than as a build/run error
# on the instance where it's cheap to diagnose.
set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
if [[ $# -lt 6 ]]; then
    echo "Usage: $0 <results_dir> <topology fc|torus> <min_bytes> <max_bytes> <dims> <collective> [<collective>...]" >&2
    exit 1
fi

results_dir="$1"; shift
topology="$1"; shift
min_bytes="$1"; shift
max_bytes="$1"; shift
dims="$1"; shift
# Remaining positional args are the list of collectives to profile in this leg.
collectives=("$@")

if [[ "$topology" != "fc" && "$topology" != "torus" ]]; then
    echo "ERROR: <topology> must be 'fc' or 'torus', got '$topology'" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Environment / defaults
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NCCL_TESTS_DIR="${NCCL_TESTS_DIR:-$HOME/nccl-tests}"
TORUS_BENCH_BIN="${TORUS_BENCH_BIN:-$HOME/torus_bench/torus_bench}"
# Design: PARSE_NCCL defaults to the copy sitting next to this script
# (rather than requiring it on PATH or hard-coding an absolute install
# path) so the pair of files can be scp'd to the instance as a unit and
# just work.
PARSE_NCCL="${PARSE_NCCL:-$SCRIPT_DIR/parse_nccl.py}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-20}"

RAW_DIR="$results_dir/raw"
CSV_DIR="$results_dir/csv"
mkdir -p "$RAW_DIR" "$CSV_DIR"

# ---------------------------------------------------------------------------
# metadata.txt: written exactly once per invocation (not once per
# collective), since it captures machine/software state that doesn't
# change across the collectives loop below.
# ---------------------------------------------------------------------------
metadata_file="$results_dir/metadata.txt"
{
    echo "=== date ==="
    date -u
    echo
    echo "=== uname -a ==="
    uname -a
    echo
    echo "=== nvidia-smi --query-gpu=name,driver_version --format=csv ==="
    nvidia-smi --query-gpu=name,driver_version --format=csv
    echo
    echo "=== nvidia-smi topo -m ==="
    nvidia-smi topo -m
    echo
    echo "=== nccl-tests git rev ==="
    # Best-effort: nccl-tests may not be a git checkout (e.g. if a tarball
    # was scp'd instead), so a failed `git rev-parse` here must not abort
    # the whole script under `set -e`.
    git -C "$NCCL_TESTS_DIR" rev-parse HEAD 2>/dev/null || echo "unknown (not a git checkout or NCCL_TESTS_DIR missing)"
} > "$metadata_file"
echo "Wrote $metadata_file"

# ---------------------------------------------------------------------------
# FC leg: stock nccl-tests binaries.
# ---------------------------------------------------------------------------
run_fc_leg() {
    local collective="$1"
    local bin="$NCCL_TESTS_DIR/build/${collective}_perf"
    local raw_log="$RAW_DIR/fc_${collective}.log"
    local out_csv="$CSV_DIR/fc_${collective}.csv"

    if [[ ! -x "$bin" ]]; then
        echo "ERROR: nccl-tests binary not found: $bin" >&2
        echo "       Build nccl-tests first, e.g. via setup_node.sh, or:" >&2
        echo "       make -j -C \"$NCCL_TESTS_DIR\" MPI=0 CUDA_HOME=\"\${CUDA_HOME:-/usr/local/cuda}\"" >&2
        exit 1
    fi

    echo "=== FC leg: $collective ==="
    "$bin" -b "$min_bytes" -e "$max_bytes" -f 2 -g 8 -w "$WARMUP" -n "$ITERS" -c 1 | tee "$raw_log"
    python3 "$PARSE_NCCL" "$raw_log" \
        --source nccl-tests \
        --collective "$collective" \
        --topology fc \
        --dims "$dims" \
        --out "$out_csv"

    # Fail loudly on a header-only CSV: parse_nccl.py can exit 0 while emitting zero data rows
    # (e.g. every swept size hit a divisibility skip for this dims/size combination), which
    # would otherwise look identical to a real, successful leg -- a silent data gap discovered
    # only much later in the correlation notebook, rather than here, where the raw log needed
    # to diagnose it is still on disk and cheap to inspect.
    csv_lines="$(wc -l < "$out_csv")"
    if [[ "$csv_lines" -le 1 ]]; then
        echo "ERROR: $out_csv has no data rows (header-only or empty) -- see $raw_log" >&2
        exit 1
    fi
    echo "Wrote $out_csv"
}

# ---------------------------------------------------------------------------
# Torus leg: custom torus_bench binary (built by the sibling work package).
# ---------------------------------------------------------------------------
run_torus_leg() {
    local collective="$1"
    local raw_log="$RAW_DIR/torus_${collective}.log"
    local out_csv="$CSV_DIR/torus_${collective}.csv"

    if [[ ! -x "$TORUS_BENCH_BIN" ]]; then
        echo "ERROR: torus_bench binary not found: $TORUS_BENCH_BIN" >&2
        echo "       Build torus_bench first (see setup_node.sh), e.g.:" >&2
        echo "       make -C \"\$TORUS_BENCH_DIR\" torus_bench" >&2
        exit 1
    fi

    echo "=== Torus leg: $collective ==="
    "$TORUS_BENCH_BIN" --collective "$collective" --dims "$dims" \
        -b "$min_bytes" -e "$max_bytes" -f 2 -w "$WARMUP" -n "$ITERS" --check \
        | tee "$raw_log"
    python3 "$PARSE_NCCL" "$raw_log" \
        --source torus_bench \
        --collective "$collective" \
        --topology torus \
        --dims "$dims" \
        --out "$out_csv"

    # See run_fc_leg's identical check above for the full rationale: a header-only CSV here
    # (e.g. every swept size hit torus_bench's own divisibility SKIP for this dims/size
    # combination) must fail the script loudly rather than silently passing as "done".
    csv_lines="$(wc -l < "$out_csv")"
    if [[ "$csv_lines" -le 1 ]]; then
        echo "ERROR: $out_csv has no data rows (header-only or empty) -- see $raw_log" >&2
        exit 1
    fi
    echo "Wrote $out_csv"
}

# ---------------------------------------------------------------------------
# Main loop: one leg (selected by $topology), all requested collectives.
# ---------------------------------------------------------------------------
for collective in "${collectives[@]}"; do
    if [[ "$topology" == "fc" ]]; then
        run_fc_leg "$collective"
    else
        run_torus_leg "$collective"
    fi
done

echo "Done. Results in $results_dir"
