#!/usr/bin/env bash
#
# setup_node.sh -- idempotent one-time (but safe-to-rerun) setup for an
# EC2 p5.48xlarge (8x H100, NVSwitch) profiling instance: arms a dead-man
# shutdown, verifies the GPU fabric is visible, builds nccl-tests and
# torus_bench if not already built, and prints a versions summary.
#
# This script runs ON the EC2 instance; it is verified here by `bash -n`
# (syntax check) only -- see the WP2 report for details.
#
# Env:
#   DEADMAN_MINUTES   Minutes until the dead-man shutdown fires. Default: 120
#   NCCL_TESTS_DIR    Where to clone/build nccl-tests. Default: $HOME/nccl-tests
#   TORUS_BENCH_DIR   Where torus_bench is scp'd/built. Default: $HOME/torus_bench
#   CUDA_HOME         CUDA toolkit root used to build nccl-tests. Default: /usr/local/cuda
set -euo pipefail

DEADMAN_MINUTES="${DEADMAN_MINUTES:-120}"
NCCL_TESTS_DIR="${NCCL_TESTS_DIR:-$HOME/nccl-tests}"
TORUS_BENCH_DIR="${TORUS_BENCH_DIR:-$HOME/torus_bench}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

# ---------------------------------------------------------------------------
# Step 1: arm the dead-man switch FIRST, before anything else can fail or
# hang.
#
# Design/WHY this must be first: the instance is launched with
# shutdown-behavior=terminate, so a `shutdown -P` here is what actually
# terminates (not just stops) the instance and stops billing. If setup
# were to fail, hang (e.g. a stuck `make`, a stalled git clone over a flaky
# network), or if the orchestrating controller process on the caller's side
# dies/loses connectivity, this is the only backstop that guarantees the
# (expensive, 8x H100) instance doesn't run forever. Arming it before any
# other step -- including the GPU sanity check below, which could itself
# hang on a broken driver -- ensures the cost cap applies unconditionally
# from the very start of setup, not only after setup "succeeds".
# ---------------------------------------------------------------------------
echo "Arming dead-man shutdown: instance will terminate in ${DEADMAN_MINUTES} minutes unless this script (or a later run of it) is used to push it back further."
# Cancel any already-pending shutdown before arming a new one: issuing a second `shutdown`
# while one is already pending errors on some systemd versions, and cancel-then-rearm also
# makes a re-run of this script push the deadline BACK (rather than erroring or stacking),
# which is the desired semantics for a legitimate re-setup (e.g. extending a long-running
# sweep with a fresh DEADMAN_MINUTES). `|| true` because there being no pending shutdown to
# cancel (the common case, e.g. this script's first run) is not an error.
sudo shutdown -c 2>/dev/null || true
sudo shutdown -P "+${DEADMAN_MINUTES}"

# ---------------------------------------------------------------------------
# Step 2: verify nvidia-smi works and all 8 GPUs are visible. Fail loudly
# (rather than proceeding to build against a broken/partial driver) since
# every downstream profiling run depends on this.
# ---------------------------------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found on PATH. Is the NVIDIA driver installed?" >&2
    exit 1
fi

if ! nvidia-smi >/dev/null; then
    echo "ERROR: nvidia-smi is present but failed to run. Driver/GPU problem?" >&2
    exit 1
fi

gpu_count="$(nvidia-smi -L | wc -l)"
if [[ "$gpu_count" -ne 8 ]]; then
    echo "ERROR: expected 8 GPUs (p5.48xlarge), found $gpu_count. Aborting." >&2
    exit 1
fi
echo "OK: nvidia-smi reports $gpu_count GPUs."

# ---------------------------------------------------------------------------
# Step 3: build nccl-tests if it isn't already built.
#
# Idempotent: only clones if NCCL_TESTS_DIR doesn't exist yet, and only
# (re)builds if the all_reduce_perf binary is missing -- a rerun of this
# script after a successful first run is a fast no-op here.
# ---------------------------------------------------------------------------
if [[ -x "$NCCL_TESTS_DIR/build/all_reduce_perf" ]]; then
    echo "OK: nccl-tests already built at $NCCL_TESTS_DIR."
else
    if [[ ! -d "$NCCL_TESTS_DIR" ]]; then
        echo "Cloning nccl-tests into $NCCL_TESTS_DIR ..."
        git clone https://github.com/NVIDIA/nccl-tests "$NCCL_TESTS_DIR"
    fi
    echo "Building nccl-tests (MPI=0, CUDA_HOME=$CUDA_HOME) ..."
    make -j -C "$NCCL_TESTS_DIR" MPI=0 CUDA_HOME="$CUDA_HOME"
fi

# ---------------------------------------------------------------------------
# Step 4: build torus_bench if it isn't already built.
#
# TORUS_BENCH_DIR is scp'd onto the instance by the orchestrator (a sibling
# work package owns the torus_bench source); if it hasn't landed yet, warn
# and continue rather than failing -- the FC leg (nccl-tests) can still run
# without it, and setup_node.sh may legitimately run before the orchestrator
# has finished copying torus_bench over.
# ---------------------------------------------------------------------------
torus_bench_bin="$TORUS_BENCH_DIR/torus_bench"
if [[ -x "$torus_bench_bin" ]]; then
    echo "OK: torus_bench already built at $torus_bench_bin."
elif [[ -d "$TORUS_BENCH_DIR" ]]; then
    echo "Building torus_bench ..."
    make -C "$TORUS_BENCH_DIR" torus_bench
else
    echo "WARNING: $TORUS_BENCH_DIR not found (expected to be scp'd there by the orchestrator)." >&2
    echo "         Skipping torus_bench build; the FC leg can still run without it." >&2
fi

# ---------------------------------------------------------------------------
# Step 5: print a versions summary for the run's metadata/provenance.
# Every lookup here is best-effort (guarded so a missing tool doesn't abort
# the script under `set -e`), since this is diagnostic output, not a hard
# requirement.
# ---------------------------------------------------------------------------
echo "=== Versions ==="

if command -v nvcc >/dev/null 2>&1; then
    nvcc --version | tail -n 1
else
    echo "nvcc: not found on PATH"
fi

nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | sed 's/^/driver_version: /'

# NCCL version discovery: prefer asking a Python-visible torch build (most
# accurate for the environment that will actually run collectives), and
# fall back to scanning the linker cache for libnccl if torch isn't
# importable. Both are best-effort.
if python3 -c "import torch; print(torch.cuda.nccl.version())" 2>/dev/null; then
    :
elif ldconfig -p | grep -qi libnccl; then
    echo "libnccl found via ldconfig:"
    ldconfig -p | grep -i libnccl
else
    echo "NCCL version: could not be determined (no torch, no libnccl in ldconfig cache)"
fi

echo "setup_node.sh complete."
