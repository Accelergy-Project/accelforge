// torus_bench.cu
//
// Single-process multi-GPU NCCL benchmark that runs collective algorithms restricted to the
// EDGES OF A LOGICAL TORUS, on hardware that is physically fully-connected (one AWS
// p5.48xlarge, 8x H100 over NVSwitch).
//
// SCIENTIFIC PURPOSE (drives the whole design)
// ---------------------------------------------
// This benchmark exists to correlate an analytical torus-network model against real
// measurements. The measurement is only valid if EVERY inter-GPU transfer travels between
// logical torus NEIGHBORS -- that constraint IS the experiment. A single misrouted transfer
// (e.g. a "shortcut" the NVSwitch fabric would happily allow but the torus topology would not)
// silently invalidates the correlation. We therefore do not trust ourselves to hand-write
// per-collective CUDA/NCCL call sequences and eyeball their correctness; instead:
//
//   SCHEDULE-AS-DATA: pure host code (no CUDA, no GPU) builds an explicit, fully materialized
//   step-by-step transfer schedule (`Schedule` = vector<Step>, each Step a set of concurrent
//   `Xfer`s followed by local reduce/copy `LocalOp`s). A single choke point -- the edge
//   assertion inside build_schedule() -- inspects every Xfer the schedule will ever contain and
//   aborts the program if any of them is not a torus-neighbor transfer. This assertion is the
//   scientific guarantee of this file and must never be disabled, in either build.
//
//   TWO EXECUTORS, ONE SCHEDULE: the identical Schedule produced by build_schedule() is handed
//   to either of two interchangeable executors with matching function signatures:
//     - a host-memory SIMULATOR (compiled here, in this environment, with no GPU/nvcc
//       available -- this is the local test vehicle and is exercised by the acceptance
//       criteria), and
//     - an NCCL/CUDA executor (built on the target GPU instance; cannot be compiled or run in
//       this environment since no nvcc/CUDA toolkit is installed here -- see the project report
//       for confirmation of this constraint).
//   Both executors are driven by the exact same schedule-building and CLI code; only the
//   "how do I actually move these bytes" implementation differs, selected at compile time via
//   the TORUS_SIM macro. Compiling with `g++ -x c++ -DTORUS_SIM` yields the simulator binary;
//   compiling with `nvcc` (TORUS_SIM undefined) yields the GPU binary. All CUDA/NCCL-specific
//   code is fenced with `#ifndef TORUS_SIM` so a plain C++ compiler never sees CUDA syntax.
//
// TOPOLOGY CONVENTION
// --------------------
// dims = [d_0, ..., d_{K-1}] is a K-dimensional torus with product(dims) = N ranks. Rank <->
// coordinate mapping is row-major with the LAST dimension fastest-varying (i.e. like a C array
// of shape `dims`). neighbor(r, dim, +-1) wraps around (mod dims[dim]). For an extent-2
// dimension, +1 and -1 land on the SAME neighbor -- the code below computes this generically via
// modular arithmetic and never special-cases extent-2 dimensions.
//
// DOCUMENTATION CONVENTIONS USED IN THIS FILE
// ---------------------------------------------
// Each function has a comment block with: a one-line summary, Parameters, Returns, and (where
// relevant) Preconditions/Notes -- the C++ analogue of NumPy-style docstrings. Each per-collective
// schedule-builder additionally documents, phase by phase, the data-placement INVARIANT that
// phase establishes; this is the load-bearing correctness argument for that collective and is
// exactly what a reader needs to convince themselves the algorithm is right.
//
// DERIVATION NOTE ON reduce_scatter (see build_reduce_scatter() below for full detail): the
// planning spec for this work package proposed a tentative per-step slot formula for
// reduce_scatter and then explicitly flagged uncertainty about it ("hold on, mirror the
// all_gather invariant exactly"). That tentative formula, worked through by hand and confirmed
// with a throwaway Python simulation across dims=[8],[2,4],[4,2],[2,2,2], places each fully
// reduced slot one hop short of its destination rank. The corrected formula and direction
// (documented at build_reduce_scatter) were derived from a time-reversal argument against the
// (spec-verified-correct) all_gather formula and independently confirmed by brute-force
// simulation before being encoded here; the C++ simulator's --check flag re-verifies this at
// runtime for every size in the sweep.

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

// =====================================================================================
// Buffer identifiers and collective enum
// =====================================================================================

// Logical per-rank buffer roles used by every schedule builder and both executors. Both
// executors allocate storage per (rank, BufId) according to buffer_sizes() below.
enum BufId : int {
    BUF_SEND = 0,
    BUF_RECV = 1,
    BUF_TMP = 2,
    BUF_WORK_A = 3,
    BUF_WORK_B = 4,
    NUM_BUFS = 5
};

enum class Collective { ALL_REDUCE, ALL_GATHER, REDUCE_SCATTER, ALLTOALL, BROADCAST, SENDRECV };

// Returns the canonical CLI/output-format name for a collective.
//
// Parameters
// ----------
// c : Collective
//
// Returns
// -------
// const char* -- a string literal, e.g. "all_reduce". Never null.
const char* collective_name(Collective c) {
    switch (c) {
        case Collective::ALL_REDUCE: return "all_reduce";
        case Collective::ALL_GATHER: return "all_gather";
        case Collective::REDUCE_SCATTER: return "reduce_scatter";
        case Collective::ALLTOALL: return "alltoall";
        case Collective::BROADCAST: return "broadcast";
        case Collective::SENDRECV: return "sendrecv";
    }
    return "?";
}

// Parses a --collective CLI argument.
//
// Parameters
// ----------
// s : const std::string& -- one of the six recognized collective names.
// out : Collective& -- set on success; left unmodified on failure.
//
// Returns
// -------
// bool -- true if `s` was recognized.
bool parse_collective(const std::string& s, Collective& out) {
    if (s == "all_reduce") { out = Collective::ALL_REDUCE; return true; }
    if (s == "all_gather") { out = Collective::ALL_GATHER; return true; }
    if (s == "reduce_scatter") { out = Collective::REDUCE_SCATTER; return true; }
    if (s == "alltoall") { out = Collective::ALLTOALL; return true; }
    if (s == "broadcast") { out = Collective::BROADCAST; return true; }
    if (s == "sendrecv") { out = Collective::SENDRECV; return true; }
    return false;
}

// Returns a human-readable name for a BufId, used only in --check diagnostic output.
const char* buf_name(int b) {
    switch (b) {
        case BUF_SEND: return "BUF_SEND";
        case BUF_RECV: return "BUF_RECV";
        case BUF_TMP: return "BUF_TMP";
        case BUF_WORK_A: return "BUF_WORK_A";
        case BUF_WORK_B: return "BUF_WORK_B";
    }
    return "?";
}

// =====================================================================================
// Schedule IR (pure data -- no CUDA dependency anywhere in this section)
// =====================================================================================

// One point-to-point transfer between two DIFFERENT ranks. Every Xfer that will ever be
// constructed by any builder in this file must satisfy is_torus_neighbor(src, dst) -- see
// check_edge_or_abort() and the final validation pass in build_schedule(). src == dst is
// forbidden by construction: same-rank data movement must be expressed as a LocalOp instead.
struct Xfer {
    int src, dst;
    int src_buf, dst_buf;
    size_t src_off, dst_off, bytes;
};

// A same-rank operation: either a byte-for-byte copy (add == false) or an elementwise
// float accumulate dst[i] += src[i] (add == true), applied over `bytes` bytes (i.e.
// bytes/sizeof(float) floats -- bytes must be a multiple of 4 whenever add == true).
struct LocalOp {
    int rank;
    int src_buf;
    size_t src_off;
    int dst_buf;
    size_t dst_off;
    size_t bytes;
    bool add;
};

// One synchronization step of a Schedule: all `xfers` are considered to execute concurrently
// (reading pre-step state), and only once every Xfer in the step has landed do the `post`
// LocalOps run (also concurrently with each other, since by construction no two post ops of
// the same step touch overlapping (rank,buf,offset) ranges).
struct Step {
    std::vector<Xfer> xfers;
    std::vector<LocalOp> post;
};

using Schedule = std::vector<Step>;

// =====================================================================================
// Torus topology helpers
// =====================================================================================

// Returns N = product(dims), the total rank count of the torus.
inline int num_ranks(const std::vector<int>& dims) {
    int n = 1;
    for (int d : dims) n *= d;
    return n;
}

// Converts a rank id to its torus coordinates.
//
// Parameters
// ----------
// r : int -- rank id, 0 <= r < product(dims).
// dims : const std::vector<int>& -- torus extents, dims[K-1] is the fastest-varying axis.
//
// Returns
// -------
// std::vector<int> -- coordinates, one per dimension, coords[j] in [0, dims[j]).
inline std::vector<int> coords_of(int r, const std::vector<int>& dims) {
    std::vector<int> c(dims.size());
    for (int j = (int)dims.size() - 1; j >= 0; --j) {
        c[j] = r % dims[j];
        r /= dims[j];
    }
    return c;
}

// Inverse of coords_of(): converts torus coordinates back to a rank id (row-major, last
// dimension fastest).
inline int rank_of(const std::vector<int>& c, const std::vector<int>& dims) {
    int r = 0;
    for (size_t j = 0; j < dims.size(); ++j) r = r * dims[j] + c[j];
    return r;
}

// Returns the rank reached from `r` by moving one hop of `delta` (+1 or -1) along dimension
// `dim`, wrapping around (mod dims[dim]). For an extent-2 dimension, delta=+1 and delta=-1
// necessarily return the same rank -- this falls out of the modular arithmetic below with no
// special-casing, matching the spec's requirement that extent-2 degeneracy not be hard-coded.
inline int neighbor(int r, int dim, int delta, const std::vector<int>& dims) {
    std::vector<int> c = coords_of(r, dims);
    int e = dims[dim];
    c[dim] = ((c[dim] + delta) % e + e) % e;
    return rank_of(c, dims);
}

// Determines whether two ranks are torus neighbors: their coordinates must differ in exactly
// one dimension, and in that dimension by +-1 modulo the dimension's extent.
//
// Notes
// -----
// For an extent-2 dimension this is trivially satisfied by any pair that differs there (both
// possible non-zero differences, 1 and (extent-1)=1, coincide), which is the intended behavior.
inline bool is_torus_neighbor(int a, int b, const std::vector<int>& dims) {
    std::vector<int> ca = coords_of(a, dims), cb = coords_of(b, dims);
    int diff_dim = -1, diff_count = 0;
    for (size_t j = 0; j < dims.size(); ++j) {
        if (ca[j] != cb[j]) { diff_dim = (int)j; ++diff_count; }
    }
    if (diff_count != 1) return false;
    int e = dims[diff_dim];
    int d = ((ca[diff_dim] - cb[diff_dim]) % e + e) % e;
    return d == 1 || d == e - 1;
}

// Aborts the program with a file:line diagnostic if (src,dst) is not a torus edge, or if
// src == dst. This is THE scientific guarantee of this benchmark (see file header) and must
// remain active in both builds.
//
// Design decision: we use an explicit check + std::abort() rather than assert() from
// <cassert>, because assert() compiles to a no-op under -DNDEBUG and we do not control every
// build environment this file might eventually be compiled in (e.g. a release-mode CI flag).
// An explicit check is unconditionally active regardless of optimization/NDEBUG flags.
inline void check_edge_or_abort(int src, int dst, const std::vector<int>& dims) {
    if (src == dst) {
        std::fprintf(stderr,
                      "%s:%d: EDGE ASSERTION FAILED: Xfer has src==dst (rank %d); same-rank "
                      "movement must be expressed as a LocalOp, not an Xfer\n",
                      __FILE__, __LINE__, src);
        std::abort();
    }
    if (!is_torus_neighbor(src, dst, dims)) {
        std::fprintf(stderr,
                      "%s:%d: EDGE ASSERTION FAILED: rank %d -> rank %d is not a torus "
                      "neighbor for the given dims; this transfer would not exist on the "
                      "logical torus and must not be scheduled\n",
                      __FILE__, __LINE__, src, dst);
        std::abort();
    }
}

// Enumerates every combination of coordinate values across dimensions [0, d), invoking `cb`
// once per combination with a coordinate vector `v` whose entries at indices >= d are left
// exactly as passed in (typically already pinned to a specific rank's own coordinates).
//
// Parameters
// ----------
// d : int -- number of leading dimensions (0..d-1) to enumerate freely; if d == 0, `cb` is
//     invoked exactly once, with `v` unchanged (the "no free dimensions" case).
// dims : const std::vector<int>& -- torus extents.
// v : std::vector<int> -- base coordinate vector (taken by value since we mutate indices < d
//     during enumeration; entries at indices >= d are the caller's fixed values).
// cb : const std::function<void(const std::vector<int>&)>& -- invoked once per combination.
//
// Notes
// -----
// Enumeration order (dimension 0 varies slowest, in this implementation) is an arbitrary
// choice: every caller in this file treats each combination as an independent, order-agnostic
// unit of work (one Xfer/LocalOp pair per combination), so the traversal order used to reach
// the same combination SET has no effect on correctness.
inline void for_each_free_combo(int d, const std::vector<int>& dims, std::vector<int> v,
                                 const std::function<void(const std::vector<int>&)>& cb) {
    std::function<void(int)> rec = [&](int j) {
        if (j == d) { cb(v); return; }
        for (int val = 0; val < dims[j]; ++val) {
            v[j] = val;
            rec(j + 1);
        }
    };
    rec(0);
}

// =====================================================================================
// Per-(collective,S) buffer sizing
// =====================================================================================

// Computes the per-rank byte size of each of the five logical buffers a given collective needs
// for total-collective-size S. Both executors call this identically to allocate storage (heap
// arrays for the simulator, cudaMalloc for the NCCL build) -- see file header for buffer roles.
//
// Parameters
// ----------
// c : Collective
// dims : const std::vector<int>& -- torus extents; N = product(dims).
// S : size_t -- total collective size in bytes for this sweep point.
//
// Returns
// -------
// std::array<size_t, NUM_BUFS> -- indexed by BufId; unused buffers are size 0.
//
// Preconditions
// -------------
// S must be divisible by N and by 4*N*N (whole-float shard/chunk boundaries); callers are
// expected to have already applied the sweep-level divisibility skip check (see main()) before
// calling this.
inline std::array<size_t, NUM_BUFS> buffer_sizes(Collective c, const std::vector<int>& dims,
                                                  size_t S) {
    int N = num_ranks(dims);
    size_t m = S / (size_t)N;  // per-rank shard, m = S/N, per the CLI spec's convention.
    int K = (int)dims.size();
    std::array<size_t, NUM_BUFS> sz{};
    sz.fill(0);
    switch (c) {
        case Collective::SENDRECV:
            sz[BUF_SEND] = m;
            sz[BUF_RECV] = (size_t)K * m;  // one m-byte region per dimension.
            break;
        case Collective::BROADCAST:
            sz[BUF_SEND] = m;  // only meaningful on root; allocated uniformly for simplicity.
            sz[BUF_RECV] = m;
            break;
        case Collective::ALL_GATHER:
            sz[BUF_SEND] = m;
            sz[BUF_RECV] = (size_t)N * m;  // N slots of m bytes each.
            break;
        case Collective::REDUCE_SCATTER:
            sz[BUF_SEND] = (size_t)N * m;  // N slots of m bytes each (destined for each rank).
            sz[BUF_RECV] = m;
            sz[BUF_WORK_A] = (size_t)N * m;  // running accumulator, one slot per destination.
            sz[BUF_TMP] = (size_t)N * m;     // staging area for incoming adds.
            break;
        case Collective::ALL_REDUCE:
            // Composed internally of reduce_scatter(shard=m/N) followed by all_gather
            // (shard=m/N); see build_all_reduce() for the full derivation. Both sub-phases'
            // internal buffer needs (N*(m/N) == m) collapse to a uniform m bytes here.
            sz[BUF_SEND] = m;
            sz[BUF_RECV] = m;
            sz[BUF_WORK_A] = m;
            sz[BUF_TMP] = m;
            break;
        case Collective::ALLTOALL:
            sz[BUF_SEND] = m;  // N chunks of c=m/N bytes each.
            sz[BUF_RECV] = m;
            sz[BUF_WORK_A] = 2 * m;  // ping-pong in-flight staging; sized generously (2x) per
            sz[BUF_WORK_B] = 2 * m;  // spec, occupancy is asserted at schedule-build time.
            break;
    }
    return sz;
}

// =====================================================================================
// Schedule builders (pure host code -- no CUDA dependency anywhere in this section)
// =====================================================================================

// Builds the schedule for `sendrecv`: one step per torus dimension, every rank exchanges its
// full BUF_SEND vector with its +1 neighbor in that dimension.
//
// Invariant per step d: after step d, every rank's BUF_RECV region [d*m, (d+1)*m) holds the
// data that neighbor(r, d, -1) sent -- i.e. its BUF_SEND contents -- since neighbor(r,d,-1)'s
// own send in this same step targets exactly rank r (send direction is always +1, and
// neighbor(neighbor(r,d,-1), d, +1) == r by construction of neighbor()).
//
// Parameters
// ----------
// dims : const std::vector<int>&
// S : size_t -- total collective bytes; m = S/N is exchanged per dimension.
//
// Returns
// -------
// Schedule -- K steps, N Xfers each, no LocalOps.
inline Schedule build_sendrecv(const std::vector<int>& dims, size_t S) {
    int N = num_ranks(dims);
    int K = (int)dims.size();
    size_t m = S / (size_t)N;
    Schedule sched;
    for (int d = 0; d < K; ++d) {
        Step step;
        for (int r = 0; r < N; ++r) {
            int dst = neighbor(r, d, +1, dims);
            check_edge_or_abort(r, dst, dims);
            step.xfers.push_back({r, dst, BUF_SEND, BUF_RECV, 0, (size_t)d * m, m});
        }
        sched.push_back(std::move(step));
    }
    return sched;
}

// Builds the schedule for `broadcast` from root rank 0.
//
// Algorithm: dimension-by-dimension forward-chain propagation. A host-side has_data[N] tracks,
// at schedule-BUILD time (not at run time), which ranks are already known to hold the
// broadcast data after each step; this bookkeeping exists purely to decide which Xfers to
// generate and is not part of the executed Schedule itself.
//
// Invariant: at the start of dimension d's phase, the set of ranks with has_data[r] == true is
// exactly the set of ranks that agree with rank 0 on every coordinate j >= d (this holds for
// d == 0 trivially: only rank 0 itself). Each of dimension d's (extent_d - 1) repeats extends
// every already-seeded "line" one hop further in the +1 direction; after all extent_d - 1
// repeats, every rank agreeing with rank 0 on coordinates j > d has data (regardless of its
// coordinate d), establishing the invariant for phase d+1. After all K dimensions, every rank
// has data.
//
// Design decision: root's own copy is written by a dedicated step-0 LocalOp (BUF_SEND ->
// BUF_RECV) with NO concurrent Xfers in that same step. This lets every subsequent send (even
// root's very first "real" send) source uniformly from BUF_RECV: had root's first send shared
// step 0 with the LocalOp, it would need to special-case sourcing from BUF_SEND instead, since
// a Step's post-LocalOps run strictly after that step's Xfers land.
//
// Parameters
// ----------
// dims : const std::vector<int>&
// S : size_t -- total collective bytes; m = S/N is the broadcast vector size.
//
// Returns
// -------
// Schedule -- 1 (LocalOp-only) + sum_d(extent_d - 1) steps.
inline Schedule build_broadcast(const std::vector<int>& dims, size_t S) {
    int N = num_ranks(dims);
    int K = (int)dims.size();
    size_t m = S / (size_t)N;
    Schedule sched;
    {
        Step step0;
        step0.post.push_back({0, BUF_SEND, 0, BUF_RECV, 0, m, false});
        sched.push_back(std::move(step0));
    }
    std::vector<char> has_data(N, 0);
    has_data[0] = 1;
    for (int d = 0; d < K; ++d) {
        int E = dims[d];
        for (int s = 0; s < E - 1; ++s) {
            Step step;
            std::vector<char> new_has = has_data;  // conditions evaluated against pre-step state.
            for (int r = 0; r < N; ++r) {
                if (!has_data[r]) continue;
                int dst = neighbor(r, d, +1, dims);
                if (has_data[dst]) continue;
                check_edge_or_abort(r, dst, dims);
                step.xfers.push_back({r, dst, BUF_RECV, BUF_RECV, 0, 0, m});
                new_has[dst] = 1;
            }
            has_data = new_has;
            sched.push_back(std::move(step));
        }
    }
    return sched;
}

// Builds the schedule for `all_gather`.
//
// Algorithm: standard ring all-gather, generalized to a mixed-radix torus by processing one
// dimension at a time (ascending order). Step 0 seeds each rank's own slot via a LocalOp.
//
// Invariant: at the START of dimension d's phase, rank r holds exactly the slots
// {u : u_j == r_j for all j >= d} (dimensions below d are already fully "free" -- rank r holds
// every value there -- while dimensions >= d are still pinned to r's own coordinate). This
// holds trivially at d == 0 (only slot r itself, u_j == r_j for ALL j) and, vacuously, means
// every rank holds every slot once d reaches K (all dimensions processed).
//
// Each phase d performs (extent_d - 1) ring-relay steps in the +1 direction: at step s, rank r
// forwards the slot batch it received on the PREVIOUS step (or, at s=1, the batch it started
// the phase with) to its +1 neighbor. This is the classic "forward only what you just received"
// ring relay, which avoids redundant retransmission and completes dimension d's expansion in
// exactly extent_d - 1 hops.
//
// Parameters
// ----------
// dims : const std::vector<int>&
// S : size_t -- total collective bytes; m = S/N is each rank's own shard size.
//
// Returns
// -------
// Schedule -- 1 + sum_d(extent_d - 1) steps.
inline Schedule build_all_gather(const std::vector<int>& dims, size_t S) {
    int N = num_ranks(dims);
    int K = (int)dims.size();
    size_t m = S / (size_t)N;
    Schedule sched;
    {
        Step step0;
        for (int r = 0; r < N; ++r)
            step0.post.push_back({r, BUF_SEND, 0, BUF_RECV, (size_t)r * m, m, false});
        sched.push_back(std::move(step0));
    }
    for (int d = 0; d < K; ++d) {
        int E = dims[d];
        for (int s = 1; s <= E - 1; ++s) {
            Step step;
            for (int r = 0; r < N; ++r) {
                std::vector<int> base = coords_of(r, dims);
                int dst = neighbor(r, d, +1, dims);
                check_edge_or_abort(r, dst, dims);
                base[d] = ((base[d] - (s - 1)) % E + E) % E;
                for_each_free_combo(d, dims, base, [&](const std::vector<int>& v) {
                    int vslot = rank_of(v, dims);
                    size_t off = (size_t)vslot * m;
                    step.xfers.push_back({r, dst, BUF_RECV, BUF_RECV, off, off, m});
                });
            }
            sched.push_back(std::move(step));
        }
    }
    return sched;
}

// Builds the schedule for `reduce_scatter`.
//
// DERIVATION NOTE (see also file header): the planning spec's own tentative formula for this
// collective was flagged mid-sentence as unreliable ("hold on, mirror the all_gather invariant
// exactly"). This implementation instead derives the per-step slot set and DIRECTION from a
// time-reversal argument against the (independently correct, spec-supplied) all_gather formula,
// and that derivation was confirmed by an exhaustive brute-force simulation (dims=[8],[2,4],
// [4,2],[2,2,2]) before being encoded here. Summary of the correction: the naive "mirror
// all_gather with the same (r_d - (s-1)) mod E slot formula and +1 direction" places each fully
// reduced slot ONE HOP SHORT of its destination rank (rank k's slot ends up fully summed at
// rank k-1, not rank k). The fix is both a different slot formula AND a different direction:
// sends go in the -1 direction, and at step t the sender's slot index is (r_d + t) mod E, not
// (r_d - (s-1)) mod E.
//
// Algorithm: work happens in BUF_WORK_A (an N-slot accumulator seeded from BUF_SEND at step 0).
// Dimensions are processed in DESCENDING order (K-1 down to 0), matching the spec.
//
// Invariant: at the END of dimension d's phase, rank r holds partial sums ONLY for slots
// {v : v_j == r_j for all j >= d} (dimensions below d are not yet reduced -- rank r's held
// slots still range over every value there -- while dimensions >= d have been fully reduced
// down to r's own coordinate). This is exactly the all_gather invariant with the phase-transition
// direction reversed: all_gather EXPANDS what a rank holds as d increases from 0 to K;
// reduce_scatter CONTRACTS what a rank holds as d decreases from K-1 to 0, so by symmetry it is
// stated as an END-of-phase (rather than start-of-phase) condition.
//
// Correctness of the per-step formula for a single dimension's ring (extent E, direction -1,
// slot index v_d = (r_d + t) mod E at step t = 1..E-1): consider dimension d as an isolated ring
// (the free dimensions j<d and pinned dimensions j>d ride along unchanged, in parallel, for
// every value). For a fixed target slot-index k (a value of v_d), the ring of E nodes must sum
// together each node's local contribution for chunk k, ending up entirely at node k. At reduce
// step t, the node currently entrusted with chunk k's running partial sum is node (k - t) mod E
// (t=1: node k-1 sends its OWN local value for chunk k to node k-2, which adds it in; t=2: node
// k-2, now holding a 2-term partial sum, forwards it to node k-3; ...; t=E-1: node (k -
// (E-1)) mod E == (k+1) mod E, holding an (E-1)-term partial sum -- every node except k itself
// -- forwards it to node k, which adds in its own remaining term to complete the sum of all E
// contributions). Restating "node (k - t) mod E sends chunk k to node (k - t - 1) mod E" from
// the SENDER's own coordinate r = (k - t) mod E gives: sender r sends chunk k = (r + t) mod E to
// receiver (r - 1) mod E == neighbor(r, d, -1). This is exactly the formula used below.
//
// Parameters
// ----------
// dims : const std::vector<int>&
// S : size_t -- total collective bytes; m = S/N, BUF_SEND holds N slots of m bytes each.
//
// Returns
// -------
// Schedule -- 1 (seed) + sum_d(extent_d - 1) (reduce) + 1 (final copy) steps.
inline Schedule build_reduce_scatter(const std::vector<int>& dims, size_t S) {
    int N = num_ranks(dims);
    int K = (int)dims.size();
    size_t m = S / (size_t)N;
    Schedule sched;
    {
        Step step0;
        for (int r = 0; r < N; ++r)
            step0.post.push_back({r, BUF_SEND, 0, BUF_WORK_A, 0, (size_t)N * m, false});
        sched.push_back(std::move(step0));
    }
    for (int d = K - 1; d >= 0; --d) {
        int E = dims[d];
        for (int t = 1; t <= E - 1; ++t) {
            Step step;
            for (int r = 0; r < N; ++r) {
                std::vector<int> cr = coords_of(r, dims);
                // Direction is -1 (see derivation above); this is the corrected direction,
                // NOT the +1 direction all_gather uses.
                int dst = neighbor(r, d, -1, dims);
                check_edge_or_abort(r, dst, dims);
                std::vector<int> base = cr;
                base[d] = ((cr[d] + t) % E + E) % E;
                for_each_free_combo(d, dims, base, [&](const std::vector<int>& v) {
                    int vslot = rank_of(v, dims);
                    size_t off = (size_t)vslot * m;
                    // Stage into the receiver's BUF_TMP at the same v*m offset (distinct
                    // offsets across combos within this step fall out automatically since each
                    // combo yields a distinct vslot), then a post LocalOp adds it into the
                    // receiver's running accumulator at that same slot.
                    step.xfers.push_back({r, dst, BUF_WORK_A, BUF_TMP, off, off, m});
                    step.post.push_back({dst, BUF_TMP, off, BUF_WORK_A, off, m, true});
                });
            }
            sched.push_back(std::move(step));
        }
    }
    {
        Step stepf;
        for (int r = 0; r < N; ++r)
            stepf.post.push_back({r, BUF_WORK_A, (size_t)r * m, BUF_RECV, 0, m, false});
        sched.push_back(std::move(stepf));
    }
    return sched;
}

// Builds the schedule for `all_reduce` by composing the (unmodified) reduce_scatter and
// all_gather builders over sub-shards of size m/N, per the spec.
//
// Design: build_reduce_scatter(dims, m) is called with reduce_scatter's OWN "S" parameter set
// to all_reduce's per-rank vector size m (not to S itself). Internally, reduce_scatter then
// computes its own m_rs = m/N == the desired sub-shard size, and its own N*m_rs == m exactly
// matches the size of all_reduce's per-rank input buffer -- so all_reduce's existing BUF_SEND
// content can be fed to reduce_scatter completely unmodified, with no data rearrangement,
// because a flat m-byte vector split into N contiguous m/N-byte pieces is precisely
// reduce_scatter's own "N contiguous slots" input convention. Symmetrically,
// build_all_gather(dims, m) produces an N*(m/N) == m byte output, exactly all_reduce's needed
// result size.
//
// A single bridging LocalOp copies reduce_scatter's own (small, m/N-byte) output out of its
// BUF_RECV into BUF_SEND, where the reused all_gather schedule's own step 0 expects to find its
// input. This reuse is safe because reduce_scatter only ever READS BUF_SEND once, in its own
// step 0; by the time its schedule finishes, BUF_SEND is dead and free to reuse as scratch.
//
// Parameters
// ----------
// dims : const std::vector<int>&
// S : size_t -- total collective bytes; m = S/N is each rank's full input/output vector size.
//
// Returns
// -------
// Schedule -- reduce_scatter(dims,m)'s steps, then 1 bridging step, then all_gather(dims,m)'s
// steps.
inline Schedule build_all_reduce(const std::vector<int>& dims, size_t S) {
    int N = num_ranks(dims);
    size_t m = S / (size_t)N;
    size_t shard = m / (size_t)N;

    Schedule sched = build_reduce_scatter(dims, m);
    {
        Step bridge;
        for (int r = 0; r < N; ++r)
            bridge.post.push_back({r, BUF_RECV, 0, BUF_SEND, 0, shard, false});
        sched.push_back(std::move(bridge));
    }
    Schedule ag = build_all_gather(dims, m);
    for (auto& step : ag) sched.push_back(std::move(step));
    return sched;
}

// Builds the schedule for `alltoall` using dimension-ordered minimal routing.
//
// Each of the N*(N-1) non-self chunks (u,v), u != v, starts at rank u (BUF_SEND offset v*c,
// c = m/N) and must reach rank v (BUF_RECV offset u*c). Self chunks (u == v) never move; they
// are resolved by a single step-0 LocalOp per rank.
//
// Routing: dimensions are processed in ascending order. Within dimension d's phase, every chunk
// whose current holder disagrees with its target on coordinate d takes one hop per Step, in the
// direction (+1 or -1) that minimizes remaining distance around that dimension's ring; the
// phase repeats until no chunk needs to move in dimension d (bounded by extent_d - 1
// iterations, the maximum possible remaining distance, with an explicit abort if that bound is
// ever exceeded -- it should not be, since minimal-direction hops need at most
// floor(extent_d/2) <= extent_d - 1 of them).
//
// In-flight buffer management: a chunk not currently at BUF_SEND or BUF_RECV lives in one of
// two ping-pong work buffers, BUF_WORK_A/BUF_WORK_B, alternating buffers each time it takes a
// hop (this guarantees a chunk's read-from buffer this step always differs from its
// write-to buffer this step, so simple sequential-then-concurrent memory semantics --
// read-all-then-write-all within a Step -- can never alias a chunk's own old and new copies of
// itself).
//
// Slot allocation within a work buffer is a genuine per-step host-side bump allocator (a
// closed-form offset formula is not safe in general: two chunks that share a source rank and
// happen to have the same destination coordinate on every already-matched dimension travel
// together and can simultaneously occupy the same intermediate rank, so slots must be assigned,
// not computed). New allocations for step t are chosen to avoid every slot occupied at the
// START of step t (including slots about to be vacated this same step, since those are still
// being read from concurrently); old slots are freed only once the step's Xfers have been
// fully built, so freed capacity becomes available starting with the NEXT step, never the
// current one. Total occupancy per (rank, buffer) is asserted <= the buffer's slot capacity
// (2*N slots of c bytes each, matching the 2*m-byte buffer size from buffer_sizes()); the spec
// notes this should stay near N by symmetry for uniform all-to-all, which is also confirmed by
// a max-occupancy check in the throwaway Python simulation used to validate this algorithm
// before writing it here (observed max was well under half the budget for all tested dims).
//
// A chunk that arrives at its destination (all dimensions matched) is routed directly into
// BUF_RECV instead of a work buffer and is marked done, removing it from further consideration.
//
// Parameters
// ----------
// dims : const std::vector<int>&
// S : size_t -- total collective bytes; m = S/N per rank, c = m/N per chunk.
//
// Returns
// -------
// Schedule -- 1 (self-chunk) step, followed by one step per routing hop actually taken.
inline Schedule build_alltoall(const std::vector<int>& dims, size_t S) {
    int N = num_ranks(dims);
    int K = (int)dims.size();
    size_t m = S / (size_t)N;
    size_t c = m / (size_t)N;
    Schedule sched;
    {
        Step step0;
        for (int r = 0; r < N; ++r)
            step0.post.push_back({r, BUF_SEND, (size_t)r * c, BUF_RECV, (size_t)r * c, c, false});
        sched.push_back(std::move(step0));
    }

    // Host-side (schedule-build-time only) bookkeeping of each in-flight chunk's current
    // position. buf: -1 == still/again at BUF_SEND (never true after the first hop, but used
    // as the initial state so the first hop's ping-pong toggle lands on BUF_WORK_A); 0 ==
    // BUF_WORK_A; 1 == BUF_WORK_B. slot is meaningful only when buf >= 0.
    struct A2AChunk {
        int u, v, holder;
        int buf;
        int slot;
        bool done;
    };
    std::vector<A2AChunk> chunks;
    chunks.reserve((size_t)N * (N - 1));
    for (int u = 0; u < N; ++u)
        for (int v = 0; v < N; ++v)
            if (u != v) chunks.push_back({u, v, u, -1, -1, false});

    const int CAP_SLOTS = 2 * N;  // 2*m bytes / c bytes-per-slot, matching buffer_sizes().
    std::vector<std::array<std::vector<char>, 2>> occupied(N);
    for (int r = 0; r < N; ++r) {
        occupied[r][0].assign(CAP_SLOTS, 0);
        occupied[r][1].assign(CAP_SLOTS, 0);
    }
    auto alloc_slot = [&](int rank, int buf) -> int {
        for (int s = 0; s < CAP_SLOTS; ++s) {
            if (!occupied[rank][buf][s]) {
                occupied[rank][buf][s] = 1;
                return s;
            }
        }
        std::fprintf(stderr,
                      "%s:%d: alltoall work-buffer overflow at rank %d buf %d (occupancy "
                      "would exceed 2*m bytes)\n",
                      __FILE__, __LINE__, rank, buf);
        std::abort();
        return -1;  // unreachable
    };

    for (int d = 0; d < K; ++d) {
        int E = dims[d];
        int iter = 0;
        while (true) {
            std::vector<size_t> movers;
            for (size_t i = 0; i < chunks.size(); ++i) {
                if (chunks[i].done) continue;
                std::vector<int> cr = coords_of(chunks[i].holder, dims);
                std::vector<int> cv = coords_of(chunks[i].v, dims);
                if (cr[d] != cv[d]) movers.push_back(i);
            }
            if (movers.empty()) break;
            ++iter;
            if (iter > E - 1) {
                std::fprintf(stderr,
                              "%s:%d: alltoall routing failed to converge in dimension %d "
                              "within %d iterations\n",
                              __FILE__, __LINE__, d, E - 1);
                std::abort();
            }

            struct Move {
                size_t chunk_idx;
                int dst_rank;
                int new_buf, new_slot;
                bool arrives;
            };
            std::vector<Move> moves;
            moves.reserve(movers.size());
            // Pass 1: decide direction/destination and allocate NEW slots against the
            // occupancy snapshot as of the start of this step (see design note above: old
            // slots are deliberately not freed until pass 3, so a slot being read from this
            // step is never handed out as someone else's new landing spot this same step).
            for (size_t i : movers) {
                A2AChunk& ch = chunks[i];
                std::vector<int> cr = coords_of(ch.holder, dims);
                std::vector<int> cv = coords_of(ch.v, dims);
                int distp = ((cv[d] - cr[d]) % E + E) % E;
                int distm = ((cr[d] - cv[d]) % E + E) % E;
                int dirn = (distp <= distm) ? +1 : -1;
                int dst = neighbor(ch.holder, d, dirn, dims);
                check_edge_or_abort(ch.holder, dst, dims);
                std::vector<int> cdst = coords_of(dst, dims);
                bool arrives = true;
                for (int j = 0; j < K; ++j)
                    if (cdst[j] != cv[j]) { arrives = false; break; }
                int new_buf = -1, new_slot = -1;
                if (!arrives) {
                    new_buf = (ch.buf == 0) ? 1 : 0;  // ping-pong: SEND or B -> A; A -> B.
                    new_slot = alloc_slot(dst, new_buf);
                }
                moves.push_back({i, dst, new_buf, new_slot, arrives});
            }
            // Pass 2: emit Xfers from each chunk's OLD position to its NEW position.
            Step step;
            for (const Move& mv : moves) {
                A2AChunk& ch = chunks[mv.chunk_idx];
                int src_buf = (ch.buf < 0) ? (int)BUF_SEND : (ch.buf == 0 ? (int)BUF_WORK_A : (int)BUF_WORK_B);
                size_t src_off = (ch.buf < 0) ? (size_t)ch.v * c : (size_t)ch.slot * c;
                int dst_buf;
                size_t dst_off;
                if (mv.arrives) {
                    dst_buf = BUF_RECV;
                    dst_off = (size_t)ch.u * c;
                } else {
                    dst_buf = (mv.new_buf == 0) ? (int)BUF_WORK_A : (int)BUF_WORK_B;
                    dst_off = (size_t)mv.new_slot * c;
                }
                step.xfers.push_back({ch.holder, mv.dst_rank, src_buf, dst_buf, src_off, dst_off, c});
            }
            // Pass 3: now that this step's Xfers are fully built, free vacated slots and
            // update chunk bookkeeping for the next iteration.
            for (const Move& mv : moves) {
                A2AChunk& ch = chunks[mv.chunk_idx];
                if (ch.buf >= 0) occupied[ch.holder][ch.buf][ch.slot] = 0;
                ch.holder = mv.dst_rank;
                if (mv.arrives) {
                    ch.done = true;
                    ch.buf = -1;
                    ch.slot = -1;
                } else {
                    ch.buf = mv.new_buf;
                    ch.slot = mv.new_slot;
                }
            }
            sched.push_back(std::move(step));
        }
    }

    for (const A2AChunk& ch : chunks) {
        if (!ch.done) {
            std::fprintf(stderr,
                          "%s:%d: alltoall chunk (u=%d,v=%d) failed to reach its destination\n",
                          __FILE__, __LINE__, ch.u, ch.v);
            std::abort();
        }
    }
    return sched;
}

// =====================================================================================
// Intra-step aliasing validation (shared, pure host code -- runs as part of build_schedule()'s
// final validation pass, so it applies identically to whichever executor -- simulator or
// NCCL/CUDA -- ends up running the assembled Schedule).
// =====================================================================================
//
// Design/WHY (see also run_schedule()'s Notes, in both the NCCL and simulator branches below,
// for the two executors' actual ordering guarantees this assertion is checked against): the
// host-memory simulator's run_schedule() gives every Step's Xfers STRONGER semantics than the
// NCCL/CUDA executor actually provides -- it snapshots every Xfer's source data before writing
// ANY Xfer's destination (see that function's own "Design decision" comment), so a schedule with
// overlapping src/dst ranges within one Step would silently produce a correct result there even
// though it would NOT on real hardware. The NCCL executor instead relies only on: (1) all of a
// Step's Xfers issued together inside one ncclGroupStart/End, with no ordering guarantee among
// DIFFERENT Xfers of that group beyond NCCL's own send/recv rendezvous (which pairs a specific
// send with its specific matching recv -- it says nothing about two unrelated Xfers of the same
// Step racing each other), and (2) per-rank CUDA stream order, which guarantees only that ops
// enqueued LATER on one rank's stream see the effects of ops enqueued EARLIER on that SAME
// stream -- in particular, a Step's post LocalOps (always enqueued, on every rank's stream,
// strictly after that Step's Xfers -- see run_schedule()) are guaranteed to see every effect of
// that Step's own Xfers, but no other cross-operation ordering is guaranteed by construction.
//
// This function makes that gap an assertion instead of a latent, hard-to-reproduce bug: every
// schedule this file ever builds is validated here to never depend on any same-step read/write
// ordering stronger than "a post-op reads a range a same-step Xfer just wrote" -- the one
// relationship the NCCL executor's fixed enqueue order (a Step's Xfers, then that Step's
// post-ops) always provides "for free", with no host synchronization required. Any OTHER
// same-step read/write overlap on the same (rank, buffer) -- e.g. one Xfer's source overlapping
// another Xfer's destination, or a post-op's destination overlapping another post-op's source --
// would only be safe under the simulator's stronger snapshot semantics, and must never occur.

// One (rank, buffer) byte range read or written by one Xfer or LocalOp within a single Step, used
// only by check_step_aliasing_or_abort() below.
struct _ByteRange {
    int rank;
    int buf;
    size_t begin, end;  // half-open [begin, end), in bytes.
    bool is_post;        // true if this range comes from a LocalOp (post); false if from an Xfer.
};

// Returns whether two half-open byte ranges overlap.
inline bool _ranges_overlap(size_t a_begin, size_t a_end, size_t b_begin, size_t b_end) {
    return a_begin < b_end && b_begin < a_end;
}

// Validates one Step against the aliasing rule described above; aborts with a diagnostic naming
// the step index and both offending ranges on the first violation found.
//
// Parameters
// ----------
// step_idx : size_t -- index of `step` within its Schedule (diagnostics only).
// step : const Step&
//
// Notes
// -----
// The single allowed exception -- a post-op's SOURCE range overlapping an Xfer's DESTINATION
// range, both of the same Step -- is exactly the reduce_scatter reduce-phase pattern (an Xfer
// lands a value in BUF_TMP, and that same Step's post-op immediately adds it out of BUF_TMP);
// see run_schedule()'s Notes for why that ordering (post-ops always after that Step's Xfers) is
// safe on both executors. Every other overlap combination aborts.
inline void check_step_aliasing_or_abort(size_t step_idx, const Step& step) {
    std::vector<_ByteRange> reads, writes;
    for (const Xfer& x : step.xfers) {
        reads.push_back({x.src, x.src_buf, x.src_off, x.src_off + x.bytes, false});
        writes.push_back({x.dst, x.dst_buf, x.dst_off, x.dst_off + x.bytes, false});
    }
    for (const LocalOp& op : step.post) {
        reads.push_back({op.rank, op.src_buf, op.src_off, op.src_off + op.bytes, true});
        writes.push_back({op.rank, op.dst_buf, op.dst_off, op.dst_off + op.bytes, true});
    }
    for (const _ByteRange& r : reads) {
        for (const _ByteRange& w : writes) {
            if (r.rank != w.rank || r.buf != w.buf) continue;
            if (!_ranges_overlap(r.begin, r.end, w.begin, w.end)) continue;
            // The one guaranteed-safe relationship: a post-op reading exactly what a same-step
            // Xfer just wrote (see this function's Notes above and run_schedule()'s Notes).
            if (r.is_post && !w.is_post) continue;
            std::fprintf(
                stderr,
                "%s:%d: INTRA-STEP ALIAS ASSERTION FAILED at step %zu: rank %d buf %s read "
                "range [%zu,%zu) (from %s) overlaps write range [%zu,%zu) (from %s); this "
                "schedule relies on same-step read/write ordering the NCCL executor does not "
                "guarantee (see check_step_aliasing_or_abort()'s Notes)\n",
                __FILE__, __LINE__, step_idx, r.rank, buf_name(r.buf), r.begin, r.end,
                r.is_post ? "a post-op" : "an Xfer", w.begin, w.end,
                w.is_post ? "a post-op" : "an Xfer");
            std::abort();
        }
    }
}

// Dispatches to the appropriate per-collective builder and then re-validates every Xfer in the
// assembled schedule against the torus-edge guarantee, and every Step against the intra-step
// aliasing rule above.
//
// Notes
// -----
// Every builder above already calls check_edge_or_abort() at each Xfer's construction site,
// which fails fastest and with the most local context. The bulk re-scan here is a deliberate
// belt-and-suspenders duplication: it makes build_schedule() itself -- not just its helpers --
// the literal authority for the scientific guarantee described in the file header, matching the
// spec's requirement that "build_schedule asserts is_torus_neighbor(...) for every Xfer". The
// aliasing pass (check_step_aliasing_or_abort()) has no earlier per-Xfer equivalent -- it is
// inherently a whole-Step check -- so build_schedule() is the only place it can run.
//
// Parameters
// ----------
// c : Collective
// dims : const std::vector<int>& -- torus extents.
// S : size_t -- total collective bytes for this sweep point.
//
// Returns
// -------
// Schedule -- fully built, edge-validated, and aliasing-validated.
inline Schedule build_schedule(Collective c, const std::vector<int>& dims, size_t S) {
    Schedule sched;
    switch (c) {
        case Collective::SENDRECV: sched = build_sendrecv(dims, S); break;
        case Collective::BROADCAST: sched = build_broadcast(dims, S); break;
        case Collective::ALL_GATHER: sched = build_all_gather(dims, S); break;
        case Collective::REDUCE_SCATTER: sched = build_reduce_scatter(dims, S); break;
        case Collective::ALL_REDUCE: sched = build_all_reduce(dims, S); break;
        case Collective::ALLTOALL: sched = build_alltoall(dims, S); break;
    }
    for (size_t i = 0; i < sched.size(); ++i) {
        for (const Xfer& x : sched[i].xfers) check_edge_or_abort(x.src, x.dst, dims);
        check_step_aliasing_or_abort(i, sched[i]);
    }
    return sched;
}

// =====================================================================================
// --check data pattern: shared, pure-host generation and verification logic
// =====================================================================================
//
// The --check input convention is uniform across all six collectives: every rank's BUF_SEND
// buffer (whatever its collective-specific size happens to be) is filled with f(rank, i) at
// flat float index i = 0 .. (buffer_bytes/4 - 1). This single convention subsumes every
// per-collective fill rule described in the spec (e.g. reduce_scatter's "rank u's BUF_SEND slot
// v holds f(u, v*elems_per_slot + i)" and alltoall's analogous per-chunk rule are both just this
// same flat-buffer fill, since a slot/chunk is by construction a contiguous sub-range of the
// flat buffer and v*elems_per_slot+i (or v*celems+i) is exactly that sub-range's flat index).
//
// Because f() is a pure, deterministic function of (source rank, index), verification never
// needs to read another rank's actual buffer contents: for every element of a rank's BUF_RECV,
// we know analytically which (source rank, source flat index) it is supposed to equal (or, for
// the reducing collectives, which SET of them it is supposed to sum) and simply recompute f()
// directly. This is implemented once, in verify_collective() below, and reused unmodified by
// both executors.

// Deterministic --check fill value.
//
// Parameters
// ----------
// r : int -- source rank.
// i : long long -- flat float index within that rank's buffer; must be >= 0 in practice (the
//     modulo-97 reduction below is defensive against negative input but this file never
//     constructs a negative index).
//
// Returns
// -------
// float -- r*100 + (i mod 97). Exact in fp32 (see verify_collective() notes on why summing up
// to N such values remains exactly representable for the N, dims this benchmark targets).
inline float f_pattern(int r, long long i) {
    long long im = i % 97;
    if (im < 0) im += 97;
    return (float)(r * 100 + (int)im);
}

// Generates the full --check fill pattern for one rank's BUF_SEND buffer.
//
// Parameters
// ----------
// rank : int
// nbytes : size_t -- BUF_SEND size for this rank (from buffer_sizes()); must be a multiple of
//     sizeof(float) (guaranteed by the S % (4*N*N) == 0 sweep-level precondition).
//
// Returns
// -------
// std::vector<float> -- nbytes/4 floats, out[i] == f_pattern(rank, i).
inline std::vector<float> gen_pattern(int rank, size_t nbytes) {
    size_t nf = nbytes / sizeof(float);
    std::vector<float> out(nf);
    for (size_t i = 0; i < nf; ++i) out[i] = f_pattern(rank, (long long)i);
    return out;
}

// Diagnostic record for the first mismatch found by verify_collective(), used to format the
// "# CHECK ... FAIL (first mismatch: ...)" line.
struct MismatchInfo {
    int rank = -1;
    int buf = BUF_RECV;
    long long idx = -1;
    float want = 0.f;
    float got = 0.f;
};

// Verifies one rank's BUF_RECV contents against the analytically-known-correct result for the
// given collective, per the check rules in the spec (all reducible, per the discussion above,
// to recomputing f_pattern() directly rather than reading other ranks' data).
//
// Parameters
// ----------
// c : Collective
// dims : const std::vector<int>&
// S : size_t -- total collective bytes for this sweep point.
// rank : int -- the rank whose BUF_RECV is being checked.
// recv : const std::vector<float>& -- that rank's full BUF_RECV contents, already copied to
//     host memory by the caller (trivial for the simulator; a cudaMemcpy D2H for the NCCL
//     build).
// mm : MismatchInfo& -- filled in on the first mismatch found; unmodified if this returns true.
//
// Returns
// -------
// bool -- true iff every checked element matched exactly.
//
// Notes
// -----
// Comparisons use exact float equality. This is intentional and safe here, not a bug: every
// f_pattern() value is a small non-negative integer (r*100 + i%97, comfortably under 2^24 for
// the rank counts and indices this benchmark exercises), and summing up to N such values (N <=
// a few hundred in any realistic torus) never leaves the range of exactly-representable
// integers in fp32 at any intermediate step -- so no rounding ever occurs, in the executor's
// float accumulation OR in this function's own reference summation, and exact comparison is
// mathematically justified rather than merely convenient.
inline bool verify_collective(Collective c, const std::vector<int>& dims, size_t S, int rank,
                               const std::vector<float>& recv, MismatchInfo& mm) {
    int N = num_ranks(dims);
    int K = (int)dims.size();
    size_t m = S / (size_t)N;

    auto check_eq = [&](size_t idx, float want, float got) -> bool {
        if (got != want) {
            mm.rank = rank;
            mm.buf = BUF_RECV;
            mm.idx = (long long)idx;
            mm.want = want;
            mm.got = got;
            return false;
        }
        return true;
    };

    switch (c) {
        case Collective::SENDRECV: {
            size_t nf = m / sizeof(float);
            for (int d = 0; d < K; ++d) {
                int src = neighbor(rank, d, -1, dims);
                for (size_t i = 0; i < nf; ++i) {
                    size_t idx = (size_t)d * nf + i;
                    if (!check_eq(idx, f_pattern(src, (long long)i), recv[idx])) return false;
                }
            }
            return true;
        }
        case Collective::BROADCAST: {
            size_t nf = m / sizeof(float);
            for (size_t i = 0; i < nf; ++i)
                if (!check_eq(i, f_pattern(0, (long long)i), recv[i])) return false;
            return true;
        }
        case Collective::ALL_GATHER: {
            size_t nf = m / sizeof(float);
            for (int u = 0; u < N; ++u) {
                for (size_t i = 0; i < nf; ++i) {
                    size_t idx = (size_t)u * nf + i;
                    if (!check_eq(idx, f_pattern(u, (long long)i), recv[idx])) return false;
                }
            }
            return true;
        }
        case Collective::REDUCE_SCATTER: {
            size_t elems_per_slot = m / sizeof(float);
            for (size_t i = 0; i < elems_per_slot; ++i) {
                float want = 0.f;
                for (int u = 0; u < N; ++u)
                    want += f_pattern(u, (long long)((size_t)rank * elems_per_slot + i));
                if (!check_eq(i, want, recv[i])) return false;
            }
            return true;
        }
        case Collective::ALL_REDUCE: {
            size_t nf = m / sizeof(float);
            for (size_t i = 0; i < nf; ++i) {
                float want = 0.f;
                for (int u = 0; u < N; ++u) want += f_pattern(u, (long long)i);
                if (!check_eq(i, want, recv[i])) return false;
            }
            return true;
        }
        case Collective::ALLTOALL: {
            size_t c_bytes = m / (size_t)N;
            size_t celems = c_bytes / sizeof(float);
            for (int u = 0; u < N; ++u) {
                for (size_t i = 0; i < celems; ++i) {
                    size_t idx = (size_t)u * celems + i;
                    float want = f_pattern(u, (long long)((size_t)rank * celems + i));
                    if (!check_eq(idx, want, recv[idx])) return false;
                }
            }
            return true;
        }
    }
    return true;
}

// Prints the single required "# CHECK ..." line for one sweep size, per the exact output
// format in the spec.
inline void print_check_line(Collective c, size_t S, bool pass, const MismatchInfo& mm) {
    if (pass) {
        std::printf("# CHECK collective=%s S=%zu: PASS\n", collective_name(c), S);
    } else {
        std::printf(
            "# CHECK collective=%s S=%zu: FAIL (first mismatch: rank=%d buf=%s idx=%lld "
            "want=%g got=%g)\n",
            collective_name(c), S, mm.rank, buf_name(mm.buf), mm.idx, (double)mm.want,
            (double)mm.got);
    }
}

// =====================================================================================
// CLI helpers
// =====================================================================================

// Parses a --dims argument of the form "2x2x2", "8", or "2x4" into per-dimension extents.
//
// Parameters
// ----------
// s : const std::string& -- 'x'-separated positive integers.
//
// Returns
// -------
// std::vector<int> -- one entry per dimension, each > 0. Empty on any parse failure (missing
// token, non-digit character, or non-positive value), which callers treat as a CLI error.
inline std::vector<int> parse_dims(const std::string& s) {
    std::vector<int> out;
    size_t pos = 0;
    while (pos <= s.size()) {
        size_t next = s.find('x', pos);
        std::string tok = (next == std::string::npos) ? s.substr(pos) : s.substr(pos, next - pos);
        if (tok.empty()) return {};
        for (char ch : tok)
            if (!std::isdigit((unsigned char)ch)) return {};
        int val = std::atoi(tok.c_str());
        if (val <= 0) return {};
        out.push_back(val);
        if (next == std::string::npos) break;
        pos = next + 1;
    }
    return out;
}

// =====================================================================================
// Executors: identical function signatures, divergent implementations.
//
// Everything above this point is pure host C++ with no CUDA dependency and is shared,
// unmodified, by both builds. Everything below is fenced per the spec: CUDA/NCCL code lives
// under `#ifndef TORUS_SIM`; the host-memory simulator (this environment's test vehicle, since
// no nvcc/GPU is available here) lives in the `#else` branch. Both branches implement the exact
// same set of function names and signatures -- alloc_buffers, free_buffers, sync_all_devices,
// fill_input_pattern, run_schedule, run_check, global_teardown -- so main() below is written
// once and never itself needs an #ifdef.
// =====================================================================================

#ifndef TORUS_SIM
// --------------------------------------------------------------------------------
// NCCL/CUDA executor. Requires nvcc + a CUDA toolkit + NCCL; NOT buildable or runnable in this
// development environment (no nvcc/CUDA installed here -- see project report). Written to
// mirror every simulator-side function signature exactly, per the spec, so this branch never
// references anything sim-only.
// --------------------------------------------------------------------------------
#include <cuda_runtime.h>
#include <nccl.h>

// CUDA error-check macro: prints file:line and aborts on any non-success cudaError_t.
#define CUDA_CHECK(call)                                                                     \
    do {                                                                                     \
        cudaError_t _e = (call);                                                             \
        if (_e != cudaSuccess) {                                                             \
            std::fprintf(stderr, "%s:%d: CUDA error: %s\n", __FILE__, __LINE__,              \
                          cudaGetErrorString(_e));                                           \
            std::abort();                                                                    \
        }                                                                                    \
    } while (0)

// NCCL error-check macro: prints file:line and aborts on any non-success ncclResult_t.
#define NCCL_CHECK(call)                                                                     \
    do {                                                                                     \
        ncclResult_t _r = (call);                                                            \
        if (_r != ncclSuccess) {                                                             \
            std::fprintf(stderr, "%s:%d: NCCL error: %s\n", __FILE__, __LINE__,              \
                          ncclGetErrorString(_r));                                           \
            std::abort();                                                                    \
        }                                                                                    \
    } while (0)

// Elementwise in-place float accumulate: dst[i] += src[i] for i in [0, n). Used to implement
// LocalOp with add == true on the device (the host-side simulator does the equivalent with a
// plain loop; see the #else branch below).
__global__ void add_inplace(float* dst, const float* src, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

static int g_N = 0;
static std::vector<std::array<void*, NUM_BUFS>> g_dev_bufs;
static std::array<size_t, NUM_BUFS> g_buf_sizes{};
static ncclComm_t* g_comms = nullptr;
static cudaStream_t* g_streams = nullptr;

// Allocates per-(rank,buffer) device storage for the upcoming sweep size, and lazily
// initializes the (persistent, reused-across-sizes) NCCL communicators and per-device streams
// on the first call.
//
// Parameters
// ----------
// N : int -- device/rank count (must match every subsequent call until the matching
//     free_buffers()).
// sizes : const size_t[NUM_BUFS] -- from buffer_sizes(); a 0 entry allocates nothing (null ptr).
inline void alloc_buffers(int N, const size_t sizes[NUM_BUFS]) {
    g_N = N;
    for (int b = 0; b < NUM_BUFS; ++b) g_buf_sizes[b] = sizes[b];
    if (!g_comms) {
        g_comms = new ncclComm_t[N];
        std::vector<int> devs(N);
        for (int i = 0; i < N; ++i) devs[i] = i;
        NCCL_CHECK(ncclCommInitAll(g_comms, N, devs.data()));
        g_streams = new cudaStream_t[N];
        for (int r = 0; r < N; ++r) {
            CUDA_CHECK(cudaSetDevice(r));
            CUDA_CHECK(cudaStreamCreate(&g_streams[r]));
        }
    }
    g_dev_bufs.assign(N, {nullptr, nullptr, nullptr, nullptr, nullptr});
    for (int r = 0; r < N; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        for (int b = 0; b < NUM_BUFS; ++b) {
            if (sizes[b] > 0) CUDA_CHECK(cudaMalloc(&g_dev_bufs[r][b], sizes[b]));
        }
    }
}

// Frees the per-(rank,buffer) device storage allocated by the matching alloc_buffers() call.
// Communicators/streams are intentionally NOT destroyed here (they are reused across sweep
// sizes); see global_teardown() for final cleanup.
inline void free_buffers(int N) {
    for (int r = 0; r < N; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        for (int b = 0; b < NUM_BUFS; ++b)
            if (g_dev_bufs[r][b]) CUDA_CHECK(cudaFree(g_dev_bufs[r][b]));
    }
    g_dev_bufs.clear();
}

// Blocks the host until every device has completed all work previously enqueued on its stream.
inline void sync_all_devices() {
    for (int r = 0; r < g_N; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// Fills every rank's BUF_SEND with the deterministic --check pattern (see gen_pattern()),
// H2D-copying a host-generated array rather than launching a fill kernel (simplest correct
// option given the CUDA path cannot be tested locally; avoids a second untestable kernel).
inline void fill_input_pattern(int N) {
    size_t send_bytes = g_buf_sizes[BUF_SEND];
    for (int r = 0; r < N; ++r) {
        std::vector<float> pat = gen_pattern(r, send_bytes);
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaMemcpy(g_dev_bufs[r][BUF_SEND], pat.data(), send_bytes,
                               cudaMemcpyHostToDevice));
    }
}

// Executes one full pass of `sched` across all N devices.
//
// Notes
// -----
// Correctness argument for the (deliberate) ABSENCE of any per-step host synchronization here
// (see the file header's SCHEDULE-AS-DATA section for the two-executor contract this satisfies):
//
//   - RANK-LOCAL ordering is preserved by CUDA stream order alone, with no host round-trip
//     needed. Every one of a given rank r's operations -- its ncclSend/ncclRecv calls (issued on
//     g_streams[r] via g_comms[r]), its cudaMemcpyAsync D2D post-ops, and its add_inplace kernel
//     launches -- are ALL enqueued on that same single stream g_streams[r], in the exact order
//     this function enqueues them (every Step's Xfers, in loop order, followed by that Step's
//     post LocalOps, in loop order, before moving to the next Step). CUDA guarantees operations
//     enqueued on one stream execute in that enqueue order; a later-enqueued op on a stream is
//     therefore guaranteed to see the effects of every earlier op on that SAME stream without
//     any explicit sync between them. This is exactly what build_schedule()'s intra-step alias
//     assertion (see that function) verifies is sufficient: it never lets a schedule reach this
//     executor if it would require a per-rank read/write ordering stronger than "post-op reads
//     what an Xfer of the same step already wrote" -- precisely the one relationship stream
//     order already provides here, for free.
//   - CROSS-RANK ordering (rank A's send must be matched by rank B's matching recv before either
//     side's dependent work proceeds) is enforced by NCCL itself, not by any host synchronization
//     this function performs: every Step's Xfers are issued inside one ncclGroupStart/End, and
//     NCCL's own send/recv rendezvous protocol is what guarantees a recv only completes (on its
//     own stream) once its matching send has actually transferred the data -- that handshake is
//     GPU-side, asynchronous, and requires no host cudaStreamSynchronize call to be correct.
//
// Given both of the above, a host round-trip is not needed between every Step, nor between a
// Step's Xfers and its post LocalOps -- it is needed exactly ONCE per call to this function, to
// give the HOST a defined point at which every device's work for this entire schedule execution
// is known to have completed (callers that need to read results back, e.g. run_check() via
// cudaMemcpy, or that are timing this call, already provide that host sync themselves --
// sync_all_devices() in main()'s check/warmup/timed-loop call sites -- but performing it once
// here too keeps this function's own postcondition self-contained rather than relying on every
// caller to remember to do it). This drops per-iteration host-sync round-trips from ~2x the
// number of Steps (the pre-fix per-step synchronize-after-Xfers +
// synchronize-after-post-LocalOps pattern) to exactly 1, with no change to the timed loop's
// measured semantics (main()'s warmup/timed loops already sync once before/after the whole w+n
// iteration count, not between individual run_schedule() calls).
inline void run_schedule(const Schedule& sched, int N) {
    for (const Step& step : sched) {
        ncclGroupStart();
        for (const Xfer& x : step.xfers) {
            char* sptr = (char*)g_dev_bufs[x.src][x.src_buf] + x.src_off;
            char* dptr = (char*)g_dev_bufs[x.dst][x.dst_buf] + x.dst_off;
            NCCL_CHECK(ncclSend(sptr, x.bytes, ncclChar, x.dst, g_comms[x.src], g_streams[x.src]));
            NCCL_CHECK(ncclRecv(dptr, x.bytes, ncclChar, x.src, g_comms[x.dst], g_streams[x.dst]));
        }
        ncclGroupEnd();
        for (const LocalOp& op : step.post) {
            CUDA_CHECK(cudaSetDevice(op.rank));
            char* sptr = (char*)g_dev_bufs[op.rank][op.src_buf] + op.src_off;
            char* dptr = (char*)g_dev_bufs[op.rank][op.dst_buf] + op.dst_off;
            if (!op.add) {
                CUDA_CHECK(cudaMemcpyAsync(dptr, sptr, op.bytes, cudaMemcpyDeviceToDevice,
                                           g_streams[op.rank]));
            } else {
                size_t nf = op.bytes / sizeof(float);
                int threads = 256;
                int blocks = (int)((nf + (size_t)threads - 1) / (size_t)threads);
                add_inplace<<<blocks, threads, 0, g_streams[op.rank]>>>((float*)dptr,
                                                                        (const float*)sptr, nf);
            }
        }
    }
    // Host synchronization happens ONCE per schedule execution, here, after the last Step --
    // see the Notes above for why nothing between Steps (or between a Step's Xfers and its
    // post-ops) needs it.
    for (int r = 0; r < N; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaStreamSynchronize(g_streams[r]));
    }
}

// Runs the --check verification for every rank, D2H-copying each rank's BUF_RECV before
// delegating to the shared verify_collective(). Prints the single required "# CHECK ..." line.
//
// Returns
// -------
// bool -- true iff every rank's BUF_RECV matched the expected result.
inline bool run_check(Collective c, const std::vector<int>& dims, size_t S, int N) {
    bool overall_pass = true;
    MismatchInfo first_mm;
    for (int r = 0; r < N; ++r) {
        size_t nbytes = g_buf_sizes[BUF_RECV];
        size_t nf = nbytes / sizeof(float);
        std::vector<float> recv(nf);
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaMemcpy(recv.data(), g_dev_bufs[r][BUF_RECV], nbytes, cudaMemcpyDeviceToHost));
        MismatchInfo mm;
        bool ok = verify_collective(c, dims, S, r, recv, mm);
        if (!ok && overall_pass) {
            overall_pass = false;
            first_mm = mm;
        }
    }
    print_check_line(c, S, overall_pass, first_mm);
    return overall_pass;
}

// Destroys the persistent NCCL communicators and CUDA streams created lazily by the first
// alloc_buffers() call. Safe to call even if alloc_buffers() was never called.
inline void global_teardown() {
    if (g_comms) {
        for (int r = 0; r < g_N; ++r) ncclCommDestroy(g_comms[r]);
        delete[] g_comms;
        g_comms = nullptr;
    }
    if (g_streams) {
        for (int r = 0; r < g_N; ++r) {
            cudaSetDevice(r);
            cudaStreamDestroy(g_streams[r]);
        }
        delete[] g_streams;
        g_streams = nullptr;
    }
}

#else
// --------------------------------------------------------------------------------
// Host-memory simulator executor (TORUS_SIM). No CUDA/NCCL dependency whatsoever -- this is
// the local, GPU-free test vehicle exercised by the acceptance criteria in this environment.
// Timings produced here are, per the spec, not scientifically meaningful (there is no real
// interconnect being modeled); the schedule-building, edge-assertion, and per-collective
// correctness logic under test is identical to what the NCCL branch would execute.
// --------------------------------------------------------------------------------

static std::vector<std::array<std::vector<char>, NUM_BUFS>> g_host_bufs;

// Allocates zero-initialized per-(rank,buffer) host storage for the upcoming sweep size.
//
// Parameters
// ----------
// N : int -- rank count.
// sizes : const size_t[NUM_BUFS] -- from buffer_sizes().
inline void alloc_buffers(int N, const size_t sizes[NUM_BUFS]) {
    g_host_bufs.assign((size_t)N, std::array<std::vector<char>, NUM_BUFS>{});
    for (int r = 0; r < N; ++r)
        for (int b = 0; b < NUM_BUFS; ++b) g_host_bufs[r][b].assign(sizes[b], 0);
}

// Releases the host storage allocated by the matching alloc_buffers() call.
inline void free_buffers(int N) {
    (void)N;
    g_host_bufs.clear();
}

// No-op: the simulator is single-threaded host code, so every operation is already
// synchronous by construction. Present only so main()'s driver loop can call the same function
// name in both builds.
inline void sync_all_devices() {}

// Fills every rank's BUF_SEND with the deterministic --check pattern (see gen_pattern()).
inline void fill_input_pattern(int N) {
    for (int r = 0; r < N; ++r) {
        std::vector<float> pat = gen_pattern(r, g_host_bufs[r][BUF_SEND].size());
        std::memcpy(g_host_bufs[r][BUF_SEND].data(), pat.data(), pat.size() * sizeof(float));
    }
}

// Executes one full pass of `sched` over host memory.
//
// Design decision: within a Step, ALL Xfer sources are first snapshotted into temporary
// buffers, and only then are all destinations written. This makes the "all xfers execute
// concurrently, reading pre-step state" semantics of Step literally true regardless of
// iteration order or any potential (believed absent, but not asserted) offset aliasing between
// a step's reads and writes -- a small, cheap robustness margin given how load-bearing exact
// schedule semantics are for this benchmark's scientific validity.
inline void run_schedule(const Schedule& sched, int N) {
    (void)N;
    for (const Step& step : sched) {
        std::vector<std::vector<char>> staged(step.xfers.size());
        for (size_t i = 0; i < step.xfers.size(); ++i) {
            const Xfer& x = step.xfers[i];
            const std::vector<char>& src = g_host_bufs[x.src][x.src_buf];
            staged[i].assign(src.begin() + (long)x.src_off, src.begin() + (long)(x.src_off + x.bytes));
        }
        for (size_t i = 0; i < step.xfers.size(); ++i) {
            const Xfer& x = step.xfers[i];
            std::memcpy(g_host_bufs[x.dst][x.dst_buf].data() + x.dst_off, staged[i].data(), x.bytes);
        }
        for (const LocalOp& op : step.post) {
            if (!op.add) {
                std::memcpy(g_host_bufs[op.rank][op.dst_buf].data() + op.dst_off,
                            g_host_bufs[op.rank][op.src_buf].data() + op.src_off, op.bytes);
            } else {
                float* dst = reinterpret_cast<float*>(g_host_bufs[op.rank][op.dst_buf].data() + op.dst_off);
                const float* src = reinterpret_cast<const float*>(g_host_bufs[op.rank][op.src_buf].data() + op.src_off);
                size_t nf = op.bytes / sizeof(float);
                for (size_t i = 0; i < nf; ++i) dst[i] += src[i];
            }
        }
    }
}

// Runs the --check verification for every rank directly against host memory (no device copy
// needed) and prints the single required "# CHECK ..." line.
//
// Returns
// -------
// bool -- true iff every rank's BUF_RECV matched the expected result.
inline bool run_check(Collective c, const std::vector<int>& dims, size_t S, int N) {
    bool overall_pass = true;
    MismatchInfo first_mm;
    for (int r = 0; r < N; ++r) {
        const std::vector<char>& raw = g_host_bufs[r][BUF_RECV];
        size_t nf = raw.size() / sizeof(float);
        std::vector<float> recv(nf);
        std::memcpy(recv.data(), raw.data(), raw.size());
        MismatchInfo mm;
        bool ok = verify_collective(c, dims, S, r, recv, mm);
        if (!ok && overall_pass) {
            overall_pass = false;
            first_mm = mm;
        }
    }
    print_check_line(c, S, overall_pass, first_mm);
    return overall_pass;
}

// No-op in the simulator: there is no persistent device/communicator state to tear down.
inline void global_teardown() {}

#endif  // TORUS_SIM

// =====================================================================================
// main(): CLI parsing, sweep loop, and output. Shared verbatim by both builds -- everything it
// calls (alloc_buffers, free_buffers, fill_input_pattern, run_schedule, sync_all_devices,
// run_check, global_teardown, build_schedule, buffer_sizes) has an identical signature in both
// the CUDA and simulator branches above, so no #ifdef is needed here at all.
// =====================================================================================

int main(int argc, char** argv) {
    std::string collective_str;
    std::string dims_str;
    long long b = -1, e = -1;
    long long f = 2;
    long long w = 5;
    long long n = 20;
    bool check = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need_val = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "torus_bench: missing value for %s\n", flag);
                std::exit(2);
            }
            return std::string(argv[++i]);
        };
        if (a == "--collective") collective_str = need_val("--collective");
        else if (a == "--dims") dims_str = need_val("--dims");
        else if (a == "-b") b = std::atoll(need_val("-b").c_str());
        else if (a == "-e") e = std::atoll(need_val("-e").c_str());
        else if (a == "-f") f = std::atoll(need_val("-f").c_str());
        else if (a == "-w") w = std::atoll(need_val("-w").c_str());
        else if (a == "-n") n = std::atoll(need_val("-n").c_str());
        else if (a == "--check") check = true;
        else {
            std::fprintf(stderr, "torus_bench: unknown argument '%s'\n", a.c_str());
            return 2;
        }
    }

    if (collective_str.empty() || dims_str.empty() || b <= 0 || e <= 0) {
        std::fprintf(stderr,
                      "usage: torus_bench --collective {all_reduce,all_gather,reduce_scatter,"
                      "alltoall,broadcast,sendrecv} --dims <e.g. 2x2x2> -b <min_total_bytes> "
                      "-e <max_total_bytes> [-f 2] [-w 5] [-n 20] [--check]\n");
        return 2;
    }
    if (f <= 1) {
        std::fprintf(stderr, "torus_bench: -f must be >= 2 (got %lld)\n", f);
        return 2;
    }

    Collective c;
    if (!parse_collective(collective_str, c)) {
        std::fprintf(stderr, "torus_bench: unknown --collective '%s'\n", collective_str.c_str());
        return 2;
    }
    std::vector<int> dims = parse_dims(dims_str);
    if (dims.empty()) {
        std::fprintf(stderr, "torus_bench: invalid --dims '%s'\n", dims_str.c_str());
        return 2;
    }
    int N = num_ranks(dims);

#ifdef TORUS_SIM
    std::printf("# torus_bench [SIMULATOR build -- host memory only, no GPU/NCCL]\n");
#else
    std::printf("# torus_bench [NCCL/CUDA build]\n");
#endif
    std::printf(
        "# collective=%s dims=%s N=%d range=[%lld,%lld] factor=%lld warmup=%lld iters=%lld "
        "check=%d\n",
        collective_name(c), dims_str.c_str(), N, b, e, f, w, n, check ? 1 : 0);

    bool any_check_failure = false;
    for (long long S = b; S <= e; S *= f) {
        size_t divisor = 4ull * (size_t)N * (size_t)N;
        if ((size_t)S % divisor != 0) {
            std::printf(
                "# SKIP S=%lld not divisible by 4*N*N=%zu (N=%d): shard/chunk boundaries "
                "would not align to whole floats\n",
                S, divisor, N);
            continue;
        }

        Schedule sched = build_schedule(c, dims, (size_t)S);
        std::array<size_t, NUM_BUFS> sizes = buffer_sizes(c, dims, (size_t)S);
        alloc_buffers(N, sizes.data());

        char check_char = '-';
        if (check) {
            fill_input_pattern(N);
            run_schedule(sched, N);
            sync_all_devices();
            bool pass = run_check(c, dims, (size_t)S, N);
            if (!pass) any_check_failure = true;
            check_char = pass ? '1' : '0';
        }

        for (long long it = 0; it < w; ++it) run_schedule(sched, N);
        sync_all_devices();
        auto t0 = std::chrono::steady_clock::now();
        for (long long it = 0; it < n; ++it) run_schedule(sched, N);
        sync_all_devices();
        auto t1 = std::chrono::steady_clock::now();
        double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        double avg_us = (n > 0) ? (total_us / (double)n) : 0.0;

        free_buffers(N);

        std::printf("TORUSBENCH,%s,%s,%lld,%.2f,%c\n", collective_name(c), dims_str.c_str(), S,
                    avg_us, check_char);
        std::fflush(stdout);
    }

    global_teardown();
    return any_check_failure ? 1 : 0;
}

