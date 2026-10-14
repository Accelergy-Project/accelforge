"""
Tests for _allow_multi_einsum_parallelism: when an architecture spatial dimension sets
it, each join is explored with the right Einsum running sequentially after the left and
in parallel with it, splitting that dimension's fanout. A sequential join serializes
each side's busy time below the shared loops into its own total. A parallel join takes
the slower branch's total, also for branches that end at different levels, since each
branch has its own instances below that fanout, and keeps the per-component latencies
of the components above it, so their busy times sum over the branches; different
components overlap. Parallel has no effect on reservations except that the live
branches' usage of that dimension is summed: a branch fenced off by a sequential join
at the same level is dead and stays out of a later parallel join there, while a branch
fenced deeper is parallel with the new Einsum, so its usage counts. The Einsum
delay is a descent at the Einsum's own level. Descent/ascent latencies sum at a join
(wind-up latency is serialized). When a free passes a level with a sequential split,
the level's descent/ascent latencies are multiplied by the # iterations of every loop
above it and added into the total latency at that level. At a level with a parallel
split or no split they are added into the descent/ascent latencies of the level above,
whose split decides; level 0 pays them once. The chip's leak power pays over the
joined runtime, so a parallel latency win carries into leak energy.

Most tests use the matmul-chain workload on the latency test arch, whose MAC X
dimension allows the parallelism: each Einsum takes one of its 4 instances.
"""

import unittest
from pathlib import Path

import accelforge as af
from accelforge.frontend.arch.arch import _spatial2resource
from accelforge.frontend.arch.spatialable import Spatialable
from accelforge.frontend.spec import Spec
from accelforge.model.main import evaluate_mapping
from accelforge.util import oset

TESTS_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_FILES_DIR = TESTS_DIR / "input_files"
LATENCY_ARCH = INPUT_FILES_DIR / "latency.arch.yaml"
WEIGHTS_INSIDE_MAPPING = INPUT_FILES_DIR / "fused_matmuls_weights_inside.mapping.yaml"
TWO_FUSED_LOOPS_MAPPING = INPUT_FILES_DIR / "fused_matmuls_two_fused_loops.mapping.yaml"
DEEP_THEN_SHALLOW_MAPPING = (
    INPUT_FILES_DIR / "fused_matmuls_deep_then_shallow.mapping.yaml"
)
SHALLOW_THEN_DEEP_MAPPING = (
    INPUT_FILES_DIR / "fused_matmuls_shallow_then_deep.mapping.yaml"
)
PAIR_THEN_DEEP_MAPPING = INPUT_FILES_DIR / "fused_matmuls_pair_then_deep.mapping.yaml"
FOUR_MATMULS_MAPPING = INPUT_FILES_DIR / "four_matmuls_three_loops.mapping.yaml"

# Four Einsums, three fused m loops of 2 iterations each, GlobalBuffer at
# 10-cycle reads and 20-cycle writes, infinitely fast compute.
CHAIN_OF_FOUR_JINJA = {
    "N_EINSUMS": 4,
    "M": 8,
    "KN": 2,
    "GB_READ_LATENCY": 10,
    "GB_WRITE_LATENCY": 20,
    "COMPUTE_THROUGHPUT": float("inf"),
}

# With these throughputs, Matmul0 is compute-bound (64 MACs / 0.2 = 320) and Matmul1
# is memory-bound (writing T2 back: 128 bits / 0.1 = 1280, plus reading W1: 32).
SLOW_MM_THROUGHPUTS = {
    "MM_READ_THROUGHPUT": 4,
    "MM_WRITE_THROUGHPUT": 0.1,
    "COMPUTE_THROUGHPUT": 0.2,
}

# Matmul0 and Matmul1 share both m loops; Matmul2 shares only the outer one. Their
# weights are held between the loops, so their GlobalBuffer nodes are at level 1.
DEEP_THEN_SHALLOW_JINJA = {
    "N_EINSUMS": 3,
    "M": 16,
    "KN": 4,
    "M_TILE": 4,
    "GB_SIZE": 2048,
    "GB_READ_LATENCY": 10,
    "GB_WRITE_LATENCY": 20,
    "COMPUTE_THROUGHPUT": float("inf"),
}

# Matmul1 and Matmul2 share both m loops; Matmul0 shares only the outer one. The
# deeper Einsums' weights are held above the inner loop, so their GlobalBuffer nodes
# are at level 1.
SHALLOW_THEN_DEEP_JINJA = {
    "N_EINSUMS": 3,
    "M": 16,
    "KN": 4,
    "M_TILE": 4,
    "GB_SIZE": 2048,
    "GB_READ_LATENCY": 10,
    "GB_WRITE_LATENCY": 20,
    "COMPUTE_THROUGHPUT": float("inf"),
}

MAC_X = f"reservation<SEP>{_spatial2resource('MAC', 'X', True)}<SEP>-1<SEP>left"
GB = "reservation<SEP>GlobalBuffer<SEP>-1<SEP>left"


def evaluate(mapping, explore_parallel, metrics=None, **jinja):
    spec = Spec.from_yaml(
        af.examples.workloads.basic.matmuls_any_einsums,
        LATENCY_ARCH,
        mapping,
        jinja_parse_data={
            "N_EINSUMS": 2,
            "M": 4,
            "KN": 4,
            "MAC_FANOUT": 4,
            "MAC_PARALLEL": explore_parallel,
            **jinja,
        },
    )
    if metrics is not None:
        spec.model.metrics = metrics
    spec.model._share_resource_latency_above_splits = True
    spec.model._include_wind_up_latency = True
    return evaluate_mapping(spec).data


class TestParallelSemantics(unittest.TestCase):
    def test_parallel_overlaps_busy_time(self):
        """Run sequentially, each Einsum's busy time serializes into the total:
        320 + 1408 = 1728. Run in parallel, each branch computes on its own MAC
        instance, so the MACs take the slower branch's 320, MainMemory's busy time
        sums over both branches, and the two overlap: max(320, 160 + 1408) = 1568,
        holding both Einsums' MAC instances at once. The sequential result stays on
        the front with fewer instances held."""
        off = evaluate(WEIGHTS_INSIDE_MAPPING, False, **SLOW_MM_THROUGHPUTS)
        self.assertEqual(list(off["Total<SEP>latency"]), [1728])

        on = evaluate(WEIGHTS_INSIDE_MAPPING, True, **SLOW_MM_THROUGHPUTS)
        front = sorted(zip(on["Total<SEP>latency"], on[MAC_X] * 4))
        self.assertEqual(front, [(1568, 2), (1728, 1)])

    def test_above_split_latency_uses_branch_slack(self):
        """The weights are held above the fused m loop, so their MainMemory reads
        may run in any branch's slack, while T0's and T2's nodes below it keep
        the rest of MainMemory's traffic in its own branch. Matmul0's branch is
        compute-bound (320 vs reading T0, 32) and Matmul1's is bound by writing T2
        back (1280 vs 320), so the branches serialize into 320 + 1280 = 1600 and
        MainMemory's whole busy time (32 + 32 + 1280 + 32 = 1376) hides under it.
        Running the branches in parallel drops the serialization and MainMemory's
        1376 sets the total. Slowing the reads to 128 bit-times raises that busy time
        to 256 + 1408 = 1664, which already exceeds the branches' 1600, so the slack
        is used up and parallelism gains nothing."""
        jinja = {"MM_WRITE_THROUGHPUT": 0.1, "COMPUTE_THROUGHPUT": 0.2}
        simple = INPUT_FILES_DIR / "fused_matmuls_to_simple.mapping.yaml"

        off = evaluate(simple, False, MM_READ_THROUGHPUT=4, **jinja)
        self.assertEqual(list(off["Total<SEP>latency"]), [1600])

        on = evaluate(simple, True, MM_READ_THROUGHPUT=4, **jinja)
        front = sorted(zip(on["Total<SEP>latency"], on[MAC_X] * 4))
        self.assertEqual(front, [(1376, 2), (1600, 1)])

        off = evaluate(simple, False, MM_READ_THROUGHPUT=1, **jinja)
        self.assertEqual(list(off["Total<SEP>latency"]), [1664])

        on = evaluate(simple, True, MM_READ_THROUGHPUT=1, **jinja)
        self.assertEqual(list(on["Total<SEP>latency"]), [1664])

    def test_unfused_join_shares_no_loops(self):
        """With every tensor backed in MainMemory the Einsums share no loops, so they
        split above all of them. Each Einsum's MainMemory busy time (reading its input
        and weights, 32 each, and writing its output back, 1280) is 1344, which its
        320 of compute hides under, and the branches serialize into 2688. Running them
        in parallel sums MainMemory's busy time to the same 2688, so nothing is gained
        and only the single-instance result stays on the front."""
        unfused = INPUT_FILES_DIR / "unfused_matmuls_to_simple.mapping.yaml"
        off = evaluate(unfused, False, **SLOW_MM_THROUGHPUTS)
        self.assertEqual(list(off["Total<SEP>latency"]), [2688])

        on = evaluate(unfused, True, **SLOW_MM_THROUGHPUTS)
        front = sorted(zip(on["Total<SEP>latency"], on[MAC_X] * 4))
        self.assertEqual(front, [(2688, 1)])

    def test_windup_paid_once_when_split_below(self):
        """With two fused m loops (8 outer x 4 inner iterations) and the weights held
        between them, each weight's GlobalBuffer node (write + read, 30) is at level
        1. The Einsums split below both loops (level 2), so level 1 has no split: the
        weights' 30 + 30 is added into level 0 and paid once (wind-up latency is
        serialized), with or without parallelism. The Einsum delays are descents at
        level 2: Matmul0 writes T0 into the GlobalBuffer and reads it to compute
        (20 + 10) and writes T1 (20), 50, and Matmul1 reads its inputs (10) and writes
        T2 (20) and reads it to MainMemory (10), 40. Each Einsum computes 512 MACs at
        one per cycle. The sequential split repays the delays on all 8 x 4 = 32
        iterations of the two loops above it, and the wind-ups add to the 512 + 512
        of compute: 90 x 32 + 60 + 1024 = 3964. The parallel split adds the delays
        into level 1 and on into level 0, which pays them once, and takes the slower
        branch's 512: 90 + 60 + 512 = 662, holding both MAC X instances. Counting
        each hop at its lower node's level would give 3944 and 642."""
        jinja = {
            "M": 32,
            "KN": 4,
            "M_TILE": 4,
            "GB_READ_LATENCY": 10,
            "GB_WRITE_LATENCY": 20,
        }
        off = evaluate(TWO_FUSED_LOOPS_MAPPING, False, **jinja)
        self.assertEqual(list(off["Total<SEP>latency"]), [3964])

        on = evaluate(TWO_FUSED_LOOPS_MAPPING, True, **jinja)
        front = sorted(zip(on["Total<SEP>latency"], on[MAC_X] * 4))
        self.assertEqual(front, [(662, 2), (3964, 1)])

    def test_sequential_split_repays_windup(self):
        """Each storage node's actions count at the node's own level, and the nodes at
        or below an Einsum's level are in its delay, a descent at that level. At
        level 2, Matmul0 writes T0 into the GlobalBuffer and reads it (20 + 10) and
        writes T1 (20), 50, and Matmul1 reads T1 (10), 10: 60. At level 1 are the
        GlobalBuffer nodes of W0 and W1 (write + read, 30 each), T2's node above the
        split, which Matmul1 writes (an ascent, 20; its read-modify-write read, 10, is
        a descent below W1's 30), and Matmul2's delay (W2 written and read, 30, and T3
        written and read to MainMemory, 30: 60), so level 1 holds
        30 + 30 + 20 + 60 = 140. A sequential split repays its level on every
        iteration of every loop above it: joining Matmul1 sequentially repays level 2
        on the 4 x 4 = 16 iterations of both loops, and joining Matmul2 sequentially
        repays level 1 on the 4 outer iterations. A parallel split adds its level into
        the level above, whose split decides, and level 0 pays once:

        - both sequential: 60 x 16 + 140 x 4 = 1520 on one MAC X instance;
        - Matmul1 in parallel: (60 + 140) x 4 = 800 on two;
        - Matmul2 in parallel: 60 x 16 + 140 = 1100 on two, dominated;
        - both in parallel: 60 + 140 = 200 on three.

        GlobalBuffer bits: sequential 448 (Matmul0's 192 beside Matmul1's 256 above
        the split, the pair's larger branch); Matmul1 in parallel 480 (both branches'
        tiles alive at once, 192 + 288); all parallel 768 (480 beside Matmul2's 288).
        Counting each hop at its lower node's level would give 1680 and 720."""
        off = evaluate(DEEP_THEN_SHALLOW_MAPPING, False, **DEEP_THEN_SHALLOW_JINJA)
        self.assertEqual(list(off["Total<SEP>latency"]), [1520])

        on = evaluate(DEEP_THEN_SHALLOW_MAPPING, True, **DEEP_THEN_SHALLOW_JINJA)
        front = sorted(zip(on["Total<SEP>latency"], on[GB] * 2048, on[MAC_X] * 4))
        self.assertEqual(front, [(200, 768, 3), (800, 480, 2), (1520, 448, 1)])

    def test_windups_at_two_levels(self):
        """MainMemory reads take 100 cycles. Every MainMemory node is at level 0, which
        has no loops above it, so each Einsum's slowest tensor there, one MainMemory
        read (100), is paid once however the Einsums split, the three summing to 300
        (wind-up latency is serialized). They add to the wind-ups of
        test_sequential_split_repays_windup: 1520 + 300 = 1820 both sequential,
        800 + 300 = 1100 with Matmul1 in parallel on two instances, and
        200 + 300 = 500 both in parallel on three. Counting each hop at its lower
        node's level would repay the reads with the GlobalBuffer nodes they fill:
        4480, 2320 and 580."""
        jinja = {**DEEP_THEN_SHALLOW_JINJA, "MM_READ_LATENCY": 100}
        off = evaluate(DEEP_THEN_SHALLOW_MAPPING, False, **jinja)
        self.assertEqual(list(off["Total<SEP>latency"]), [1820])

        on = evaluate(DEEP_THEN_SHALLOW_MAPPING, True, **jinja)
        front = sorted(zip(on["Total<SEP>latency"], on[MAC_X] * 4))
        self.assertEqual(front, [(500, 3), (1100, 2), (1820, 1)])

    def test_sequential_split_repays_windups_joined_later(self):
        """Matmul0 splits from the Matmul1-Matmul2 pair above the inner m loop
        (level 1), and that pair splits below it (level 2). At level 2 are the pair's
        Einsum delays: Matmul1 writes T2 (20) and reads it to accumulate (10), 30, and
        Matmul2 reads T2 (10) and writes T3 and reads it to MainMemory (20 + 10), 40:
        70. At level 1 are Matmul0's delay (T0 written into the GlobalBuffer and read,
        20 + 10, and T1 written, 20: 50) and the GlobalBuffer nodes of W1 and W2 for
        the deeper Einsums (write + read, 30 each; Matmul1's read of T1, 10, is below
        W1's 30), 110. W2's node only arrives at the second join, after the first join
        has already recorded its split, and a sequential first split still repays it.
        A sequential split repays its level on every iteration of every loop above it,
        whatever the other split is: level 1 on the 4 outer iterations, level 2 on the
        4 x 4 = 16 iterations of both loops. A parallel split adds its level into the
        level above, whose split decides, and level 0 pays once:

        - both sequential: 110 x 4 + 70 x 16 = 1560 on one MAC X instance;
        - Matmul0 split off sequentially, Matmul2 in parallel: (110 + 70) x 4 = 720
          on two;
        - Matmul0 split off in parallel, Matmul2 joined sequentially below it:
          110 + 70 x 16 = 1230 on two, dominated;
        - both in parallel: 110 + 70 = 180 on three.

        GlobalBuffer bits: sequential 448 (Matmul2's 192 beside Matmul1's 256 above
        the pair's split); Matmul2 in parallel 480 (288 + 192, both branches' tiles
        alive at once); all parallel 744 (480 beside Matmul0's 264). Counting each hop
        at its lower node's level would give 1480 and 640."""
        off = evaluate(SHALLOW_THEN_DEEP_MAPPING, False, **SHALLOW_THEN_DEEP_JINJA)
        self.assertEqual(list(off["Total<SEP>latency"]), [1560])

        on = evaluate(SHALLOW_THEN_DEEP_MAPPING, True, **SHALLOW_THEN_DEEP_JINJA)
        front = sorted(zip(on["Total<SEP>latency"], on[GB] * 2048, on[MAC_X] * 4))
        self.assertEqual(front, [(180, 744, 3), (720, 480, 2), (1560, 448, 1)])

    def test_reservations_summed_above_split_maxed_below(self):
        """With two fused m loops, each Einsum's weights are held above the inner
        loop and its T tiles below it, and the Einsums split below both loops. Both
        weights are above the split, so they are summed (128 x 2 = 256); the T tiles
        are below it, so only one branch's are counted (32 x 2 = 64). 320 bits is the
        exact GlobalBuffer footprint either way. Holding the weights inside each branch
        instead puts everything below the split, and the 192-bit branch footprint
        (128 + 32 x 2) is held once rather than twice."""
        jinja = {"M": 32, "KN": 4, "M_TILE": 4, "GB_SIZE": 320}
        for explore_parallel in [False, True]:
            data = evaluate(TWO_FUSED_LOOPS_MAPPING, explore_parallel, **jinja)
            self.assertEqual(sorted(set(data[GB] * 320)), [320])

        jinja = {"GB_SIZE": 192, **SLOW_MM_THROUGHPUTS}
        for explore_parallel in [False, True]:
            data = evaluate(WEIGHTS_INSIDE_MAPPING, explore_parallel, **jinja)
            self.assertEqual(sorted(set(data[GB] * 192)), [192])

    def test_parallel_sums_reservations_above_the_fanout(self):
        """Parallel branches' reservations of components above the
        _allow_multi_einsum_parallelism fanout are alive at once and sum, like the
        branches' usage of that dimension (Ctrl-F for
        ALWAYS_PARALLEL_BELOW_MULTI_EINSUM_PARALLEL_SPATIAL). GlobalBuffer bits from
        test_sequential_split_repays_windup: 448 sequential, 480 with Matmul1 in
        parallel, 768 with all three in parallel. A 448-bit GlobalBuffer keeps only the
        sequential arm, 480 bits admit the two-instance arm, 768 all three."""
        for size, front in [
            (448, [(1520, 448, 1)]),
            (480, [(800, 480, 2), (1520, 448, 1)]),
            (768, [(200, 768, 3), (800, 480, 2), (1520, 448, 1)]),
        ]:
            with self.subTest(size=size):
                jinja = {**DEEP_THEN_SHALLOW_JINJA, "GB_SIZE": size}
                on = evaluate(DEEP_THEN_SHALLOW_MAPPING, True, **jinja)
                rows = sorted(
                    zip(on["Total<SEP>latency"], (on[GB] * size).round(), on[MAC_X] * 4)
                )
                self.assertEqual(rows, front)

    def test_leak_power_paid_over_joined_runtime(self):
        """The whole chip leaks no matter which Einsum runs: leak power maxes across
        branches and pays over the joined runtime at the final free, so the fastest
        parallel result's wind-up saving carries into its leak energy: 2 x 200 against
        2 x 1520 sequentially (latencies from test_sequential_split_repays_windup)."""
        jinja = {**DEEP_THEN_SHALLOW_JINJA, "GB_LEAK_POWER": 2}
        off = evaluate(DEEP_THEN_SHALLOW_MAPPING, False, **jinja)
        self.assertEqual(list(off["Total<SEP>leak_energy"]), [2 * 1520])

        on = evaluate(DEEP_THEN_SHALLOW_MAPPING, True, **jinja)
        fastest = on.loc[on["Total<SEP>latency"].idxmin()]
        self.assertEqual(fastest["Total<SEP>leak_energy"], 2 * 200)
        dynamic = off["Total<SEP>energy"].iloc[0] - 2 * 1520
        self.assertEqual(fastest["Total<SEP>energy"], dynamic + 2 * 200)

    def test_leak_energy_without_a_latency_objective(self):
        """Leak power pays over the joined runtime at the final fence, so the
        parallel latency win still carries into the energy when latency is not
        reported: 28928 of dynamic energy plus 2 x 1520 sequentially against
        2 x 200 with both joins in parallel."""
        jinja = {**DEEP_THEN_SHALLOW_JINJA, "GB_LEAK_POWER": 2}
        energy_only = af.mapper.Metrics.ENERGY

        off = evaluate(DEEP_THEN_SHALLOW_MAPPING, False, metrics=energy_only, **jinja)
        self.assertEqual(list(off["Total<SEP>energy"]), [28928 + 2 * 1520])

        on = evaluate(DEEP_THEN_SHALLOW_MAPPING, True, metrics=energy_only, **jinja)
        self.assertEqual(on["Total<SEP>energy"].min(), 28928 + 2 * 200)

    def test_windups_repaid_across_two_joins(self):
        """Matmul0 and Matmul1 split below all three fused m loops (level 3), Matmul2
        below two (level 2), and Matmul3 below one (level 1); every loop has 2
        iterations. Each storage node's actions count at the node's own level. At
        level 3 are Matmul0's delay (T0 written into the GlobalBuffer and read,
        20 + 10, and T1 written, 20: 50) and Matmul1's (T1 read, 10), 60; at level 2
        Matmul1's W1 node (write + read, 30) and T2 node (T2 written, an ascent, 20;
        its read-modify-write read, 10, is a descent below W1's 30), and Matmul2's
        delay (W2 written and read, 30), 30 + 20 + 30 = 80; at level 1 Matmul2's T3
        node (written, 20, and read to accumulate, 10) and Matmul3's delay (W3
        written and read, 30; T4 goes to MainMemory at no latency), 60. A parallel
        join adds its level into the level above, and level 0 pays once. A sequential
        join repays its level, with whatever was added into it, on every iteration of
        every loop above it: 2 x 2 x 2 = 8 at level 3, 2 x 2 = 4 at level 2 and 2 at
        level 1. Matmul3's parallel join at level 1 is parallel with everything below
        it, so its instance adds to the max usage below it, the fenced pair included:

        - all sequential: 60 x 8 + 80 x 4 + 60 x 2 = 920 on one MAC X instance;
        - Matmul0 and Matmul1 in parallel: (60 + 80) x 4 + 60 x 2 = 680 on two;
        - Matmul1 and Matmul2 in parallel: (60 + 80 + 60) x 2 = 400 on three;
        - all parallel: 60 + 80 + 60 = 200 on four.

        The other combinations are dominated: 760 and 860 on two, 620 on three twice
        (Matmul1 and Matmul3 in parallel, (60 + 80) x 4 + 60, beside the pair's two
        instances; Matmul2 and Matmul3 in parallel beside Matmul1's one). Keeping the
        fenced pair out of Matmul3's join would put the first 620 on two."""
        off = evaluate(FOUR_MATMULS_MAPPING, False, **CHAIN_OF_FOUR_JINJA)
        self.assertEqual(list(off["Total<SEP>latency"]), [920])

        on = evaluate(FOUR_MATMULS_MAPPING, True, **CHAIN_OF_FOUR_JINJA)
        front = sorted(zip(on["Total<SEP>latency"], on[MAC_X] * 4))
        self.assertEqual(front, [(200, 4), (400, 3), (680, 2), (920, 1)])

    def test_a_branch_repays_only_its_own_windups(self):
        """Matmul0 and Matmul1 split below both of their m loops (level 2, under the
        outer loop of 2 and an inner loop of 4); Matmul2 splits from the pair below
        the outer loop (level 1); Matmul3 is fused deeper inside Matmul2's branch
        (level 3, under three loops of 2). Each storage node's actions count at the
        node's own level. The pair's Einsum delays are at level 2: Matmul0 writes T0
        into the GlobalBuffer and reads it (20 + 10) and writes T1 (20), 50, and
        Matmul1 reads T1 (10), 10: 60. T2's node is at level 1: Matmul1 writes T2 (an
        ascent, 20) and reads it to accumulate (a descent, 10), and Matmul2 reads it
        (10), 40. In Matmul2's branch, W2's node (write + read, 30) is at its own level
        2 and the delays of Matmul2 (T3 written, 20, and read to accumulate, 10: 30)
        and Matmul3 (T3 read, 10) at level 3, 40. The pair's level-2 split is in the
        pair's branch, not Matmul2's, so Matmul2's level 2 has no split and W2's node
        is added into level 1. A parallel join adds its level into the level above,
        and level 0 pays once. A sequential join repays its level, with whatever was
        added into it, on every iteration of every loop above it: 2 x 4 = 8 for the
        pair's split, 2 at level 1 and 2 x 2 x 2 = 8 at level 3:

        - all sequential: 60 x 8 + (40 + 30) x 2 + 40 x 8 = 940 on one MAC X instance;
        - Matmul1 and Matmul3 in parallel: (60 + 40 + 30 + 40) x 2 = 340 on two;
        - all parallel: 60 + 40 + 30 + 40 = 170 on four.

        The other combinations are dominated: 550, 580 and 700 on two, 450 and 590 on
        three. With only Matmul2 in parallel, the pair's 60 x 8 = 480 and the
        Matmul2-Matmul3 branch's 40 x 8 = 320 run at once and max, plus the level-1 70
        paid once: 550; adding the two branches would give 870. Taking the pair's
        split as a split at Matmul2's level 2 would repay W2's node there, 30 x 4,
        giving 1000 all sequential."""
        off = evaluate(PAIR_THEN_DEEP_MAPPING, False, **CHAIN_OF_FOUR_JINJA)
        self.assertEqual(list(off["Total<SEP>latency"]), [940])

        on = evaluate(PAIR_THEN_DEEP_MAPPING, True, **CHAIN_OF_FOUR_JINJA)
        front = sorted(zip(on["Total<SEP>latency"], on[MAC_X] * 4))
        self.assertEqual(front, [(170, 4), (340, 2), (940, 1)])

    def test_pair_under_parallel_keeps_the_busy_time_bound(self):
        """The chain is GlobalBuffer-bound (5632 + 5632 + 6144 = 17408 of reads,
        derived in test_whole_chain_busy_time_survives_a_mid_chain_fence), which no
        schedule beats: every join sums the GlobalBuffer's busy time over its
        branches, sequential or parallel, so every arm ties at 17408 and only the
        single-instance one stays on the front."""
        jinja = {
            "N_EINSUMS": 3,
            "M": 16,
            "M_TILE": 4,
            "GB_SIZE": 2048,
            "GB_READ_THROUGHPUT": 1,
            "COMPUTE_THROUGHPUT": float("inf"),
        }
        off = evaluate(DEEP_THEN_SHALLOW_MAPPING, False, **jinja)
        self.assertEqual(list(off["Total<SEP>latency"]), [17408])

        on = evaluate(DEEP_THEN_SHALLOW_MAPPING, True, **jinja)
        front = sorted(zip(on["Total<SEP>latency"], on[MAC_X] * 4))
        self.assertEqual(front, [(17408, 1)])


class TestParallelMapper(unittest.TestCase):
    def map_workload(self, explore_parallel):
        spec = Spec.from_yaml(
            af.examples.workloads.basic.matmuls_any_einsums,
            LATENCY_ARCH,
            jinja_parse_data={
                "N_EINSUMS": 3,
                "M": 4,
                "KN": 4,
                "GB_SIZE": 2048,
                "GB_LEAK_POWER": 1,
                "MAC_FANOUT": 4,
                "MAC_PARALLEL": explore_parallel,
                **SLOW_MM_THROUGHPUTS,
            },
        )
        spec.mapper.metrics = af.mapper.Metrics.LATENCY | af.mapper.Metrics.ENERGY
        spec.model._share_resource_latency_above_splits = True
        return spec.map_workload_to_arch(print_progress=False)

    def test_parallel_never_beats_the_busy_time_bound(self):
        """This chain is MainMemory-bound: every Einsum moves
        32 + 32 + 1280 = 1344 bit-times through it (inputs at 4 bits per cycle, the
        output at 0.1), and the whole-chain busy time 3 x 1344 = 4032 is the same
        no matter how the Einsums are scheduled. Parallelism cannot beat it, so the
        parallel arms tie the sequential result exactly, and the leak (paid over
        equal runtimes) ties the energy too. The final model evaluation re-explores
        sequential/parallel combinations and reproduces the mapper's result."""
        off = self.map_workload(False)
        on = self.map_workload(True)
        self.assertEqual(off.data["Total<SEP>latency"].min(), 4032)
        for c in ["Total<SEP>latency", "Total<SEP>energy"]:
            self.assertEqual(on.data[c].min(), off.data[c].min())


class TestTpuZParallelism(unittest.TestCase):
    def make_spec(self, z_parallel):
        spec = Spec.from_yaml(
            af.examples.workloads.basic.matmuls_any_einsums,
            af.examples.arches.tpu_v4i,
            jinja_parse_data={"N_EINSUMS": 2, "M": 16, "KN": 16},
        )
        # Every Einsum must fit beside another in the Z fanout, so full usage cannot
        # be required of each.
        for node in spec.arch.get_nodes_of_type(Spatialable):
            for s in node.spatial:
                s.min_usage = 0
                if s.name == "Z":
                    s._allow_multi_einsum_parallelism = z_parallel
        spec.mapper.metrics = af.mapper.Metrics.LATENCY | af.mapper.Metrics.ENERGY
        spec.model._share_resource_latency_above_splits = True
        return spec

    def test_tpu_parallelizes_across_z(self):
        """The TPU's memories and other spatial dimensions sit below the Z fanout, so
        parallel Einsums use their own instances of them: the mapper tracks only the
        shared usage of Z itself across Einsums."""
        pmappings = af.mapper.FFM.make_pmappings(
            self.make_spec(True), print_progress=False
        )
        tracked = oset(
            c.split("<SEP>")[1]
            for groups in pmappings.einsum2pmappings.values()
            for g in groups
            for c in g.mappings.data.columns
            if c.startswith("reservation")
        )
        self.assertEqual(tracked, oset([_spatial2resource("LocalBuffer", "Z", True)]))

        # map_workload_to_arch re-evaluates its results through the model, so this
        # covers the model path's tracking too.
        on = self.make_spec(True).map_workload_to_arch(print_progress=False)
        evaluated = [c for c in on.data.columns if c.startswith("reservation")]
        self.assertFalse(any("Register" in c for c in evaluated))
        self.assertTrue(any("GlobalBuffer" in c for c in evaluated))

        off = self.make_spec(False).map_workload_to_arch(print_progress=False)
        for c in ["Total<SEP>latency", "Total<SEP>energy"]:
            self.assertLessEqual(on.data[c].min(), off.data[c].min() * (1 + 1e-6))

    def test_at_most_one_parallelism_spatial(self):
        spec = self.make_spec(True)
        for node in spec.arch.get_nodes_of_type(Spatialable):
            for s in node.spatial:
                s._allow_multi_einsum_parallelism = True
        with self.assertRaises(ValueError):
            spec.arch._multi_einsum_parallel_spatial()
