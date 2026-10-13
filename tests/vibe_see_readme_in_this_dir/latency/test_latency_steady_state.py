"""
Steady-state latency, with every action latency zero so no wind-up or wind-down is
paid. A resource's actions add into its busy time, a branch takes as long as its
slowest resource, sequential branches add, and at each split every resource's busy time
over the branch and its sub-branches bounds the branch's total latency. A parallel join
takes the slower branch on the components below the _allow_multi_einsum_parallelism
fanout, which each branch has its own instances of, and sums the busy time of the
components above it over the branches.

All tests use the matmul-chain workload on the latency test arch. With M = KN = 4 each
Einsum does 64 MACs and each tensor is 128 bits.
"""

import unittest

import accelforge as af
from tests.vibe_see_readme_in_this_dir.latency.test_latency import INPUT_FILES_DIR
from tests.vibe_see_readme_in_this_dir.latency.test_latency import (
    LATENCY_ARCH,
    total_latency,
)
from tests.vibe_see_readme_in_this_dir.latency.test_parallel_mappings import (
    FOUR_MATMULS_MAPPING,
    MAC_X,
    SHALLOW_THEN_DEEP_MAPPING,
    SLOW_MM_THROUGHPUTS,
    evaluate,
)

SIMPLE_MAPPING = INPUT_FILES_DIR / "fused_matmuls_to_simple.mapping.yaml"

# Every MainMemory and GlobalBuffer action moves one bit per cycle.
UNIT_THROUGHPUTS = {
    "MM_READ_THROUGHPUT": 1,
    "MM_WRITE_THROUGHPUT": 1,
    "GB_READ_THROUGHPUT": 1,
    "GB_WRITE_THROUGHPUT": 1,
}

# Four Einsums under three fused m loops of 2 iterations each: 32 MACs per Einsum, each
# T tensor 128 bits and each W tensor 32 bits.
FOUR_EINSUMS = {"N_EINSUMS": 4, "M": 8, "KN": 2}


class TestResourceBusyTime(unittest.TestCase):
    def test_steady_state_is_the_slowest_resource(self):
        """One Einsum, every action one bit-time. The GlobalBuffer holds W0 above the
        m loop and T0 and T1 below it. It is filled with W0 once (128 writes) and with
        T0's 32-bit row on each of the 4 m iterations (128 writes); it reads one value
        of each input per MAC (512 + 512), is written one value of T1 per MAC (512),
        re-reads T1 on 3 of every 4 MACs to accumulate (384) and reads T1 once to drain
        it to MainMemory (128): 1536 reads + 768 writes = 2304. MainMemory reads T0
        and W0 once (256) and is written T1 once (128): 384. The MACs take 64. The
        steady state is the slowest resource's, 2304; summing the resources would give
        2752."""
        row = total_latency(LATENCY_ARCH, SIMPLE_MAPPING, 1, **UNIT_THROUGHPUTS)
        self.assertEqual(row["Total<SEP>latency"], 2304)
        self.assertEqual(row["Matmul0<SEP>component_latency<SEP>GlobalBuffer"], 2304)
        self.assertEqual(row["Matmul0<SEP>component_latency<SEP>MainMemory"], 384)
        self.assertEqual(row["Matmul0<SEP>component_latency<SEP>MAC"], 64)

    def test_a_resources_actions_add(self):
        """With GlobalBuffer reads at two bits per cycle and writes at one, the
        GlobalBuffer's busy time is the sum over its actions: 1536 / 2 + 768 = 1536
        (counts from test_steady_state_is_the_slowest_resource), above the MACs' 64.
        Counting only the reads, or only the slowest action, would give 768."""
        row = total_latency(
            LATENCY_ARCH,
            SIMPLE_MAPPING,
            1,
            GB_READ_THROUGHPUT=2,
            GB_WRITE_THROUGHPUT=1,
        )
        self.assertEqual(row["Total<SEP>latency"], 1536)

    def test_separate_ports_are_two_resources(self):
        """MainMemory reads T0 and W0 (256 bits) and is written T1 (128 bits), each at
        0.1 bits per cycle. With separate ports the read port is busy for 2560 and the
        write port for 1280, and the two overlap: 2560. Sharing one port, the two add:
        3840."""
        jinja = {"MM_READ_THROUGHPUT": 0.1, "MM_WRITE_THROUGHPUT": 0.1}
        row = total_latency(
            LATENCY_ARCH, SIMPLE_MAPPING, 1, MM_SEPARATE_PORTS=True, **jinja
        )
        self.assertEqual(row["Total<SEP>latency"], 2560)
        self.assertEqual(
            row["Matmul0<SEP>component_latency<SEP>MainMemory (read)"], 2560
        )
        self.assertEqual(
            row["Matmul0<SEP>component_latency<SEP>MainMemory (write)"], 1280
        )

        row = total_latency(LATENCY_ARCH, SIMPLE_MAPPING, 1, **jinja)
        self.assertEqual(row["Total<SEP>latency"], 3840)


class TestParallelBusyTime(unittest.TestCase):
    def test_a_port_used_by_one_einsum_overlaps_in_parallel(self):
        """Matmul0 is compute-bound: 64 MACs / 0.2 = 320 against reading T0 and W0,
        256. Only Matmul1 uses MainMemory's write port, writing T2 back from its
        GlobalBuffer node below the fused m loop: 128 bits / 0.1 = 1280.
        Sequentially the branches serialize into 320 + 1280 = 1600
        (test_separate_read_write_ports). In parallel the MACs take the slower
        branch's 320, the read port's busy time sums over both branches
        (256 + 128 = 384), and both hide under the write port's 1280. Sharing one port
        would give 1664 either way."""
        on = evaluate(
            SIMPLE_MAPPING,
            True,
            MM_SEPARATE_PORTS=True,
            MM_READ_THROUGHPUT=1,
            MM_WRITE_THROUGHPUT=0.1,
            COMPUTE_THROUGHPUT=0.2,
        )
        front = sorted(zip(on["Total<SEP>latency"], on[MAC_X] * 4))
        self.assertEqual(front, [(1280, 2), (1600, 1)])

    def test_parallel_compute_takes_the_slower_branch(self):
        """Both Einsums are compute-bound, 64 MACs / 0.2 = 320 each, with infinitely
        fast memories. Sequentially 320 + 320 = 640 on one MAC X instance. In parallel
        each branch computes on its own instance, so the parallel join takes the
        slower right: 320 on two. Summing the MACs' busy time would tie the sequential
        640."""
        on = evaluate(SIMPLE_MAPPING, True, COMPUTE_THROUGHPUT=0.2)
        front = sorted(zip(on["Total<SEP>latency"], on[MAC_X] * 4))
        self.assertEqual(front, [(320, 2), (640, 1)])

    def test_fanout_limits_the_parallel_results(self):
        """Three Einsums, each computing for 64 MACs / 0.2 = 320. MainMemory reads T0
        (128 bits / 4 = 32), held below the fused m loop so private to Matmul0's
        branch, and each weight (32), held above it so free to fill any branch's
        slack, and is written T3 back from Matmul2's branch (128 / 0.1 = 1280): 1408
        in all. A parallel join takes the slower branch's MACs and sums MainMemory's
        busy time.

        - All sequential: 320 + 320 + 1280 = 1920 on one instance.
        - Matmul0 and Matmul1 in parallel, then Matmul2: max(320, 32) + 1280 = 1600 on
          two instances.
        - Matmul0, then Matmul1 and Matmul2 in parallel: 320 + max(320, 1280) = 1600
          on two instances.
        - All parallel: max(320, 32 + 1280) = 1312 below the loop, and MainMemory's
          1408 sets the total, on three instances.

        With 4 MAC X instances the front holds 1408, 1600 and 1920. With 2 the
        three-instance result does not fit, and with 1 no parallel result does. Not
        checking the usage against the fanout would keep (1408, 3) at 2."""
        off = evaluate(SIMPLE_MAPPING, False, N_EINSUMS=3, **SLOW_MM_THROUGHPUTS)
        self.assertEqual(list(off["Total<SEP>latency"]), [1920])

        for fanout, expected in [
            (4, [(1408, 3), (1600, 2), (1920, 1)]),
            (2, [(1600, 2), (1920, 1)]),
            (1, [(1920, 1)]),
        ]:
            on = evaluate(
                SIMPLE_MAPPING,
                True,
                N_EINSUMS=3,
                MAC_FANOUT=fanout,
                **SLOW_MM_THROUGHPUTS,
            )
            front = sorted(zip(on["Total<SEP>latency"], on[MAC_X] * fanout))
            self.assertEqual(front, expected)


class TestBusyTimeAcrossLevels(unittest.TestCase):
    def test_memory_bound_chain_of_four(self):
        """Four Einsums under three fused m loops of 2 iterations each, MainMemory
        reads at 0.1 bits per cycle and 32 MACs / 0.2 = 160 per Einsum. MainMemory
        writes each GlobalBuffer node again on every iteration of the loops above it:
        Matmul0's T0 tile (16 bits) and W0 (32 bits) on all 8, 384 bits; W1 and W2 on
        4 each, 128 bits each; W3 on 2, 64 bits. T4 has no GlobalBuffer node, so
        MainMemory reads back the first of each output's 2 partial sums to add the
        second, during Matmul3's compute: 128 bits. 832 bits take 8320.

        All sequential, Matmul0 and Matmul1's sub-branches take 3840 + 160 (Matmul0 is
        bound by its own reads); with Matmul2 the level needs MainMemory's
        3840 + 1280 + 1280 = 6400, more than 3840 + 160 + 1280 of sub-branches; and
        Matmul3's branch follows with 640 of W3 reads and 1280 of T4's:
        6400 + 1920 = 8320, MainMemory's whole busy time. In parallel each branch keeps
        its own MACs (160) and MainMemory's 8320 bounds every result, so only the
        single-instance result stays on the front."""
        jinja = {**FOUR_EINSUMS, "MM_READ_THROUGHPUT": 0.1, "COMPUTE_THROUGHPUT": 0.2}
        off = evaluate(FOUR_MATMULS_MAPPING, False, **jinja)
        self.assertEqual(list(off["Total<SEP>latency"]), [8320])

        on = evaluate(FOUR_MATMULS_MAPPING, True, **jinja)
        front = sorted(zip(on["Total<SEP>latency"], on[MAC_X] * 4))
        self.assertEqual(front, [(8320, 1)])

    def test_a_fenced_branch_slack_is_not_reused(self):
        """Matmul0 splits off above the inner m loop, and Matmul1 and Matmul2 split
        below it. GlobalBuffer reads take one bit-time and each Einsum computes for
        64 / 0.045 = 1422.2. Reads into compute happen during that Einsum's compute,
        so each branch holds its own GlobalBuffer reads: Matmul0 1408 and Matmul1 1408,
        both compute-bound at 1422.2; Matmul2 1536 (W2 and T2 reads, T3's accumulate
        and its drain), GlobalBuffer-bound. The Matmul1/Matmul2 level is
        1422.2 + 1536 = 2958.2, more than its 2944 of GlobalBuffer reads. Every
        GlobalBuffer node is below the outer m loop, so none of that traffic can fill
        Matmul0's 14.2 of slack: 1422.2 + 2958.2 = 4380.4. Letting it would give the
        GlobalBuffer's whole-chain busy time, 4352."""
        row = total_latency(
            LATENCY_ARCH,
            SHALLOW_THEN_DEEP_MAPPING,
            3,
            M_TILE=1,
            GB_READ_THROUGHPUT=1,
            COMPUTE_THROUGHPUT=0.045,
        )
        self.assertAlmostEqual(
            row["Total<SEP>latency"], 64 / 0.045 + 64 / 0.045 + 1536, places=2
        )

    def test_fenced_busy_time_still_bounds_the_total(self):
        """Four Einsums under three fused m loops of 2 iterations each (M = 8, KN = 2),
        GlobalBuffer reads at one bit-time, 32 MACs / 0.2 = 160 per Einsum. The
        GlobalBuffer reads each input once per MAC (256 bits) and re-reads each output
        it holds on 1 of every 2 MACs to accumulate (128): Matmul0 640 (T0, W0, T1),
        Matmul1 640 (T1, W1, T2), Matmul2 640 (T2, W2, T3) and Matmul3 512 (T3, W3;
        T4 goes straight to MainMemory), 2432 in all. These reads go to compute, so
        they count in each Einsum's own branch: every branch is GlobalBuffer-bound and
        the branches serialize into 640 + 640 + 640 + 512 = 2432, also the
        GlobalBuffer's whole busy time."""
        row = total_latency(
            LATENCY_ARCH,
            FOUR_MATMULS_MAPPING,
            4,
            M=8,
            KN=2,
            GB_READ_THROUGHPUT=1,
            COMPUTE_THROUGHPUT=0.2,
        )
        self.assertEqual(row["Total<SEP>latency"], 2432)

    def test_fenced_busy_time_bounds_every_parallel_result(self):
        """As test_fenced_busy_time_still_bounds_the_total, with the MAC X dimension
        allowing parallelism. Running Einsums in parallel cannot shorten the
        GlobalBuffer's 2432 of reads, so every result is 2432 and only the
        single-instance one stays on the front."""
        on = evaluate(
            FOUR_MATMULS_MAPPING,
            True,
            **FOUR_EINSUMS,
            GB_READ_THROUGHPUT=1,
            COMPUTE_THROUGHPUT=0.2,
        )
        front = sorted(zip(on["Total<SEP>latency"], on[MAC_X] * 4))
        self.assertEqual(front, [(2432, 1)])

    def test_output_written_from_compute_stays_in_its_branch(self):
        """Four Einsums (M = 8, KN = 2), 32 MACs / 0.2 = 160 each. MainMemory reads at 4
        bits per cycle and is written at 0.1. It writes Matmul0's T0 tile and W0 again on
        all 8 innermost iterations ((128 + 256) / 4 = 96), W1 and W2 on 4 each (32
        each) and W3 on 2 (16). T4 has no GlobalBuffer node: Matmul3's compute
        writes each of its 32 partial sums to MainMemory (256 bits / 0.1 = 2560), and
        MainMemory reads back the first of each output's two (128 / 4 = 32). Those
        transfers need Matmul3's compute, so they happen in Matmul3's branch:
        max(160, 16 + 2560 + 32) = 2608. The other three branches are compute-bound,
        their MainMemory reads fitting in their slack: 160 + 160 + 160 + 2608 = 3088,
        more than MainMemory's whole-chain busy time,
        96 + 32 + 32 + 16 + 2560 + 32 = 2768. Counting T4's traffic at MainMemory's
        level, above every split, lets it fill the other branches' slack: 2768."""
        row = total_latency(
            LATENCY_ARCH,
            FOUR_MATMULS_MAPPING,
            4,
            M=8,
            KN=2,
            MM_READ_THROUGHPUT=4,
            MM_WRITE_THROUGHPUT=0.1,
            COMPUTE_THROUGHPUT=0.2,
        )
        self.assertEqual(row["Total<SEP>latency"], 3088)
