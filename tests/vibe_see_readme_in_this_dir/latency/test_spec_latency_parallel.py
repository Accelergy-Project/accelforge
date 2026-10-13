"""
The parallel arm of a join, derived from the spec only. With _allow_multi_einsum_
parallelism on the MAC's X dimension, each join explores a sequential and a parallel
arm; the parallel arm splits the fanout between the Einsums (usage sums and must fit).
The total for parallelized Einsums is the sum of their ascent/descent latencies plus
the max of per-branch steady states and per-resource cumulative latency. The MAC sits at
the fanout, so it is never shared and its steady states max across parallel branches;
MainMemory and GlobalBuffer are above it and sum. Parallelized branches' totals max,
also across levels; only fenced sequential branches add. A branch fenced at the join's
level is dead: a later parallel join there sums the live branches' fanout usage only; a
branch fenced deeper is parallel with the new Einsum, so its usage counts. A parallel
split hands its wind-ups to the level above, whose split decides when they are paid.

Each test reads the kept arms as {PEs used: total latency}: 1 PE is the sequential
arm, 2 PEs the parallel one, 3 PEs three Einsums all in parallel.
"""

import unittest

from tests.vibe_see_readme_in_this_dir.latency.test_spec_latency_helpers import (
    DEEP,
    DEEP_THEN_SHALLOW,
    FUSED,
    GB_ONLY_FIRST,
    PAIR_THEN_SHALLOW,
    UNFUSED_THEN_FUSED,
    arms,
)

# MainMemory 1 bit per unit on one port: 256 per Einsum. 64 MACs take 320.
MM_BOUND_PAIR = {
    "MM_READ_THROUGHPUT": 1,
    "MM_WRITE_THROUGHPUT": 1,
    "COMPUTE_THROUGHPUT": 0.2,
}
MM_AND_COMPUTE = {"MM_READ_LATENCY": 1000, "COMPUTE_LATENCY": 10, "MM_WRITE_LATENCY": 1}


class TestParallelSteadyState(unittest.TestCase):
    def test_branch_steady_states_max(self):
        """
        128 of compute per Einsum. Sequential: 256 on one PE. Parallel: each Einsum has
        its own MAC instance, max(128, 128) = 128 on two PEs.
        """
        self.assertEqual(arms(FUSED, 2, COMPUTE_THROUGHPUT=0.5), {1: 256, 2: 128})

    def test_shared_resource_above_fanout_sums(self):
        """
        Sequential: max(320 + 320, MainMemory 512) = 640. Parallel: MainMemory is above
        the fanout so its 256 + 256 = 512 sums, against max(320, 320): 512. Maxing
        MainMemory too would give 320.
        """
        self.assertEqual(arms(FUSED, 2, **MM_BOUND_PAIR), {1: 640, 2: 512})

    def test_share_flag_off_pays_at_compute_node(self):
        """
        RULING: with _share_resource_latency_above_splits off every resource is paid
        at the Einsum's compute node, so parallel branches max it.
        Matmul0 max(MainMemory 256, MAC 320) = 320, Matmul1 the same: parallel max(320,
        320) = 320, sequential 640. Summing MainMemory across the branches would give
        512.
        """
        row = arms(FUSED, 2, share=False, **MM_BOUND_PAIR)
        self.assertEqual(row, {1: 640, 2: 320})

    def test_fanout_limits_the_arms(self):
        """
        Two Einsums need two PEs to run in parallel. Fanout 1 keeps only the
        sequential arm; fanout 2 keeps both, with the parallel arm using the whole
        fanout.
        """
        self.assertEqual(arms(FUSED, 2, fanout=1, COMPUTE_THROUGHPUT=0.5), {1: 256})
        row = arms(FUSED, 2, fanout=2, COMPUTE_THROUGHPUT=0.5)
        self.assertEqual(row, {1: 256, 2: 128})

    def test_component_used_by_one_einsum(self):
        """
        GlobalBuffer serves only Matmul0: 2304 against MAC 640. Sequential 2304 + 640 =
        2944. Parallel: max(2304, 640) with GlobalBuffer's sum being Matmul0's alone:
        2304.
        """
        row = arms(
            GB_ONLY_FIRST,
            2,
            GB_READ_THROUGHPUT=1,
            GB_WRITE_THROUGHPUT=1,
            COMPUTE_THROUGHPUT=0.1,
        )
        self.assertEqual(row, {1: 2944, 2: 2304})

    def test_three_einsums(self):
        """
        128 of compute each. All sequential 384 on one PE; two parallel and one
        sequential 128 + 128 = 256 on two PEs; all parallel 128 on three PEs.
        """
        row = arms(FUSED, 3, COMPUTE_THROUGHPUT=0.5)
        self.assertEqual(row, {1: 384, 2: 256, 3: 128})


class TestParallelAcrossLevels(unittest.TestCase):
    def test_parallel_branch_maxes_across_levels(self):
        """
        Matmul0 reads T0, W0 and writes T1 straight to MainMemory, 1920 bits at 1 per
        unit; Matmul1 and Matmul2 share loop m and move 256 bits each. With the share
        flag off each is paid at its compute node. Matmul0 || Matmul1 then Matmul2
        after Matmul1: the completed parallel branch maxes with the deeper sequential
        pair, max(1920, 256 + 256) = 1920 on two PEs. Summing across levels would give
        2432 and leave Matmul0 -> (Matmul1 || Matmul2) = 2176 as the two-PE arm. All
        sequential 2432; three PEs also 1920 and dominated. With the flag on
        MainMemory's 2432 is cumulative in every arm, so only the one-PE arm survives.
        """
        jinja = {"MM_READ_THROUGHPUT": 1, "MM_WRITE_THROUGHPUT": 1}
        row = arms(UNFUSED_THEN_FUSED, 3, share=False, **jinja)
        self.assertEqual(row, {1: 2432, 2: 1920})
        self.assertEqual(arms(UNFUSED_THEN_FUSED, 3, share=True, **jinja), {1: 2432})

    def test_dead_branches_stay_out(self):
        """
        Four fused Einsums of 128 each. Matmul0 || Matmul1, then Matmul2 alone, then
        Matmul3 || Matmul2: the fenced pair is dead, so the last join sums only the
        live branches' usage, two PEs, for 128 + 128 = 256. Front {1: 512, 2: 256, 4:
        128}; the three-PE arms are 256 too and dominated. Counting the dead pair as
        well would need three PEs and leave 384 as the two-PE arm.
        """
        row = arms(FUSED, 4, COMPUTE_THROUGHPUT=0.5)
        self.assertEqual(row, {1: 512, 2: 256, 4: 128})

    def test_deeper_fenced_branch_is_parallel_with_a_later_join(self):
        """
        Matmul0 on all four MAC instances, Matmul1 after it below both m loops, then
        Matmul2 beside the pair below the outer loop only: Matmul2 is parallel with
        both. 16 MACs per instance for Matmul0: 32; Matmul1 and Matmul2 64 each: 128.
        Freeing the pair to level 1 maxes the fenced Matmul0 with the live Matmul1 into
        the right leaf, 4 PEs, and the parallel join adds Matmul2's: 5 PEs. Fanout 4
        keeps only all-sequential, 32 + 128 + 128 = 288 on 4 PEs; fanout 8 adds
        max(32 + 128, 128) = 160 on 5 PEs and everything parallel, 128 on 6 PEs.
        Leaving the fenced Matmul0 on the left would give 160 on 4 PEs at both fanouts.
        """
        row = arms(PAIR_THEN_SHALLOW, 3, fanout=4, COMPUTE_THROUGHPUT=0.5)
        self.assertEqual(row, {4: 288})
        row = arms(PAIR_THEN_SHALLOW, 3, fanout=8, COMPUTE_THROUGHPUT=0.5)
        self.assertEqual(row, {4: 288, 5: 160, 6: 128})


class TestParallelWindup(unittest.TestCase):
    def test_paid_once_across_repetitions(self):
        """
        GlobalBuffer writes at 100. Sequential: W0, W1 fills 200 once + the level-1
        writes (Matmul0's T0 fill and T1 output, Matmul1's T2 output: 300) x 4 = 1400.
        Parallel: the split at level 1 is parallel, so its 300 joins the fills at level
        0 and is paid once: 500.
        """
        self.assertEqual(arms(FUSED, 2, GB_WRITE_LATENCY=100), {1: 1400, 2: 500})

    def test_windups_summed_across_parallel_einsums(self):
        """
        One compute of 10 per Einsum. Sequential (10 + 10) x 4 = 80. Parallel: summed,
        not maxed, and paid once: 20. A max would give 10.
        """
        self.assertEqual(arms(FUSED, 2, COMPUTE_LATENCY=10), {1: 80, 2: 20})

    def test_main_memory_write_in_both_arms(self):
        """
        T2's write into MainMemory sits at MainMemory's level 0, so both arms pay it
        once: 100. The parallel arm then costs the same on more PEs and is dominated,
        leaving the sequential arm alone.
        """
        self.assertEqual(arms(FUSED, 2, MM_WRITE_LATENCY=100), {1: 100})

    def test_deep_then_shallow_arms(self):
        """
        MainMemory's reads and T3's write sit at level 0 in every arm: 3001. All
        sequential: computes 10 x 16 + 10 x 16 + 10 x 4 = 360 -> 3361. Matmul0 ||
        Matmul1 then Matmul2 sequential: level 2's 20 moves up to level 1 and is repaid
        x 4 with Matmul2's 10: 120 -> 3121 on two PEs (the other two-PE arm, sequential
        at level 2 and parallel at level 1, is 320 + 10 -> 3331 and is dominated).
        Everything parallel: 30 once -> 3031 on three PEs.
        """
        row = arms(DEEP_THEN_SHALLOW, 3, **DEEP, **MM_AND_COMPUTE)
        self.assertEqual(row, {1: 3361, 2: 3121, 3: 3031})


if __name__ == "__main__":
    unittest.main()
