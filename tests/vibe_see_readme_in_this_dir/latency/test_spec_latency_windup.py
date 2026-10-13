"""
Wind-up and wind-down rules of the Latency Model (accelforge/frontend/model.py,
"Ascent and descent latencies" and "Aggregating latency") and the WINDUP LATENCY column
family (pmapping_dataframe.py), derived from the spec only. All throughputs are infinite
here, so the total latency is wind-up alone.

Rules exercised. Descent and ascent latency at a level is the time to pass through the
components at that level: one read + one write of every buffer the tensor passes
through there, summed per tensor, the slowest tensor setting the level, in the direction
the data moves (a write from the parent and a read to the child are descent, a write from the
child and a read to the parent are ascent; an output's read-modify-write reads are
descent and sit on the input side of the slowest-tensor max). So a storage node's actions
sit at the node's own level: MainMemory, above every loop, at level 0; a GlobalBuffer
tile inside the shared loop at level 1. Splits go highest in a slot, so every tile at a
level sits below that level's split. Ascent/descent is summed across the Einsums in
the split even if parallel, multiplied by the iterations of every loop above the
component that has a sequential split between the loop and the component, then added to
the steady state of the branch it is in. A private branch pays the slowest input's
propagation to compute, one compute, and the slowest output's propagation back, as the
descent at its number of fused loops. A parallel split hands its wind-ups to the level
above, whose split decides.

Shapes (M = KN = 4 unless noted):
  fused_matmuls_to_simple: W_i in GlobalBuffer above the shared m loop (level 0, 4
  iterations); T_i, T_{i+1} tiles in GlobalBuffer inside it (level 1); the sequential
  split is at level 1; loops n_i, n_{i+1} and compute are private below it.
  fused_matmuls_weights_inside: as above but W_i inside the loop too (level 1).
  unfused_matmuls_to_simple: every tensor in GlobalBuffer at level 0 of its own branch,
  m inside; no shared loops.
  fused_matmuls_two_fused_loops (M_TILE 2): m tile 2 (level 0, 2 iterations), W_i in
  GlobalBuffer at level 1, m tile 1 (level 1, 2 iterations), split at level 2.
  deep_then_shallow / shallow_then_deep (M = 16, KN = 4, M_TILE 4): outer m of 4
  iterations, inner m of 4 iterations; splits at levels 2 and 1 (deep) or 1 and 2
  (shallow).
  four_matmuls_three_loops (M = 8, KN = 2): three nested m loops of 2 iterations;
  splits at levels 3, 2 and 1.
"""

import unittest

from tests.vibe_see_readme_in_this_dir.latency.test_spec_latency_helpers import (
    DEEP,
    DEEP_THEN_SHALLOW,
    FOUR,
    FOUR_MATMULS,
    FUSED,
    PAIR_THEN_DEEP,
    SHALLOW_THEN_DEEP,
    TWO_FUSED_LOOPS,
    UNFUSED,
    WEIGHTS_INSIDE,
    arms,
    latency,
)

ONLY_MM_READ = {"MM_READ_LATENCY": 100}
ONLY_MM_WRITE = {"MM_WRITE_LATENCY": 100}
ONLY_COMPUTE = {"COMPUTE_LATENCY": 10}
ONLY_GB_READ = {"GB_READ_LATENCY": 100}
ONLY_GB_WRITE = {"GB_WRITE_LATENCY": 100}
# One power of ten per action so a total reads as a list of the actions it contains.
ALL_HOPS = {
    "MM_READ_LATENCY": 10000,
    "GB_WRITE_LATENCY": 1000,
    "GB_READ_LATENCY": 100,
    "COMPUTE_LATENCY": 10,
    "MM_WRITE_LATENCY": 1,
}
# GlobalBuffer actions are free, so only MainMemory's actions (level 0, paid once) and
# the one compute (the private branch's descent) carry latency.
MM_AND_COMPUTE = {"MM_READ_LATENCY": 1000, "COMPUTE_LATENCY": 10, "MM_WRITE_LATENCY": 1}
SEPARATE_PORTS = {
    "MM_SEPARATE_PORTS": True,
    "MM_READ_THROUGHPUT": 1,
    "MM_WRITE_THROUGHPUT": 1,
    "COMPUTE_THROUGHPUT": 0.32,
}


class TestPrivateBranch(unittest.TestCase):
    def test_path_hops_sum(self):
        """
        One Einsum, every tensor in GlobalBuffer at level 0, no splits. Slowest input
        T0 or W0: MainMemory read 10000 + GlobalBuffer write 1000 + GlobalBuffer read
        100 = 11100; one compute 10; output T1: GlobalBuffer write 1000 + read 100 +
        MainMemory write 1 = 1101. Total 12211, paid once. Summing the two inputs would
        give 23311.
        """
        self.assertEqual(latency(UNFUSED, 1, **ALL_HOPS), 12211)

    def test_slowest_tensor_not_sum(self):
        """
        T0 and W0 are both read from MainMemory at 100. The slowest tensor is 100; a
        sum over tensors would give 200.
        """
        self.assertEqual(latency(UNFUSED, 1, **ONLY_MM_READ), 100)

    def test_one_compute(self):
        """One compute of 10 is in the wind-up, once (not 64 times)."""
        self.assertEqual(latency(UNFUSED, 1, **ONLY_COMPUTE), 10)

    def test_wind_down_to_backing_storage(self):
        """T1's last write to MainMemory, 100, is the wind-down."""
        self.assertEqual(latency(UNFUSED, 1, **ONLY_MM_WRITE), 100)

    def test_paid_once_not_per_loop_iteration(self):
        """
        One Einsum of the fused shape: the m loop (4 iterations) is inside the only
        branch, so the compute of 10 is paid once when switching to the branch, not 40.
        """
        self.assertEqual(latency(FUSED, 1, **ONLY_COMPUTE), 10)

    def test_cross_level_descents(self):
        """
        AMBIGUITY: for one Einsum with no split, does the private-branch rule (slowest
        tensor from its highest branch-private reservation to compute, one compute,
        slowest output back; the 2026-10-08 ruling) or the per-level rule (each
        storage's actions at its own level, levels added) apply when the storages sit
        at different levels; asserting the private-branch rule. GlobalBuffer writes at
        100: W0's write at level 0 and T0's write at level 1 are both inputs, slowest
        100; T1's write from the compute 100. Total 200. Adding the levels would give
        100 + (100 + 100) = 300.
        """
        self.assertEqual(latency(FUSED, 1, **ONLY_GB_WRITE), 200)


class TestSequentialSplit(unittest.TestCase):
    def test_main_memory_reads_paid_once(self):
        """
        MainMemory's read to the child sits at MainMemory's level 0, above every loop
        and split, so it is paid once and the slowest tensor is taken per Einsum.
        Fused pair: Matmul0 max(T0, W0) 100 + Matmul1 W1 100 = 200; weights inside: the
        same 200; one fused Einsum: 100. Multiplying by the shared loop would give 800;
        summing the tensors 300.
        """
        for mapping, n, total in [
            (FUSED, 2, 200),
            (WEIGHTS_INSIDE, 2, 200),
            (FUSED, 1, 100),
        ]:
            with self.subTest(mapping=mapping.name, n=n):
                self.assertEqual(latency(mapping, n, **ONLY_MM_READ), total)

    def test_main_memory_write_paid_once(self):
        """
        T2's write from the child into MainMemory is an ascent at MainMemory's level
        0: paid once, 100, not 100 x 4.
        """
        self.assertEqual(latency(FUSED, 2, **ONLY_MM_WRITE), 100)

    def test_private_compute_repaid_as_descent(self):
        """
        Each Einsum's private branch below the split pays one compute of 10 as its
        descent at the split level; the split at level 1 repays it 4 times: (10 + 10) x
        4 = 80. Paid once it would be 20; a max across Einsums would be 40.
        """
        self.assertEqual(latency(FUSED, 2, **ONLY_COMPUTE), 80)

    def test_weights_inside_slowest_tensor(self):
        """
        W_i inside the shared loop, GlobalBuffer writes at 100. Level 1, x 4: Matmul0's
        T0 and W0 writes from the parent are both inputs, slowest 100, plus T1's write
        from the compute 100; Matmul1's W1 write 100 plus T2's write 100. (200 + 200) x
        4 = 1600. A sum over tensors would be (300 + 200) x 4 = 2000.
        """
        self.assertEqual(latency(WEIGHTS_INSIDE, 2, **ONLY_GB_WRITE), 1600)

    def test_global_buffer_writes_at_their_levels(self):
        """
        GlobalBuffer writes at 100. Level 0, once: W0's and W1's writes from
        MainMemory, 200. Level 1, x 4: Matmul0's T0 write from MainMemory 100 and T1's
        write from the compute 100; Matmul1's T2 write from the compute 100 (T1 is
        produced in place): 300 x 4 = 1200. Total 1400. Everything once would be 500.
        """
        self.assertEqual(latency(FUSED, 2, **ONLY_GB_WRITE), 1400)

    def test_global_buffer_reads_at_their_levels(self):
        """
        GlobalBuffer reads at 100. Level 0, once: W0's and W1's reads to the compute,
        200. Level 1, x 4: Matmul0's T0 read and T1's read-modify-write read are both
        inputs, slowest 100; Matmul1's T1 read and T2's read-modify-write read, 100;
        T2's read to MainMemory is an ascent, 100: 300 x 4 = 1200. Total 1400. Summing
        the inputs would give 200 + 500 x 4 = 2200.
        """
        self.assertEqual(latency(FUSED, 2, **ONLY_GB_READ), 1400)

    def test_all_actions_fused_pair(self):
        """
        Every action latency set, one power of ten each. A tensor pays all of its
        storages at a level and the slowest tensor sets the level. Level 0, once: W0 =
        MainMemory read 10000 + GlobalBuffer write 1000 + read 100 = 11100 beats T0's
        MainMemory read 10000; W1 11100; T2's MainMemory write 1: 22201. Level 1, x 4:
        Matmul0 slowest input T0 (GlobalBuffer write 1000 + read 100 = 1100, over T1's
        read-modify-write read 100) + compute 10 + T1's write from the compute 1000 =
        2110; Matmul1 slowest input 100 (T1's read, T2's read-modify-write read) + 10 +
        T2's write 1000 and read to MainMemory 100 = 1210: 3320 x 4 = 13280. Total
        35481. Taking W0's slowest storage instead of their sum would give 33281.
        """
        self.assertEqual(latency(FUSED, 2, **ALL_HOPS), 35481)

    def test_windup_joins_steady_state_at_its_level(self):
        """
        MainMemory at 1 bit per unit on one port. Inside the split: Matmul0 T0 reads
        128, Matmul1 T2 writes 128, plus repaid computes 80 -> 336 at level 1. Level 0
        maxes that against the whole MainMemory time 128 + 128 + 128 + 128 = 512: total
        512. Adding the wind-up above the max would give 592.
        """
        row = latency(
            FUSED, 2, MM_READ_THROUGHPUT=1, MM_WRITE_THROUGHPUT=1, **ONLY_COMPUTE
        )
        self.assertEqual(row, 512)

    def test_level_zero_windup_added_after_the_max(self):
        """
        As above with MainMemory reads at 100 latency: level 1 is 128 + 128 + 80 =
        336, maxed against MainMemory 512 -> 512, then the level-0 reads (Matmul0
        max(T0, W0) 100 + Matmul1 W1 100 = 200) are added once: 712. Adding them before
        the max would give 536.
        """
        row = latency(
            FUSED,
            2,
            MM_READ_THROUGHPUT=1,
            MM_WRITE_THROUGHPUT=1,
            **ONLY_COMPUTE,
            **ONLY_MM_READ,
        )
        self.assertEqual(row, 712)

    def test_two_fused_loops_every_loop_above(self):
        """
        Split at level 2 under m tile 2 (2 iterations) and m tile 1 (2 iterations).
        The computes sit at the split level and are repaid x 2 x 2 = 4: (10 + 10) x 4 =
        80. MainMemory's reads (Matmul0 max(T0, W0) 1000, Matmul1 W1 1000) and T2's
        write 1 sit at level 0, once: 2001. Total 2081. Using only the nearest loop
        gives 2041.
        """
        row = latency(TWO_FUSED_LOOPS, 2, M_TILE=2, **MM_AND_COMPUTE)
        self.assertEqual(row, 2081)


class TestFlags(unittest.TestCase):
    def test_windup_flag_off(self):
        """With _include_wind_up_latency off no action latency counts: 0, and 256 with
        128 of compute per Einsum."""
        self.assertEqual(latency(FUSED, 2, windup=False, **ALL_HOPS), 0)
        row = latency(FUSED, 2, windup=False, COMPUTE_THROUGHPUT=0.5, **ALL_HOPS)
        self.assertEqual(row, 256)

    def test_four_flag_combinations(self):
        """
        Separate ports at 1 bit per unit, MAC 200, compute latency 10. Steady state
        with slack 400, without 456 (see test_spec_latency_steady_state). Wind-up (10 +
        10) x 4 = 80 is added at level 1 before the level-0 max against the read port's
        384, so it survives: on/on 480, off/on 536, on/off 400, off/off 456.
        """
        expected = {(True, True): 480, (False, True): 536}
        expected |= {(True, False): 400, (False, False): 456}
        for (share, windup), total in expected.items():
            with self.subTest(share=share, windup=windup):
                row = latency(
                    FUSED,
                    2,
                    share=share,
                    windup=windup,
                    **SEPARATE_PORTS,
                    **ONLY_COMPUTE,
                )
                self.assertEqual(row, total)


class TestDeeperChains(unittest.TestCase):
    def test_deep_then_shallow(self):
        """
        MainMemory, level 0, once: Matmul0 max(T0, W0) 1000, Matmul1 W1 1000, Matmul2
        W2 1000, T3's write 1: 3001. Computes: Matmul0 and Matmul1 at the level-2
        split under two m loops, 10 x 16 each; Matmul2 at the level-1 split under one,
        10 x 4: 360. Total 3361.
        """
        row = latency(DEEP_THEN_SHALLOW, 3, **DEEP, **MM_AND_COMPUTE)
        self.assertEqual(row, 3361)

    def test_deep_then_shallow_gb_writes(self):
        """
        GlobalBuffer writes at 100. Splits go highest in a slot, so every tile at a
        level sits below that level's split and the multiplier is the iterations of
        every loop above the split. Level 1 (x 4): W0, W1 and T2's write from Matmul1's
        compute in the Matmul0 Matmul1 branch, W2 and T3's write from the compute in
        the Matmul2 branch: 500 x 4 = 2000. Level 2 (x 16): T0's write and T1's write
        from the compute: 200 x 16 = 3200. Total 5200. Paying T2 once would give 4900.
        """
        row = latency(DEEP_THEN_SHALLOW, 3, **DEEP, **ONLY_GB_WRITE)
        self.assertEqual(row, 5200)

    def test_shallow_then_deep(self):
        """
        MainMemory, level 0, once: Matmul0 max(T0, W0) 1000, Matmul1 W1 1000, Matmul2
        W2 1000, T3's write 1: 3001. Computes: Matmul0 at the level-1 split 10 x 4;
        Matmul1 and Matmul2 at the level-2 split 10 x 16 each: 360. Total 3361.
        """
        row = latency(SHALLOW_THEN_DEEP, 3, **DEEP, **MM_AND_COMPUTE)
        self.assertEqual(row, 3361)

    def test_shallow_then_deep_slowest_tensor(self):
        """
        MainMemory reads only, level 0, once: Matmul0's T0 and W0 are both read,
        slowest 100, plus W1 and W2: 300. Summing T0 and W0 would give 400.
        """
        row = latency(SHALLOW_THEN_DEEP, 3, **DEEP, **ONLY_MM_READ)
        self.assertEqual(row, 300)

    def test_four_einsums_three_split_levels(self):
        """
        MainMemory, level 0, once: Matmul0 max(T0, W0) 1000, W1 1000, W2 1000, Matmul3
        max(W3, T4's read-modify-write read) 1000, T4's write 1: 4001. Computes:
        Matmul0 and Matmul1 under three loops 10 x 8 each, Matmul2 under two 10 x 4,
        Matmul3 under one 10 x 2: 220. Total 4001 + 220 = 4221.
        """
        row = latency(FOUR_MATMULS, 4, **FOUR, **MM_AND_COMPUTE)
        self.assertEqual(row, 4221)

    def test_pair_then_deep(self):
        """
        Compute 10 and MainMemory write 1 only. Matmul0 | Matmul1 at level 2 under m
        loops of 2 and 4 iterations: 20 x 8 = 160. Matmul2 | Matmul3 at level 3 under
        three loops of 2: 20 x 8 = 160. T4's write into MainMemory at level 0: 1.
        Total 321.
        """
        row = latency(PAIR_THEN_DEEP, 4, **FOUR, COMPUTE_LATENCY=10, MM_WRITE_LATENCY=1)
        self.assertEqual(row, 321)


class TestUnfusedJoin(unittest.TestCase):
    def test_each_einsum_once(self):
        """
        No shared loops: the split at level 0 has no loop above it, so each Einsum's
        private wind-up of 12211 (test_path_hops_sum) is paid once: 24422.
        """
        self.assertEqual(latency(UNFUSED, 2, **ALL_HOPS), 24422)

    def test_parallel_arm_sums_windups(self):
        """
        64 MACs at 1 per unit. Sequential: 64 + 64 + 24422 = 24550 on one PE. Parallel:
        wind-ups still sum (pessimism) over max(64, 64): 24486 on two PEs.
        """
        row = arms(UNFUSED, 2, COMPUTE_THROUGHPUT=1, **ALL_HOPS)
        self.assertEqual(row, {1: 24550, 2: 24486})


if __name__ == "__main__":
    unittest.main()
