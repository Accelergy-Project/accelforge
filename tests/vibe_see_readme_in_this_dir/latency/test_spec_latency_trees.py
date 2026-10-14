"""
Latency trees derived from the spec: the Latency Model docstring in
accelforge/frontend/model.py and the free_tree / free_to_loop_index /
sequential_fence_to_level / finish / merge_next docstrings in
accelforge/mapper/FFM/_join_pmappings/pmapping_dataframe.py. Every expected number is
derived by hand in the test docstring; the pictures are in the tree_pass/TREES.md notes.

Per-Einsum inputs, from the spec: with only COMPUTE_LATENCY = D set, an Einsum's own
delay (its descent at its own level) is D; the private MAC's steady state is computes per
instance / COMPUTE_THROUGHPUT; a tracked component's busy time (MainMemory, GlobalBuffer:
above the multi-Einsum fanout) counts at the level of the storage it talks to; fanout
usage is instances / MAC_FANOUT.

Join rules, from the spec: wind-ups and tracked busy times sum at every join; a parallel
join maxes the total-latency rights, a sequential fence adds the right into the left. At a
free, a wind-up at a level with a sequential split is multiplied by every loop above it
and paid at that level; under a parallel split it moves one level up and asks again; at
level 0 it is paid once. Tracked busy time is maxed into the total when freed or fenced.
Fanout usage sums across parallelized Einsums and maxes across sequentialized ones; other
reservations max across branches.

arms(...) returns {PEs used: total latency} of the kept front, so an arm is visible only
when it beats every arm with fewer PEs.
"""

import unittest

from tests.vibe_see_readme_in_this_dir.latency.test_spec_latency_helpers import (
    FOUR_MATMULS,
    FUSED,
    INPUT_FILES_DIR,
    PAIR_THEN_DEEP,
    PAIR_THEN_SHALLOW,
    UNFUSED_THEN_FUSED,
    WEIGHTS_INSIDE,
    arms,
    latency,
)

PAIR_THEN_SHALLOW_X4 = INPUT_FILES_DIR / "trees_pair_then_shallow.mapping.yaml"
TWO_PAIRS = INPUT_FILES_DIR / "trees_two_pairs.mapping.yaml"
UNFUSED_THEN_FUSED_X2 = INPUT_FILES_DIR / "trees_unfused_then_fused.mapping.yaml"

# Delay 10 per Einsum; 64 MACs (M = 4) -> steady state 4, 128 MACs (M = 8) -> 8.
DELAY = {"COMPUTE_LATENCY": 10, "COMPUTE_THROUGHPUT": 16}


class TestSpecLatencyTrees(unittest.TestCase):
    def test_pair_sequential_at_one_level(self):
        """
        Two Einsums below loop m (4 iterations), split sequential at level 1. Each
        Einsum: delay 10 at level 1, MAC steady state 4.

        Wind-ups sum at the join: 10 + 10 = 20 at level 1. The split at level 1 is
        sequential, so they are multiplied by loop 0: 20 x 4 = 80. Sequential branches'
        steady states sum: 4 + 4 = 8. Total 80 + 8 = 88.

        Wrong rule: steady states maxed across sequential branches -> 80 + 4 = 84.
        """
        self.assertEqual(latency(FUSED, **DELAY), 88)

    def test_pair_parallel_at_one_level(self):
        """
        The same pair with every sequential/parallel choice explored. Parallel at
        level 1: wind-ups still sum (20) but the split is parallel, so they move to
        level 0 and are paid once; the rights are maxed: max(4, 4) = 4. Total 24 on 2
        PEs (fanout usage 1/4 + 1/4). Sequential: 88 on 1 PE (max(1/4, 1/4)).

        Wrong rules: parallel wind-ups maxed -> 10 + 4 = 14; parallel wind-ups repaid
        per iteration of loop m -> 80 + 4 = 84.
        """
        self.assertEqual(arms(FUSED, **DELAY), {1: 88, 2: 24})

    def test_three_einsums_mixed_splits_at_one_level(self):
        """
        Three Einsums below loop m (4 iterations), two joins both at level 1. Wind-ups
        30 at level 1; multiplied by 4 = 120 whenever any join at the level is
        sequential, paid once when both are parallel.

        All sequential: 120 + 4 + 4 + 4 = 132 on 1 PE. One parallel join, (M0 || M1); M2
        or M0; (M1 || M2): 120 + max(4, 4) + 4 = 128 on max(2/4, 1/4) = 2 PEs. Both
        parallel: 30 + max(4, 4, 4) = 34 on 3 PEs.

        Wrong rule: a parallel pair's wind-ups paid once although the level has a
        sequential split -> 20 + 10 x 4 + 8 = 68 on 2 PEs.
        """
        self.assertEqual(arms(FUSED, 3, **DELAY), {1: 132, 2: 128, 3: 34})

    def test_four_einsums_mixed_splits_at_one_level(self):
        """
        Four Einsums below loop m (4 iterations), three joins at level 1. Wind-ups 40,
        x4 = 160 whenever a join is sequential.

        All sequential: 160 + 16 = 176 on 1 PE. (M0 || M1); (M2 || M3): 160 + 4 + 4 =
        168 on max(2, 2) = 2 PEs; it beats one parallel join (160 + 12 = 172, 2 PEs) and
        the 3-PE groups ((M0 || M1 || M2); M3 at 168). All parallel: 40 + 4 = 44 on 4
        PEs.

        Wrong rule: parallel steady states summed -> 176 on 2 PEs.
        """
        self.assertEqual(arms(FUSED, 4, **DELAY), {1: 176, 2: 168, 4: 44})

    def test_parallel_split_above_sequential_split_with_loop_between(self):
        """
        Loop 0 (m, 2 iterations); split at level 1; branch A holds loop 1 (m, 4
        iterations) and a pair split at level 2 (Matmul0 and Matmul1, 4 MAC instances
        each: steady state 32/16 = 2); branch B is Matmul2 (steady 8). Delays 10. Fanout
        16: usages 4/16, 4/16, 1/16.

        Arm (sequential at 2, parallel at 1): A's wind-ups 20 are multiplied by loops 1
        and 0: 20 x 4 x 2 = 160, paid at level 2; A's steady states sum: 2 + 2; A = 164.
        The parallel join at level 1 maxes the rights: max(164, 8) = 164, and Matmul2's
        delay 10 moves to level 0 and is paid once: 174 on max(4, 4) + 1 = 5 PEs.
        All sequential: 160 + 10 x 2 + 2 + 2 + 8 = 192 on 4 PEs.

        Wrong rules for the 5-PE arm: Matmul2's delay repaid per iteration of loop 0 ->
        184; A's total and Matmul2's steady state summed -> 182.
        """
        a = arms(PAIR_THEN_SHALLOW_X4, 3, fanout=16, M=8, **DELAY)
        self.assertEqual(a[5], 174)
        self.assertEqual(a[4], 192)

    def test_sequential_split_above_parallel_split_with_loop_between(self):
        """
        Same tree as above, arm (parallel at 2, sequential at 1). A's pair is parallel:
        its wind-ups 20 are not repaid for loop 1 and move to level 1; steady max(2, 2)
        = 2. Level 1 is sequential: the wind-ups now there, 20 + Matmul2's 10 = 30, are
        multiplied by loop 0 only: 60, paid at level 1. Steady states sum: 2 + 8. Total
        70 on max(4 + 4, 1) = 8 PEs. Both parallel: 30 + max(2, 8) = 38 on 9 PEs.

        Wrong rules for the 8-PE arm: the moved wind-ups also multiplied by loop 1 ->
        20 x 8 + 10 x 2 + 10 = 190; wind-ups paid once -> 40.
        """
        a = arms(PAIR_THEN_SHALLOW_X4, 3, fanout=16, M=8, **DELAY)
        self.assertEqual(a[8], 70)
        self.assertEqual(a[9], 38)

    def test_splits_at_three_levels(self):
        """
        Loop 0 (2 iterations); split at level 1: Matmul3 beside [loop 1 (2); split at
        level 2: Matmul2 beside [loop 2 (2); split at level 3: Matmul0, Matmul1]].
        Steady 8, delay 10, usage 1/8 each. A sequential level multiplies its wind-ups
        by every loop above it (level 3: 8, level 2: 4, level 1: 2); a parallel level
        passes them one level up.

        All sequential: 20 x 8 + 10 x 4 + 10 x 2 + 4 x 8 = 252 on 1 PE. Pair parallel,
        rest sequential: the pair's 20 move to level 2: (20 + 10) x 4 + 10 x 2 + max(8,
        8) + 8 + 8 = 164 on 2 PEs. Pair and level 2 parallel, level 1 sequential: (20 +
        10 + 10) x 2 + max(8, 8) + 8 = 96 on 3 PEs. All parallel: 40 + 8 = 48 on 4 PEs.

        Wrong rule for the 2-PE arm: the pair's wind-ups also multiplied by loop 2 ->
        160 + 40 + 20 + 24 = 244.
        """
        a = arms(FOUR_MATMULS, 4, fanout=8, M=8, **DELAY)
        self.assertEqual(a, {1: 252, 2: 164, 3: 96, 4: 48})

    def test_multi_instance_einsum_beside_single_instance_einsums(self):
        """
        The pair-then-shallow tree with Matmul0 on 4 MAC instances (steady 32/16 = 2,
        usage 4/8; its delay is not multiplied by instances), Matmul1 and Matmul2 on 1
        (steady 8, usage 1/8). Loops: 0 has 2 iterations, 1 has 4.

        All sequential: 20 x 8 + 10 x 2 + 2 + 8 + 8 = 198 on max(4, 1, 1) = 4 PEs. Pair
        parallel, level 1 sequential: (20 + 10) x 2 + max(2, 8) + 8 = 76 on max(4 + 1,
        1) = 5 PEs. All parallel: 30 + max(2, 8, 8) = 38 on 6 PEs. The (sequential,
        parallel) arm, max(170, 8) + 10 = 180 on 5 PEs, is beaten by 76.

        Wrong rules: usage summed across the sequential pair -> 198 on 5 PEs; the
        4-instance delay multiplied by 4 -> 318 on 4 PEs.
        """
        a = arms(PAIR_THEN_SHALLOW, 3, fanout=8, M=8, **DELAY)
        self.assertEqual(a, {4: 198, 5: 76, 6: 38})

    def test_two_pairs_at_one_level_under_separate_loops(self):
        """
        Loop 0 (2 iterations); split at level 1; branch A: loop 1 (4 iterations), pair
        Matmul0, Matmul1 at level 2 on 4 instances each (steady 32/8 = 4); branch B: its
        own loop 1 (4 iterations), pair Matmul2, Matmul3 at level 2 on 1 instance each
        (steady 16). Delays 10. Fanout 16.

        A sequential: 20 x 8 + 8 = 168; A parallel: 4, with 20 moving up. B sequential:
        160 + 32 = 192; B parallel: 16, with 20 moving up. Arms (A, level 1, B):
        (S,S,P): 168 + 16 + 20 x 2 = 224 on 4 PEs; (S,P,S): max(168, 192) = 192 on 5;
        (S,P,P): max(168, 16) + 20 = 188 on 6: B's parallel wind-ups pass level 2,
        where only the sibling branch A has a sequential split, and level 1 (parallel),
        and are paid once; (P,S,P): (4 + 40) + (16 + 40) = 100 on 8; (P,P,P): 16 + 40 =
        56 on 10.

        Wrong rule for the 6-PE arm: B's wind-ups multiplied by B's loops because A's
        split at the same level is sequential -> max(168, 16 + 160) = 176.
        """
        a = arms(TWO_PAIRS, 4, fanout=16, M=8, COMPUTE_THROUGHPUT=8, COMPUTE_LATENCY=10)
        self.assertEqual(a, {4: 224, 5: 192, 6: 188, 8: 100, 10: 56})

    def test_tracked_component_busy_in_several_branches(self):
        """
        Three Einsums below loop m (8 iterations) with their weights inside the
        branches, so every MainMemory action is exchanged with a GlobalBuffer at level
        1: Matmul0 1280 bits, Matmul1 1024, Matmul2 1280; at throughput 256 the
        MainMemory busy times are 5, 4, 5. MAC steady state 8 each. No wind-ups.

        All sequential: each fence maxes the branch's busy time into its total:
        max(8, 5) + max(8, 4) + max(8, 5) = 24 on 1 PE; the summed busy time 14 is
        below it. (M0 || M1); M2: max(8, 5 + 4) + 8 = 17 on 2 PEs. All parallel: the
        MainMemory busy time sums across all three branches and exceeds the maxed steady
        state: max(8, 14) = 14 on 3 PEs.

        Wrong rules for the 3-PE arm: busy time maxed across parallel branches -> 8;
        busy time added to the steady state -> 22.
        """
        a = arms(
            WEIGHTS_INSIDE,
            3,
            fanout=8,
            M=8,
            COMPUTE_THROUGHPUT=16,
            MM_READ_THROUGHPUT=256,
            MM_WRITE_THROUGHPUT=256,
        )
        self.assertEqual(a, {1: 24, 2: 17, 3: 14})

    def test_own_delay_and_wind_ups_above_it(self):
        """
        AMBIGUITY: 186 / 63 pinned (ruling c: each storage's actions at the storage's
        own level); reading model.py's "propagate data from reservations above this
        split to reservations below this split" as placing the MainMemory read of the T0
        tile and the MainMemory write of the T2 tile at level 1 (they recur per m
        iteration) gives Matmul0 25 and Matmul1 28 at level 1, 7 + 7 at level 0: (25 +
        28) x 4 + 14 = 226 sequential, 67 parallel.

        The fused pair below loop m (4 iterations). GlobalBuffer read 1, write 2;
        MainMemory read 4, write 8; compute 16; no steady state.

        Matmul0 at level 1: inputs T0 (GlobalBuffer parent-write 2 + child-read 1 = 3),
        W0 (no storage at level 1), T1's read-modify-write reads (1): slowest 3; one
        compute 16; output T1 (child-write 2): 3 + 16 + 2 = 21. Level 0: W0 pays
        MainMemory read 4 + GlobalBuffer write 2 + read 1 = 7, T0 pays MainMemory read
        4; slowest 7. Matmul1 at level 1: inputs T1 (read 1), T2's read-modify-write
        reads (1); compute 16; output T2 (GlobalBuffer write 2 + the read that sends it
        up 1 = 3): 20. Level 0: W1 7 (descent), T2's MainMemory write 8 (ascent).

        Sequential: level 1 (21 + 20) x 4 = 164; level 0 paid once 7 + 7 + 8 = 22; 186.
        Parallel: 41 + 22 = 63.

        Wrong rule: the slowest input and output maxed instead of summed -> Matmul0 19,
        Matmul1 19: 152 + 22 = 174 sequential.
        """
        a = arms(
            FUSED,
            GB_READ_LATENCY=1,
            GB_WRITE_LATENCY=2,
            MM_READ_LATENCY=4,
            MM_WRITE_LATENCY=8,
            COMPUTE_LATENCY=16,
        )
        self.assertEqual(a, {1: 186, 2: 63})

    def test_reservations_summed_across_parallel_branches(self):
        """
        RULING: reservations of components above the multi-Einsum parallel spatial are
        alive in every parallel branch at once, so a parallel join sums them; only
        components below it are maxed. Ctrl-F for
        ALWAYS_PARALLEL_BELOW_MULTI_EINSUM_PARALLEL_SPATIAL.

        The fused pair below loop m, GlobalBuffer sized to the bit. Per Einsum: W 128
        bits at level 0; W + a 32-bit row tile of T_i + one of T_i+1 = 192 bits at level
        1. Joined: level 0 holds both weights all run, 256. Sequential: each branch
        holds its 192 plus the other branch's weight 128, and the fence maxes the
        branches: 320. Parallel: both branches' row tiles are alive at once, 256 + 64 +
        64 = 384.

        GB_SIZE 320: only the sequential arm fits, {1: 80} (wind-ups 20 x 4). GB_SIZE
        384: both arms, {1: 80, 2: 20} (wind-ups 20 once parallel); 383 drops the
        parallel arm again. GB_SIZE 319: nothing fits and the join raises.

        Wrong rule: parallel branches maxed too, 320, so the 2-PE arm would fit at 320.
        """
        self.assertEqual(arms(FUSED, GB_SIZE=320, COMPUTE_LATENCY=10), {1: 80})
        self.assertEqual(arms(FUSED, GB_SIZE=384, COMPUTE_LATENCY=10), {1: 80, 2: 20})
        self.assertEqual(arms(FUSED, GB_SIZE=383, COMPUTE_LATENCY=10), {1: 80})
        with self.assertRaises(ValueError):
            arms(FUSED, GB_SIZE=319, COMPUTE_LATENCY=10)

    def test_split_above_every_loop_then_fused_pair(self):
        """
        Matmul0 with no GlobalBuffer joins at shared loop index -1 (split at level 0,
        above every loop; its own level is 0); Matmul1 and Matmul2 share loop m (8
        iterations) with T2 in GlobalBuffer at level 1, on 2 MAC instances each. Steady:
        8, 4, 4. Delays 10 at levels 0, 1, 1. Fanout 8: usages 1/8, 2/8, 2/8.

        (S,S): Matmul0's delay is at level 0 and paid once: 10 + 8 = 18; the pair 20 x 8
        + 4 + 4 = 168; 186 on 2 PEs. (S,P): 18 + 20 + max(4, 4) = 42 on 4 PEs. (P,S):
        Matmul0 parallel with the whole pair: max(8, 168) + Matmul0's 10 = 178 on 1 +
        2 = 3 PEs. (P,P): max(8, 4, 4) + 30 = 38 on 5 PEs.

        Wrong rules: Matmul0's level-0 delay multiplied by loop m -> 256 for (S,S); the
        parallel delay maxed into the pair's total instead of added -> 168 for (P,S).
        """
        a = arms(UNFUSED_THEN_FUSED_X2, 3, fanout=8, M=8, **DELAY)
        self.assertEqual(a, {2: 186, 3: 178, 4: 42, 5: 38})

    def test_pair_at_level_two_beside_pair_at_level_three(self):
        """
        Loop 0 (2 iterations); split at level 1; branch A: loop 1 (4), pair at level 2;
        branch B: loop 1 (2), W2 at level 2, loop 2 (2), pair at level 3. Steady 8,
        delay 10, usage 1/8 each. The joins happen at levels 2, 1, 3 in that order.

        A sequential: 20 x 8 + 16 = 176; parallel: 8 with 20 moving up. B sequential:
        20 x (2 x 2 x 2) + 16 = 176; parallel: 8 with 20 moving up. All sequential: 352
        on 1 PE. Both pairs parallel, level 1 sequential: (8 + 20 x 2) + (8 + 20 x 2) =
        96 on 2 PEs. All parallel: 8 + 40 = 48 on 4 PEs. The 3-PE arms, max(8, 176) +
        20 = 196, are beaten by 96.

        Wrong rule for the 2-PE arm: the moved wind-ups also multiplied by the loops
        below level 1 -> 176 + 176 = 352.
        """
        a = arms(PAIR_THEN_DEEP, 4, fanout=8, M=8, **DELAY)
        self.assertEqual(a, {1: 352, 2: 96, 4: 48})

    def test_tracked_component_busy_above_and_inside_branches(self):
        """
        The fused pair below loop m with weights at level 0. MainMemory at throughput
        64: Matmul0 fills T0 (256 bits) into a GlobalBuffer at level 1 -> 4 at level 1,
        and W0 (128) into a GlobalBuffer at level 0 -> 2 at level 0; upward inclusive 6
        at level 0. Matmul1 writes T2 (256) at level 1 -> 4, W1 -> 2; 6 at level 0. MAC
        steady state 128/16 = 8. No wind-ups.

        Sequential: max(8, 4) + max(8, 4) = 16; the MainMemory total 12 is below it:
        16 on 1 PE. Parallel: at level 1 max(8, 4 + 4) = 8; at level 0 the MainMemory
        total including the fills, 4 + 4 + 2 + 2 = 12, exceeds it: 12 on 2 PEs.

        Wrong rule: the level-0 fills not counted -> 8 on 2 PEs.
        """
        a = arms(
            FUSED,
            fanout=8,
            M=8,
            COMPUTE_THROUGHPUT=16,
            MM_READ_THROUGHPUT=64,
            MM_WRITE_THROUGHPUT=64,
        )
        self.assertEqual(a, {1: 16, 2: 12})

    def test_wind_ups_at_a_level_between(self):
        """
        The pair-then-shallow tree (4 instances for Matmul0 and Matmul1) with only
        GlobalBuffer read 1 and write 2; no steady state. Matmul0 at level 2: T0 and W0
        each write 2 + read 1 = 3, T1's reads 1; output T1 write 2: 5. Matmul1 at level
        2: T1 read 1: 1; its output T2 sits at level 1, above its own level, so T2 pays
        there: descent 1 (read-modify-write read) and ascent 2 (write). Matmul2 at
        level 1: T2 read 1, W2 3; T3 goes to MainMemory at zero latency: 3.

        Wind-ups: level 2: 5 + 1 = 6; level 1: 1 + 2 + 3 = 6. Loops: 0 has 2 iterations,
        1 has 4. (S,S): 6 x 8 + 6 x 2 = 60 on 4 PEs. (S,P): 48 + 6 once = 54 on 5 PEs.
        (P,S): (6 + 6) x 2 = 24 on 8 PEs. (P,P): 12 on 9 PEs.

        Wrong rule for the 5-PE arm: level-1 wind-ups repaid per iteration of loop 0
        under a parallel split -> 60.
        """
        a = arms(
            PAIR_THEN_SHALLOW_X4,
            3,
            fanout=16,
            M=8,
            GB_READ_LATENCY=1,
            GB_WRITE_LATENCY=2,
        )
        self.assertEqual(a, {4: 60, 5: 54, 8: 24, 9: 12})

    def test_tracked_busy_time_above_a_branch_steady_state(self):
        """
        AMBIGUITY: 76 pinned for the 1-PE arm; free_to_loop_index's "max(sum(per-branch
        steady state latency), per-resource cumulative latency)" with "per-branch steady
        state" read as the MAC alone gives max(24, 72) = 72, while model.py's "the
        steady-state latency is the max steady state of all resources used in that
        branch" and the fence rule (each component's busy time maxed into the total at
        the fence) give 76.

        Matmul0 with no GlobalBuffer, split at level 0, then Matmul1 and Matmul2 fused
        below loop m (8 iterations), single instances. Every Matmul0 value goes through
        MainMemory at level 0: T0 read per n1 (128 values), W0 read per m (128), T1
        read-modify-write (96 reads + 128 writes): 480 values = 3840 bits, 60 at
        throughput 64. Matmul1, Matmul2: 4 at level 1, 6 at level 0. MAC 8 each.

        (S,S): max(8, 60) + max(8, 4) + max(8, 4) = 76; the pair's MainMemory 12 is
        below 16. (S,P): 60 + max(8, 4 + 4) at level 1, then max(8, 12) at level 0 =
        72 on 2 PEs. (P,S): max(8, 16) = 16 against the MainMemory total 60 + 12 = 72:
        72 on 2 PEs.
        """
        a = arms(
            UNFUSED_THEN_FUSED,
            3,
            fanout=8,
            M=8,
            COMPUTE_THROUGHPUT=16,
            MM_READ_THROUGHPUT=64,
            MM_WRITE_THROUGHPUT=64,
        )
        self.assertEqual(a, {1: 76, 2: 72})


if __name__ == "__main__":
    unittest.main()
