"""
Passages of the latency spec (the Latency Model docstring in accelforge/frontend/model.py
and the free_tree / free_to_loop_index / sequential_fence_to_level / finish / merge_next
docstrings in accelforge/mapper/FFM/_join_pmappings/pmapping_dataframe.py) that can be
read two ways, or cases the text is silent on, where the readings give different
numbers. Each test builds the smallest tree that separates the readings, derives both
numbers by hand in its docstring, and pins the reading most consistent with the rest of
the text. The notes are in the ambiguity_pass/CANDIDATES.md and REPORT.md scratchpad
files.
"""

import unittest

import accelforge as af
from accelforge.frontend.spec import Spec
from accelforge.model.main import evaluate_mapping
from tests.vibe_see_readme_in_this_dir.latency.test_spec_latency_helpers import (
    FUSED,
    GB,
    INF,
    INPUT_FILES_DIR,
    LATENCY,
    LATENCY_ARCH,
    MAC_X_PARALLEL,
    SMALL,
    TWO_FUSED_LOOPS,
    arms,
    evaluate,
    latency,
)

LEVEL_WITHOUT_SPLIT = INPUT_FILES_DIR / "ambiguity_level_without_split.mapping.yaml"
WEIGHT_INSIDE_ONE_BRANCH = (
    INPUT_FILES_DIR / "ambiguity_weight_inside_one_branch.mapping.yaml"
)
NETWORKED_ARCH = INPUT_FILES_DIR / "networked_latency.arch.yaml"
NETWORKED_INPUT_ABOVE = INPUT_FILES_DIR / "ambiguity_networked_input_above.mapping.yaml"

# MainMemory 1 bit per unit on one port: 512 for the fused pair, 256 of it (T0's reads
# and T2's writes) inside the shared loop. Delay 10 per Einsum.
MM_BOUND_WITH_DELAY = {
    "MM_READ_THROUGHPUT": 1,
    "MM_WRITE_THROUGHPUT": 1,
    "COMPUTE_LATENCY": 10,
}


def evaluate_spec(mapping, n_einsums=2, arch=LATENCY_ARCH, n_instances=None, **jinja):
    """As the helpers' evaluate, with a choice of arch and Matmul0's n_instances."""
    spec = Spec.from_yaml(
        af.examples.workloads.basic.matmuls_any_einsums,
        arch,
        mapping,
        jinja_parse_data={
            "N_EINSUMS": n_einsums,
            **SMALL,
            "COMPUTE_THROUGHPUT": INF,
            **jinja,
        },
    )
    spec.model._share_resource_latency_above_splits = True
    spec.model._include_wind_up_latency = True
    if n_instances is not None:
        spec.workload.einsums["Matmul0"].n_instances = n_instances
    return evaluate_mapping(spec).data


class TestSpecLatencyAmbiguities(unittest.TestCase):
    def test_wind_up_at_a_level_with_no_split_between_loops(self):
        """
        AMBIGUITY: 4000 pinned. "Ultimately, ascent/descent latency is multiplied by
        (total # iterations of all loops above iff there is a sequential split above
        this component in the tree and below the loop)"; the free rule says what a
        sequential or a parallel split does with the wind-ups at its level and is silent
        on a level with no split. Reading A (pinned): a loop multiplies the wind-ups of
        a storage only when a sequential split lies between that loop and the storage,
        so a level with no split hands its wind-ups to the level above as a parallel
        split does, 4000. Reading B1 ("the loop above" is the nearest loop, with no
        split between it and the storage: paid once): 3800. Reading B2 (all loops above
        once any sequential split is above the storage): 5200.

        Loop 0 (m tile 8, 2 iterations); sequential split at level 1. Branch A: T2 in
        GlobalBuffer at level 1, loop 1 (m tile 2, 4 iterations), W0 and W1 in
        GlobalBuffer at level 2 with no split there, loop 2 (m tile 1, 2 iterations),
        Matmul0 and Matmul1 split sequentially at level 3 sharing T1. Branch B: T2 and
        W2 in GlobalBuffer at level 1, its own m loop, Matmul2. Only GlobalBuffer
        writes cost, 100.

        Matmul0's delay at level 3: T0's write 100 + compute 0 + T1's write 100 = 200.
        Matmul1's delay at level 3 is 0 (T1 is read for free); its output T2 sits at
        level 1 and pays its write there, an ascent of 100. Matmul2's delay at level 1:
        W2's write 100, T3 goes to MainMemory for free. Level 2: W0's write 100 and W1's
        write 100, 200 in all. Level 3 is sequential under three loops: 200 x 16 = 3200.
        Level 1 is sequential under loop 0: (100 + 100 + 200 moved up) x 2 = 800.
        Total 4000. B1: 3200 + 400 + 200 = 3800. B2: 3200 + 400 + 200 x 8 = 5200.
        """
        self.assertEqual(
            latency(LEVEL_WITHOUT_SPLIT, 3, M=16, GB_WRITE_LATENCY=100), 4000
        )

    def test_sequential_wind_ups_under_the_max_of_the_level_above(self):
        """
        AMBIGUITY: {1: 512} pinned. model.py: "The total latency of a branch is the sum
        of the following: Any ascent/descent latencies, if the branch has sub-branches;
        The max of the following: The total latency of each sub-branch ...; The max,
        of all resources used in the branch and sub-branches, the sum steady-state
        latency of the resource." free_to_loop_index: "multiply by the #iterations at
        every level above & add to the steady state latency at this level", and
        free_tree: "added into the level AFTER the max. Used to pay wind-up latency into
        the branch's steady state" (that level is then maxed at the level above).
        Reading A (pinned, pmapping_dataframe): the wind-ups are part of level 1's
        steady state and level 0 maxes it against MainMemory's busy time: sequential
        512, and the parallel arm, 532, is beaten: {1: 512}. Reading B (model.py, and
        "We do not allow pre-fetching between branches"): the wind-ups are added on top
        of the max: sequential 592, arms {1: 592, 2: 532}.

        The fused pair below loop m (4 iterations). MainMemory at 1 bit per unit: T0's
        reads 128 and T2's writes 128 at level 1, W0's and W1's reads 128 each at level
        0, 512 in all. MAC 128 each (COMPUTE_THROUGHPUT 0.5). Delays 10 at level 1.

        Sequential: level 1 is max(128, 128) + max(128, 128) + (10 + 10) x 4 = 336;
        level 0 is max(336, 512) = 512. Parallel: level 1 is max(128, 128, 128 + 128) =
        256; the wind-ups 20 move to level 0 and are paid once after its max: max(256,
        512) + 20 = 532 on 2 PEs, beaten by 512 on 1 PE. Under B the sequential arm is
        80 + max(256, 512) = 592, so both arms stay.
        """
        row = arms(FUSED, COMPUTE_THROUGHPUT=0.5, **MM_BOUND_WITH_DELAY)
        self.assertEqual(row, {1: 512})

    def test_parallel_wind_ups_moved_above_the_level_zero_max(self):
        """
        AMBIGUITY: {1: 592, 2: 532} pinned. free_to_loop_index TOTAL LATENCY: "Total
        latency for parallelized Einsums will be (sum ascent/descent latencies) +
        max(per-branch steady state, per-branch steady state latency, per-resource
        cumulative latency)." WINDUP Freeing: "If this split is parallel, then add to the
        ascent/descent latency at the level above", and free_tree: "added into the level
        AFTER the max". Reading A (pinned): the parallel pair's wind-ups move to level
        0 and are paid once after level 0's max, 532 on 2 PEs. Reading B (the
        parallelized Einsums' total at level 1 already holds their wind-ups, 20 + 256 =
        276, and level 0 maxes it against MainMemory): 512 on 2 PEs.

        The fused pair as in the test above with MAC 256 each (COMPUTE_THROUGHPUT 0.25)
        so the parallel arm survives. Sequential: 256 + 256 + 80 = 592 at level 1,
        above MainMemory's 512, so 592 under both readings. Parallel: level 1 is
        max(256, 256, 256) = 256; A: max(256, 512) + 20 = 532; B: max(276, 512) = 512.
        """
        row = arms(FUSED, COMPUTE_THROUGHPUT=0.25, **MM_BOUND_WITH_DELAY)
        self.assertEqual(row, {1: 592, 2: 532})

    def test_n_instances_repay_the_delay(self):
        """
        AMBIGUITY: {1: 132, 2: 38} pinned. Ruling (c): "The delay is paid once per
        switch into the branch." Einsum.n_instances: "Number of times to repeat the
        Einsum ... Energy, latency, and other summable metrics are multiplied by this
        value." The latency spec is silent on instances. Reading A (pinned): each
        repeat of the Einsum is a switch into it and the delay is a latency, so the
        wind-up is multiplied like the steady state. Reading B: one switch into the
        branch, only the steady state doubles: {1: 92, 2: 28}.

        The fused pair below loop m (4 iterations), Matmul0 with n_instances 2. Delay 10
        each; MAC 64 / 16 = 4 per instance, so Matmul0's steady state is 8 and its
        wind-up 20 (A) or 10 (B); Matmul1 4 and 10. Fanout 4, usage 1/4 each (the
        instances are not parallel: "non-persistent reservations are not" multiplied).

        A: sequential (20 + 10) x 4 + 8 + 4 = 132; parallel 30 + max(8, 4) = 38 on 2
        PEs. B: (10 + 10) x 4 + 12 = 92; 20 + 8 = 28.
        """
        data = evaluate_spec(
            FUSED,
            n_instances=2,
            COMPUTE_LATENCY=10,
            COMPUTE_THROUGHPUT=16,
            MAC_FANOUT=4,
            MAC_PARALLEL=True,
        )
        row = {round(u * 4): t for u, t in zip(data[MAC_X_PARALLEL], data[LATENCY])}
        self.assertEqual(row, {1: 132, 2: 38})

    def test_network_hops_paid_at_the_lower_storage_level(self):
        """
        AMBIGUITY: 8000 pinned. "Descent and ascent latency at a given level is how
        much time it takes to pass through all components at the level of the split.
        For example, if data passes through one buffer, it'll be the latency of one read
        + one write of the buffer." A network has no storage node, so no level of its
        own, and the text is silent on where a hop is paid. Reading A (pinned, from
        "data movement between A<->B occurs when the lower-in-the-tree reservation is
        ready"): a hop is paid at the level of the storage it fills, below the network.
        Reading B: at the level of the storage above the network: 7200.

        The networked arch (MainMemory, GlobalBuffer, PeArray, Scratchpad x4, MacArray,
        MAC x4), hop latency 100, everything else free. T0, W0 and W1 are held whole in
        GlobalBuffer above the shared m loop (4 iterations); each Einsum fills its
        Scratchpads inside its branch over the PeArray (4 hops) and the MAC over the
        MacArray (1 hop); T1 is exchanged in GlobalBuffer at level 1.

        A: every input reaches the Scratchpad and the MAC inside the branch, 400 + 100,
        and every output goes back the same way, 500: each delay is 1000, repaid on the
        4 iterations by the sequential split: 1000 x 4 x 2 = 8000. B: Matmul0's T0 and
        W0 cross the PeArray from level 0, 400 once; its delay is 100 + 500 = 600, x 4
        = 2400. Matmul1's W1 crosses from level 0, 400 once; T1 crosses from level 1 so
        its delay stays 1000, x 4 = 4000. Level 0: 400 + 400 = 800. 2400 + 4000 + 800 =
        7200.
        """
        data = evaluate_spec(
            NETWORKED_INPUT_ABOVE, arch=NETWORKED_ARCH, HOP_LATENCY=100
        )
        self.assertEqual(list(data[LATENCY]), [8000])

    def test_imperfect_loop_asserts_with_wind_up_latency(self):
        """
        RULING: imperfect loops are not supported with wind-up latency; the model
        asserts. Ctrl-F for WINDUP_LATENCY_PERFECT_ASSUMPTION. The wind-ups are
        multiplied by the product of the per-loop iteration counts, 2 x 4 = 8 here,
        while the split is entered 4 + 2 = 6 times: 160 against 120.

        Two fused m loops with M = 6: loop 0 with tile 4 (2 iterations, the second
        holding 2 rows), W0 and W1 in GlobalBuffer at level 1, loop 1 with tile 1 (4
        iterations, 2 inside the second outer iteration), the sequential split at level
        2. With wind-up latency off the same mapping evaluates, and nothing costs
        anything, so its latency is 0.
        """
        with self.assertRaises(AssertionError):
            latency(TWO_FUSED_LOOPS, M=6, M_TILE=4, COMPUTE_LATENCY=10)
        self.assertEqual(latency(TWO_FUSED_LOOPS, M=6, M_TILE=4, windup=False), 0)

    def test_join_adds_reservations_above_the_split_into_both_branches(self):
        """
        AMBIGUITY: 320 of 1000 pinned in both cases. free_to_loop_index RESERVATIONS
        Joining: "Don't touch left; Sum reservations on the right (shared between
        Einsums) with the next". The merge_next figure: the left's C becomes C+A2 and
        the right's E2 becomes E2+D, each side gaining the other side's reservation
        above the shared loop index. Reading A (pinned, the figure): a reservation
        above the split lives during every branch, so the dead left gains the next
        Einsum's level-0 reservation and the live right gains the left tree's. Reading B
        (the bullets): the left is untouched and the right sums only the same level:
        192 of 1000 in both cases, and GB_SIZE 192 would fit.

        The fused pair below loop m with one weight inside its own branch. INSIDE 0: W0
        in GlobalBuffer inside Matmul0's branch (192 bits with the T0 and T1 tiles), W1
        above the loop (128), Matmul1's branch 64. A: dead left 192 + 128 = 320, live
        right 64 + 128 = 192, max 320. INSIDE 1: W0 above the loop, W1 inside
        Matmul1's branch. A: left 192 (W0 is in its own column), right 192 + 128 = 320.
        B: 192 both. With GB_SIZE 319 nothing fits under A and the join raises.
        """
        for inside in (0, 1):
            with self.subTest(inside=inside):
                data = evaluate(WEIGHT_INSIDE_ONE_BRANCH, GB_SIZE=1000, INSIDE=inside)
                self.assertAlmostEqual(data[GB].iloc[0], 0.32, places=4)
                with self.assertRaises(ValueError):
                    evaluate(WEIGHT_INSIDE_ONE_BRANCH, GB_SIZE=319, INSIDE=inside)


if __name__ == "__main__":
    unittest.main()
