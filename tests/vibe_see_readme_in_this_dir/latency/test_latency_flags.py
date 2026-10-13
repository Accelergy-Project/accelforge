"""
The two latency flags. With _share_resource_latency_above_splits off, resource slack
cannot propagate up above a split: each Einsum pays the steady state of every resource
it uses at its compute node. With _include_wind_up_latency off, wind-up and wind-down
latencies are not included in the total latency.

All tests use the fused matmul chain on the latency test arch, two Einsums with
M = KN = 4: 64 MACs per Einsum and 128 bits per tensor.
"""

import unittest

from accelforge.model.main import evaluate_mapping
from tests.vibe_see_readme_in_this_dir.latency.test_latency import (
    LATENCY_ARCH,
    make_spec,
)
from tests.vibe_see_readme_in_this_dir.latency.test_latency_windup_private import (
    ACTION_LATENCIES,
    SIMPLE_MAPPING,
)
from tests.vibe_see_readme_in_this_dir.latency.test_parallel_mappings import (
    MAC_X,
    SLOW_MM_THROUGHPUTS,
)


def evaluate_with_flags(mapping, share, windup, n_einsums=2, **jinja):
    spec = make_spec(LATENCY_ARCH, mapping, n_einsums, **jinja)
    spec.model._share_resource_latency_above_splits = share
    spec.model._include_wind_up_latency = windup
    return evaluate_mapping(spec).data


class TestShareFlag(unittest.TestCase):
    def test_share_off_pays_steady_state_at_compute(self):
        """Matmul0 computes for 64 MACs / 0.2 = 320 and reads T0 and W0 from
        MainMemory (256 bits at one per cycle); Matmul1 computes for 320, reads W1
        (128) and writes T2 back (128 bits / 0.1 = 1280). T0's and T2's GlobalBuffer
        node is below the fused m loop and the weights' above it. Sharing, the
        branches serialize into max(320, 128) + max(320, 1280) = 1600 and MainMemory's
        whole busy time, 256 + 1408 = 1664, bounds the total
        (test_fused_memory_bound_overlaps_compute). Not sharing, each Einsum pays all
        of its steady state at its compute node: max(320, 256) + max(320, 1408) =
        1728."""
        jinja = {
            "MM_READ_THROUGHPUT": 1,
            "MM_WRITE_THROUGHPUT": 0.1,
            "COMPUTE_THROUGHPUT": 0.2,
        }
        for share, expected in [(False, 1728), (True, 1664)]:
            data = evaluate_with_flags(SIMPLE_MAPPING, share, True, **jinja)
            self.assertEqual(list(data["Total<SEP>latency"]), [expected])

    def test_share_off_parallel_takes_the_slower_branch(self):
        """MainMemory reads at 4 bits per cycle and is written at 0.1, and each
        Einsum computes for 320. Not sharing, nothing is tracked per level, so each
        Einsum's whole steady state is its right: Matmul0 max(320, 32 + 32) = 320 and
        Matmul1 max(320, 32 + 1280) = 1312. Sequentially they add: 1632 on one MAC X
        instance. The parallel join takes the slower right: 1312 on two, although
        MainMemory is busy for 64 + 1312 = 1376 over both branches. Summing the
        rights would tie the sequential 1632."""
        on = evaluate_with_flags(
            SIMPLE_MAPPING,
            False,
            True,
            MAC_FANOUT=4,
            MAC_PARALLEL=True,
            **SLOW_MM_THROUGHPUTS,
        )
        front = sorted(zip(on["Total<SEP>latency"], on[MAC_X] * 4))
        self.assertEqual(front, [(1312, 2), (1632, 1)])


class TestWindupFlag(unittest.TestCase):
    def test_four_flag_combinations(self):
        """MainMemory reads at 4 bits per cycle and takes 100 cycles per read; nothing
        else has a latency. Matmul0 computes for 320 and reads T0 (32, held below
        the fused m loop) and W0 (32, held above it); Matmul1 computes for 320,
        reads W1 (32, above) and writes T2 back (1280, below). Every MainMemory node
        is at level 0, where each Einsum's slowest tensor takes one MainMemory read
        (100), a level-0 descent paid once, 200 across the Einsums; the GlobalBuffer
        and the MACs have no latency, so the Einsum delays are 0.

        - Sharing, with wind-ups: max(320 + 1280, 32 + 32 + 32 + 1280) + 200 = 1800.
        - Sharing, without wind-ups: 1600.
        - Not sharing, with wind-ups: max(320, 64) + max(320, 1312) + 200 = 1832.
        - Not sharing, without wind-ups: 1632.

        Counting each hop at its lower node's level would put T0's read in Matmul0's
        delay, repaid on the 4 m iterations: 2200 and 2232."""
        jinja = {**SLOW_MM_THROUGHPUTS, "MM_READ_LATENCY": 100}
        for share, windup, expected in [
            (True, True, 1800),
            (True, False, 1600),
            (False, True, 1832),
            (False, False, 1632),
        ]:
            data = evaluate_with_flags(SIMPLE_MAPPING, share, windup, **jinja)
            self.assertEqual(list(data["Total<SEP>latency"]), [expected])

    def test_windup_off_leaves_windup_out(self):
        """With infinitely fast memories each Einsum computes for 64, so sequentially
        64 + 64 = 128. Including wind-up latency, the action latencies would add 844
        of wind-ups: 972 (test_fused_repays_communication_latency_each_switch); with
        it off none of that is paid."""
        data = evaluate_with_flags(SIMPLE_MAPPING, True, False, **ACTION_LATENCIES)
        self.assertEqual(list(data["Total<SEP>latency"]), [128])

    def test_windup_off_with_a_parallel_dimension(self):
        """As test_windup_off_leaves_windup_out, with the MAC X dimension allowing
        parallelism. Wind-up latency is off, so no result includes it: 64 + 64 = 128
        sequentially on one instance, and the parallel join takes the slower right,
        64 on two. With the wind-ups the results would be 972 and 620."""
        on = evaluate_with_flags(
            SIMPLE_MAPPING,
            True,
            False,
            MAC_FANOUT=4,
            MAC_PARALLEL=True,
            **ACTION_LATENCIES,
        )
        front = sorted(zip(on["Total<SEP>latency"], on[MAC_X] * 4))
        self.assertEqual(front, [(64, 2), (128, 1)])
