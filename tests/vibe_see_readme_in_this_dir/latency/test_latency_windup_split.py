"""
Descent and ascent latencies of a split shared by several Einsums: the time to move
data from reservations above the split to reservations below it, and back up. They are
summed across the Einsums in the split (also in parallel), added on top of the
branch's steady state, and repaid on every iteration of every loop above the split when
the split is sequential. When it is parallel they are added into the level above,
whose split decides; level 0 pays them once.

All tests use the matmul-chain workload on the latency test arch.
"""

import unittest

from tests.vibe_see_readme_in_this_dir.latency.test_latency import (
    TWO_FUSED_LOOPS_MAPPING,
)
from tests.vibe_see_readme_in_this_dir.latency.test_latency_windup_private import (
    ACTION_LATENCIES,
    INFINITELY_FAST_COMPUTE,
    SIMPLE_MAPPING,
)
from tests.vibe_see_readme_in_this_dir.latency.test_parallel_mappings import (
    MAC_X,
    SLOW_MM_THROUGHPUTS,
    evaluate,
)


class TestSplitWindup(unittest.TestCase):
    def test_descents_sum_across_three_einsums(self):
        """Three Einsums under one fused m loop (4 iterations, M = KN = 4). The Einsum
        delays are descents at level 1. Matmul0 writes T0 into the GlobalBuffer and
        reads it (20 + 10 = 30), one MAC (3), and writes T1 (20): 53. Matmul1 reads T1
        (10), one MAC, and writes T2 (20): 33. Matmul2 reads T2 (10), one MAC, and T3
        is written and read to MainMemory (20 + 10 = 30): 43. At level 0 a tensor
        pays all of its nodes and the slowest tensor sets the level: each Einsum's
        weight takes a MainMemory read and a GlobalBuffer write + read (130), and T3's
        MainMemory write is an ascent (200): 590, which nothing repays. Wind-ups sum
        at every join, sequential or parallel. If either join is sequential, the
        split repays the delays' 129 on each of the 4 iterations: 516 + 590 = 1106,
        and only the single-instance result of those stays on the front. All
        parallel, the delays are added into level 0 and paid once: 129 + 590 = 719 on
        three instances. Taking the slowest node at level 0 would give 1016 and
        629."""
        on = evaluate(
            SIMPLE_MAPPING,
            True,
            N_EINSUMS=3,
            **INFINITELY_FAST_COMPUTE,
            **ACTION_LATENCIES,
        )
        front = sorted(zip(on["Total<SEP>latency"], on[MAC_X] * 4))
        self.assertEqual(front, [(719, 3), (1106, 1)])

    def test_windup_adds_on_top_of_shared_slack(self):
        """MainMemory reads at 4 bits per cycle and takes 100 cycles per read. The
        steady state is that of test_above_split_latency_uses_branch_slack: the
        branches serialize into 320 + 1280 = 1600, over MainMemory's 1376, and in
        parallel MainMemory's 1376 sets it, over the MACs' 320 (one instance each).
        Every MainMemory node is at level 0, where each Einsum's slowest tensor takes
        one MainMemory read (100), a level-0 descent paid once (100 + 100), and the
        descents are added on top of the max: 1800 sequentially, 1576 in parallel.
        Counting each hop at its lower node's level would put T0's read in Matmul0's
        delay, repaid on the 4 m iterations: 2200 and 1676."""
        jinja = {**SLOW_MM_THROUGHPUTS, "MM_READ_LATENCY": 100}
        off = evaluate(SIMPLE_MAPPING, False, **jinja)
        self.assertEqual(list(off["Total<SEP>latency"]), [1800])

        on = evaluate(SIMPLE_MAPPING, True, **jinja)
        front = sorted(zip(on["Total<SEP>latency"], on[MAC_X] * 4))
        self.assertEqual(front, [(1576, 2), (1800, 1)])

    def test_repeated_parallel_branches_pay_windup_once(self):
        """Two fused m loops (M = 32, M_TILE = 4: 8 outer x 4 inner iterations) with the
        split below both, GlobalBuffer reads at 10 cycles and writes at 20,
        infinitely fast compute. The Einsum delays are descents at level 2: Matmul0
        writes T0 into the GlobalBuffer and reads it to compute (20 + 10) and writes
        T1 (20), 50, and Matmul1 reads its inputs (10) and writes T2 (20) and reads it
        to MainMemory (10), 40. Each weight's GlobalBuffer node between the loops
        (write + read, 30) is at level 1, which has no split. Sequentially the split
        repays the delays on every iteration of both loops, and the weights' 60 is
        added into level 0 and paid once: 90 x 32 + 60 = 2940. In parallel the
        branches are repeated without switching to a sequential split: the delays are
        added into level 1 and with the weights' nodes into level 0, which pays them
        once: 90 + 60 = 150, holding both MAC X instances. Counting each hop at its
        lower node's level would give 130 and 2920."""
        on = evaluate(
            TWO_FUSED_LOOPS_MAPPING,
            True,
            M=32,
            KN=4,
            M_TILE=4,
            GB_READ_LATENCY=10,
            GB_WRITE_LATENCY=20,
            **INFINITELY_FAST_COMPUTE,
        )
        front = sorted(zip(on["Total<SEP>latency"], on[MAC_X] * 4))
        self.assertEqual(front, [(150, 2), (2940, 1)])
