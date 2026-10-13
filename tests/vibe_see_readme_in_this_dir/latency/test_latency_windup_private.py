"""
Wind-up and wind-down latency of a branch private to one Einsum: the time for the
slowest tensor to propagate from its highest branch-private reservation to compute, one
compute, and the time for the slowest output to propagate back up. Each storage node on
a tensor's path pays one read and one write at its own level, in the direction the data
moves: a write from the parent and a read to the child are descent, a write from the
child and a read to the parent are ascent, so an output's read-modify-write reads count
with the inputs. At a level above the fused loops a tensor pays all of its nodes there,
and the slowest tensor sets the level. The Einsum delay is a descent at the Einsum's
own level, paid each time its branch is switched to: a sequential split there repays
it on every iteration of every loop above it, and a parallel split or no split adds it
into the level above, whose split decides.

All tests use the matmul-chain workload on the latency test arch with infinitely fast
compute, so there is no steady state to compare the wind-up against (except where
noted).
"""

import unittest

import accelforge as af
from tests.vibe_see_readme_in_this_dir.latency.test_latency import INPUT_FILES_DIR
from tests.vibe_see_readme_in_this_dir.latency.test_latency import (
    LATENCY_ARCH,
    TWO_FUSED_LOOPS_MAPPING,
    total_latency,
)
from tests.vibe_see_readme_in_this_dir.latency.test_parallel_mappings import (
    MAC_X,
    evaluate,
)

SIMPLE_MAPPING = INPUT_FILES_DIR / "fused_matmuls_to_simple.mapping.yaml"
UNFUSED_MAPPING = INPUT_FILES_DIR / "unfused_matmuls_to_simple.mapping.yaml"
INFINITELY_FAST_COMPUTE = {"COMPUTE_THROUGHPUT": float("inf")}

# MainMemory reads take 100 cycles and writes 200, GlobalBuffer reads 10 and writes 20,
# and one MAC 3.
ACTION_LATENCIES = {
    "MM_READ_LATENCY": 100,
    "MM_WRITE_LATENCY": 200,
    "GB_READ_LATENCY": 10,
    "GB_WRITE_LATENCY": 20,
    "COMPUTE_LATENCY": 3,
}


class TestPrivateWindup(unittest.TestCase):
    def test_slowest_input_one_compute_slowest_output(self):
        """One Einsum, so no tensor is shared and its whole branch is private, paid
        once. T0 and W0 each reach compute through a MainMemory read and a
        GlobalBuffer write + read: 100 + 20 + 10 = 130, the slowest input. One MAC: 3.
        T1 goes up through the GlobalBuffer's write + read (30) and MainMemory's write
        (200; the first tile has nothing to read-modify, so no read): 230.
        130 + 3 + 230 = 363. Summing the two inputs would give 493; paying it on each of
        the 4 m iterations, 1452."""
        row = total_latency(
            LATENCY_ARCH,
            SIMPLE_MAPPING,
            1,
            **INFINITELY_FAST_COMPUTE,
            **ACTION_LATENCIES,
        )
        self.assertEqual(row["Total<SEP>latency"], 363)

    def test_windup_adds_to_the_steady_state(self):
        """As test_slowest_input_one_compute_slowest_output, with the MACs at 0.1 per
        cycle, so the compute steady state is 64 / 0.1 = 640 (the memories are
        infinitely fast). Wind-up latency is the additional time from switching to the
        Einsum to the steady state, and wind-down the additional time from the end of
        the computations to the last output value written to backing storage, so both
        add to the steady state: 640 + 363 = 1003. Taking the max of the two gives
        640."""
        row = total_latency(
            LATENCY_ARCH,
            SIMPLE_MAPPING,
            1,
            COMPUTE_THROUGHPUT=0.1,
            **ACTION_LATENCIES,
        )
        self.assertEqual(row["Total<SEP>latency"], 1003)

    def test_slowest_input_not_the_sum_of_inputs(self):
        """Only MainMemory reads have a latency (100). T0 and W0 each take one
        MainMemory read to reach compute, so the slowest input is 100, and compute and
        the output path add nothing: 100. Summing the inputs would give 200."""
        row = total_latency(
            LATENCY_ARCH,
            SIMPLE_MAPPING,
            1,
            MM_READ_LATENCY=100,
            **INFINITELY_FAST_COMPUTE,
        )
        self.assertEqual(row["Total<SEP>latency"], 100)

    def test_wind_down_reaches_backing_storage(self):
        """Wind-down lasts until the last output value is written to its backing
        storage, MainMemory. With only MainMemory writes taking 200, the inputs and
        compute take 0 and T1's write into MainMemory 200: 200.

        With only GlobalBuffer writes taking 20, T0 and W0 are each written into the
        GlobalBuffer on their way to compute (20, the slowest input), and T1 is written
        into the GlobalBuffer on its way up (20) and into MainMemory at no latency:
        20 + 20 = 40. Counting only the output would give 20; summing all three
        GlobalBuffer storages, 60."""
        row = total_latency(
            LATENCY_ARCH,
            SIMPLE_MAPPING,
            1,
            MM_WRITE_LATENCY=200,
            **INFINITELY_FAST_COMPUTE,
        )
        self.assertEqual(row["Total<SEP>latency"], 200)

        row = total_latency(
            LATENCY_ARCH,
            SIMPLE_MAPPING,
            1,
            GB_WRITE_LATENCY=20,
            **INFINITELY_FAST_COMPUTE,
        )
        self.assertEqual(row["Total<SEP>latency"], 40)

    def test_private_windup_repaid_on_each_switch(self):
        """Two Einsums under one fused m loop (4 iterations). The GlobalBuffer nodes
        below the loop are at level 1, in each Einsum's delay, a descent at level 1
        that the sequential split repays on each iteration. Matmul0 writes T0 into the
        GlobalBuffer and reads it (20 + 10 = 30, the slowest input; T1's
        read-modify-write read, 10, counts with the inputs), one MAC (3), and writes
        T1 (20): 53 x 4 = 212. Matmul1 reads T1 (10), one MAC (3), and T2 is written
        (20) and read to MainMemory (10): 43 x 4 = 172. The MainMemory nodes and the
        weights' GlobalBuffer nodes are above the loop, at level 0, paid once. There a
        tensor pays all of its nodes and the slowest tensor sets the level: each
        Einsum's weight takes a MainMemory read and a GlobalBuffer write + read
        (100 + 30 = 130, over T0's 100), and T2's MainMemory write is an ascent (200):
        212 + 172 + 130 + 130 + 200 = 844.

        With two fused m loops (M = 32, M_TILE = 4: 8 outer x 4 inner iterations) and
        the split below both, the delays are at level 2 and the sequential split
        repays them on every iteration of both loops: (53 + 43) x 32 = 3072. The
        weights' GlobalBuffer nodes between the loops (30 + 30) are at level 1, which
        has no split, so they are added into level 0 and paid once with the MainMemory
        reads (100 + 100) and T2's write (200): 3072 + 60 + 400 = 3532. Taking the
        slowest node at level 0 instead of the slowest tensor would give 784."""
        row = total_latency(
            LATENCY_ARCH,
            SIMPLE_MAPPING,
            2,
            **INFINITELY_FAST_COMPUTE,
            **ACTION_LATENCIES,
        )
        self.assertEqual(row["Total<SEP>latency"], 844)

        row = total_latency(
            LATENCY_ARCH,
            TWO_FUSED_LOOPS_MAPPING,
            2,
            M=32,
            M_TILE=4,
            **INFINITELY_FAST_COMPUTE,
            **ACTION_LATENCIES,
        )
        self.assertEqual(row["Total<SEP>latency"], 3532)

    def test_unfused_einsums_pay_their_windup_once(self):
        """With every tensor backed in MainMemory the Einsums share no loops, so each
        branch is private and entered once. Matmul0: T0 and W0 each reach compute in
        100 + 30 = 130, one MAC takes 3, and T1 goes up in 30 + 200 = 230: 363.
        Matmul1 reads T1 back from MainMemory the same way: 363. Sequentially 726; in
        parallel the wind-ups sum, also 726, so only the single-instance result stays
        on the front. Paying them on each of the 4 m iterations would give 2904;
        taking the slower parallel branch would put (363, 2) on the front."""
        off = evaluate(
            UNFUSED_MAPPING, False, **INFINITELY_FAST_COMPUTE, **ACTION_LATENCIES
        )
        self.assertEqual(list(off["Total<SEP>latency"]), [726])

        on = evaluate(
            UNFUSED_MAPPING, True, **INFINITELY_FAST_COMPUTE, **ACTION_LATENCIES
        )
        front = sorted(zip(on["Total<SEP>latency"], on[MAC_X] * 4))
        self.assertEqual(front, [(726, 1)])
