"""
Busy time across sequential splits: a component's latency above a split sums with the
Einsums that share that level, a fenced branch does not shorten the levels after it,
and the whole chain's busy time on one component bounds the total.
"""

import unittest

from tests.vibe_see_readme_in_this_dir.latency.test_latency import (
    DEEP_THEN_SHALLOW_MAPPING,
    LATENCY_ARCH,
    SHALLOW_THEN_DEEP_MAPPING,
    total_latency,
)


class TestBusyTimeAcrossSplits(unittest.TestCase):
    def test_busy_time_above_a_split_sums_with_a_later_einsum(self):
        """Matmul0 splits off above the inner m loop, and Matmul1 and Matmul2 split
        below it.

        With only the GlobalBuffer reads slowed to one bit-time, each Einsum reads one
        value per MAC from each input (64 x 8 = 512 each) and re-reads its output on 3
        of every 4 MACs to accumulate (48 x 8 = 384); Matmul2 also drains T3 to
        MainMemory from its node below the inner loop (16 values, 128). Reads to
        compute count at the Einsum's own level, so every read is in its Einsum's own
        branch and none is above a split: Matmul0 512 + 512 + 384 = 1408, Matmul1 the
        same 1408, and Matmul2 1408 + 128 = 1536. The branches serialize into
        1408 + 1408 + 1536 = 4352, also the GlobalBuffer's whole-chain busy time."""
        row = total_latency(
            LATENCY_ARCH,
            SHALLOW_THEN_DEEP_MAPPING,
            3,
            M_TILE=1,
            GB_READ_THROUGHPUT=1,
            COMPUTE_THROUGHPUT=float("inf"),
        )
        self.assertEqual(row["Total<SEP>latency"], 4352)

    def test_a_fenced_branch_does_not_shorten_a_later_level(self):
        """Matmul0 splits off above the inner m loop, and Matmul1 and Matmul2 split
        below it. Matmul0's branch is fenced at the first split, and the Matmul1/Matmul2
        level that follows still runs for as long as its own busy time, so the two add.

        With M = 64 and KN = 2 each Einsum does 256 MACs, each T tensor is 1024 bits and
        each W tensor 32 bits. A GlobalBuffer read costs one bit-time and a MainMemory
        read ten, on a port of its own.

        Matmul0's branch is bound by its MainMemory reads: T0's 512-bit tile is written
        on each of the 2 outer m iterations (1024 bits) and W0 on each of the 2 outer ones
        (64 bits), so 1088 x 10 = 10880. Its GlobalBuffer traffic is only 5120 (T0 and
        W0 once per MAC, 2048 each, plus 1024 re-reading T1 to accumulate).

        Reads to compute count in each Einsum's own branch. Matmul1 reads T1 and W1
        (2048 each) and re-reads T2 (1024), so 5120; Matmul2 reads W2 and T2 (2048
        each), re-reads T3 (1024) and drains T3 to MainMemory from its node below the
        inner loop (1024), so 6144. Both are GlobalBuffer-bound, so the level is
        5120 + 6144 = 11264 and the total is 10880 + 11264 = 22144. Letting the level
        use Matmul0's GlobalBuffer slack would give the GlobalBuffer's whole-chain busy
        time, 5120 + 5120 + 6144 = 16384."""
        row = total_latency(
            LATENCY_ARCH,
            SHALLOW_THEN_DEEP_MAPPING,
            3,
            M=64,
            KN=2,
            M_TILE=32,
            MM_SEPARATE_PORTS=True,
            MM_READ_THROUGHPUT=0.1,
            GB_READ_THROUGHPUT=1,
            COMPUTE_THROUGHPUT=float("inf"),
        )
        self.assertEqual(row["Total<SEP>latency"], 22144)

    def test_whole_chain_busy_time_survives_a_mid_chain_fence(self):
        """Matmul0 and Matmul1 share both m loops and split below them; the pair shares
        only the outer loop with Matmul2, so the pair's branch is fenced at the second
        split. Its GlobalBuffer traffic keeps counting toward the port afterwards: the
        chain's whole-chain busy time is the bandwidth it needs, and no schedule beats
        it.

        With M = 16 and KN = 4 each Einsum does 256 MACs, each T tensor is 512 bits and
        each W tensor 128 bits, and a GlobalBuffer read costs one bit-time. Reads to
        compute count in each Einsum's own branch. Matmul0 reads W0 and T0 (one value
        per MAC, 2048 each) and re-reads T1 on 3 of every 4 MACs to accumulate (1536),
        so 5632. Matmul1 reads W1 and T1 (2048 each) and re-reads T2 (1536), so 5632
        again. Matmul2 reads W2 and T2 (2048 each), re-reads T3 (1536) and drains T3
        to MainMemory (512), so 6144.

        5632 + 5632 + 6144 = 17408 sets the total, and the branches serialize into
        the same 17408."""
        row = total_latency(
            LATENCY_ARCH,
            DEEP_THEN_SHALLOW_MAPPING,
            3,
            M=16,
            M_TILE=4,
            GB_SIZE=2048,
            GB_READ_THROUGHPUT=1,
            COMPUTE_THROUGHPUT=float("inf"),
        )
        self.assertEqual(row["Total<SEP>latency"], 17408)
