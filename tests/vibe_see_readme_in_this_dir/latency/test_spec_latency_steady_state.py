"""
Steady-state rules of the Latency Model (accelforge/frontend/model.py), derived from the
spec only. Each resource's steady state is #actions / throughput, summed over its
actions; a memory with separate read/write ports is two resources. Within a branch the
steady state is the max over its resources. Sequential branches sum. The total of a
branch is max(sum of sub-branch totals, each resource's summed steady state), so slack
propagates upward when _share_resource_latency_above_splits is on; when it is off every
resource an Einsum uses is paid at the Einsum's compute node.

Action counts (bits, 8 per value) with M = KN = 4, 64 MACs per Einsum:
  fused_matmuls_to_simple: Matmul0 reads T0 128 and W0 128 from MainMemory, Matmul1
  reads W1 128 and writes T2 128. The W fills sit above the shared m loop (level 0); the
  T0 reads and T2 writes sit inside it (level 1). GlobalBuffer, N = 1: reads 1536,
  writes 768.
  unfused_matmuls_to_simple: each Einsum reads T_i 128 and W_i 128, writes T_{i+1} 128.
  spec_gb_only_first: Matmul0 GlobalBuffer reads 1536, writes 768; Matmul1 uses none.
  four_matmuls_three_loops (M = 8, KN = 2, 32 MACs each): MainMemory reads T0 128,
  W0 256 (level 3), W1 128 (level 2), W2 128 (level 2), W3 64 and T4 128 (level 1).
"""

import unittest

from tests.vibe_see_readme_in_this_dir.latency.test_spec_latency_helpers import (
    FOUR,
    FOUR_MATMULS,
    FUSED,
    GB_ONLY_FIRST,
    UNFUSED,
    latency,
)

# Read and write ports each move 1 bit per unit time; 64 MACs take 200.
SEPARATE_PORTS = {
    "MM_SEPARATE_PORTS": True,
    "MM_READ_THROUGHPUT": 1,
    "MM_WRITE_THROUGHPUT": 1,
    "COMPUTE_THROUGHPUT": 0.32,
}


class TestSingleEinsum(unittest.TestCase):
    def test_compute_bound(self):
        """64 MACs / 0.5 = 128. Memories are infinitely fast."""
        self.assertEqual(latency(FUSED, 1, COMPUTE_THROUGHPUT=0.5), 128)

    def test_memory_bound(self):
        """MainMemory reads (128 + 128) / 1 = 256; compute is infinitely fast."""
        self.assertEqual(latency(FUSED, 1, MM_READ_THROUGHPUT=1), 256)

    def test_one_port_sums_reads_and_writes(self):
        """
        One port: reads 256 / 1 + writes 128 / 0.5 = 512. A max reading would give
        256.
        """
        row = latency(FUSED, 1, MM_READ_THROUGHPUT=1, MM_WRITE_THROUGHPUT=0.5)
        self.assertEqual(row, 512)

    def test_separate_ports_are_separate_resources(self):
        """
        Separate ports: read port 256 / 1 = 256, write port 128 / 0.5 = 256, each its
        own resource, so max = 256. A summed reading would give 512.
        """
        row = latency(
            FUSED,
            1,
            MM_SEPARATE_PORTS=True,
            MM_READ_THROUGHPUT=1,
            MM_WRITE_THROUGHPUT=0.5,
        )
        self.assertEqual(row, 256)

    def test_branch_is_max_over_resources(self):
        """
        MainMemory 256 / 1 = 256, MAC 64 / 0.2 = 320, GlobalBuffer (1536 + 768) / 16 =
        144. Max = 320. A summed reading would give 720.
        """
        row = latency(
            FUSED,
            1,
            MM_READ_THROUGHPUT=1,
            COMPUTE_THROUGHPUT=0.2,
            GB_READ_THROUGHPUT=16,
            GB_WRITE_THROUGHPUT=16,
        )
        self.assertEqual(row, 320)


class TestSequentialBranches(unittest.TestCase):
    def test_two_einsums_sum(self):
        """Two compute-bound Einsums of 128 each run one after the other: 256."""
        self.assertEqual(latency(FUSED, 2, COMPUTE_THROUGHPUT=0.5), 256)

    def test_three_einsums_fill_overlaps_compute(self):
        """
        Inside the shared loop: Matmul0 max(T0 reads 128, MAC 128) = 128, Matmul1 128,
        Matmul2 128, sum 384. MainMemory in total reads T0 128 + W0 + W1 + W2 384 = 512,
        and the W fills above the loop overlap the compute, so max(384, 512) = 512.
        Paying the fills on top would give 384 + 384 = 768.
        """
        row = latency(FUSED, 3, MM_READ_THROUGHPUT=1, COMPUTE_THROUGHPUT=0.5)
        self.assertEqual(row, 512)

    def test_unfused_pair_no_shared_loops(self):
        """
        No shared loops: each Einsum is one private branch of max(reads 256, MAC 128) =
        256, in sequence 512. There is nothing above a split to share, so both flag
        settings agree.
        """
        for share in (True, False):
            with self.subTest(share=share):
                row = latency(
                    UNFUSED,
                    2,
                    share=share,
                    MM_READ_THROUGHPUT=1,
                    COMPUTE_THROUGHPUT=0.5,
                )
                self.assertEqual(row, 512)

    def test_component_used_by_one_einsum(self):
        """
        GlobalBuffer serves only Matmul0: (1536 + 768) / 1 = 2304 against MAC 64 / 0.1 =
        640, so Matmul0 is 2304. Matmul1 never touches GlobalBuffer: 640. Sum 2944.
        """
        row = latency(
            GB_ONLY_FIRST,
            2,
            GB_READ_THROUGHPUT=1,
            GB_WRITE_THROUGHPUT=1,
            COMPUTE_THROUGHPUT=0.1,
        )
        self.assertEqual(row, 2944)

    def test_four_einsums_compute_bound(self):
        """
        32 MACs / 0.1 = 320 per Einsum. MainMemory reads at 1 bit per unit: Matmul0
        reads T0 128 + W0 256 = 384 inside its own branch (level 3), so it is 384; the
        others are compute bound at 320. Level 3: 384 + 320 = 704 vs MainMemory 384.
        Level 2: 704 + 320 = 1024 vs 384 + 128 + 128 = 640. Level 1: 1024 + 320 = 1344
        vs 640 + 192 = 832. Total 1344.
        """
        row = latency(
            FOUR_MATMULS, 4, **FOUR, MM_READ_THROUGHPUT=1, COMPUTE_THROUGHPUT=0.1
        )
        self.assertEqual(row, 1344)

    def test_four_einsums_memory_bound(self):
        """
        Compute infinitely fast. Matmul0 384, Matmul1 0 (its W1 fill is above its
        split), Matmul2 128, Matmul3 64 + 128 = 192. Level 3: 384 vs 384. Level 2: 384 +
        128 = 512 vs 640 -> 640. Level 1: 640 + 192 = 832 vs 832. Total 832, the whole
        MainMemory busy time.
        """
        row = latency(FOUR_MATMULS, 4, **FOUR, MM_READ_THROUGHPUT=1)
        self.assertEqual(row, 832)


class TestShareResourceLatencyAboveSplits(unittest.TestCase):
    def test_slack_propagates_above_split(self):
        """
        Inside the shared loop: Matmul0 max(T0 reads 128, MAC 200) = 200, Matmul1
        max(T2 writes 128, MAC 200) = 200, sum 400. Read port total T0 + W0 + W1 = 384,
        write port 128, both under 400. Total 400.
        """
        self.assertEqual(latency(FUSED, 2, share=True, **SEPARATE_PORTS), 400)

    def test_no_slack_above_split(self):
        """
        Flag off: every resource is paid at the compute node. Matmul0 max(reads 256,
        MAC 200) = 256, Matmul1 max(reads 128, writes 128, MAC 200) = 200. Total 456.
        """
        self.assertEqual(latency(FUSED, 2, share=False, **SEPARATE_PORTS), 456)

    def test_flags_agree_when_fill_dominates(self):
        """
        Read port at 0.5: T0 256, W0 256, W1 256, total 768. On: max(256 + 200, 768) =
        768. Off: max(512, 200) + max(256, 128, 200) = 768.
        """
        jinja = {**SEPARATE_PORTS, "MM_READ_THROUGHPUT": 0.5}
        for share in (True, False):
            with self.subTest(share=share):
                self.assertEqual(latency(FUSED, 2, share=share, **jinja), 768)

    def test_three_einsums(self):
        """
        On: 200 + 200 + 200 = 600 vs read port 128 + 384 = 512 -> 600. Off: max(256,
        200) + max(128, 200) + max(128, 128, 200) = 656.
        """
        self.assertEqual(latency(FUSED, 3, share=True, **SEPARATE_PORTS), 600)
        self.assertEqual(latency(FUSED, 3, share=False, **SEPARATE_PORTS), 656)


if __name__ == "__main__":
    unittest.main()
