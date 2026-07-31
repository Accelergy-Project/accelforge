"""
Tests for the latency model: communication (wind-up/down) latency, and per-level
component latency that lets a shared memory's busy time overlap with other Einsums'
compute during fusion.

All tests use the matmul-chain workload with M = KN = 4 and 8 bits per value, so each
Einsum does 64 MACs and each tensor is 128 bits.
"""

import unittest
from pathlib import Path

import accelforge as af
from accelforge.frontend.spec import Spec
from accelforge.model.main import evaluate_mapping

TESTS_DIR = Path(__file__).resolve().parent
INPUT_FILES_DIR = TESTS_DIR / "input_files"
LATENCY_ARCH = INPUT_FILES_DIR / "latency.arch.yaml"
NETWORKED_ARCH = INPUT_FILES_DIR / "networked_latency.arch.yaml"
NETWORKED_MAPPING = INPUT_FILES_DIR / "fused_matmuls_to_networked.mapping.yaml"


def total_latency(arch, mapping, n_einsums, **jinja):
    spec = Spec.from_yaml(
        af.examples.workloads.basic.matmuls,
        arch,
        mapping,
        jinja_parse_data={"N_EINSUMS": n_einsums, "M": 4, "KN": 4, **jinja},
    )
    result = evaluate_mapping(spec)
    return result.data.iloc[0]


class TestCommunicationLatency(unittest.TestCase):
    def test_communication_latency_in_total(self):
        """Action latencies add wind-up/down (communication) latency to the total."""
        # With zero action latencies there is no communication latency and the total
        # is the compute steady state: 64 MACs / 1 MAC per second.
        row = total_latency(
            LATENCY_ARCH, af.examples.mappings.fused_matmuls_to_simple, 1
        )
        self.assertEqual(row["Total<SEP>latency"], 64)

        row = total_latency(
            LATENCY_ARCH,
            af.examples.mappings.fused_matmuls_to_simple,
            1,
            MM_READ_LATENCY=100,
            MM_WRITE_LATENCY=200,
            GB_READ_LATENCY=10,
            GB_WRITE_LATENCY=20,
            COMPUTE_LATENCY=3,
        )
        # The worst input reaches compute in one MainMemory read + one GlobalBuffer
        # write + read (100 + 20 + 10 = 130). The output follows with one MAC (3),
        # winds back up through the GlobalBuffer (20 + 10 = 30), and is read-modify-
        # written at MainMemory (100 + 200 = 300). 130 + 3 + 30 + 300 = 463, which
        # dominates the 64-cycle compute steady state.
        self.assertEqual(row["Total<SEP>latency"], 463)

    def test_fused_repays_communication_latency_each_switch(self):
        """Fused Einsums exchange their intermediate tensor through the shared buffer
        once per shared-loop iteration, and the communication latency of that exchange
        is paid on every switch."""
        row = total_latency(
            LATENCY_ARCH,
            af.examples.mappings.fused_matmuls_to_simple,
            2,
            MM_READ_LATENCY=100,
            MM_WRITE_LATENCY=200,
            GB_READ_LATENCY=10,
            GB_WRITE_LATENCY=20,
            COMPUTE_LATENCY=3,
        )
        # T1 is backed at the GlobalBuffer below the fused m loop, so its wind-down
        # repeats for each of the 4 m iterations. Matmul0: worst input 130, then
        # (one MAC + GlobalBuffer write + read = 33) x 4 iterations = 262. Matmul1
        # is the unfused 463 from above (T1's inputs wind down 30 x 4 = 120 < 130
        # from MainMemory). 262 + 463 = 725.
        self.assertEqual(row["Total<SEP>latency"], 725)

    def test_fused_through_slow_interconnect(self):
        """Two Einsums fused through networks pay the hop latency of the intermediate
        tensor's route on every shared-loop iteration."""
        mesh_hops = 4  # All 4 Scratchpad positions are used: 4 hops on the PeArray
        switch_hops = 1  # The all-to-all MacArray is one hop for any route
        down = mesh_hops + switch_hops  # backing -> compute, and compute -> backing
        # Matmul0: inputs wind down from MainMemory once (5 hops), then T1 winds up
        # to its GlobalBuffer backing below the fused m loop on every one of the 4
        # iterations: 5 + 5 x 4 = 25. Matmul1 mirrors it: T1 winds down 5 x 4 = 20,
        # then T2 winds up to MainMemory once: 20 + 5 = 25.
        expected = 2 * (down + down * 4)
        for hop_latency in [0, 1, 100]:
            row = total_latency(
                NETWORKED_ARCH,
                NETWORKED_MAPPING,
                2,
                MAC_TILE=1,
                HOP_LATENCY=hop_latency,
            )
            self.assertEqual(row["Total<SEP>latency"], expected * hop_latency)


class TestSharedMemoryOverlap(unittest.TestCase):
    def test_fused_memory_bound_overlaps_compute(self):
        """When a compute-bound Einsum is fused with a memory-bound Einsum, the shared
        memory's busy time is summed across the Einsums and overlapped with their
        compute: the total is the max of the two, not the sum of per-Einsum maxes.
        The tensors at MainMemory are backed above the shared m loop, so their
        reservations are co-resident and their transfers may fill any slice of the
        fused execution."""
        row = total_latency(
            LATENCY_ARCH,
            af.examples.mappings.fused_matmuls_to_simple,
            2,
            MM_READ_THROUGHPUT=1,
            MM_WRITE_THROUGHPUT=0.1,
            COMPUTE_THROUGHPUT=0.2,
        )
        # Matmul0 is compute-bound: 64 MACs / 0.2 = 320 vs reading T0 and W0 from
        # MainMemory (256 bits / 1 = 256). Matmul1 is memory-bound: writing T2 back
        # (128 bits / 0.1 = 1280) plus reading W1 (128 / 1 = 128) is 1408 vs 320.
        self.assertEqual(row["Matmul0<SEP>component_latency<SEP>MAC"], 320)
        self.assertEqual(row["Matmul0<SEP>component_latency<SEP>MainMemory"], 256)
        self.assertEqual(row["Matmul1<SEP>component_latency<SEP>MAC"], 320)
        self.assertEqual(row["Matmul1<SEP>component_latency<SEP>MainMemory"], 1408)

        # max(320 + 320, 256 + 1408) = 1664: Matmul0's MainMemory slack absorbs part
        # of Matmul1's traffic. Summing per-Einsum maxes would give 320 + 1408 = 1728.
        self.assertEqual(row["Total<SEP>latency"], 1664)


if __name__ == "__main__":
    unittest.main()
