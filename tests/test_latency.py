"""
Tests for the latency model: communication (wind-up/down) latency, and per-level
component latency that lets a shared memory's busy time overlap with other Einsums'
compute during fusion.

All tests use the matmul-chain workload with M = KN = 4 and 8 bits per value, so each
Einsum does 64 MACs and each tensor is 128 bits.
"""

import copy
import unittest

import pandas as pd
from pathlib import Path

import accelforge as af
from accelforge.frontend.spec import Spec
from accelforge.model.main import evaluate_mapping

TESTS_DIR = Path(__file__).resolve().parent
INPUT_FILES_DIR = TESTS_DIR / "input_files"
LATENCY_ARCH = INPUT_FILES_DIR / "latency.arch.yaml"
NETWORKED_ARCH = INPUT_FILES_DIR / "networked_latency.arch.yaml"
NETWORKED_MAPPING = INPUT_FILES_DIR / "fused_matmuls_to_networked.mapping.yaml"
WEIGHTS_INSIDE_MAPPING = INPUT_FILES_DIR / "fused_matmuls_weights_inside.mapping.yaml"


def make_spec(arch, mapping, n_einsums, **jinja):
    return Spec.from_yaml(
        af.examples.workloads.basic.matmuls,
        arch,
        mapping,
        jinja_parse_data={"N_EINSUMS": n_einsums, "M": 4, "KN": 4, **jinja},
    )


def total_latency(arch, mapping, n_einsums, **jinja):
    result = evaluate_mapping(make_spec(arch, mapping, n_einsums, **jinja))
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
        # Each storage's connection is one of each action it performs. The worst
        # input reaches compute in one MainMemory read + one GlobalBuffer write +
        # read (100 + 20 + 10 = 130). The output follows: one MAC (3), the
        # GlobalBuffer's write + read (30), and the MainMemory write (200; the
        # first tile has nothing to read-modify, so its read is skipped).
        # 130 + 3 + 30 + 200 = 363, which dominates the 64-cycle compute steady
        # state.
        self.assertEqual(row["Total<SEP>latency"], 363)

    def test_fused_repays_communication_latency_each_switch(self):
        """The exchange of the intermediate through the shared buffer happens below
        the fused loop and is paid on every switch; the weights' fills and the
        output's drain cross the fused loop, so they are paid once for the whole
        fused group (descent/ascent latency), bounded by the slowest Einsum."""
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
        # Storages above the fused m loop are reserved once and can pre-send
        # ahead of it, so their connections are paid once (descent/ascent, maxed
        # across Einsums); storages below repeat for each of the 4 m iterations
        # via the Einsum delay. Matmul0: T0's and T1's GlobalBuffer connections
        # (write + read = 30 each) plus one MAC: (30 + 3 + 30) x 4 = 252.
        # Matmul1 mirrors it with T1's and T2's GlobalBuffer connections:
        # (30 + 3 + 30) x 4 = 252. Paid once: the inputs' MainMemory reads
        # (descent, max = 100) and T2's MainMemory write (ascent, 200; the first
        # tile skips the read-modify-write read). 252 + 252 + 100 + 200 = 804.
        self.assertEqual(row["Total<SEP>latency"], 804)

    def test_fused_through_slow_interconnect(self):
        """Two Einsums fused through networks pay the hop latency of the intermediate
        tensor's route on every shared-loop iteration."""
        mesh_hops = 4  # All 4 Scratchpad positions are used: 4 hops on the PeArray
        switch_hops = 1  # The all-to-all MacArray is one hop for any route
        down = mesh_hops + switch_hops  # backing -> compute, and compute -> backing
        # Each Einsum's delay is its slowest input's wind-down plus its output's
        # wind-up (5 hops each way), repeated for each of the 4 iterations of the
        # fused loop above the intermediate's GlobalBuffer backing.
        expected = 2 * (down + down) * 4
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
        A MainMemory transfer runs while the GlobalBuffer reservation on its other
        end is alive: the weights' GlobalBuffer staging is above the shared m loop,
        so their reads may fill any slice of the fused execution, while T0/T2's
        GlobalBuffer reservations live below it, confining that traffic to its own
        Einsum. MainMemory's total busy time still bounds the total either way."""
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

    def test_separate_read_write_ports(self):
        """With separate MainMemory ports, reads and writes are independent
        components that may overlap: W1's reads no longer serialize behind T2's
        writeback, saving their 64 bit-times versus the shared port's 1664."""
        row = total_latency(
            LATENCY_ARCH,
            af.examples.mappings.fused_matmuls_to_simple,
            2,
            MM_READ_THROUGHPUT=1,
            MM_WRITE_THROUGHPUT=0.1,
            COMPUTE_THROUGHPUT=0.2,
            MM_SEPARATE_PORTS=True,
        )
        self.assertEqual(
            row["Matmul0<SEP>component_latency<SEP>MainMemory (read)"], 256
        )
        self.assertEqual(
            row["Matmul1<SEP>component_latency<SEP>MainMemory (read)"], 128
        )
        self.assertEqual(
            row["Matmul1<SEP>component_latency<SEP>MainMemory (write)"], 1280
        )
        # T2's writes go to its GlobalBuffer staging below the fused m loop, so
        # the write port's 1280 is confined to Matmul1's block and sets its
        # width; the shareable reads (256 + 128 = 384) and the MACs (320 + 320)
        # hide beneath it. 320 + 1280 = 1600.
        self.assertEqual(row["Total<SEP>latency"], 1600)

    def test_transfer_needs_deeper_reservation(self):
        """With every GlobalBuffer staging below the fused m loop, MainMemory
        transfers can only run during their own Einsum's per-iteration windows, so
        each Einsum's MainMemory traffic is confined to its block and the overlap
        credit above is lost. Matmul0 is compute-bound (320 vs reading T0 once and
        W0 per m iteration: (128 + 4 x 128) / 4 = 160); Matmul1's writeback still
        dominates its block (128 + 1280 = 1408). 320 + 1408 = 1728, though
        MainMemory is only busy for 160 + 1408 = 1568 of it."""
        row = total_latency(
            LATENCY_ARCH,
            WEIGHTS_INSIDE_MAPPING,
            2,
            MM_READ_THROUGHPUT=4,
            MM_WRITE_THROUGHPUT=0.1,
            COMPUTE_THROUGHPUT=0.2,
        )
        self.assertEqual(row["Total<SEP>latency"], 1728)


class TestLatencyTimeline(unittest.TestCase):
    """_latency_timeline asserts internally that its total matches the model's
    Total<SEP>latency, so each call here also cross-checks the layout math."""

    def test_timeline_totals(self):
        from accelforge.plotting.latency import _latency_timeline

        scenarios = [
            (LATENCY_ARCH, af.examples.mappings.fused_matmuls_to_simple, 1, 363),
            (LATENCY_ARCH, af.examples.mappings.fused_matmuls_to_simple, 2, 804),
            (NETWORKED_ARCH, NETWORKED_MAPPING, 2, 8000),
        ]
        jinja = {
            "MM_READ_LATENCY": 100,
            "MM_WRITE_LATENCY": 200,
            "GB_READ_LATENCY": 10,
            "GB_WRITE_LATENCY": 20,
            "COMPUTE_LATENCY": 3,
            "MAC_TILE": 1,
            "HOP_LATENCY": 100,
        }
        for arch, mapping, n_einsums, expected in scenarios:
            blocks, _, total = _latency_timeline(
                make_spec(arch, mapping, n_einsums, **jinja)
            )
            self.assertEqual(total, expected)
            self.assertEqual(len(blocks), n_einsums)

    def test_timeline_overlap_layout(self):
        """The layout from test_fused_memory_bound_overlaps_compute, bar by bar.
        T0/T2 traffic sits at level 1 (their GlobalBuffer reservations live below
        the fused m loop, confining it to its own Einsum's block); weight reads sit
        at level 0 and may fill slack anywhere. Matmul1's T2 writeback fills its
        block, and MainMemory's total busy time (256 + 1408 = 1664) sets the end."""
        from accelforge.plotting.latency import _latency_timeline

        blocks, bars, total = _latency_timeline(
            make_spec(
                LATENCY_ARCH,
                af.examples.mappings.fused_matmuls_to_simple,
                2,
                MM_READ_THROUGHPUT=1,
                MM_WRITE_THROUGHPUT=0.1,
                COMPUTE_THROUGHPUT=0.2,
            )
        )
        self.assertEqual(total, 1664)
        self.assertEqual(
            [(b.einsum, b.start, b.end) for b in blocks],
            [("Matmul0", 0, 320), ("Matmul1", 320, 1664)],
        )
        mm = [
            (b.einsum, b.level, b.start, b.end)
            for b in bars
            if b.component == "MainMemory"
        ]
        self.assertEqual(
            sorted(mm),
            [
                ("Matmul0", 0, 128, 256),  # W0 reads, shareable
                ("Matmul0", 1, 0, 128),  # T0 reads, private
                # W1 reads, shareable; right-aligned to the 1664 deadline
                ("Matmul1", 0, 1536, 1664),
                ("Matmul1", 1, 320, 1600),  # T2 writeback, private
            ],
        )
        # Compute has its own lane now, and nothing else (communication is zero
        # here) is left for the Other lane.
        mac = [(b.einsum, b.start, b.end) for b in bars if b.component == "MAC"]
        self.assertEqual(mac, [("Matmul0", 0, 320), ("Matmul1", 320, 640)])
        self.assertEqual([b for b in bars if b.component is None], [])

    def test_plot_latency(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from accelforge.plotting.latency import plot_latency

        fig, _ = plot_latency(
            make_spec(
                LATENCY_ARCH,
                af.examples.mappings.fused_matmuls_to_simple,
                2,
                MM_READ_THROUGHPUT=1,
                MM_WRITE_THROUGHPUT=0.1,
                COMPUTE_THROUGHPUT=0.2,
            )
        )
        plt.close(fig)

    def test_timeline_matches_mapper_output(self):
        """Timelines built from mapper results match the mapper-reported latency
        (and, via the internal assert, a fresh model evaluation)."""
        from accelforge.plotting.latency import _latency_timeline

        spec = Spec.from_yaml(
            af.examples.arches.simple,
            af.examples.workloads.basic.matmuls,
            jinja_parse_data={"N_EINSUMS": 2, "M": 16, "KN": 16},
        )
        spec.mapper.metrics = af.mapper.Metrics.LATENCY
        result = spec.map_workload_to_arch(print_progress=False)
        two = copy.copy(result)
        two.data = pd.concat([result.data, result.data])
        with self.assertRaises(ValueError):
            _latency_timeline(spec, two)
        for i in range(len(result.data)):
            one = copy.copy(result)
            one.data = result.data.iloc[[i]]
            _, _, total = _latency_timeline(spec, one)
            self.assertEqual(total, result.data["Total<SEP>latency"].iloc[i])


if __name__ == "__main__":
    unittest.main()
