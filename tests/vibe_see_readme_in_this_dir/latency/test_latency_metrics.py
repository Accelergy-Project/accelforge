"""
Which Total columns each metric leaves after the final free and fence, and that the
GlobalBuffer's leak power pays over the joined runtime in every energy metric.

Every test evaluates the chain with each Einsum's weights' GlobalBuffer node below the fused m loop
(WEIGHTS_INSIDE_MAPPING), slow MainMemory, a GlobalBuffer leaking 2 per cycle, and the
MAC X dimension allowing parallelism. Latency: sequentially 320 + 1408 = 1728, in
parallel max(320, 160 + 1408) = 1568 (test_parallel_overlaps_busy_time).

Dynamic energy, 1 per bit moved and 1 per MAC. Matmul0: MainMemory reads T0 once (128)
and W0 on each of the 4 m iterations (512); the GlobalBuffer is written T0 (128) and
reads it once per MAC (512), is written W0 (512) and reads it (512), and is written T1
(512) and re-reads it on 3 of every 4 MACs to accumulate (384); 64 MACs. That is
640 + 2560 + 64 = 3264. Matmul1: MainMemory reads W1 (512) and is written T2 (128); the
GlobalBuffer reads T1 (512), is written W1 (512) and reads it (512), and is written T2
(512), re-reads it (384) and reads it to drain it (128); 64 MACs. That is also
640 + 2560 + 64 = 3264, 6528 in all.

Leak energy is the leak power over the joined runtime: 2 x 1568 = 3136 in parallel,
2 x 1728 = 3456 sequentially. Without RESOURCE_USAGE the reservations are not reported,
so the parallel result, faster and cheaper, is the only one left.
"""

import unittest

from accelforge.frontend.mapper.metrics import Metrics
from tests.vibe_see_readme_in_this_dir.latency.test_parallel_mappings import (
    GB,
    MAC_X,
    SLOW_MM_THROUGHPUTS,
    WEIGHTS_INSIDE_MAPPING,
    evaluate,
)

MM = "reservation<SEP>MainMemory<SEP>-1<SEP>left"

# Slow MainMemory and a GlobalBuffer that leaks 2 per cycle.
LEAKY_SLOW_MM = {**SLOW_MM_THROUGHPUTS, "GB_LEAK_POWER": 2}


def evaluate_metrics(metrics):
    return evaluate(WEIGHTS_INSIDE_MAPPING, True, metrics=metrics, **LEAKY_SLOW_MM)


def total_columns(data):
    return sorted(c for c in data.columns if c.startswith("Total<SEP>"))


def reservation_columns(data):
    return sorted(c for c in data.columns if c.startswith("reservation<SEP>"))


class TestMetrics(unittest.TestCase):
    def test_latency(self):
        """Only the total latency is reported, and the parallel result's 1568 is the
        only one left."""
        data = evaluate_metrics(Metrics.LATENCY)
        self.assertEqual(
            total_columns(data), ["Total<SEP>latency", "Total<SEP>mapping"]
        )
        self.assertEqual(reservation_columns(data), [])
        self.assertEqual(list(data["Total<SEP>latency"]), [1568])

    def test_energy(self):
        """Only the total energy is reported: 6528 + 2 x 1568 = 9664 in parallel,
        cheaper than the sequential 6528 + 2 x 1728 = 9984."""
        data = evaluate_metrics(Metrics.ENERGY)
        self.assertEqual(total_columns(data), ["Total<SEP>energy", "Total<SEP>mapping"])
        self.assertEqual(list(data["Total<SEP>energy"]), [9664])

    def test_energy_delay_product(self):
        """Only the energy-delay product is reported, with the leak energy in the
        energy: 9664 x 1568 = 15153152 in parallel, against 9984 x 1728 sequentially.
        Leaving the leak out would give 6528 x 1568."""
        data = evaluate_metrics(Metrics.ENERGY_DELAY_PRODUCT)
        self.assertEqual(
            total_columns(data),
            ["Total<SEP>energy_delay_product", "Total<SEP>mapping"],
        )
        self.assertEqual(list(data["Total<SEP>energy_delay_product"]), [15153152])

    def test_leak_energy(self):
        """Only the leak energy is reported: 2 x 1568 = 3136 in parallel."""
        data = evaluate_metrics(Metrics.LEAK_ENERGY)
        self.assertEqual(
            total_columns(data), ["Total<SEP>leak_energy", "Total<SEP>mapping"]
        )
        self.assertEqual(list(data["Total<SEP>leak_energy"]), [3136])

    def test_dynamic_energy(self):
        """Only the dynamic energy is reported, 6528 either way. No latency is
        tracked, and the final fence needs none."""
        data = evaluate_metrics(Metrics.DYNAMIC_ENERGY)
        self.assertEqual(
            total_columns(data), ["Total<SEP>dynamic_energy", "Total<SEP>mapping"]
        )
        self.assertEqual(list(data["Total<SEP>dynamic_energy"]), [6528])

    def test_latency_and_energy(self):
        """The parallel result is both faster and cheaper: (1568, 9664) alone."""
        data = evaluate_metrics(Metrics.LATENCY | Metrics.ENERGY)
        rows = list(zip(data["Total<SEP>latency"], data["Total<SEP>energy"]))
        self.assertEqual(rows, [(1568, 9664)])

    def test_latency_and_resource_usage(self):
        """Reporting resource usage keeps the reservations at the top level, so the
        sequential result stays on the front with one MAC X instance against the
        parallel result's two: (1568, 2) and (1728, 1). Every memory is reported; with
        infinite sizes they read 0."""
        data = evaluate_metrics(Metrics.LATENCY | Metrics.RESOURCE_USAGE)
        self.assertEqual(reservation_columns(data), sorted([GB, MAC_X, MM]))
        front = sorted(zip(data["Total<SEP>latency"], data[MAC_X] * 4))
        self.assertEqual(front, [(1568, 2), (1728, 1)])
        self.assertEqual(list(data[MM]), [0, 0])
        self.assertEqual(list(data[GB]), [0, 0])
