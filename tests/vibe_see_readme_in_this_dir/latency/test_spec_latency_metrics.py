"""
Latency under each metric selection and the leak power rule of finish(): leak power
times total latency is added into Total<SEP>energy and Total<SEP>leak_energy and the
leak power column is dropped. Energy is 1 per action bit: fused_matmuls_to_simple with
M = KN = 4 does 4992 bits of memory actions plus MACs (2752 for Matmul0 alone).
"""

import unittest

from accelforge.frontend.mapper.metrics import Metrics
from tests.vibe_see_readme_in_this_dir.latency.test_spec_latency_helpers import (
    ENERGY,
    FUSED,
    LATENCY,
    MAC_X_PARALLEL,
    evaluate,
)

# 128 of compute per Einsum: 256 sequential, 128 parallel.
COMPUTE_BOUND = {"COMPUTE_THROUGHPUT": 0.5}
EDP = "Total<SEP>energy_delay_product"


class TestMetrics(unittest.TestCase):
    def test_latency_only(self):
        """Only latency is reported: 256."""
        data = evaluate(FUSED, 2, metrics=Metrics.LATENCY, **COMPUTE_BOUND)
        self.assertEqual(list(data[LATENCY]), [256])
        self.assertNotIn(ENERGY, data.columns)

    def test_energy_only(self):
        """Only energy is reported: 4992, no latency column."""
        data = evaluate(FUSED, 2, metrics=Metrics.ENERGY, **COMPUTE_BOUND)
        self.assertEqual(list(data[ENERGY]), [4992])
        self.assertNotIn(LATENCY, data.columns)

    def test_energy_delay_product(self):
        """4992 x 256 = 1277952."""
        data = evaluate(FUSED, 2, metrics=Metrics.ENERGY_DELAY_PRODUCT, **COMPUTE_BOUND)
        self.assertEqual(list(data[EDP]), [1277952])

    def test_leak_power_times_latency(self):
        """
        GlobalBuffer leaks 2 per unit time over 256: 512 of leak energy. Energy 4992 +
        512 = 5504, leak energy 512, and no leak power column remains.
        """
        data = evaluate(FUSED, 2, GB_LEAK_POWER=2, **COMPUTE_BOUND)
        self.assertEqual(list(data[ENERGY]), [5504])
        self.assertEqual(list(data["Total<SEP>leak_energy"]), [512])
        self.assertNotIn("Total<SEP>leak_power", data.columns)

    def test_leak_energy_per_arm(self):
        """
        The parallel arm finishes in 128, so it leaks 256: energy 5248 on two PEs
        against 5504 on one.
        """
        data = evaluate(
            FUSED, 2, GB_LEAK_POWER=2, MAC_FANOUT=4, MAC_PARALLEL=True, **COMPUTE_BOUND
        )
        rows = {
            round(u * 4): (l, e)
            for u, l, e in zip(data[MAC_X_PARALLEL], data[LATENCY], data[ENERGY])
        }
        self.assertEqual(rows, {1: (256, 5504), 2: (128, 5248)})

    def test_energy_delay_product_prefers_parallel_arm(self):
        """Under EDP the parallel arm scores 4992 x 128 = 638976, the best row."""
        data = evaluate(
            FUSED,
            2,
            metrics=Metrics.ENERGY_DELAY_PRODUCT,
            MAC_FANOUT=4,
            MAC_PARALLEL=True,
            **COMPUTE_BOUND,
        )
        self.assertEqual(min(data[EDP]), 638976)


if __name__ == "__main__":
    unittest.main()
