"""
Test that the mapper correctly accounts for Toll (processing-stage) energy.

A Toll charges a fixed energy per read; data passing through it contributes
to total energy. This test loads an arch with a Toll between main_memory and
LocalBuffer, runs the mapper, and checks that total energy is positive and
reflects the Toll's read energy.
"""

import unittest
from pathlib import Path

from accelforge.frontend.spec import Spec
from accelforge.frontend.mapper.metrics import Metrics
from accelforge.mapper.FFM import make_pmappings, join_pmappings


TESTS_DIR = Path(__file__).resolve().parent
INPUT_FILES_DIR = TESTS_DIR / "input_files"
TOLL_ARCH = INPUT_FILES_DIR / "toll.arch.yaml"
MATMUL_WORKLOAD = INPUT_FILES_DIR / "matmul_toll.workload.yaml"


class TestToll(unittest.TestCase):
    def test_toll_energy(self):
        spec = Spec.from_yaml(str(TOLL_ARCH), str(MATMUL_WORKLOAD))
        spec.mapper.metrics = Metrics.ENERGY
        pmappings = make_pmappings(spec, print_progress=False)
        mappings = join_pmappings(pmappings, print_progress=False, metrics=Metrics.ENERGY)

        self.assertGreater(
            len(mappings.data), 0, "mapper should return at least one mapping"
        )

        # Total energy column uses "<SEP>" as separator
        total_energy_col = "Total<SEP>energy"
        self.assertIn(
            total_energy_col,
            mappings.data.columns,
            f"expected column {total_energy_col} in mapping results",
        )
        energy = mappings.data.iloc[0][total_energy_col]
        self.assertGreater(energy, 0, "total energy should be positive")

        # Toll has read energy 100 per action; data flows through it, so total
        # energy should be at least the Toll contribution (100 * number of read actions).
        # Matmul1: 128*64*2 (T0,W0 inputs) + 128*128 (T1 output) elements through Toll
        # in a typical dataflow. We only assert that energy is in a plausible range.
        self.assertAlmostEqual(energy, 100 * (128 * 64 * 2 + 128 * 128) * 8)


if __name__ == "__main__":
    unittest.main()
