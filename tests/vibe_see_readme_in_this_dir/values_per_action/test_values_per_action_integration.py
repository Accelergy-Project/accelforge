"""
End-to-end integration tests for values_per_action / bits_per_action precedence.

Each test loads a Toll-based arch with a different vpa/bpa configuration, runs
the mapper, and checks the resulting total energy against the closed-form
expectation computed from the precedence-selected vpa.
"""

import unittest
from pathlib import Path

from accelforge.frontend.spec import Spec
from accelforge.frontend.mapper.metrics import Metrics

TESTS_DIR = Path(__file__).resolve().parent
INPUT_FILES_DIR = TESTS_DIR / "input_files"
WORKLOAD = INPUT_FILES_DIR / "vpa_precedence.workload.yaml"

# Matmul1: T0[m,n0] + W0[n0,n1] + T1[m,n1] with M=128, N0=64, N1=128.
# Every tensor element traverses the Toll exactly once.
TOTAL_ELEMENTS = 128 * 64 + 64 * 128 + 128 * 128  # 32768
READ_ENERGY = 100


def _energy_for_vpa(vpa):
    return TOTAL_ELEMENTS / vpa * READ_ENERGY


class TestValuesPerActionIntegration(unittest.TestCase):
    def _run(self, arch_yaml):
        spec = Spec.from_yaml(str(arch_yaml), str(WORKLOAD))
        spec.mapper.metrics = Metrics.ENERGY
        mappings = spec.map_workload_to_arch(print_progress=False)
        self.assertGreater(
            len(mappings.data), 0, "mapper should return at least one mapping"
        )
        return mappings.data.iloc[0]["Total<SEP>energy"]

    def test_action_vpa_wins(self):
        """action.values_per_action overrides component.values_per_action."""
        energy = self._run(INPUT_FILES_DIR / "vpa_precedence_action_wins.arch.yaml")
        self.assertAlmostEqual(energy, _energy_for_vpa(2))

    def test_component_vpa_used_when_action_empty(self):
        """component.values_per_action used when action's is empty."""
        energy = self._run(INPUT_FILES_DIR / "vpa_precedence_component_wins.arch.yaml")
        self.assertAlmostEqual(energy, _energy_for_vpa(5))

    def test_action_bpa_wins_when_both_vpa_empty(self):
        """action.bits_per_action overrides component.bits_per_action; vpa = bpa/bpv."""
        energy = self._run(INPUT_FILES_DIR / "vpa_precedence_action_bpa.arch.yaml")
        # action.bpa=8, component.bpv=4 -> vpa=2
        self.assertAlmostEqual(energy, _energy_for_vpa(2))

    def test_component_bpa_used_when_action_bpa_and_vpa_empty(self):
        """component.bits_per_action used when nothing else set; vpa = bpa/bpv."""
        energy = self._run(INPUT_FILES_DIR / "vpa_precedence_component_bpa.arch.yaml")
        # component.bpa=16, component.bpv=4 -> vpa=4
        self.assertAlmostEqual(energy, _energy_for_vpa(4))

    def test_default_bpa_is_one(self):
        """Nothing set: action.bpa defaults to 1; vpa = 1/workload_bpv."""
        energy = self._run(INPUT_FILES_DIR / "vpa_precedence_default.arch.yaml")
        # default bpa=1, workload bpv=8 -> vpa = 0.125
        self.assertAlmostEqual(energy, _energy_for_vpa(0.125))


if __name__ == "__main__":
    unittest.main()
