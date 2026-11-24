"""
Architectures with `configurations`: the mapper generates pmapping templates under each
configuration allowed for the Einsum, records the configuration on each Compute node,
and the model evaluates a mapping under the recorded configuration.
"""

import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from accelforge.frontend.mapper.metrics import Metrics
from accelforge.frontend.mapping import Compute
from accelforge.frontend.spec import Spec
from accelforge.mapper.FFM import make_pmappings
from accelforge.model.main import evaluate_mapping

INPUT_FILES_DIR = Path(__file__).resolve().parent / "input_files"
ARCH = INPUT_FILES_DIR / "configurations.arch.yaml"
WORKLOAD = INPUT_FILES_DIR / "configurations.workload.yaml"


def _spec():
    spec = Spec.from_yaml(str(ARCH), str(WORKLOAD))
    spec.mapper.metrics = Metrics.ENERGY
    spec.model.metrics = Metrics.ENERGY
    return spec


class TestConfigurations(unittest.TestCase):
    def test_templates_per_configuration(self):
        pmappings = make_pmappings(
            _spec(), can_combine_multiple_runs=False, print_progress=False
        )
        counts = {
            einsum: Counter(j.mapping.nodes[-1].arch_configuration for j in jobs)
            for einsum, jobs in pmappings.einsum2jobs.items()
        }
        # a_and_b is only allowed for matmuls
        self.assertEqual(set(counts["Matmul"]), {"a", "b", "a_and_b"})
        self.assertEqual(set(counts["Bias"]), {"a", "b"})
        # An Einsum gets the same templates under each of its configurations
        for count in counts.values():
            self.assertEqual(len(set(count.values())), 1)

    def test_only_flags_of_active_configuration_are_true(self):
        spec = _spec()
        mac_energy = {}
        for configuration in [c.name for c in spec.arch.configurations]:
            evaluated = spec._spec_eval_expressions(
                einsum_name="Matmul", configuration=configuration
            )
            mac_energy[configuration] = evaluated.arch["mac"].actions["compute"].energy
        # The matmul costs 4 - A - 2 * B. a_and_b shares its flags with a and b
        self.assertEqual(mac_energy, {"a": 3, "b": 2, "a_and_b": 1})

    def test_configuration_is_required(self):
        with self.assertRaisesRegex(ValueError, "no configuration was given"):
            _spec()._spec_eval_expressions(einsum_name="Matmul")

    def test_best_mapping_uses_cheapest_configuration(self):
        spec = _spec()
        mappings = spec.map_workload_to_arch(print_progress=False)
        energy = np.atleast_1d(mappings.energy())
        best = int(np.argmin(energy))
        mapping = mappings.data.iloc[best]["Total<SEP>mapping"](_for_model=True)
        computes = {c.einsum: c for c in mapping.get_nodes_of_type(Compute)}
        self.assertEqual(computes["Matmul"].arch_configuration, "a_and_b")
        self.assertIn("Arch configuration a_and_b", str(computes["Matmul"]))

        spec.mapping = mapping
        evaluated = evaluate_mapping(
            spec,
            flattened_arches=mappings._flattened_arches,
            evaluated_specs=mappings._evaluated_specs,
        )
        self.assertAlmostEqual(
            float(evaluated.data["Total<SEP>energy"].iloc[0]), float(energy[best])
        )


if __name__ == "__main__":
    unittest.main()
