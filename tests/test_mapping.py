"""
Tests that evaluate golden mappings extracted from the mapper's optimal results.

These tests load pre-computed optimal mappings (from tests/input_files/mapper_golden_mappings/)
and verify that the model evaluates them to the expected energy and latency values. This
validates the model evaluation pipeline without needing to run the slow mapper.
"""
import unittest

import accelforge as af
from accelforge.frontend.spec import Spec
from accelforge.frontend.mapping.mapping import Spatial
from accelforge.model.main import evaluate_mapping
from accelforge.util.parallel import set_n_parallel_jobs

try:
    from .paths import EXAMPLES_DIR, CURRENT_DIR
except ImportError:
    from paths import EXAMPLES_DIR, CURRENT_DIR

set_n_parallel_jobs(1)

GOLDEN_DIR = CURRENT_DIR / "input_files" / "mapper_golden_mappings"

M_SHAPE = 64
KN_SHAPE = 64
FANOUT = 4


def _populate_spatial_component_objects(spec):
    """Populate component_object on Spatial mapping nodes from the architecture.

    When mappings are loaded from YAML, Spatial nodes have component_object=None.
    The model requires this to be populated for evaluation.
    """
    spatial_nodes = spec.mapping.get_nodes_of_type(Spatial)
    if not spatial_nodes:
        return

    for node in spatial_nodes:
        node.component_object = spec.arch.nodes[node.component]


class TestGoldenMappingSimple(unittest.TestCase):
    """Tests golden mappings on the simple architecture (no fanout)."""

    def test_one_matmul_energy(self):
        spec = Spec.from_yaml(
            af.examples.arches.simple,
            af.examples.workloads.matmuls,
            GOLDEN_DIR / "one_matmul_simple.yaml",
            jinja_parse_data={"N_EINSUMS": 1, "M": M_SHAPE, "KN": KN_SHAPE},
        )
        result = evaluate_mapping(spec)
        self.assertAlmostEqual(float(result.energy()), 8617984.0, places=0)

    def test_one_matmul_latency(self):
        spec = Spec.from_yaml(
            af.examples.arches.simple,
            af.examples.workloads.matmuls,
            GOLDEN_DIR / "one_matmul_simple.yaml",
            jinja_parse_data={"N_EINSUMS": 1, "M": M_SHAPE, "KN": KN_SHAPE},
        )
        result = evaluate_mapping(spec)
        latency = result.latency(per_einsum=True)
        self.assertAlmostEqual(float(latency["Matmul0"]), M_SHAPE * KN_SHAPE**2)

    def test_two_matmuls_energy(self):
        spec = Spec.from_yaml(
            af.examples.arches.simple,
            af.examples.workloads.matmuls,
            GOLDEN_DIR / "two_matmuls_simple.yaml",
            jinja_parse_data={"N_EINSUMS": 2, "M": M_SHAPE, "KN": KN_SHAPE},
        )
        result = evaluate_mapping(spec)
        self.assertAlmostEqual(float(result.energy()), 17235968.0, places=0)

    def test_two_matmuls_latency(self):
        spec = Spec.from_yaml(
            af.examples.arches.simple,
            af.examples.workloads.matmuls,
            GOLDEN_DIR / "two_matmuls_simple.yaml",
            jinja_parse_data={"N_EINSUMS": 2, "M": M_SHAPE, "KN": KN_SHAPE},
        )
        result = evaluate_mapping(spec)
        latency = result.latency(per_einsum=True)
        self.assertAlmostEqual(float(latency["Matmul0"]), M_SHAPE * KN_SHAPE**2)
        self.assertAlmostEqual(float(latency["Matmul1"]), M_SHAPE * KN_SHAPE**2)


class TestGoldenMappingFanoutOneMatmul(unittest.TestCase):
    """Tests golden mappings on fanout architectures with 1 matmul."""

    def _check_latency(self, arch_name):
        spec = Spec.from_yaml(
            EXAMPLES_DIR / "arches" / "fanout_variations" / f"{arch_name}.yaml",
            EXAMPLES_DIR / "workloads" / "matmuls.yaml",
            GOLDEN_DIR / f"fanout_{arch_name}_1matmul.yaml",
            jinja_parse_data={"N_EINSUMS": 1, "M": M_SHAPE, "KN": KN_SHAPE},
        )
        _populate_spatial_component_objects(spec)
        result = evaluate_mapping(spec)
        latency = result.latency(per_einsum=True)
        self.assertIn("Matmul0", latency)
        self.assertAlmostEqual(
            float(latency["Matmul0"]),
            M_SHAPE * KN_SHAPE**2 / FANOUT,
        )

    def test_at_mac(self):
        self._check_latency("at_mac")

    def test_at_glb(self):
        self._check_latency("at_glb")

    def test_at_mac_with_fanout_node(self):
        self._check_latency("at_mac_with_fanout_node")

    def test_at_glb_with_fanout_node(self):
        self._check_latency("at_glb_with_fanout_node")


class TestGoldenMappingFanoutTwoMatmuls(unittest.TestCase):
    """Tests golden mappings on fanout architectures with 2 matmuls."""

    def _check_latency(self, arch_name):
        spec = Spec.from_yaml(
            EXAMPLES_DIR / "arches" / "fanout_variations" / f"{arch_name}.yaml",
            EXAMPLES_DIR / "workloads" / "matmuls.yaml",
            GOLDEN_DIR / f"fanout_{arch_name}_2matmuls.yaml",
            jinja_parse_data={"N_EINSUMS": 2, "M": M_SHAPE, "KN": KN_SHAPE},
        )
        _populate_spatial_component_objects(spec)
        result = evaluate_mapping(spec)
        latency = result.latency(per_einsum=True)
        self.assertIn("Matmul0", latency)
        self.assertIn("Matmul1", latency)
        self.assertAlmostEqual(
            float(latency["Matmul0"]),
            M_SHAPE * KN_SHAPE**2 / FANOUT,
        )
        self.assertAlmostEqual(
            float(latency["Matmul1"]),
            M_SHAPE * KN_SHAPE**2 / FANOUT,
        )

    def test_at_mac(self):
        self._check_latency("at_mac")

    def test_at_glb(self):
        self._check_latency("at_glb")

    def test_at_mac_with_fanout_node(self):
        self._check_latency("at_mac_with_fanout_node")

    def test_at_glb_with_fanout_node(self):
        self._check_latency("at_glb_with_fanout_node")


class TestGoldenMappingFanoutConstraints(unittest.TestCase):
    """Tests golden mapping on fanout architecture with constraints."""

    def test_at_mac_constraints(self):
        spec = Spec.from_yaml(
            EXAMPLES_DIR / "arches" / "fanout_variations" / "at_mac_with_constraints.yaml",
            EXAMPLES_DIR / "workloads" / "matmuls.yaml",
            GOLDEN_DIR / "fanout_at_mac_constraints.yaml",
            jinja_parse_data={"N_EINSUMS": 1, "M": M_SHAPE, "KN": KN_SHAPE},
        )
        _populate_spatial_component_objects(spec)
        result = evaluate_mapping(spec)
        latency = result.latency(per_einsum=True)
        self.assertIn("Matmul0", latency)
        self.assertAlmostEqual(
            float(latency["Matmul0"]),
            M_SHAPE * KN_SHAPE**2 / FANOUT,
        )


if __name__ == "__main__":
    unittest.main()
