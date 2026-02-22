"""
Tests for the top-level Spec class:
  - Spec defaults
  - Spec.from_yaml
  - Spec._spec_eval_expressions
  - Spec._get_flattened_architecture
  - Integration with workload, arch, config, variables, renames, mapper, model
"""

import unittest
from pathlib import Path

from accelforge.frontend.spec import Spec, Specification
from accelforge.frontend.workload import Workload
from accelforge.frontend.arch import Arch, Memory, Compute, Container, Fork
from accelforge.frontend.arch import Spatial as ArchSpatial
from accelforge.frontend.config import Config
from accelforge.frontend.variables import Variables
from accelforge.frontend.renames import Renames
from accelforge.frontend.mapper.ffm import FFM
from accelforge.frontend.model import Model
from accelforge.frontend.mapping import Mapping

_REPO_ROOT = Path(__file__).parent.parent.parent
EXAMPLES_DIR = _REPO_ROOT / "examples"


# ============================================================================
# Spec Defaults
# ============================================================================


class TestSpecDefaults(unittest.TestCase):
    """Tests for Spec default construction."""

    def test_default_creation(self):
        s = Spec()
        self.assertIsNotNone(s)

    def test_default_arch(self):
        s = Spec()
        self.assertIsInstance(s.arch, Arch)

    def test_default_workload(self):
        s = Spec()
        self.assertIsInstance(s.workload, Workload)

    def test_default_mapping(self):
        s = Spec()
        self.assertIsInstance(s.mapping, Mapping)

    def test_default_config(self):
        s = Spec()
        self.assertIsInstance(s.config, Config)

    def test_default_variables(self):
        s = Spec()
        self.assertIsInstance(s.variables, Variables)

    def test_default_renames(self):
        s = Spec()
        self.assertIsInstance(s.renames, Renames)

    def test_default_mapper(self):
        s = Spec()
        self.assertIsInstance(s.mapper, FFM)

    def test_default_model(self):
        s = Spec()
        self.assertIsInstance(s.model, Model)

    def test_specification_alias(self):
        self.assertIs(Specification, Spec)


# ============================================================================
# Spec Construction with Inputs
# ============================================================================


class TestSpecConstruction(unittest.TestCase):
    """Tests for constructing Spec with explicit components."""

    def test_spec_with_workload(self):
        w = Workload(
            rank_sizes={"M": 64, "K": 32, "N": 64},
            bits_per_value={"All": 8},
            einsums=[
                {
                    "name": "Matmul",
                    "tensor_accesses": [
                        {"name": "A", "projection": ["m", "k"]},
                        {"name": "B", "projection": ["k", "n"]},
                        {"name": "C", "projection": ["m", "n"], "output": True},
                    ],
                }
            ],
        )
        s = Spec(workload=w)
        self.assertEqual(len(s.workload.einsums), 1)

    def test_spec_with_arch(self):
        a = Arch(
            nodes=[
                Memory(name="MainMem", size=1_000_000),
                Compute(name="MAC"),
            ]
        )
        s = Spec(arch=a)
        self.assertEqual(len(s.arch.nodes), 2)

    def test_spec_with_variables(self):
        v = Variables(x=10, y=20)
        s = Spec(variables=v)
        self.assertEqual(s.variables.x, 10)

    def test_spec_with_config(self):
        c = Config(use_installed_component_models=False)
        s = Spec(config=c)
        self.assertFalse(s.config.use_installed_component_models)


# ============================================================================
# Spec from YAML
# ============================================================================


class TestSpecFromYAML(unittest.TestCase):
    """Tests for Spec.from_yaml with real YAML files."""

    def test_load_concise_gpt(self):
        yaml_path = EXAMPLES_DIR / "workloads" / "gpt3_6.7B.yaml"
        if not yaml_path.exists():
            self.skipTest(f"YAML file not found: {yaml_path}")
        spec = Spec.from_yaml(yaml_path)
        self.assertIsNotNone(spec.workload)
        self.assertGreater(len(spec.workload.einsums), 0)

    def test_load_regular_gpt(self):
        yaml_path = EXAMPLES_DIR / "workloads" / "gpt3_6.7B.yaml"
        if not yaml_path.exists():
            self.skipTest(f"YAML file not found: {yaml_path}")
        spec = Spec.from_yaml(yaml_path)
        self.assertIsNotNone(spec.workload)
        self.assertGreater(len(spec.workload.einsums), 0)

    def test_load_three_matmuls(self):
        yaml_path = EXAMPLES_DIR / "workloads" / "three_matmuls_annotated.yaml"
        if not yaml_path.exists():
            self.skipTest(f"YAML file not found: {yaml_path}")
        spec = Spec.from_yaml(yaml_path)
        self.assertEqual(len(spec.workload.einsums), 3)

    def test_load_matmuls_with_jinja(self):
        yaml_path = EXAMPLES_DIR / "workloads" / "matmuls.yaml"
        if not yaml_path.exists():
            self.skipTest(f"YAML file not found: {yaml_path}")
        spec = Spec.from_yaml(yaml_path, jinja_parse_data={"N_EINSUMS": 2})
        self.assertEqual(len(spec.workload.einsums), 2)

    def test_jinja_variables_propagate(self):
        yaml_path = EXAMPLES_DIR / "workloads" / "matmuls.yaml"
        if not yaml_path.exists():
            self.skipTest(f"YAML file not found: {yaml_path}")
        spec = Spec.from_yaml(
            yaml_path, jinja_parse_data={"N_EINSUMS": 3, "M": 256, "KN": 64}
        )
        self.assertEqual(len(spec.workload.einsums), 3)


# ============================================================================
# Spec Evaluation
# ============================================================================


class TestSpecEvaluation(unittest.TestCase):
    """Tests for Spec._spec_eval_expressions."""

    def test_eval_basic_spec(self):
        yaml_path = EXAMPLES_DIR / "workloads" / "three_matmuls_annotated.yaml"
        if not yaml_path.exists():
            self.skipTest(f"YAML file not found: {yaml_path}")
        spec = Spec.from_yaml(yaml_path)
        evaluated = spec._spec_eval_expressions()
        self.assertIsNotNone(evaluated)
        self.assertTrue(getattr(evaluated, "_evaluated", False))

    def test_eval_with_einsum_name(self):
        yaml_path = EXAMPLES_DIR / "workloads" / "three_matmuls_annotated.yaml"
        if not yaml_path.exists():
            self.skipTest(f"YAML file not found: {yaml_path}")
        spec = Spec.from_yaml(yaml_path)
        evaluated = spec._spec_eval_expressions(einsum_name="Matmul1")
        self.assertTrue(getattr(evaluated, "_evaluated", False))

    def test_eval_concise_gpt(self):
        yaml_path = EXAMPLES_DIR / "workloads" / "gpt3_6.7B.yaml"
        if not yaml_path.exists():
            self.skipTest(f"YAML file not found: {yaml_path}")
        spec = Spec.from_yaml(yaml_path)
        evaluated = spec._spec_eval_expressions()
        # Check that bits_per_value was properly resolved
        for einsum in evaluated.workload.einsums:
            for ta in einsum.tensor_accesses:
                self.assertIsNotNone(ta.bits_per_value)
                self.assertEqual(ta.bits_per_value, 8)

    def test_eval_partial_arch_only(self):
        yaml_path = EXAMPLES_DIR / "workloads" / "three_matmuls_annotated.yaml"
        if not yaml_path.exists():
            self.skipTest(f"YAML file not found: {yaml_path}")
        spec = Spec.from_yaml(yaml_path)
        evaluated = spec._spec_eval_expressions(eval_arch=True, eval_non_arch=False)
        # Should still work; workload is not evaluated
        self.assertIsNotNone(evaluated)

    def test_eval_partial_non_arch_only(self):
        yaml_path = EXAMPLES_DIR / "workloads" / "three_matmuls_annotated.yaml"
        if not yaml_path.exists():
            self.skipTest(f"YAML file not found: {yaml_path}")
        spec = Spec.from_yaml(yaml_path)
        evaluated = spec._spec_eval_expressions(eval_arch=False, eval_non_arch=True)
        self.assertIsNotNone(evaluated)


# ============================================================================
# Spec with Persistent Tensors
# ============================================================================


class TestSpecPersistentTensors(unittest.TestCase):
    """Test that persistent_tensors field properly marks tensors."""

    def test_persistent_tensors_evaluated(self):
        yaml_path = EXAMPLES_DIR / "workloads" / "gpt3_6.7B.yaml"
        if not yaml_path.exists():
            self.skipTest(f"YAML file not found: {yaml_path}")
        spec = Spec.from_yaml(yaml_path)
        evaluated = spec._spec_eval_expressions()

        weight_tensors = {"WV", "WK", "WQ", "WZ", "WFFA", "WFFB"}
        for einsum in evaluated.workload.einsums:
            for ta in einsum.tensor_accesses:
                if ta.name in weight_tensors:
                    self.assertTrue(
                        ta.persistent,
                        f"{ta.name} in {einsum.name} should be persistent",
                    )


# ============================================================================
# Spec with Concise and Regular YAML Equivalence
# ============================================================================


class TestSpecConciseRegularEquivalence(unittest.TestCase):
    """Test that concise and regular YAML produce equivalent results."""

    def test_same_einsum_names(self):
        concise_path = EXAMPLES_DIR / "workloads" / "gpt3_6.7B.yaml"
        regular_path = EXAMPLES_DIR / "misc" / "gpt3_6.7B_verbose_annotated.yaml"
        if not concise_path.exists() or not regular_path.exists():
            self.skipTest("YAML files not found")

        concise = Spec.from_yaml(concise_path)._spec_eval_expressions()
        regular = Spec.from_yaml(regular_path)._spec_eval_expressions()

        concise_names = {e.name for e in concise.workload.einsums}
        regular_names = {e.name for e in regular.workload.einsums}
        self.assertEqual(concise_names, regular_names)

    def test_same_rank_size_keys(self):
        concise_path = EXAMPLES_DIR / "workloads" / "gpt3_6.7B.yaml"
        regular_path = EXAMPLES_DIR / "misc" / "gpt3_6.7B_verbose_annotated.yaml"
        if not concise_path.exists() or not regular_path.exists():
            self.skipTest("YAML files not found")

        concise = Spec.from_yaml(concise_path)
        regular = Spec.from_yaml(regular_path)
        self.assertEqual(
            set(concise.workload.rank_sizes.keys()),
            set(regular.workload.rank_sizes.keys()),
        )


if __name__ == "__main__":
    unittest.main()
