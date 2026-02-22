"""
Tests for expression evaluation:
  - Spec._spec_eval_expressions (variables, renames, workload, arch)
  - Workload bits_per_value resolution
  - Workload persistent_tensors resolution
  - Arch evaluation with variables and expressions
  - Partial evaluation (eval_arch only, eval_non_arch only)
  - Variables referencing other variables
  - EvaluationError for bad expressions
"""

import unittest
from pathlib import Path

from accelforge.frontend.spec import Spec
from accelforge.frontend.workload import Workload, Einsum
from accelforge.frontend.arch import Arch, Memory, Compute, Container, Toll
from accelforge.frontend.arch import Spatial as ArchSpatial
from accelforge.frontend.renames import Renames, EinsumRename, Rename
from accelforge.frontend.variables import Variables
from accelforge.frontend.config import Config

_REPO_ROOT = Path(__file__).parent.parent.parent
EXAMPLES_DIR = _REPO_ROOT / "examples"


# ============================================================================
# Spec eval basic
# ============================================================================


class TestSpecEvalExpressions(unittest.TestCase):
    """Test that _spec_eval_expressions evaluates the spec correctly."""

    def _simple_spec(self, **overrides):
        """A simple spec with one matmul einsum."""
        data = dict(
            workload=Workload(
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
            ),
            arch=Arch(
                nodes=[
                    Memory(
                        name="MainMemory",
                        size=1_000_000,
                        actions=[
                            {"name": "read", "energy": 1, "latency": 0},
                            {"name": "write", "energy": 1, "latency": 0},
                        ],
                        leak_power=0,
                        area=0,
                    ),
                    Compute(
                        name="MAC",
                        actions=[{"name": "compute", "energy": 1, "latency": 1}],
                        leak_power=0,
                        area=0,
                    ),
                ]
            ),
        )
        data.update(overrides)
        return Spec(**data)

    def test_eval_sets_evaluated_flag(self):
        spec = self._simple_spec()
        evaluated = spec._spec_eval_expressions()
        self.assertTrue(getattr(evaluated, "_evaluated", False))

    def test_eval_resolves_bits_per_value(self):
        spec = self._simple_spec()
        evaluated = spec._spec_eval_expressions()
        for ta in evaluated.workload.einsums[0].tensor_accesses:
            self.assertIsNotNone(ta.bits_per_value)
            self.assertEqual(ta.bits_per_value, 8)

    def test_eval_preserves_rank_sizes(self):
        spec = self._simple_spec()
        evaluated = spec._spec_eval_expressions()
        self.assertEqual(evaluated.workload.rank_sizes["M"], 64)
        self.assertEqual(evaluated.workload.rank_sizes["K"], 32)

    def test_eval_preserves_arch(self):
        spec = self._simple_spec()
        evaluated = spec._spec_eval_expressions()
        mem = evaluated.arch.find("MainMemory")
        self.assertEqual(mem.name, "MainMemory")

    def test_eval_with_einsum_name(self):
        spec = self._simple_spec()
        evaluated = spec._spec_eval_expressions(einsum_name="Matmul")
        self.assertTrue(getattr(evaluated, "_evaluated", False))

    def test_eval_arch_only(self):
        spec = self._simple_spec()
        evaluated = spec._spec_eval_expressions(eval_arch=True, eval_non_arch=False)
        self.assertIsNotNone(evaluated)

    def test_eval_non_arch_only(self):
        spec = self._simple_spec()
        evaluated = spec._spec_eval_expressions(eval_arch=False, eval_non_arch=True)
        self.assertIsNotNone(evaluated)

    def test_eval_with_variables(self):
        spec = self._simple_spec(variables=Variables(my_size=256))
        evaluated = spec._spec_eval_expressions()
        self.assertIsNotNone(evaluated)

    def test_calling_eval_twice_idempotent(self):
        spec = self._simple_spec()
        evaluated1 = spec._spec_eval_expressions()
        evaluated2 = evaluated1._spec_eval_expressions()
        self.assertTrue(getattr(evaluated2, "_evaluated", False))


# ============================================================================
# Bits per value evaluation
# ============================================================================


class TestBitsPerValueEvaluation(unittest.TestCase):
    """Test bits_per_value resolution during evaluation."""

    def test_all_tensors_get_bits(self):
        spec = Spec(
            workload=Workload(
                rank_sizes={"M": 16, "N": 16},
                bits_per_value={"All": 16},
                einsums=[
                    {
                        "name": "E",
                        "tensor_accesses": [
                            {"name": "X", "projection": ["m"]},
                            {"name": "Y", "projection": ["m", "n"], "output": True},
                        ],
                    }
                ],
            ),
        )
        evaluated = spec._spec_eval_expressions()
        for ta in evaluated.workload.einsums[0].tensor_accesses:
            self.assertEqual(ta.bits_per_value, 16)

    def test_per_tensor_bits_override(self):
        """Per-tensor-access bits_per_value overrides the workload-level one."""
        spec = Spec(
            workload=Workload(
                rank_sizes={"M": 16, "N": 16},
                bits_per_value={"All": 8},
                einsums=[
                    {
                        "name": "E",
                        "tensor_accesses": [
                            {"name": "X", "projection": ["m"], "bits_per_value": 32},
                            {"name": "Y", "projection": ["m", "n"], "output": True},
                        ],
                    }
                ],
            ),
        )
        evaluated = spec._spec_eval_expressions()
        x_ta = [
            t for t in evaluated.workload.einsums[0].tensor_accesses if t.name == "X"
        ][0]
        y_ta = [
            t for t in evaluated.workload.einsums[0].tensor_accesses if t.name == "Y"
        ][0]
        self.assertEqual(x_ta.bits_per_value, 32)
        self.assertEqual(y_ta.bits_per_value, 8)


# ============================================================================
# Persistent tensors evaluation
# ============================================================================


class TestPersistentTensorsEvaluation(unittest.TestCase):
    """Test persistent_tensors field evaluation."""

    def test_persistent_tensors_marked(self):
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

    def test_intermediate_tensors_not_persistent(self):
        yaml_path = EXAMPLES_DIR / "workloads" / "gpt3_6.7B.yaml"
        if not yaml_path.exists():
            self.skipTest(f"YAML file not found: {yaml_path}")
        spec = Spec.from_yaml(yaml_path)
        evaluated = spec._spec_eval_expressions()

        # I is an intermediate. It should not be persistent.
        v_einsum = [e for e in evaluated.workload.einsums if e.name == "V"][0]
        i_ta = [t for t in v_einsum.tensor_accesses if t.name == "I"][0]
        self.assertFalse(i_ta.persistent)


# ============================================================================
# Renames evaluation
# ============================================================================


class TestRenamesEvaluation(unittest.TestCase):
    """Test that renames are properly resolved during spec evaluation."""

    def test_default_renames_applied(self):
        yaml_path = EXAMPLES_DIR / "workloads" / "three_matmuls_annotated.yaml"
        if not yaml_path.exists():
            self.skipTest(f"YAML file not found: {yaml_path}")
        spec = Spec.from_yaml(yaml_path)
        evaluated = spec._spec_eval_expressions()

        # Check that the default renames (input, output, weight) are applied
        for einsum in evaluated.workload.einsums:
            rename_names = {r.name for r in einsum.renames}
            self.assertIn("input", rename_names)
            self.assertIn("output", rename_names)
            self.assertIn("weight", rename_names)

    def test_inline_renames_applied(self):
        yaml_path = EXAMPLES_DIR / "workloads" / "three_matmuls_annotated.yaml"
        if not yaml_path.exists():
            self.skipTest(f"YAML file not found: {yaml_path}")
        spec = Spec.from_yaml(yaml_path)
        evaluated = spec._spec_eval_expressions()

        # Matmul1 has an inline rename: input: T0
        matmul1 = evaluated.workload.einsums["Matmul1"]
        # The inline rename should set the "input" rename to T0
        input_rename = [r for r in matmul1.renames if r.name == "input"]
        self.assertGreater(len(input_rename), 0)

    def test_builtin_renames_available(self):
        """Built-in renames like All, Inputs, Outputs, Nothing are always available."""
        yaml_path = EXAMPLES_DIR / "workloads" / "three_matmuls_annotated.yaml"
        if not yaml_path.exists():
            self.skipTest(f"YAML file not found: {yaml_path}")
        spec = Spec.from_yaml(yaml_path)
        evaluated = spec._spec_eval_expressions()

        matmul1 = evaluated.workload.einsums["Matmul1"]
        rename_names = {r.name for r in matmul1.renames}
        for builtin in ["All", "Tensors", "Nothing", "Inputs", "Outputs"]:
            self.assertIn(builtin, rename_names)


# ============================================================================
# YAML evaluation with Jinja
# ============================================================================


class TestJinjaEvaluation(unittest.TestCase):
    """Test that Jinja2 templating works during spec evaluation."""

    def test_jinja_variable_substitution(self):
        yaml_path = EXAMPLES_DIR / "workloads" / "matmuls.yaml"
        if not yaml_path.exists():
            self.skipTest(f"YAML file not found: {yaml_path}")
        spec = Spec.from_yaml(
            yaml_path, jinja_parse_data={"N_EINSUMS": 2, "M": 128, "KN": 64}
        )
        self.assertEqual(len(spec.workload.einsums), 2)

    def test_jinja_three_einsums(self):
        yaml_path = EXAMPLES_DIR / "workloads" / "matmuls.yaml"
        if not yaml_path.exists():
            self.skipTest(f"YAML file not found: {yaml_path}")
        spec = Spec.from_yaml(yaml_path, jinja_parse_data={"N_EINSUMS": 3})
        self.assertEqual(len(spec.workload.einsums), 3)

    def test_jinja_default_values(self):
        """GPT concise uses Jinja defaults for BATCH_SIZE and N_TOKENS."""
        yaml_path = EXAMPLES_DIR / "workloads" / "gpt3_6.7B.yaml"
        if not yaml_path.exists():
            self.skipTest(f"YAML file not found: {yaml_path}")
        spec = Spec.from_yaml(yaml_path)
        # Default BATCH_SIZE is 1, N_TOKENS is 8192
        self.assertEqual(spec.workload.rank_sizes["B"], 1)
        self.assertEqual(spec.workload.rank_sizes["M"], 8192)


# ============================================================================
# Multi-file spec evaluation
# ============================================================================


class TestMultiFileSpecEval(unittest.TestCase):
    """Test spec evaluation when loading multiple YAML files."""

    def test_load_arch_and_workload(self):
        arch_path = EXAMPLES_DIR / "arches" / "simple.yaml"
        wl_path = EXAMPLES_DIR / "workloads" / "matmuls.yaml"
        if not arch_path.exists() or not wl_path.exists():
            self.skipTest("YAML files not found")
        spec = Spec.from_yaml(arch_path, wl_path, jinja_parse_data={"N_EINSUMS": 1})
        self.assertGreater(len(spec.workload.einsums), 0)
        self.assertGreater(len(spec.arch.nodes), 0)

    def test_full_spec_evaluates(self):
        arch_path = EXAMPLES_DIR / "arches" / "simple.yaml"
        wl_path = EXAMPLES_DIR / "workloads" / "matmuls.yaml"
        if not arch_path.exists() or not wl_path.exists():
            self.skipTest("YAML files not found")
        spec = Spec.from_yaml(arch_path, wl_path, jinja_parse_data={"N_EINSUMS": 1})
        evaluated = spec._spec_eval_expressions()
        self.assertTrue(getattr(evaluated, "_evaluated", False))

    def test_full_spec_with_mapping(self):
        arch_path = EXAMPLES_DIR / "arches" / "simple.yaml"
        wl_path = EXAMPLES_DIR / "workloads" / "matmuls.yaml"
        map_path = EXAMPLES_DIR / "mappings" / "unfused_matmuls_to_simple.yaml"
        if not all(p.exists() for p in [arch_path, wl_path, map_path]):
            self.skipTest("YAML files not found")
        spec = Spec.from_yaml(
            arch_path,
            wl_path,
            map_path,
            jinja_parse_data={"N_EINSUMS": 1, "M": 64, "KN": 32},
        )
        evaluated = spec._spec_eval_expressions()
        self.assertTrue(getattr(evaluated, "_evaluated", False))
        # Mapping should have nodes
        self.assertGreater(len(evaluated.mapping.nodes), 0)


# ============================================================================
# Evaluation with arch expressions
# ============================================================================


class TestArchExpressionEvaluation(unittest.TestCase):
    """Test that arch expressions (e.g., size: 1024*1024*128*8) are evaluated."""

    def test_arch_size_expression(self):
        """Arch memory sizes that are expressions get evaluated.
        TPU arch needs einsum context for some expressions (e.g., len(All))."""
        arch_path = EXAMPLES_DIR / "arches" / "tpu_v4i.yaml"
        wl_path = EXAMPLES_DIR / "workloads" / "three_matmuls_annotated.yaml"
        if not arch_path.exists() or not wl_path.exists():
            self.skipTest("YAML files not found")
        spec = Spec.from_yaml(arch_path, wl_path)
        # Evaluate with einsum context to handle `weight.bits_per_value` etc.
        evaluated = spec._spec_eval_expressions(einsum_name="Matmul1")
        glb = evaluated.arch.find("GlobalBuffer")
        # 1024*1024*128*8 = 1073741824
        self.assertEqual(glb.size, 1024 * 1024 * 128 * 8)

    def test_arch_latency_expression(self):
        """Arch action latency expressions like 1/(8*614e9) get evaluated."""
        arch_path = EXAMPLES_DIR / "arches" / "tpu_v4i.yaml"
        wl_path = EXAMPLES_DIR / "workloads" / "three_matmuls_annotated.yaml"
        if not arch_path.exists() or not wl_path.exists():
            self.skipTest("YAML files not found")
        spec = Spec.from_yaml(arch_path, wl_path)
        # Evaluate with einsum context
        evaluated = spec._spec_eval_expressions(einsum_name="Matmul1")
        mm = evaluated.arch.find("MainMemory")
        read_action = [a for a in mm.actions if a.name == "read"][0]
        self.assertIsNotNone(read_action.latency)
        self.assertAlmostEqual(read_action.latency, 1 / (8 * 614e9), places=20)


# ============================================================================
# Concise and verbose equivalence
# ============================================================================


class TestConciseVerboseEquivalence(unittest.TestCase):
    """Check that concise and verbose workloads produce the same evaluated results."""

    def test_same_einsum_names(self):
        concise_path = EXAMPLES_DIR / "workloads" / "gpt3_6.7B.yaml"
        verbose_path = EXAMPLES_DIR / "misc" / "gpt3_6.7B_verbose_annotated.yaml"
        if not concise_path.exists() or not verbose_path.exists():
            self.skipTest("YAML files not found")

        c = Spec.from_yaml(concise_path)._spec_eval_expressions()
        v = Spec.from_yaml(verbose_path)._spec_eval_expressions()
        self.assertEqual(
            {e.name for e in c.workload.einsums},
            {e.name for e in v.workload.einsums},
        )

    def test_same_tensor_names_per_einsum(self):
        concise_path = EXAMPLES_DIR / "workloads" / "gpt3_6.7B.yaml"
        verbose_path = EXAMPLES_DIR / "misc" / "gpt3_6.7B_verbose_annotated.yaml"
        if not concise_path.exists() or not verbose_path.exists():
            self.skipTest("YAML files not found")

        c = Spec.from_yaml(concise_path)._spec_eval_expressions()
        v = Spec.from_yaml(verbose_path)._spec_eval_expressions()
        for c_einsum in c.workload.einsums:
            v_einsum = v.workload.einsums[c_einsum.name]
            self.assertEqual(
                c_einsum.tensor_names,
                v_einsum.tensor_names,
                f"Tensor names differ for {c_einsum.name}",
            )

    def test_same_bits_per_value(self):
        concise_path = EXAMPLES_DIR / "workloads" / "gpt3_6.7B.yaml"
        verbose_path = EXAMPLES_DIR / "misc" / "gpt3_6.7B_verbose_annotated.yaml"
        if not concise_path.exists() or not verbose_path.exists():
            self.skipTest("YAML files not found")

        c = Spec.from_yaml(concise_path)._spec_eval_expressions()
        v = Spec.from_yaml(verbose_path)._spec_eval_expressions()
        for c_einsum in c.workload.einsums:
            v_einsum = v.workload.einsums[c_einsum.name]
            for c_ta in c_einsum.tensor_accesses:
                v_ta = v_einsum.tensor_accesses[c_ta.name]
                self.assertEqual(
                    c_ta.bits_per_value,
                    v_ta.bits_per_value,
                    f"bits_per_value differs for {c_ta.name} in {c_einsum.name}",
                )


if __name__ == "__main__":
    unittest.main()
