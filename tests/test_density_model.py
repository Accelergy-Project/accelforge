"""Tests for Phase 1: Hypergeometric density model and workload density spec.

Validation tests sourced from ARTIFACT_EVALUATION.md and IMPLEMENTATION_PLAN.md.
"""

import math
import unittest

from accelforge.model.density_model import (
    HypergeometricDensityModel,
    effectual_operations,
)


class TestHypergeometricPMF(unittest.TestCase):
    """Test the hypergeometric PMF calculations."""

    def test_tiny_hand_computed_k0(self):
        """N=10, r=3, tile=4, k=0 -> C(3,0)*C(7,4)/C(10,4) = 1/6."""
        model = HypergeometricDensityModel(density=0.3, tensor_size=10)
        self.assertEqual(model.r, 3)
        self.assertAlmostEqual(model.prob(4, 0), 1 / 6, places=10)

    def test_tiny_hand_computed_k1(self):
        """N=10, r=3, tile=4, k=1 -> C(3,1)*C(7,3)/C(10,4) = 3*35/210 = 1/2."""
        model = HypergeometricDensityModel(density=0.3, tensor_size=10)
        self.assertAlmostEqual(model.prob(4, 1), 0.5, places=10)

    def test_tiny_hand_computed_k2(self):
        """N=10, r=3, tile=4, k=2 -> C(3,2)*C(7,2)/C(10,4) = 3*21/210 = 3/10."""
        model = HypergeometricDensityModel(density=0.3, tensor_size=10)
        self.assertAlmostEqual(model.prob(4, 2), 3 / 10, places=10)

    def test_tiny_hand_computed_k3(self):
        """N=10, r=3, tile=4, k=3 -> C(3,3)*C(7,1)/C(10,4) = 7/210 = 1/30."""
        model = HypergeometricDensityModel(density=0.3, tensor_size=10)
        self.assertAlmostEqual(model.prob(4, 3), 1 / 30, places=10)

    def test_pmf_sums_to_one(self):
        """PMF over all possible k values should sum to 1."""
        model = HypergeometricDensityModel(density=0.3, tensor_size=10)
        total = sum(model.prob(4, k) for k in range(5))  # k=0..4
        self.assertAlmostEqual(total, 1.0, places=10)


class TestProbEmpty(unittest.TestCase):
    """Test P(tile is all zeros)."""

    def test_fig1_scalar_empty(self):
        """ARTIFACT_EVAL §2.5: N=16384, d=0.1015625, tile=1, P(empty)=0.8984375."""
        model = HypergeometricDensityModel(density=0.1015625, tensor_size=16384)
        self.assertEqual(model.r, 1664)
        self.assertAlmostEqual(model.prob_empty(1), 0.8984375, places=6)

    def test_density_zero_always_empty(self):
        """d=0 -> tile is always empty."""
        model = HypergeometricDensityModel(density=0.0, tensor_size=1000)
        self.assertEqual(model.r, 0)
        self.assertAlmostEqual(model.prob_empty(10), 1.0)

    def test_density_one_never_empty(self):
        """d=1.0 -> tile is never empty (all nonzero)."""
        model = HypergeometricDensityModel(density=1.0, tensor_size=1000)
        self.assertEqual(model.r, 1000)
        self.assertAlmostEqual(model.prob_empty(10), 0.0)


class TestExpectedOccupancy(unittest.TestCase):
    """Test E[nnz in tile]."""

    def test_fig1_buffer_tile(self):
        """ARTIFACT_EVAL §2.2: N=16384, d=0.1015625, tile=128 -> 13."""
        model = HypergeometricDensityModel(density=0.1015625, tensor_size=16384)
        self.assertAlmostEqual(model.expected_occupancy(128), 13.0)

    def test_fig1_full_tensor(self):
        """ARTIFACT_EVAL §2.2: N=16384, d=0.1015625, tile=16384 -> 1664."""
        model = HypergeometricDensityModel(density=0.1015625, tensor_size=16384)
        self.assertAlmostEqual(model.expected_occupancy(16384), 1664.0)

    def test_lab4_data_capacity_d02(self):
        """ARTIFACT_EVAL §4 Part 4: N=64, d=0.2, tile=64 -> ceil(12.8) = 13."""
        model = HypergeometricDensityModel(density=0.2, tensor_size=64)
        self.assertEqual(model.r, 13)
        self.assertEqual(model.expected_occupancy_ceil(64), 13)

    def test_lab4_data_capacity_d04(self):
        """ARTIFACT_EVAL §4 Part 4: N=64, d=0.4, tile=64 -> ceil(25.6) = 26."""
        model = HypergeometricDensityModel(density=0.4, tensor_size=64)
        self.assertEqual(model.r, 26)
        self.assertEqual(model.expected_occupancy_ceil(64), 26)

    def test_lab4_data_capacity_d06(self):
        """ARTIFACT_EVAL §4 Part 4: N=64, d=0.6, tile=64 -> ceil(38.4) = 39."""
        model = HypergeometricDensityModel(density=0.6, tensor_size=64)
        self.assertEqual(model.r, 39)
        self.assertEqual(model.expected_occupancy_ceil(64), 39)

    def test_lab4_data_capacity_d08(self):
        """ARTIFACT_EVAL §4 Part 4: N=64, d=0.8, tile=64 -> ceil(51.2) = 52."""
        model = HypergeometricDensityModel(density=0.8, tensor_size=64)
        self.assertEqual(model.r, 52)
        self.assertEqual(model.expected_occupancy_ceil(64), 52)

    def test_lab4_data_capacity_d10(self):
        """ARTIFACT_EVAL §4 Part 4: N=64, d=1.0, tile=64 -> 64."""
        model = HypergeometricDensityModel(density=1.0, tensor_size=64)
        self.assertEqual(model.r, 64)
        self.assertEqual(model.expected_occupancy_ceil(64), 64)

    def test_tile_equals_tensor_deterministic(self):
        """When tile=N, result is deterministic: E[nnz] = r."""
        model = HypergeometricDensityModel(density=0.5, tensor_size=100)
        self.assertEqual(model.r, 50)
        self.assertAlmostEqual(model.expected_occupancy(100), 50.0)

    def test_proportional_scaling(self):
        """E[nnz in tile] = tile * r / N (proportional to tile size)."""
        model = HypergeometricDensityModel(density=0.25, tensor_size=400)
        self.assertEqual(model.r, 100)
        self.assertAlmostEqual(model.expected_occupancy(40), 10.0)
        self.assertAlmostEqual(model.expected_occupancy(200), 50.0)


class TestEdgeCases(unittest.TestCase):
    """Edge cases for the density model."""

    def test_density_zero(self):
        model = HypergeometricDensityModel(density=0.0, tensor_size=1000)
        self.assertEqual(model.r, 0)
        self.assertAlmostEqual(model.expected_occupancy(10), 0.0)
        self.assertAlmostEqual(model.prob_empty(10), 1.0)

    def test_density_one(self):
        model = HypergeometricDensityModel(density=1.0, tensor_size=1000)
        self.assertEqual(model.r, 1000)
        self.assertAlmostEqual(model.expected_occupancy(10), 10.0)
        self.assertAlmostEqual(model.prob(10, 10), 1.0)

    def test_tensor_size_zero(self):
        model = HypergeometricDensityModel(density=0.5, tensor_size=0)
        self.assertEqual(model.r, 0)
        self.assertAlmostEqual(model.expected_occupancy(0), 0.0)
        self.assertAlmostEqual(model.prob(0, 0), 1.0)

    def test_tile_larger_than_tensor(self):
        """tile_shape > N should clamp to N."""
        model = HypergeometricDensityModel(density=0.5, tensor_size=100)
        # Requesting tile of 200 from tensor of 100 should give r=50
        self.assertAlmostEqual(model.expected_occupancy(200), 50.0)
        self.assertAlmostEqual(model.prob(200, 50), 1.0)

    def test_pigeonhole_minimum(self):
        """When density is high, minimum nnz in tile = max(0, n+r-N).
        N=100, d=0.9, r=90, tile=20: min = max(0, 20+90-100) = 10.
        So P(k < 10) = 0."""
        model = HypergeometricDensityModel(density=0.9, tensor_size=100)
        self.assertEqual(model.r, 90)
        # k < 10 should have probability 0
        for k in range(10):
            self.assertAlmostEqual(model.prob(20, k), 0.0)
        # P(k >= 10) should be 1
        self.assertAlmostEqual(model.prob_at_least(20, 10), 1.0)

    def test_r_calculation_ceiling(self):
        """r = ceil(d * N), not round or floor."""
        model = HypergeometricDensityModel(density=0.2, tensor_size=64)
        # 0.2 * 64 = 12.8, ceil = 13
        self.assertEqual(model.r, 13)

    def test_exact_density_13_over_128(self):
        """0.1015625 = 13/128 exactly, so r = ceil(13/128 * 16384) = 1664."""
        model = HypergeometricDensityModel(density=13 / 128, tensor_size=16384)
        self.assertEqual(model.r, 1664)


class TestEffectualOperations(unittest.TestCase):
    """Test the effectual operations calculator."""

    def test_lab4_part1(self):
        """ARTIFACT_EVAL §4 Part 1: total=512, d_A=0.25, d_B=0.5 -> 64."""
        self.assertEqual(effectual_operations(512, 0.25, 0.5), 64)

    def test_fig1_effectual_computes(self):
        """Fig1: total=2097152, d_A=d_B=0.1015625 -> 21632.
        Note: Sparseloop reports 21633 due to 3-state remainder (Phase 5/6)."""
        result = effectual_operations(2097152, 0.1015625, 0.1015625)
        self.assertEqual(result, 21632)

    def test_all_dense(self):
        """All density=1.0 -> all operations effectual."""
        self.assertEqual(effectual_operations(1000, 1.0, 1.0), 1000)

    def test_one_zero(self):
        """One density=0 -> no effectual operations."""
        self.assertEqual(effectual_operations(1000, 0.5, 0.0), 0)

    def test_single_operand(self):
        """Single density (e.g., element-wise)."""
        self.assertEqual(effectual_operations(1000, 0.5), 500)

    def test_three_operands(self):
        """Three operand densities."""
        self.assertEqual(effectual_operations(1000, 0.5, 0.5, 0.5), 125)


class TestWorkloadDensitySpec(unittest.TestCase):
    """Test that TensorAccess density field parses correctly."""

    def test_density_field_exists(self):
        """TensorAccess should have a density field."""
        from accelforge.frontend.workload import TensorAccess

        ta = TensorAccess(name="A", projection=["m", "k"])
        self.assertIsNone(ta.density)

    def test_density_field_set(self):
        """TensorAccess density can be set to a float."""
        from accelforge.frontend.workload import TensorAccess

        ta = TensorAccess(name="A", projection=["m", "k"], density=0.25)
        self.assertEqual(ta.density, 0.25)

    def test_density_none_means_dense(self):
        """None density is interpreted as dense (1.0) by convention."""
        from accelforge.frontend.workload import TensorAccess

        ta = TensorAccess(name="A", projection=["m", "k"])
        # None means dense -- consumers should treat None as 1.0
        self.assertIsNone(ta.density)

    def test_workload_densities_field_exists(self):
        """Workload should have a densities field."""
        from accelforge.frontend.workload import Workload

        w = Workload()
        self.assertIsNotNone(w.densities)


class TestDensityPropagation(unittest.TestCase):
    """Test that workload.densities propagates to TensorAccess.density
    through the _spec_eval_expressions pipeline.

    Uses real YAML files (matmuls example) so this exercises the actual
    Spec evaluation path: Workload._eval_expressions -> Einsum._eval_expressions
    -> density_dict resolution -> TensorAccess.density assignment.
    """

    @classmethod
    def setUpClass(cls):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from paths import EXAMPLES_DIR
        cls.EXAMPLES_DIR = EXAMPLES_DIR

    def _load_spec(self):
        """Load a single-matmul Spec (T0, W0 -> T1)."""
        from accelforge.frontend.spec import Spec
        return Spec.from_yaml(
            self.EXAMPLES_DIR / "arches" / "simple.yaml",
            self.EXAMPLES_DIR / "workloads" / "matmuls.yaml",
            self.EXAMPLES_DIR / "mappings" / "unfused_matmuls_to_simple.yaml",
            jinja_parse_data={"N_EINSUMS": 1, "M": 8, "KN": 8},
        )

    def _get_densities(self, spec):
        """Return {tensor_name: density} for the first Einsum."""
        return {
            ta.name: ta.density
            for ta in spec.workload.einsums[0].tensor_accesses
        }

    def test_no_densities_all_none(self):
        """Without densities set, all tensors have density=None."""
        spec = self._load_spec()
        evaluated = spec._spec_eval_expressions(einsum_name="Matmul0")
        dens = self._get_densities(evaluated)
        for name, d in dens.items():
            self.assertIsNone(d, f"Tensor {name} should be None")

    def test_workload_densities_per_tensor(self):
        """densities: {T0: 0.3, W0: 0.5} -> T0=0.3, W0=0.5, T1=None."""
        from accelforge.util._basetypes import EvalableDict
        spec = self._load_spec()
        spec.workload.densities = EvalableDict({"T0": 0.3, "W0": 0.5})
        evaluated = spec._spec_eval_expressions(einsum_name="Matmul0")
        dens = self._get_densities(evaluated)
        self.assertEqual(dens["T0"], 0.3)
        self.assertEqual(dens["W0"], 0.5)
        self.assertIsNone(dens["T1"])

    def test_workload_densities_all(self):
        """densities: {All: 0.5} -> all tensors get density 0.5."""
        from accelforge.util._basetypes import EvalableDict
        spec = self._load_spec()
        spec.workload.densities = EvalableDict({"All": 0.5})
        evaluated = spec._spec_eval_expressions(einsum_name="Matmul0")
        dens = self._get_densities(evaluated)
        for name, d in dens.items():
            self.assertEqual(d, 0.5, f"Tensor {name}")

    def test_per_tensor_overrides_workload(self):
        """Per-TensorAccess density overrides workload-level density."""
        from accelforge.util._basetypes import EvalableDict
        spec = self._load_spec()
        spec.workload.densities = EvalableDict({"All": 0.5})
        # Set per-tensor density on T0 before evaluation
        for einsum in spec.workload.einsums:
            for ta in einsum.tensor_accesses:
                if ta.name == "T0":
                    ta.density = 0.1
        evaluated = spec._spec_eval_expressions(einsum_name="Matmul0")
        dens = self._get_densities(evaluated)
        self.assertEqual(dens["T0"], 0.1)   # per-tensor wins
        self.assertEqual(dens["W0"], 0.5)   # from workload
        self.assertEqual(dens["T1"], 0.5)   # from workload

    def test_cross_einsum_consistency_ok(self):
        """Two Einsums sharing a tensor with same density -> no error."""
        from accelforge.frontend.spec import Spec
        from accelforge.util._basetypes import EvalableDict
        spec = Spec.from_yaml(
            self.EXAMPLES_DIR / "arches" / "simple.yaml",
            self.EXAMPLES_DIR / "workloads" / "matmuls.yaml",
            self.EXAMPLES_DIR / "mappings" / "unfused_matmuls_to_simple.yaml",
            jinja_parse_data={"N_EINSUMS": 2, "M": 8, "KN": 8},
        )
        # T1 is shared between Matmul0 (output) and Matmul1 (input).
        # Setting consistent density should work.
        spec.workload.densities = EvalableDict({"T1": 0.5})
        evaluated = spec._spec_eval_expressions(einsum_name="Matmul0")
        # Should not raise
        dens0 = {
            ta.name: ta.density
            for ta in evaluated.workload.einsums[0].tensor_accesses
        }
        dens1 = {
            ta.name: ta.density
            for ta in evaluated.workload.einsums[1].tensor_accesses
        }
        self.assertEqual(dens0["T1"], 0.5)
        self.assertEqual(dens1["T1"], 0.5)

    def test_cross_einsum_inconsistency_raises(self):
        """Two Einsums sharing a tensor with different densities -> raises."""
        from accelforge.frontend.spec import Spec
        spec = Spec.from_yaml(
            self.EXAMPLES_DIR / "arches" / "simple.yaml",
            self.EXAMPLES_DIR / "workloads" / "matmuls.yaml",
            self.EXAMPLES_DIR / "mappings" / "unfused_matmuls_to_simple.yaml",
            jinja_parse_data={"N_EINSUMS": 2, "M": 8, "KN": 8},
        )
        # T1 is shared. Set different densities per-Einsum.
        for ta in spec.workload.einsums[0].tensor_accesses:
            if ta.name == "T1":
                ta.density = 0.3
        for ta in spec.workload.einsums[1].tensor_accesses:
            if ta.name == "T1":
                ta.density = 0.7
        with self.assertRaises(ValueError) as ctx:
            spec._spec_eval_expressions(einsum_name="Matmul0")
        self.assertIn("T1", str(ctx.exception))
        self.assertIn("density", str(ctx.exception).lower())


class TestDensityModelRepr(unittest.TestCase):
    """Test string representation."""

    def test_repr(self):
        model = HypergeometricDensityModel(density=0.5, tensor_size=100)
        s = repr(model)
        self.assertIn("0.5", s)
        self.assertIn("100", s)
        self.assertIn("50", s)


if __name__ == "__main__":
    unittest.main()
