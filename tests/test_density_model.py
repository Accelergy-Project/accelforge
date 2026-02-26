"""Tests for hypergeometric and structured density models.

Core math validation: PMF, prob_empty, expected_occupancy, edge cases,
effectual_operations, and structured density model.
Workload density propagation tests removed — covered by reproduction tests.
"""

import unittest

from accelforge.model.density_model import (
    HypergeometricDensityModel,
    StructuredDensityModel,
    create_density_model,
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
        """N=10, r=3, tile=4, k=1 -> 1/2."""
        model = HypergeometricDensityModel(density=0.3, tensor_size=10)
        self.assertAlmostEqual(model.prob(4, 1), 0.5, places=10)

    def test_tiny_hand_computed_k2(self):
        """N=10, r=3, tile=4, k=2 -> 3/10."""
        model = HypergeometricDensityModel(density=0.3, tensor_size=10)
        self.assertAlmostEqual(model.prob(4, 2), 3 / 10, places=10)

    def test_tiny_hand_computed_k3(self):
        """N=10, r=3, tile=4, k=3 -> 1/30."""
        model = HypergeometricDensityModel(density=0.3, tensor_size=10)
        self.assertAlmostEqual(model.prob(4, 3), 1 / 30, places=10)

    def test_pmf_sums_to_one(self):
        model = HypergeometricDensityModel(density=0.3, tensor_size=10)
        total = sum(model.prob(4, k) for k in range(5))
        self.assertAlmostEqual(total, 1.0, places=10)


class TestProbEmpty(unittest.TestCase):
    """Test P(tile is all zeros)."""

    def test_fig1_scalar_empty(self):
        """N=16384, d=0.1015625, tile=1, P(empty)=0.8984375."""
        model = HypergeometricDensityModel(density=0.1015625, tensor_size=16384)
        self.assertEqual(model.r, 1664)
        self.assertAlmostEqual(model.prob_empty(1), 0.8984375, places=6)

    def test_density_zero_always_empty(self):
        model = HypergeometricDensityModel(density=0.0, tensor_size=1000)
        self.assertAlmostEqual(model.prob_empty(10), 1.0)

    def test_density_one_never_empty(self):
        model = HypergeometricDensityModel(density=1.0, tensor_size=1000)
        self.assertAlmostEqual(model.prob_empty(10), 0.0)


class TestExpectedOccupancy(unittest.TestCase):
    """Test E[nnz in tile]."""

    def test_fig1_buffer_tile(self):
        """N=16384, d=0.1015625, tile=128 -> 13."""
        model = HypergeometricDensityModel(density=0.1015625, tensor_size=16384)
        self.assertAlmostEqual(model.expected_occupancy(128), 13.0)

    def test_fig1_full_tensor(self):
        """N=16384, d=0.1015625, tile=16384 -> 1664."""
        model = HypergeometricDensityModel(density=0.1015625, tensor_size=16384)
        self.assertAlmostEqual(model.expected_occupancy(16384), 1664.0)

    def test_lab4_data_capacity_d02(self):
        """N=64, d=0.2, tile=64 -> ceil(12.8) = 13."""
        model = HypergeometricDensityModel(density=0.2, tensor_size=64)
        self.assertEqual(model.r, 13)
        self.assertEqual(model.expected_occupancy_ceil(64), 13)

    def test_lab4_data_capacity_d04(self):
        model = HypergeometricDensityModel(density=0.4, tensor_size=64)
        self.assertEqual(model.r, 26)
        self.assertEqual(model.expected_occupancy_ceil(64), 26)

    def test_lab4_data_capacity_d06(self):
        model = HypergeometricDensityModel(density=0.6, tensor_size=64)
        self.assertEqual(model.r, 39)
        self.assertEqual(model.expected_occupancy_ceil(64), 39)

    def test_lab4_data_capacity_d08(self):
        model = HypergeometricDensityModel(density=0.8, tensor_size=64)
        self.assertEqual(model.r, 52)
        self.assertEqual(model.expected_occupancy_ceil(64), 52)

    def test_lab4_data_capacity_d10(self):
        model = HypergeometricDensityModel(density=1.0, tensor_size=64)
        self.assertEqual(model.r, 64)
        self.assertEqual(model.expected_occupancy_ceil(64), 64)


class TestEdgeCases(unittest.TestCase):
    """Edge cases for the density model."""

    def test_tensor_size_zero(self):
        model = HypergeometricDensityModel(density=0.5, tensor_size=0)
        self.assertEqual(model.r, 0)
        self.assertAlmostEqual(model.expected_occupancy(0), 0.0)

    def test_tile_larger_than_tensor(self):
        """tile_shape > N should clamp to N."""
        model = HypergeometricDensityModel(density=0.5, tensor_size=100)
        self.assertAlmostEqual(model.expected_occupancy(200), 50.0)
        self.assertAlmostEqual(model.prob(200, 50), 1.0)

    def test_pigeonhole_minimum(self):
        """N=100, d=0.9, r=90, tile=20: min = max(0, 20+90-100) = 10."""
        model = HypergeometricDensityModel(density=0.9, tensor_size=100)
        for k in range(10):
            self.assertAlmostEqual(model.prob(20, k), 0.0)
        self.assertAlmostEqual(model.prob_at_least(20, 10), 1.0)

    def test_r_calculation_ceiling(self):
        """r = ceil(d * N), not round or floor."""
        model = HypergeometricDensityModel(density=0.2, tensor_size=64)
        self.assertEqual(model.r, 13)

    def test_exact_density_13_over_128(self):
        model = HypergeometricDensityModel(density=13 / 128, tensor_size=16384)
        self.assertEqual(model.r, 1664)


class TestEffectualOperations(unittest.TestCase):
    """Test the effectual operations calculator."""

    def test_lab4_part1(self):
        """total=512, d_A=0.25, d_B=0.5 -> 64."""
        self.assertEqual(effectual_operations(512, 0.25, 0.5), 64)

    def test_fig1_effectual_computes(self):
        """total=2097152, d_A=d_B=0.1015625 -> 21632."""
        result = effectual_operations(2097152, 0.1015625, 0.1015625)
        self.assertEqual(result, 21632)

    def test_all_dense(self):
        self.assertEqual(effectual_operations(1000, 1.0, 1.0), 1000)

    def test_one_zero(self):
        self.assertEqual(effectual_operations(1000, 0.5, 0.0), 0)

    def test_single_operand(self):
        self.assertEqual(effectual_operations(1000, 0.5), 500)

    def test_three_operands(self):
        self.assertEqual(effectual_operations(1000, 0.5, 0.5, 0.5), 125)


class TestStructuredDensityModel(unittest.TestCase):
    """Test the deterministic structured density model."""

    def test_prob_empty_always_zero(self):
        model = StructuredDensityModel(density=0.5, tensor_size=1000)
        self.assertEqual(model.prob_empty(1), 0.0)
        self.assertEqual(model.prob_empty(10), 0.0)
        self.assertEqual(model.prob_empty(100), 0.0)

    def test_prob_empty_zero_density(self):
        model = StructuredDensityModel(density=0.0, tensor_size=1000)
        self.assertEqual(model.prob_empty(10), 1.0)

    def test_exact_occupancy(self):
        model = StructuredDensityModel(density=0.5, tensor_size=1000)
        self.assertEqual(model.expected_occupancy(100), 50.0)
        self.assertEqual(model.expected_occupancy(4), 2.0)

    def test_occupancy_2_4(self):
        """2:4 sparsity: density=0.5, every group of 4 has exactly 2 nonzeros."""
        model = StructuredDensityModel(density=0.5, tensor_size=1024)
        self.assertEqual(model.expected_occupancy(4), 2.0)
        self.assertEqual(model.expected_occupancy_ceil(4), 2)

    def test_occupancy_ceil(self):
        model2 = StructuredDensityModel(density=0.33, tensor_size=100)
        self.assertAlmostEqual(model2.expected_occupancy(10), 3.3)
        self.assertEqual(model2.expected_occupancy_ceil(10), 4)


class TestConditioned(unittest.TestCase):
    """Test the conditioned() density model re-parameterization."""

    def test_basic_conditioning(self):
        """After conditioning, N=parent_shape and r=ceil(parent_occupancy)."""
        model = HypergeometricDensityModel(density=0.5, tensor_size=1000)
        ennz = model.expected_occupancy(100)  # 50.0
        child = model.conditioned(100, ennz)
        self.assertEqual(child.N, 100)
        self.assertEqual(child.r, 50)
        self.assertAlmostEqual(child.density, 0.5)

    def test_full_density(self):
        """d=1.0: conditioned model should have r=parent_shape."""
        model = HypergeometricDensityModel(density=1.0, tensor_size=500)
        child = model.conditioned(100, 100.0)
        self.assertEqual(child.N, 100)
        self.assertEqual(child.r, 100)
        self.assertAlmostEqual(child.density, 1.0)

    def test_zero_occupancy(self):
        """Zero parent_occupancy → r=0."""
        model = HypergeometricDensityModel(density=0.0, tensor_size=1000)
        child = model.conditioned(100, 0.0)
        self.assertEqual(child.r, 0)
        self.assertAlmostEqual(child.prob_empty(10), 1.0)

    def test_r_capped_at_n(self):
        """r should never exceed N."""
        model = HypergeometricDensityModel(density=0.9, tensor_size=1000)
        # Force parent_occupancy > parent_shape
        child = model.conditioned(10, 15.0)
        self.assertEqual(child.N, 10)
        self.assertEqual(child.r, 10)

    def test_structured_preserves_density(self):
        """Structured conditioning narrows N but keeps density."""
        model = StructuredDensityModel(density=0.5, tensor_size=1000)
        child = model.conditioned(100, 50.0)
        self.assertIsInstance(child, StructuredDensityModel)
        self.assertEqual(child.N, 100)
        self.assertAlmostEqual(child.density, 0.5)

    def test_conditioned_prob_empty_differs(self):
        """Conditioned model should produce a valid but different prob_empty."""
        model = HypergeometricDensityModel(density=0.1, tensor_size=10000)
        global_pe = model.prob_empty(10)
        # Condition on a 100-element parent with ~10 nonzeros
        child = model.conditioned(100, model.expected_occupancy(100))
        child_pe = child.prob_empty(10)
        # Both should be in the same ballpark (same effective density)
        self.assertGreater(global_pe, 0.0)
        self.assertGreater(child_pe, 0.0)
        self.assertLess(abs(global_pe - child_pe), 0.05)

    def test_chained_conditioning(self):
        """Conditioning twice should produce valid models."""
        model = HypergeometricDensityModel(density=0.5, tensor_size=10000)
        ennz1 = model.expected_occupancy(100)  # 50.0
        child1 = model.conditioned(100, ennz1)
        ennz2 = child1.expected_occupancy(10)  # 5.0
        child2 = child1.conditioned(10, ennz2)
        self.assertEqual(child2.N, 10)
        self.assertEqual(child2.r, 5)
        self.assertAlmostEqual(child2.density, 0.5)


class TestCreateDensityModel(unittest.TestCase):
    """Test the factory function."""

    def test_none_returns_hypergeometric(self):
        model = create_density_model(0.5, 1000)
        self.assertIsInstance(model, HypergeometricDensityModel)

    def test_structured_returns_structured(self):
        model = create_density_model(0.5, 1000, distribution="structured")
        self.assertIsInstance(model, StructuredDensityModel)

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            create_density_model(0.5, 1000, distribution="unknown")


if __name__ == "__main__":
    unittest.main()
