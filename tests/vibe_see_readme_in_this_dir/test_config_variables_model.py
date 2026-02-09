"""
Tests for:
  - Config
  - Variables
  - Model
  - FFM (Fast and Fusiest Mapper config)
  - Metrics
"""

import unittest
import math

from accelforge.frontend.config import Config
from accelforge.frontend.variables import Variables
from accelforge.frontend.model import Model
from accelforge.frontend.mapper.ffm import FFM
from accelforge.frontend.mapper.metrics import Metrics


# ============================================================================
# Config
# ============================================================================


class TestConfig(unittest.TestCase):
    """Tests for the Config model."""

    def test_default_creation(self):
        c = Config()
        self.assertEqual(len(c.expression_custom_functions), 0)
        self.assertEqual(len(c.component_models), 0)

    def test_use_installed_component_models_default(self):
        c = Config()
        self.assertTrue(c.use_installed_component_models)

    def test_use_installed_component_models_false(self):
        c = Config(use_installed_component_models=False)
        self.assertFalse(c.use_installed_component_models)

    def test_custom_functions_list(self):
        c = Config(expression_custom_functions=["path/to/file.py"])
        self.assertEqual(len(c.expression_custom_functions), 1)

    def test_component_models_list(self):
        c = Config(component_models=["path/to/model.py"])
        self.assertEqual(len(c.component_models), 1)


# ============================================================================
# Variables
# ============================================================================


class TestVariables(unittest.TestCase):
    """Tests for the Variables model."""

    def test_empty_creation(self):
        v = Variables()
        self.assertIsNotNone(v)

    def test_custom_attributes(self):
        """Variables supports arbitrary extra attributes via EvalExtras."""
        v = Variables(a=123, b="a + 5")
        self.assertEqual(v.a, 123)
        self.assertEqual(v.b, "a + 5")


# ============================================================================
# Model
# ============================================================================


class TestModel(unittest.TestCase):
    """Tests for the Model configuration."""

    def test_default_creation(self):
        m = Model()
        self.assertIsNotNone(m.metrics)

    def test_default_metrics_all(self):
        m = Model()
        self.assertEqual(m.metrics, Metrics.all_metrics())


# ============================================================================
# Metrics
# ============================================================================


class TestMetrics(unittest.TestCase):
    """Tests for the Metrics flag enum."""

    def test_latency_exists(self):
        self.assertIsNotNone(Metrics.LATENCY)

    def test_energy_exists(self):
        self.assertIsNotNone(Metrics.ENERGY)

    def test_resource_usage_exists(self):
        self.assertIsNotNone(Metrics.RESOURCE_USAGE)

    def test_actions_exists(self):
        self.assertIsNotNone(Metrics.ACTIONS)

    def test_all_metrics_includes_all(self):
        all_m = Metrics.all_metrics()
        self.assertIn(Metrics.LATENCY, all_m)
        self.assertIn(Metrics.ENERGY, all_m)
        self.assertIn(Metrics.RESOURCE_USAGE, all_m)
        self.assertIn(Metrics.ACTIONS, all_m)

    def test_combining_metrics(self):
        combined = Metrics.LATENCY | Metrics.ENERGY
        self.assertIn(Metrics.LATENCY, combined)
        self.assertIn(Metrics.ENERGY, combined)
        self.assertNotIn(Metrics.RESOURCE_USAGE, combined)

    def test_individual_metric_values_are_unique(self):
        vals = [
            Metrics.LATENCY,
            Metrics.ENERGY,
            Metrics.RESOURCE_USAGE,
            Metrics.ACTIONS,
        ]
        self.assertEqual(len(vals), len(set(vals)))


# ============================================================================
# FFM
# ============================================================================


class TestFFM(unittest.TestCase):
    """Tests for the FFM mapper configuration."""

    def test_default_creation(self):
        f = FFM()
        self.assertIsNotNone(f)

    def test_default_metrics(self):
        f = FFM()
        self.assertEqual(f.metrics, Metrics.ENERGY)

    def test_default_info_metrics(self):
        f = FFM()
        self.assertEqual(f.info_metrics, Metrics.all_metrics())

    def test_force_memory_hierarchy_order_default(self):
        f = FFM()
        self.assertTrue(f.force_memory_hierarchy_order)

    def test_max_fused_loops_per_rank_variable(self):
        f = FFM()
        self.assertEqual(f.max_fused_loops_per_rank_variable, 1)

    def test_max_fused_loops_default_inf(self):
        f = FFM()
        self.assertEqual(f.max_fused_loops, float("inf"))

    def test_max_loops_default_inf(self):
        f = FFM()
        self.assertEqual(f.max_loops, float("inf"))

    def test_max_loops_minus_ranks_default_inf(self):
        f = FFM()
        self.assertEqual(f.max_loops_minus_ranks, float("inf"))

    def test_memory_limit_default_inf(self):
        f = FFM()
        self.assertEqual(f.memory_limit, float("inf"))

    def test_memory_limit_per_process_default_inf(self):
        f = FFM()
        self.assertEqual(f.memory_limit_per_process, float("inf"))

    def test_time_limit_default_inf(self):
        f = FFM()
        self.assertEqual(f.time_limit, float("inf"))

    def test_time_limit_per_pmapping_template_default_inf(self):
        f = FFM()
        self.assertEqual(f.time_limit_per_pmapping_template, float("inf"))

    def test_max_pmapping_templates_default_inf(self):
        f = FFM()
        self.assertEqual(f.max_pmapping_templates_per_einsum, float("inf"))

    def test_custom_metrics(self):
        f = FFM(metrics=Metrics.LATENCY)
        self.assertEqual(f.metrics, Metrics.LATENCY)

    def test_combined_metrics(self):
        f = FFM(metrics=Metrics.LATENCY | Metrics.ENERGY)
        self.assertIn(Metrics.LATENCY, f.metrics)
        self.assertIn(Metrics.ENERGY, f.metrics)

    def test_out_of_order_default_false(self):
        f = FFM()
        self.assertFalse(
            f.out_of_order_hierarchy_explore_removing_spatials_for_more_temporals
        )

    def test_custom_max_fused_loops(self):
        f = FFM(max_fused_loops=3)
        self.assertEqual(f.max_fused_loops, 3)


if __name__ == "__main__":
    unittest.main()
