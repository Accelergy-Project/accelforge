"""
Tests for additional component fields documented in component_energy_area.rst
and accelerator_energy_latency.rst:

  - Component.energy_scale, area_scale, leak_power_scale
  - Component.n_parallel_instances
  - Component.total_area, total_leak_power
  - Action.energy_scale, Action.bits_per_action
  - Component.total_latency expression
  - TensorHolder.tensors.keep / may_keep
  - Memory.size and related fields
"""

import math
import os
import tempfile
import unittest
from pathlib import Path

from accelforge.frontend.arch import (
    Action,
    Arch,
    Compute,
    Container,
    Memory,
    Toll,
    Spatial as ArchSpatial,
    Tensors,
    TensorHolderAction,
)
from accelforge.frontend.spec import Spec

_REPO_ROOT = Path(__file__).parent.parent.parent
EXAMPLES_DIR = _REPO_ROOT / "examples"


def _yaml_spec(yaml_text: str, **jinja_parse_data) -> Spec:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir=str(_REPO_ROOT)
    ) as f:
        f.write(yaml_text)
        f.flush()
        try:
            return Spec.from_yaml(f.name, jinja_parse_data=jinja_parse_data)
        finally:
            os.unlink(f.name)


# ============================================================================
# Action fields
# ============================================================================


class TestActionFields(unittest.TestCase):
    """Test Action model fields."""

    def test_action_name_and_energy(self):
        a = Action(name="read", energy=1.5, latency=0)
        self.assertEqual(a.name, "read")
        self.assertEqual(a.energy, 1.5)
        self.assertEqual(a.latency, 0)

    def test_action_energy_scale_default(self):
        a = Action(name="read", energy=1.0, latency=0)
        self.assertEqual(a.energy_scale, 1)

    def test_action_energy_scale_custom(self):
        a = Action(name="read", energy=1.0, latency=0, energy_scale=2.0)
        self.assertEqual(a.energy_scale, 2.0)


class TestTensorHolderActionFields(unittest.TestCase):
    """Test TensorHolderAction (subclass of Action for Memory/Toll)."""

    def test_bits_per_action_default_is_expression(self):
        """Default bits_per_action is an expression string, not numeric 1."""
        a = TensorHolderAction(name="read", energy=1.0, latency=0)
        self.assertIsInstance(a.bits_per_action, str)
        self.assertIn("bits_per_action", a.bits_per_action)

    def test_bits_per_action_custom(self):
        a = TensorHolderAction(name="read", energy=1.0, latency=0, bits_per_action=32)
        self.assertEqual(a.bits_per_action, 32)


# ============================================================================
# Component fields: energy_scale, area_scale, leak_power_scale, n_parallel
# ============================================================================


class TestComputeComponentFields(unittest.TestCase):
    """Test Compute component fields."""

    def test_energy_scale_default(self):
        c = Compute(
            name="MAC",
            leak_power=0,
            area=0,
            actions=[{"name": "compute", "energy": 1, "latency": 1}],
        )
        self.assertEqual(c.energy_scale, 1)

    def test_area_scale_default(self):
        c = Compute(
            name="MAC",
            leak_power=0,
            area=0,
            actions=[{"name": "compute", "energy": 1, "latency": 1}],
        )
        self.assertEqual(c.area_scale, 1)

    def test_leak_power_scale_default(self):
        c = Compute(
            name="MAC",
            leak_power=0,
            area=0,
            actions=[{"name": "compute", "energy": 1, "latency": 1}],
        )
        self.assertEqual(c.leak_power_scale, 1)

    def test_n_parallel_instances_default(self):
        c = Compute(
            name="MAC",
            leak_power=0,
            area=0,
            actions=[{"name": "compute", "energy": 1, "latency": 1}],
        )
        self.assertEqual(c.n_parallel_instances, 1)

    def test_custom_scales(self):
        c = Compute(
            name="MAC",
            leak_power=0,
            area=100,
            actions=[{"name": "compute", "energy": 1, "latency": 1}],
            energy_scale=0.5,
            area_scale=2,
            leak_power_scale=0.1,
        )
        self.assertEqual(c.energy_scale, 0.5)
        self.assertEqual(c.area_scale, 2)
        self.assertEqual(c.leak_power_scale, 0.1)

    def test_n_parallel_instances_custom(self):
        c = Compute(
            name="MAC",
            leak_power=0,
            area=0,
            actions=[{"name": "compute", "energy": 1, "latency": 1}],
            n_parallel_instances=4,
        )
        self.assertEqual(c.n_parallel_instances, 4)


class TestMemoryComponentFields(unittest.TestCase):
    """Test Memory component fields."""

    def test_memory_size(self):
        m = Memory(
            name="Buf",
            size=1024,
            leak_power=0,
            area=0,
            actions=[
                {"name": "read", "energy": 1, "latency": 0},
                {"name": "write", "energy": 1, "latency": 0},
            ],
        )
        self.assertEqual(m.size, 1024)

    def test_memory_size_inf(self):
        m = Memory(
            name="MainMem",
            size="inf",
            leak_power=0,
            area=0,
            actions=[
                {"name": "read", "energy": 1, "latency": 0},
                {"name": "write", "energy": 1, "latency": 0},
            ],
        )
        self.assertEqual(m.size, "inf")

    def test_memory_total_latency_default(self):
        m = Memory(
            name="Buf",
            size=1024,
            leak_power=0,
            area=0,
            actions=[
                {"name": "read", "energy": 1, "latency": 0},
                {"name": "write", "energy": 1, "latency": 0},
            ],
        )
        self.assertIsNotNone(m.total_latency)

    def test_memory_total_latency_expression(self):
        m = Memory(
            name="Buf",
            size=1024,
            leak_power=0,
            area=0,
            actions=[
                {"name": "read", "energy": 1, "latency": 5},
                {"name": "write", "energy": 1, "latency": 3},
            ],
            total_latency="max(read_latency, write_latency)",
        )
        self.assertEqual(m.total_latency, "max(read_latency, write_latency)")

    def test_memory_energy_scale_custom(self):
        m = Memory(
            name="Buf",
            size=1024,
            leak_power=0,
            area=0,
            actions=[
                {"name": "read", "energy": 1, "latency": 0},
                {"name": "write", "energy": 1, "latency": 0},
            ],
            energy_scale=0.5,
        )
        self.assertEqual(m.energy_scale, 0.5)

    def test_memory_area_and_leak(self):
        m = Memory(
            name="Buf",
            size=1024,
            leak_power=0.01,
            area=100,
            actions=[
                {"name": "read", "energy": 1, "latency": 0},
                {"name": "write", "energy": 1, "latency": 0},
            ],
        )
        self.assertEqual(m.area, 100)
        self.assertEqual(m.leak_power, 0.01)


# ============================================================================
# Tensors fields (keep / may_keep)
# ============================================================================


class TestTensorsFields(unittest.TestCase):
    """Test Tensors model."""

    def test_tensors_keep(self):
        t = Tensors(keep="All")
        self.assertEqual(t.keep, "All")

    def test_tensors_may_keep(self):
        t = Tensors(keep="All", may_keep="Inputs")
        self.assertEqual(t.may_keep, "Inputs")

    def test_tensors_on_memory(self):
        m = Memory(
            name="Buf",
            size=1024,
            leak_power=0,
            area=0,
            tensors={"keep": "~Intermediates", "may_keep": "All"},
            actions=[
                {"name": "read", "energy": 1, "latency": 0},
                {"name": "write", "energy": 1, "latency": 0},
            ],
        )
        self.assertEqual(m.tensors.keep, "~Intermediates")
        self.assertEqual(m.tensors.may_keep, "All")

    def test_tensors_keep_only(self):
        t = Tensors(keep="Inputs")
        self.assertEqual(t.keep, "Inputs")
        # may_keep should have a default
        self.assertIsNotNone(t.may_keep)


# ============================================================================
# Component from YAML: TPU v4i detailed field checks
# ============================================================================


class TestTPUComponentFieldsParsed(unittest.TestCase):
    """Test parsed component fields from tpu_v4i.yaml."""

    @classmethod
    def setUpClass(cls):
        arch_path = EXAMPLES_DIR / "arches" / "tpu_v4i.yaml"
        wl_path = EXAMPLES_DIR / "workloads" / "three_matmuls_annotated.yaml"
        if not arch_path.exists() or not wl_path.exists():
            raise unittest.SkipTest("YAML not found")
        cls.spec = Spec.from_yaml(arch_path, wl_path)

    def test_main_memory_size_inf(self):
        mm = self.spec.arch.find("MainMemory")
        self.assertEqual(mm.size, "inf")

    def test_global_buffer_has_read_write_actions(self):
        gb = self.spec.arch.find("GlobalBuffer")
        action_names = {a.name for a in gb.actions}
        self.assertIn("read", action_names)
        self.assertIn("write", action_names)

    def test_global_buffer_total_latency_is_expression(self):
        gb = self.spec.arch.find("GlobalBuffer")
        self.assertIsInstance(gb.total_latency, str)
        self.assertIn("max", gb.total_latency)

    def test_array_fanout_spatial(self):
        """ProcessingElement has spatial fanouts for reuse."""
        pe = self.spec.arch.find("ProcessingElement")
        self.assertGreater(len(pe.spatial), 0)
        spatial_names = [s.name for s in pe.spatial]
        self.assertIn("reuse_input", spatial_names)
        self.assertIn("reuse_output", spatial_names)

    def test_local_buffer_spatial(self):
        lb = self.spec.arch.find("LocalBuffer")
        self.assertEqual(len(lb.spatial), 1)
        self.assertEqual(lb.spatial[0].name, "Z")
        self.assertEqual(lb.spatial[0].fanout, 4)

    def test_mac_actions(self):
        mac = self.spec.arch.find("MAC")
        action_names = {a.name for a in mac.actions}
        self.assertIn("compute", action_names)

    def test_scalar_unit_actions(self):
        su = self.spec.arch.find("ScalarUnit")
        action_names = {a.name for a in su.actions}
        self.assertIn("compute", action_names)


class TestTPUComponentFieldsEvaluated(unittest.TestCase):
    """Test evaluated component fields from tpu_v4i.yaml."""

    @classmethod
    def setUpClass(cls):
        arch_path = EXAMPLES_DIR / "arches" / "tpu_v4i.yaml"
        wl_path = EXAMPLES_DIR / "workloads" / "three_matmuls_annotated.yaml"
        if not arch_path.exists() or not wl_path.exists():
            raise unittest.SkipTest("YAML not found")
        cls.spec = Spec.from_yaml(arch_path, wl_path)._spec_eval_expressions(
            einsum_name="Matmul1"
        )

    def test_main_memory_size_evaluates_to_inf(self):
        mm = self.spec.arch.find("MainMemory")
        self.assertEqual(mm.size, math.inf)

    def test_global_buffer_total_latency_stays_expression(self):
        """total_latency is NOT evaluated by _spec_eval_expressions; it stays as a string."""
        gb = self.spec.arch.find("GlobalBuffer")
        self.assertIsInstance(gb.total_latency, str)
        self.assertIn("max", gb.total_latency)

    def test_global_buffer_size_evaluated(self):
        gb = self.spec.arch.find("GlobalBuffer")
        self.assertIsInstance(gb.size, (int, float))
        self.assertGreater(gb.size, 0)

    def test_register_size_evaluated(self):
        reg = self.spec.arch.find("Register")
        self.assertIsInstance(reg.size, (int, float))
        self.assertGreater(reg.size, 0)

    def test_action_energies_evaluated(self):
        """All action energies should be numeric after evaluation."""
        for node in self.spec.arch.get_nodes_of_type(Memory):
            for action in node.actions:
                self.assertIsInstance(
                    action.energy,
                    (int, float),
                    f"{node.name}.{action.name}.energy is {action.energy}",
                )

    def test_mac_enabled_evaluated(self):
        """MAC 'enabled' field should evaluate to a boolean after eval."""
        mac = self.spec.arch.find("MAC")
        self.assertIsInstance(mac.enabled, bool)


# ============================================================================
# Simple arch: all fields should be directly numeric
# ============================================================================


class TestSimpleArchComponentFields(unittest.TestCase):
    """Test simple.yaml where all fields are directly numeric."""

    @classmethod
    def setUpClass(cls):
        arch_path = EXAMPLES_DIR / "arches" / "simple.yaml"
        if not arch_path.exists():
            raise unittest.SkipTest("YAML not found")
        cls.spec = Spec.from_yaml(arch_path)

    def test_main_memory_actions(self):
        mm = self.spec.arch.find("MainMemory")
        read = [a for a in mm.actions if a.name == "read"][0]
        write = [a for a in mm.actions if a.name == "write"][0]
        self.assertEqual(read.energy, 1)
        self.assertEqual(write.energy, 1)
        self.assertEqual(read.latency, 0)

    def test_global_buffer_actions(self):
        buf = self.spec.arch.find("GlobalBuffer")
        read = [a for a in buf.actions if a.name == "read"][0]
        self.assertEqual(read.energy, 1)

    def test_mac_compute_energy(self):
        mac = self.spec.arch.find("MAC")
        compute = [a for a in mac.actions if a.name == "compute"][0]
        self.assertEqual(compute.energy, 1)
        self.assertEqual(compute.latency, 1)


# ============================================================================
# Toll component
# ============================================================================


class TestTollComponent(unittest.TestCase):
    """Test Toll component construction and fields."""

    def test_toll_with_actions(self):
        t = Toll(
            name="Quantizer",
            direction="down",
            leak_power=0,
            area=0,
            actions=[
                {"name": "read", "energy": 0.5, "latency": 1},
                {"name": "write", "energy": 0.3, "latency": 1},
            ],
        )
        self.assertEqual(len(t.actions), 2)
        self.assertEqual(t.actions[0].energy, 0.5)

    def test_toll_tensors_field(self):
        t = Toll(
            name="Quantizer",
            direction="up",
            leak_power=0,
            area=0,
            tensors={"keep": "All"},
        )
        self.assertEqual(t.tensors.keep, "All")

    def test_toll_energy_scale(self):
        t = Toll(
            name="Quantizer",
            direction="down",
            leak_power=0,
            area=0,
            energy_scale=2.0,
        )
        self.assertEqual(t.energy_scale, 2.0)


# ============================================================================
# Arch: get_fanout, get_nodes_of_type
# ============================================================================


class TestArchUtilityMethods(unittest.TestCase):
    """Test utility methods on Arch."""

    @classmethod
    def setUpClass(cls):
        arch_path = EXAMPLES_DIR / "arches" / "simple.yaml"
        if not arch_path.exists():
            raise unittest.SkipTest("YAML not found")
        cls.spec = Spec.from_yaml(arch_path)

    def test_find_by_name(self):
        mm = self.spec.arch.find("MainMemory")
        self.assertIsInstance(mm, Memory)
        self.assertEqual(mm.name, "MainMemory")

    def test_find_nonexistent(self):
        with self.assertRaises(Exception):
            self.spec.arch.find("NonExistent")

    def test_get_all_memories(self):
        mems = self.spec.arch.get_nodes_of_type(Memory)
        names = {m.name for m in mems}
        self.assertIn("MainMemory", names)
        self.assertIn("GlobalBuffer", names)

    def test_get_all_computes(self):
        computes = self.spec.arch.get_nodes_of_type(Compute)
        names = {c.name for c in computes}
        self.assertIn("MAC", names)


class TestArchWithFanout(unittest.TestCase):
    """Test arch with Container node."""

    @classmethod
    def setUpClass(cls):
        path = (
            EXAMPLES_DIR
            / "arches"
            / "fanout_variations"
            / "at_glb_with_fanout_node.yaml"
        )
        if not path.exists():
            raise unittest.SkipTest("YAML not found")
        cls.spec = Spec.from_yaml(path)

    def test_fanout_found(self):
        fanout = self.spec.arch.find("GlobalBufferArray")
        self.assertIsInstance(fanout, Container)

    def test_fanout_get_fanout(self):
        fanout = self.spec.arch.find("GlobalBufferArray")
        self.assertEqual(fanout.get_fanout(), 4)

    def test_all_leaf_count(self):
        """Should have: MainMemory, GlobalBufferArray, GlobalBuffer, MAC."""
        from accelforge.frontend.arch import Leaf

        leaves = list(self.spec.arch.get_nodes_of_type(Leaf))
        self.assertEqual(len(leaves), 4)


if __name__ == "__main__":
    unittest.main()
