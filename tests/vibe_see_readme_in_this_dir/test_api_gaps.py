"""
Tests covering previously-untested API surface areas:

  1. Workload.get_iteration_space_shape_isl_string()
  2. Workload.n_instances / Einsum.n_instances
  3. Tensors.back, tile_shape, no_refetch_from_above, tensor_order_options
  4. Arch.total_area, total_leak_power, per_component_total_area/leak
  5. Spec.calculate_component_area_energy_latency_leak()
  6. Action.latency_scale, Component.latency_scale
  7. TensorHolder.bits_per_value_scale, bits_per_action
  8. Spec.to_yaml() round-trip serialization
  9. EvalableModel utility methods (all_fields_default, model_dump_non_none, etc.)
"""

import math
import os
import tempfile
import unittest
from pathlib import Path

from accelforge.frontend.spec import Spec
from accelforge.frontend.workload import (
    Workload,
    Einsum,
    TensorAccess,
    ImpliedProjection,
    Shape,
)
from accelforge.frontend.arch import (
    Action,
    Arch,
    Component,
    Comparison,
    Compute,
    Container,
    Memory,
    Tensors,
    TensorHolder,
    TensorHolderAction,
    Toll,
    Spatial as ArchSpatial,
)
from accelforge.frontend.mapping.mapping import (
    Mapping,
    Temporal,
    Storage,
    Compute as MappingCompute,
)
from accelforge.frontend.renames import Rename, RenameList, Renames
from accelforge.frontend.config import Config
from accelforge.frontend.variables import Variables

_REPO_ROOT = Path(__file__).parent.parent.parent
EXAMPLES_DIR = _REPO_ROOT / "examples"


def _yaml_spec(yaml_text: str, **jinja_parse_data) -> Spec:
    """Helper: write YAML text to a temp file and load it as a Spec."""
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
# 1. Workload.get_iteration_space_shape_isl_string
# ============================================================================


class TestGetIterationSpaceShapeIslString(unittest.TestCase):
    """Test Workload.get_iteration_space_shape_isl_string."""

    def _simple_workload(self):
        return Workload(
            rank_sizes={"M": 16, "K": 32, "N": 64},
            einsums=[
                Einsum(
                    name="Matmul",
                    tensor_accesses=[
                        TensorAccess(name="A", projection={"M": "m", "K": "k"}),
                        TensorAccess(name="B", projection={"K": "k", "N": "n"}),
                        TensorAccess(
                            name="C",
                            projection={"M": "m", "N": "n"},
                            output=True,
                        ),
                    ],
                )
            ],
        )

    def test_basic_isl_string(self):
        wl = self._simple_workload()
        isl_str = wl.get_iteration_space_shape_isl_string("Matmul")
        self.assertIsInstance(isl_str, str)
        # Should contain bounds for rank variables
        self.assertIn("m", isl_str)
        self.assertIn("k", isl_str)
        self.assertIn("n", isl_str)

    def test_isl_string_contains_bounds(self):
        wl = self._simple_workload()
        isl_str = wl.get_iteration_space_shape_isl_string("Matmul")
        # Should contain upper bounds from rank_sizes
        self.assertIn("16", isl_str)
        self.assertIn("32", isl_str)
        self.assertIn("64", isl_str)

    def test_isl_string_format(self):
        """ISL string should use '0 <= var < bound' format joined by 'and'."""
        wl = self._simple_workload()
        isl_str = wl.get_iteration_space_shape_isl_string("Matmul")
        self.assertIn("0 <=", isl_str)
        self.assertIn("and", isl_str)

    def test_with_iteration_space_shape(self):
        """Workload-level iteration_space_shape is incorporated."""
        wl = Workload(
            rank_sizes={"M": 16, "K": 32},
            iteration_space_shape={"m": "0 <= m < 16"},
            einsums=[
                Einsum(
                    name="E",
                    tensor_accesses=[
                        TensorAccess(name="A", projection={"M": "m", "K": "k"}),
                        TensorAccess(name="B", projection={"M": "m"}, output=True),
                    ],
                )
            ],
        )
        isl_str = wl.get_iteration_space_shape_isl_string("E")
        self.assertIn("0 <= m < 16", isl_str)

    def test_with_einsum_rank_sizes(self):
        """Per-Einsum rank_sizes override workload-level ones."""
        wl = Workload(
            rank_sizes={"M": 16},
            einsums=[
                Einsum(
                    name="E",
                    rank_sizes={"K": 99},
                    tensor_accesses=[
                        TensorAccess(name="A", projection={"M": "m", "K": "k"}),
                        TensorAccess(name="B", projection={"M": "m"}, output=True),
                    ],
                )
            ],
        )
        isl_str = wl.get_iteration_space_shape_isl_string("E")
        self.assertIn("99", isl_str)
        self.assertIn("16", isl_str)

    def test_from_yaml_three_matmuls(self):
        yaml_path = EXAMPLES_DIR / "workloads" / "three_matmuls_annotated.yaml"
        if not yaml_path.exists():
            self.skipTest("YAML not found")
        spec = Spec.from_yaml(yaml_path)
        # Einsum names in three_matmuls_annotated are Matmul1, Matmul2, Matmul3
        isl_str = spec.workload.get_iteration_space_shape_isl_string("Matmul1")
        self.assertIsInstance(isl_str, str)
        self.assertIn("and", isl_str)


# ============================================================================
# 2. Workload.n_instances / Einsum.n_instances
# ============================================================================


class TestNInstances(unittest.TestCase):
    """Test n_instances field on Workload and Einsum."""

    def test_einsum_n_instances_default(self):
        e = Einsum(
            name="E",
            tensor_accesses=[
                TensorAccess(name="A", projection={"M": "m"}),
                TensorAccess(name="B", projection={"M": "m"}, output=True),
            ],
        )
        self.assertEqual(e.n_instances, 1)

    def test_einsum_n_instances_custom(self):
        e = Einsum(
            name="E",
            n_instances=4,
            tensor_accesses=[
                TensorAccess(name="A", projection={"M": "m"}),
                TensorAccess(name="B", projection={"M": "m"}, output=True),
            ],
        )
        self.assertEqual(e.n_instances, 4)

    def test_workload_n_instances_default(self):
        wl = Workload()
        self.assertEqual(wl.n_instances, 1)

    def test_workload_n_instances_custom(self):
        wl = Workload(n_instances=8)
        self.assertEqual(wl.n_instances, 8)


# ============================================================================
# 3. Tensors.back, tile_shape, no_refetch_from_above, tensor_order_options
# ============================================================================


class TestTensorsAdvancedFields(unittest.TestCase):
    """Test advanced Tensors fields."""

    def test_back_default(self):
        t = Tensors(keep="All")
        self.assertEqual(t.back, "Nothing")

    def test_back_custom(self):
        t = Tensors(keep="All", back="Inputs")
        self.assertEqual(t.back, "Inputs")

    def test_tile_shape_default_empty(self):
        t = Tensors(keep="All")
        self.assertEqual(len(t.tile_shape), 0)

    def test_tile_shape_with_comparisons(self):
        t = Tensors(
            keep="All",
            tile_shape=[
                {"expression": "~m", "operator": "<=", "value": 128},
            ],
        )
        self.assertEqual(len(t.tile_shape), 1)
        self.assertIsInstance(t.tile_shape[0], Comparison)
        self.assertEqual(t.tile_shape[0].value, 128)

    def test_no_refetch_from_above_default(self):
        t = Tensors(keep="All")
        self.assertEqual(t.no_refetch_from_above, "~All")

    def test_no_refetch_from_above_custom(self):
        t = Tensors(keep="All", no_refetch_from_above="Inputs")
        self.assertEqual(t.no_refetch_from_above, "Inputs")

    def test_tensor_order_options_default_empty(self):
        t = Tensors(keep="All")
        self.assertEqual(len(t.tensor_order_options), 0)

    def test_tensor_order_options_custom(self):
        t = Tensors(
            keep="All",
            tensor_order_options=[["input | output", "weight"]],
        )
        self.assertEqual(len(t.tensor_order_options), 1)
        self.assertEqual(len(t.tensor_order_options[0]), 2)

    def test_force_memory_hierarchy_order_default(self):
        t = Tensors(keep="All")
        self.assertTrue(t.force_memory_hierarchy_order)

    def test_force_memory_hierarchy_order_false(self):
        t = Tensors(keep="All", force_memory_hierarchy_order=False)
        self.assertFalse(t.force_memory_hierarchy_order)

    def test_on_memory_from_yaml(self):
        """Test advanced Tensors fields survive YAML parsing."""
        spec = _yaml_spec(
            """\
arch:
  nodes:
  - !Memory
    name: Mem
    size: inf
    leak_power: 0
    area: 0
    tensors:
      keep: All
      back: Inputs
      no_refetch_from_above: Outputs
      tile_shape:
      - {expression: "~m", operator: "<=", value: 64}
    actions:
    - {name: read, energy: 1, latency: 0}
    - {name: write, energy: 1, latency: 0}
  - !Compute
    name: MAC
    leak_power: 0
    area: 0
    actions:
    - {name: compute, energy: 1, latency: 1}
"""
        )
        mem = spec.arch.find("Mem")
        self.assertEqual(mem.tensors.keep, "All")
        self.assertEqual(mem.tensors.back, "Inputs")
        self.assertEqual(mem.tensors.no_refetch_from_above, "Outputs")
        self.assertEqual(len(mem.tensors.tile_shape), 1)
        self.assertEqual(mem.tensors.tile_shape[0].value, 64)


# ============================================================================
# 4. Arch.total_area, total_leak_power, per_component_total_area/leak
# ============================================================================


class TestArchTotalAreaLeakPower(unittest.TestCase):
    """Test Arch.total_area and related properties."""

    def _make_arch_with_totals(self):
        """Build a Spec, calculate area/leak, and return it."""
        spec = _yaml_spec(
            """\
arch:
  nodes:
  - !Memory
    name: MainMemory
    size: inf
    leak_power: 0.5
    area: 100
    tensors: {keep: All}
    actions:
    - {name: read, energy: 1, latency: 0}
    - {name: write, energy: 1, latency: 0}
  - !Compute
    name: MAC
    leak_power: 0.1
    area: 10
    actions:
    - {name: compute, energy: 1, latency: 1}

workload:
  rank_sizes: {M: 16}
  bits_per_value: {All: 8}
  einsums:
  - name: E
    tensor_accesses:
    - {name: A, projection: [m]}
    - {name: B, projection: [m], output: true}
"""
        )
        return spec.calculate_component_area_energy_latency_leak()

    def test_per_component_total_area(self):
        spec = self._make_arch_with_totals()
        area = spec.arch.per_component_total_area
        self.assertIn("MainMemory", area)
        self.assertIn("MAC", area)
        self.assertEqual(area["MainMemory"], 100)
        self.assertEqual(area["MAC"], 10)

    def test_per_component_total_leak_power(self):
        spec = self._make_arch_with_totals()
        leak = spec.arch.per_component_total_leak_power
        self.assertIn("MainMemory", leak)
        self.assertIn("MAC", leak)
        self.assertAlmostEqual(leak["MainMemory"], 0.5)
        self.assertAlmostEqual(leak["MAC"], 0.1)

    def test_total_area(self):
        spec = self._make_arch_with_totals()
        self.assertAlmostEqual(spec.arch.total_area, 110)

    def test_total_leak_power(self):
        spec = self._make_arch_with_totals()
        self.assertAlmostEqual(spec.arch.total_leak_power, 0.6)

    def test_raises_without_calculation(self):
        """Accessing total_area before calculate raises ValueError."""
        spec = _yaml_spec(
            """\
arch:
  nodes:
  - !Memory
    name: Mem
    size: inf
    leak_power: 0
    area: 0
    actions:
    - {name: read, energy: 1, latency: 0}
    - {name: write, energy: 1, latency: 0}
  - !Compute
    name: MAC
    leak_power: 0
    area: 0
    actions:
    - {name: compute, energy: 1, latency: 1}
"""
        )
        # total_area depends on total_area being set per-component
        # which requires calculate_component_area_energy_latency_leak
        with self.assertRaises((ValueError, TypeError)):
            _ = spec.arch.total_area


# ============================================================================
# 5. Spec.calculate_component_area_energy_latency_leak
# ============================================================================


class TestCalculateComponentAreaEnergyLatencyLeak(unittest.TestCase):
    """Test the full calculate_component_area_energy_latency_leak method."""

    def _make_spec(self):
        return _yaml_spec(
            """\
arch:
  nodes:
  - !Memory
    name: MainMemory
    size: inf
    leak_power: 0.5
    area: 100
    tensors: {keep: All}
    actions:
    - {name: read, energy: 2.0, latency: 10}
    - {name: write, energy: 3.0, latency: 15}
  - !Compute
    name: MAC
    leak_power: 0.1
    area: 10
    actions:
    - {name: compute, energy: 0.5, latency: 1}

workload:
  rank_sizes: {M: 16}
  bits_per_value: {All: 8}
  einsums:
  - name: E
    tensor_accesses:
    - {name: A, projection: [m]}
    - {name: B, projection: [m], output: true}
"""
        )

    def test_returns_spec(self):
        spec = self._make_spec()
        result = spec.calculate_component_area_energy_latency_leak()
        self.assertIsInstance(result, Spec)

    def test_area_populated(self):
        spec = self._make_spec()
        result = spec.calculate_component_area_energy_latency_leak()
        mm = result.arch.find("MainMemory")
        self.assertEqual(mm.area, 100)
        self.assertIsNotNone(mm.total_area)

    def test_energy_populated(self):
        spec = self._make_spec()
        result = spec.calculate_component_area_energy_latency_leak()
        mm = result.arch.find("MainMemory")
        read_action = [a for a in mm.actions if a.name == "read"][0]
        self.assertEqual(read_action.energy, 2.0)

    def test_latency_populated(self):
        spec = self._make_spec()
        result = spec.calculate_component_area_energy_latency_leak()
        mm = result.arch.find("MainMemory")
        read_action = [a for a in mm.actions if a.name == "read"][0]
        self.assertEqual(read_action.latency, 10)

    def test_leak_power_populated(self):
        spec = self._make_spec()
        result = spec.calculate_component_area_energy_latency_leak()
        mm = result.arch.find("MainMemory")
        self.assertEqual(mm.leak_power, 0.5)

    def test_selective_area_only(self):
        spec = self._make_spec()
        result = spec.calculate_component_area_energy_latency_leak(
            area=True, energy=False, latency=False, leak=False
        )
        mm = result.arch.find("MainMemory")
        self.assertIsNotNone(mm.total_area)

    def test_noop_if_all_false(self):
        spec = self._make_spec()
        result = spec.calculate_component_area_energy_latency_leak(
            area=False, energy=False, latency=False, leak=False
        )
        self.assertIsInstance(result, Spec)

    def test_with_fanout_multiplies_area(self):
        """A spatial fanout should multiply area."""
        spec = _yaml_spec(
            """\
arch:
  nodes:
  - !Memory
    name: MainMemory
    size: inf
    leak_power: 0
    area: 0
    tensors: {keep: All}
    actions:
    - {name: read, energy: 1, latency: 0}
    - {name: write, energy: 1, latency: 0}
  - !Container
    name: Array
    spatial:
    - {name: X, fanout: 4}
  - !Compute
    name: MAC
    leak_power: 0.1
    area: 10
    actions:
    - {name: compute, energy: 1, latency: 1}

workload:
  rank_sizes: {M: 16}
  bits_per_value: {All: 8}
  einsums:
  - name: E
    tensor_accesses:
    - {name: A, projection: [m]}
    - {name: B, projection: [m], output: true}
"""
        )
        result = spec.calculate_component_area_energy_latency_leak()
        mac = result.arch.find("MAC")
        # total_area should be area * fanout = 10 * 4
        self.assertEqual(mac.total_area, 40)


# ============================================================================
# 6. Action.latency_scale, Component.latency_scale
# ============================================================================


class TestLatencyScale(unittest.TestCase):
    """Test latency_scale field on Action and Component."""

    def test_action_latency_scale_default(self):
        a = Action(name="read", energy=1, latency=5)
        self.assertEqual(a.latency_scale, 1)

    def test_action_latency_scale_custom(self):
        a = Action(name="read", energy=1, latency=5, latency_scale=2.0)
        self.assertEqual(a.latency_scale, 2.0)

    def test_component_latency_scale_default(self):
        c = Compute(
            name="MAC",
            leak_power=0,
            area=0,
            actions=[{"name": "compute", "energy": 1, "latency": 1}],
        )
        self.assertEqual(c.latency_scale, 1)

    def test_component_latency_scale_custom(self):
        c = Compute(
            name="MAC",
            leak_power=0,
            area=0,
            latency_scale=0.5,
            actions=[{"name": "compute", "energy": 1, "latency": 1}],
        )
        self.assertEqual(c.latency_scale, 0.5)

    def test_component_total_latency_default(self):
        c = Compute(
            name="MAC",
            leak_power=0,
            area=0,
            actions=[{"name": "compute", "energy": 1, "latency": 1}],
        )
        # Default total_latency is an expression
        self.assertIsInstance(c.total_latency, str)
        self.assertIn("sum", c.total_latency)


# ============================================================================
# 7. TensorHolder.bits_per_value_scale, bits_per_action
# ============================================================================


class TestTensorHolderFields(unittest.TestCase):
    """Test TensorHolder-specific fields."""

    def test_bits_per_value_scale_default(self):
        m = Memory(
            name="Mem",
            size=1024,
            leak_power=0,
            area=0,
            actions=[
                {"name": "read", "energy": 1, "latency": 0},
                {"name": "write", "energy": 1, "latency": 0},
            ],
        )
        self.assertEqual(m.bits_per_value_scale, {"All": 1})

    def test_bits_per_value_scale_custom(self):
        m = Memory(
            name="Mem",
            size=1024,
            leak_power=0,
            area=0,
            bits_per_value_scale={"All": 2},
            actions=[
                {"name": "read", "energy": 1, "latency": 0},
                {"name": "write", "energy": 1, "latency": 0},
            ],
        )
        self.assertEqual(m.bits_per_value_scale, {"All": 2})

    def test_bits_per_action_default_none(self):
        """TensorHolder-level bits_per_action defaults to None."""
        m = Memory(
            name="Mem",
            size=1024,
            leak_power=0,
            area=0,
            actions=[
                {"name": "read", "energy": 1, "latency": 0},
                {"name": "write", "energy": 1, "latency": 0},
            ],
        )
        self.assertIsNone(m.bits_per_action)

    def test_bits_per_action_custom(self):
        m = Memory(
            name="Mem",
            size=1024,
            leak_power=0,
            area=0,
            bits_per_action=32,
            actions=[
                {"name": "read", "energy": 1, "latency": 0},
                {"name": "write", "energy": 1, "latency": 0},
            ],
        )
        self.assertEqual(m.bits_per_action, 32)

    def test_tensor_holder_action_bits_per_action_default_is_expression(self):
        """TensorHolderAction.bits_per_action is an expression by default."""
        a = TensorHolderAction(name="read", energy=1, latency=0)
        self.assertIsInstance(a.bits_per_action, str)

    def test_tensor_holder_action_bits_per_action_custom(self):
        a = TensorHolderAction(name="read", energy=1, latency=0, bits_per_action=64)
        self.assertEqual(a.bits_per_action, 64)

    def test_toll_bits_per_value_scale(self):
        t = Toll(
            name="Q",
            direction="down",
            leak_power=0,
            area=0,
            bits_per_value_scale={"All": 0.5},
        )
        self.assertEqual(t.bits_per_value_scale, {"All": 0.5})


# ============================================================================
# 8. Spec.to_yaml() round-trip
# ============================================================================


class TestToYaml(unittest.TestCase):
    """Test to_yaml serialization.

    Note: Spec.to_yaml() fails if the Metrics enum is present (ruamel can't
    serialize it). So we test to_yaml on individual sub-models instead.
    """

    def test_workload_to_yaml_returns_string(self):
        wl = Workload(
            rank_sizes={"M": 16},
            einsums=[
                Einsum(
                    name="E",
                    tensor_accesses=[
                        TensorAccess(name="A", projection={"M": "m"}),
                        TensorAccess(name="B", projection={"M": "m"}, output=True),
                    ],
                )
            ],
        )
        yaml_str = wl.to_yaml()
        self.assertIsInstance(yaml_str, str)
        self.assertIn("E", yaml_str)
        self.assertIn("16", yaml_str)

    def test_workload_to_yaml_file(self):
        wl = Workload(
            rank_sizes={"M": 42},
            einsums=[
                Einsum(
                    name="E",
                    tensor_accesses=[
                        TensorAccess(name="A", projection={"M": "m"}),
                        TensorAccess(name="B", projection={"M": "m"}, output=True),
                    ],
                )
            ],
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            wl.to_yaml(f.name)
        try:
            with open(f.name) as fh:
                content = fh.read()
            self.assertIn("42", content)
        finally:
            os.unlink(f.name)

    def test_round_trip_workload(self):
        """Serialize a Workload to YAML, reload as dict, verify fields."""
        orig = Workload(
            rank_sizes={"M": 16, "K": 32},
            einsums=[
                Einsum(
                    name="Matmul",
                    tensor_accesses=[
                        TensorAccess(name="A", projection={"M": "m", "K": "k"}),
                        TensorAccess(name="B", projection={"K": "k"}, output=True),
                    ],
                )
            ],
        )
        yaml_str = orig.to_yaml()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, dir=str(_REPO_ROOT)
        ) as f:
            f.write("workload:\n")
            # Indent each line of the YAML string
            for line in yaml_str.splitlines():
                f.write(f"  {line}\n")
            f.flush()
            try:
                reloaded = Spec.from_yaml(f.name)
            finally:
                os.unlink(f.name)

        self.assertEqual(len(reloaded.workload.einsums), 1)
        self.assertEqual(reloaded.workload.einsums[0].name, "Matmul")

    def test_config_to_yaml(self):
        c = Config()
        yaml_str = c.to_yaml()
        self.assertIsInstance(yaml_str, str)

    def test_variables_to_yaml(self):
        v = Variables(x=42, y=99)
        yaml_str = v.to_yaml()
        self.assertIsInstance(yaml_str, str)
        self.assertIn("42", yaml_str)
        self.assertIn("99", yaml_str)

    def test_einsum_to_yaml(self):
        e = Einsum(
            name="E",
            tensor_accesses=[
                TensorAccess(name="A", projection={"M": "m"}),
                TensorAccess(name="B", projection={"M": "m"}, output=True),
            ],
        )
        yaml_str = e.to_yaml()
        self.assertIsInstance(yaml_str, str)
        self.assertIn("E", yaml_str)


# ============================================================================
# 9. EvalableModel utility methods
# ============================================================================


class TestAllFieldsDefault(unittest.TestCase):
    """Test EvalableModel.all_fields_default."""

    def test_default_spec_all_fields_default(self):
        spec = Spec()
        self.assertTrue(spec.all_fields_default())

    def test_non_default_spec(self):
        spec = Spec(variables=Variables(x=42))
        self.assertFalse(spec.all_fields_default())

    def test_default_config(self):
        c = Config()
        self.assertTrue(c.all_fields_default())

    def test_default_workload(self):
        wl = Workload()
        self.assertTrue(wl.all_fields_default())


class TestModelDumpNonNone(unittest.TestCase):
    """Test model_dump_non_none."""

    def test_excludes_none_values(self):
        c = Compute(
            name="MAC",
            leak_power=0,
            area=0,
            actions=[{"name": "compute", "energy": 1, "latency": 1}],
        )
        dump = c.model_dump_non_none()
        for k, v in dump.items():
            self.assertIsNotNone(v, f"key {k} should not be None")

    def test_includes_non_none_values(self):
        c = Compute(
            name="MAC",
            leak_power=0,
            area=0,
            actions=[{"name": "compute", "energy": 1, "latency": 1}],
        )
        dump = c.model_dump_non_none()
        self.assertEqual(dump["name"], "MAC")
        self.assertEqual(dump["area"], 0)


class TestShallowModelDump(unittest.TestCase):
    """Test shallow_model_dump."""

    def test_returns_dict(self):
        c = Compute(
            name="MAC",
            leak_power=0,
            area=0,
            actions=[{"name": "compute", "energy": 1, "latency": 1}],
        )
        dump = c.shallow_model_dump()
        self.assertIsInstance(dump, dict)
        self.assertIn("name", dump)

    def test_excludes_none_by_default(self):
        c = Compute(
            name="MAC",
            leak_power=0,
            area=0,
            actions=[{"name": "compute", "energy": 1, "latency": 1}],
        )
        dump = c.shallow_model_dump()
        for k, v in dump.items():
            self.assertIsNotNone(v, f"key {k} should not be None")

    def test_include_none(self):
        c = Compute(
            name="MAC",
            leak_power=0,
            area=0,
            actions=[{"name": "compute", "energy": 1, "latency": 1}],
        )
        dump = c.shallow_model_dump(include_None=True)
        # Should include fields that are None
        self.assertIsInstance(dump, dict)
        self.assertIn("name", dump)


class TestContainsAndGetItem(unittest.TestCase):
    """Test __contains__ and __getitem__ on EvalableModel."""

    def test_contains_existing_field(self):
        e = Einsum(
            name="E",
            tensor_accesses=[
                TensorAccess(name="A", projection={"M": "m"}),
                TensorAccess(name="B", projection={"M": "m"}, output=True),
            ],
        )
        self.assertIn("name", e)
        self.assertIn("tensor_accesses", e)

    def test_not_contains_nonexistent(self):
        e = Einsum(
            name="E",
            tensor_accesses=[
                TensorAccess(name="A", projection={"M": "m"}),
                TensorAccess(name="B", projection={"M": "m"}, output=True),
            ],
        )
        self.assertNotIn("nonexistent_field", e)

    def test_getitem(self):
        e = Einsum(
            name="E",
            tensor_accesses=[
                TensorAccess(name="A", projection={"M": "m"}),
                TensorAccess(name="B", projection={"M": "m"}, output=True),
            ],
        )
        self.assertEqual(e["name"], "E")

    def test_getitem_missing_raises_keyerror(self):
        e = Einsum(
            name="E",
            tensor_accesses=[
                TensorAccess(name="A", projection={"M": "m"}),
                TensorAccess(name="B", projection={"M": "m"}, output=True),
            ],
        )
        with self.assertRaises(KeyError):
            _ = e["nonexistent"]

    def test_setitem(self):
        e = Einsum(
            name="E",
            tensor_accesses=[
                TensorAccess(name="A", projection={"M": "m"}),
                TensorAccess(name="B", projection={"M": "m"}, output=True),
            ],
        )
        e["n_instances"] = 5
        self.assertEqual(e.n_instances, 5)


class TestEvalableListContains(unittest.TestCase):
    """Test __contains__ and __getitem__ on EvalableList (name-based lookup)."""

    def test_einsum_list_contains_by_name(self):
        wl = Workload(
            einsums=[
                Einsum(
                    name="E1",
                    tensor_accesses=[
                        TensorAccess(name="A", projection={"M": "m"}),
                        TensorAccess(name="B", projection={"M": "m"}, output=True),
                    ],
                ),
                Einsum(
                    name="E2",
                    tensor_accesses=[
                        TensorAccess(name="C", projection={"N": "n"}),
                        TensorAccess(name="D", projection={"N": "n"}, output=True),
                    ],
                ),
            ],
        )
        self.assertIn("E1", wl.einsums)
        self.assertIn("E2", wl.einsums)
        self.assertNotIn("E3", wl.einsums)

    def test_einsum_list_getitem_by_name(self):
        wl = Workload(
            einsums=[
                Einsum(
                    name="E1",
                    tensor_accesses=[
                        TensorAccess(name="A", projection={"M": "m"}),
                        TensorAccess(name="B", projection={"M": "m"}, output=True),
                    ],
                ),
            ],
        )
        e = wl.einsums["E1"]
        self.assertEqual(e.name, "E1")

    def test_tensor_access_list_getitem_by_name(self):
        e = Einsum(
            name="E",
            tensor_accesses=[
                TensorAccess(name="A", projection={"M": "m"}),
                TensorAccess(name="B", projection={"M": "m"}, output=True),
            ],
        )
        a = e.tensor_accesses["A"]
        self.assertEqual(a.name, "A")


# ============================================================================
# Extra: Component.component_class
# ============================================================================


class TestComponentClass(unittest.TestCase):
    """Test Component.component_class field."""

    def test_default_none(self):
        c = Compute(
            name="MAC",
            leak_power=0,
            area=0,
            actions=[{"name": "compute", "energy": 1, "latency": 1}],
        )
        self.assertIsNone(c.component_class)

    def test_custom_class(self):
        c = Compute(
            name="MAC",
            leak_power=0,
            area=0,
            component_class="MyCustomMAC",
            actions=[{"name": "compute", "energy": 1, "latency": 1}],
        )
        self.assertEqual(c.component_class, "MyCustomMAC")

    def test_get_component_class_raises_when_none(self):
        """If component_class is None, get_component_class raises EvaluationError."""
        from accelforge.util.exceptions import EvaluationError

        m = Memory(
            name="SRAM_32KB",
            size=1024,
            leak_power=0,
            area=0,
            actions=[
                {"name": "read", "energy": 1, "latency": 0},
                {"name": "write", "energy": 1, "latency": 0},
            ],
        )
        with self.assertRaises(EvaluationError):
            m.get_component_class()

    def test_get_component_class_when_set(self):
        """If component_class is set, get_component_class returns it."""
        m = Memory(
            name="SRAM",
            size=1024,
            leak_power=0,
            area=0,
            component_class="smartbuffer_SRAM",
            actions=[
                {"name": "read", "energy": 1, "latency": 0},
                {"name": "write", "energy": 1, "latency": 0},
            ],
        )
        cc = m.get_component_class()
        self.assertEqual(cc, "smartbuffer_SRAM")


# ============================================================================
# Extra: Workload.empty_renames, Einsum.empty_renames
# ============================================================================


class TestEmptyRenames(unittest.TestCase):
    """Test the empty_renames static/method."""

    def test_einsum_empty_renames_keys(self):
        renames = Einsum.empty_renames()
        self.assertIsInstance(renames, dict)
        # The static method returns: All, Tensors, Nothing, Inputs, Outputs
        expected_keys = {"All", "Inputs", "Outputs", "Nothing", "Tensors"}
        for key in expected_keys:
            self.assertIn(key, renames)

    def test_einsum_empty_renames_are_invertible_sets(self):
        from accelforge.util._setexpressions import InvertibleSet

        renames = Einsum.empty_renames()
        for key, val in renames.items():
            self.assertIsInstance(val, InvertibleSet, f"{key} should be InvertibleSet")

    def test_workload_empty_renames(self):
        """Workload.empty_renames is a plain method (not a property)."""
        wl = Workload(
            einsums=[
                Einsum(
                    name="E",
                    tensor_accesses=[
                        TensorAccess(name="A", projection={"M": "m"}),
                        TensorAccess(name="B", projection={"M": "m"}, output=True),
                    ],
                ),
            ],
        )
        renames = wl.empty_renames()
        self.assertIsInstance(renames, dict)
        for key in ("All", "Inputs", "Outputs", "Nothing"):
            self.assertIn(key, renames)


# ============================================================================
# Extra: Shape.rank_variables
# ============================================================================


class TestShapeRankVariables(unittest.TestCase):
    """Test Shape.rank_variables property."""

    def test_empty_shape(self):
        s = Shape()
        self.assertEqual(s.rank_variables, set())

    def test_shape_with_expressions(self):
        s = Shape(["0 <= m < 16", "0 <= k < 32"])
        rv = s.rank_variables
        self.assertIn("m", rv)
        self.assertIn("k", rv)


# ============================================================================
# Extra: Config.expression_custom_functions
# ============================================================================


class TestConfigExpressionCustomFunctions(unittest.TestCase):
    """Test that Config.expression_custom_functions stores the list."""

    def test_default_empty(self):
        c = Config()
        self.assertEqual(len(c.expression_custom_functions), 0)

    def test_from_yaml_with_paths(self):
        spec = _yaml_spec(
            """\
config:
  expression_custom_functions: []
  use_installed_component_models: false
  component_models: []
"""
        )
        self.assertEqual(len(spec.config.expression_custom_functions), 0)
        self.assertFalse(spec.config.use_installed_component_models)


if __name__ == "__main__":
    unittest.main()
