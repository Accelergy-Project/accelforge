"""
Tests for rendering:
  - Workload.render() -> SVG
  - Arch.render() (Hierarchical.render) -> SVG
  - Mapping.render() -> SVG
  - Mapping.render_pydot() -> pydot.Dot
  - Arch _render_node_name, _render_node_label, _render_node_shape
  - MappingNode _render_node_name, _render_node_label
  - Rendering from YAML examples
"""

import unittest
from pathlib import Path

import pydot

from accelforge.frontend.spec import Spec
from accelforge.frontend.workload import Workload
from accelforge.frontend.arch import (
    Arch,
    Hierarchical,
    Memory,
    Compute,
    Container,
    Toll,
    Leaf,
)
from accelforge.frontend.mapping.mapping import (
    Mapping,
    Temporal,
    Spatial,
    Storage,
    Compute as MappingCompute,
    Nested,
    Sequential,
    Pipeline,
    Reservation,
)

_REPO_ROOT = Path(__file__).parent.parent.parent
EXAMPLES_DIR = _REPO_ROOT / "examples"


# ============================================================================
# Workload rendering
# ============================================================================


class TestWorkloadRender(unittest.TestCase):
    def _make_workload(self):
        return Workload(
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

    def test_render_returns_svg_string(self):
        wl = self._make_workload()
        svg = wl.render()
        self.assertIsInstance(svg, str)
        self.assertIn("<svg", str(svg))

    def test_render_contains_einsum_name(self):
        wl = self._make_workload()
        svg = str(wl.render())
        self.assertIn("Matmul", svg)

    def test_render_contains_tensor_names(self):
        wl = self._make_workload()
        svg = str(wl.render())
        for name in ["A", "B", "C"]:
            self.assertIn(name, svg)

    def test_render_multiple_einsums(self):
        wl = Workload(
            rank_sizes={"M": 16, "K": 16, "N": 16, "P": 16},
            bits_per_value={"All": 8},
            einsums=[
                {
                    "name": "Matmul0",
                    "tensor_accesses": [
                        {"name": "T0", "projection": ["m", "k"]},
                        {"name": "W0", "projection": ["k", "n"]},
                        {"name": "T1", "projection": ["m", "n"], "output": True},
                    ],
                },
                {
                    "name": "Matmul1",
                    "tensor_accesses": [
                        {"name": "T1", "projection": ["m", "n"]},
                        {"name": "W1", "projection": ["n", "p"]},
                        {"name": "T2", "projection": ["m", "p"], "output": True},
                    ],
                },
            ],
        )
        svg = str(wl.render())
        self.assertIn("<svg", svg)
        self.assertIn("Matmul0", svg)
        self.assertIn("Matmul1", svg)

    def test_render_repr_svg(self):
        """_repr_svg_ should return the same SVG as render()."""
        wl = self._make_workload()
        self.assertEqual(str(wl._repr_svg_()), str(wl.render()))

    def test_render_from_yaml_workload(self):
        yaml_path = EXAMPLES_DIR / "workloads" / "three_matmuls_annotated.yaml"
        if not yaml_path.exists():
            self.skipTest("YAML file not found")
        spec = Spec.from_yaml(yaml_path)
        evaluated = spec._spec_eval_expressions()
        svg = str(evaluated.workload.render())
        self.assertIn("<svg", svg)


# ============================================================================
# Architecture rendering
# ============================================================================


class TestArchRender(unittest.TestCase):
    def _make_arch(self):
        return Arch(
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
                Memory(
                    name="GlobalBuffer",
                    size=100_000,
                    actions=[
                        {"name": "read", "energy": 0.5, "latency": 0},
                        {"name": "write", "energy": 0.5, "latency": 0},
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
        )

    def test_render_returns_svg(self):
        arch = self._make_arch()
        svg = arch.render()
        self.assertIsInstance(svg, str)
        self.assertIn("<svg", str(svg))

    def test_render_contains_node_names(self):
        arch = self._make_arch()
        svg = str(arch.render())
        for name in ["MainMemory", "GlobalBuffer", "MAC"]:
            self.assertIn(name, svg)

    def test_render_repr_svg(self):
        arch = self._make_arch()
        self.assertEqual(str(arch._repr_svg_()), str(arch.render()))

    def test_render_from_yaml_arch(self):
        arch_path = EXAMPLES_DIR / "arches" / "simple.yaml"
        wl_path = EXAMPLES_DIR / "workloads" / "matmuls.yaml"
        if not arch_path.exists() or not wl_path.exists():
            self.skipTest("YAML files not found")
        spec = Spec.from_yaml(arch_path, wl_path, jinja_parse_data={"N_EINSUMS": 1})
        evaluated = spec._spec_eval_expressions()
        svg = str(evaluated.arch.render())
        self.assertIn("<svg", svg)
        self.assertIn("MainMemory", svg)

    def test_render_tpu_arch(self):
        """TPU arch needs einsum context for expressions like len(All) == 2."""
        arch_path = EXAMPLES_DIR / "arches" / "tpu_v4i.yaml"
        wl_path = EXAMPLES_DIR / "workloads" / "three_matmuls_annotated.yaml"
        if not arch_path.exists() or not wl_path.exists():
            self.skipTest("YAML files not found")
        spec = Spec.from_yaml(arch_path, wl_path)
        # TPU requires einsum context for evaluation
        evaluated = spec._spec_eval_expressions(einsum_name="Matmul1")
        svg = str(evaluated.arch.render())
        self.assertIn("<svg", svg)
        self.assertIn("MAC", svg)


# ============================================================================
# Architecture render helpers
# ============================================================================


class TestArchRenderHelpers(unittest.TestCase):
    def test_render_node_name_memory(self):
        """_render_node_name uses class name + object id, not the component name."""
        mem = Memory(
            name="TestMem",
            size=100,
            actions=[
                {"name": "read", "energy": 1, "latency": 0},
                {"name": "write", "energy": 1, "latency": 0},
            ],
            leak_power=0,
            area=0,
        )
        name = mem._render_node_name()
        self.assertIsInstance(name, str)
        self.assertIn("Memory", name)

    def test_render_node_label_memory(self):
        mem = Memory(
            name="TestMem",
            size=100,
            actions=[
                {"name": "read", "energy": 1, "latency": 0},
                {"name": "write", "energy": 1, "latency": 0},
            ],
            leak_power=0,
            area=0,
        )
        label = mem._render_node_label()
        self.assertIsInstance(label, str)
        self.assertIn("TestMem", label)

    def test_render_node_shape_memory(self):
        mem = Memory(
            name="TestMem",
            size=100,
            actions=[
                {"name": "read", "energy": 1, "latency": 0},
                {"name": "write", "energy": 1, "latency": 0},
            ],
            leak_power=0,
            area=0,
        )
        shape = mem._render_node_shape()
        self.assertIsInstance(shape, str)

    def test_render_node_compute(self):
        comp = Compute(
            name="MAC",
            actions=[{"name": "compute", "energy": 1, "latency": 1}],
            leak_power=0,
            area=0,
        )
        node = comp._render_node()
        self.assertIsInstance(node, pydot.Node)


# ============================================================================
# Mapping rendering
# ============================================================================


class TestMappingRender(unittest.TestCase):
    def _make_mapping(self):
        return Mapping(
            nodes=[
                Storage(tensors=["A", "B", "C"], component="MainMemory"),
                Temporal(rank_variable="m", tile_shape=4),
                Temporal(rank_variable="n", tile_shape=4),
                Storage(tensors=["A", "B", "C"], component="GlobalBuffer"),
                Temporal(rank_variable="k", tile_shape=1),
                MappingCompute(einsum="Matmul", component="MAC"),
            ]
        )

    def test_render_returns_svg(self):
        mapping = self._make_mapping()
        svg = mapping.render()
        self.assertIsInstance(svg, str)
        self.assertIn("<svg", str(svg))

    def test_render_pydot_returns_dot(self):
        mapping = self._make_mapping()
        graph = mapping.render_pydot()
        self.assertIsInstance(graph, pydot.Dot)

    def test_render_contains_components(self):
        mapping = self._make_mapping()
        svg = str(mapping.render())
        self.assertIn("MainMemory", svg)
        self.assertIn("GlobalBuffer", svg)

    def test_render_repr_svg(self):
        mapping = self._make_mapping()
        self.assertEqual(str(mapping._repr_svg_()), str(mapping.render()))

    def test_render_without_reservations(self):
        mapping = Mapping(
            nodes=[
                Storage(tensors=["A"], component="MainMemory"),
                Reservation(purposes=["A"], resource="MainMemory"),
                Temporal(rank_variable="m", tile_shape=1),
                MappingCompute(einsum="E", component="MAC"),
            ]
        )
        graph = mapping.render_pydot(with_reservations=False)
        self.assertIsInstance(graph, pydot.Dot)

    def test_render_with_reservations(self):
        mapping = Mapping(
            nodes=[
                Storage(tensors=["A"], component="MainMemory"),
                Reservation(purposes=["A"], resource="MainMemory"),
                Temporal(rank_variable="m", tile_shape=1),
                MappingCompute(einsum="E", component="MAC"),
            ]
        )
        graph = mapping.render_pydot(with_reservations=True)
        self.assertIsInstance(graph, pydot.Dot)

    def test_render_without_tile_shape(self):
        mapping = self._make_mapping()
        graph = mapping.render_pydot(with_tile_shape=False)
        self.assertIsInstance(graph, pydot.Dot)

    def test_render_nested_mapping(self):
        mapping = Mapping(
            nodes=[
                Storage(tensors=["A", "B"], component="MainMemory"),
                Sequential(
                    nodes=[
                        Nested(
                            nodes=[
                                Storage(tensors=["A"], component="Buf"),
                                Temporal(rank_variable="m", tile_shape=1),
                                MappingCompute(einsum="E1", component="MAC"),
                            ]
                        ),
                        Nested(
                            nodes=[
                                Storage(tensors=["B"], component="Buf"),
                                Temporal(rank_variable="n", tile_shape=1),
                                MappingCompute(einsum="E2", component="MAC"),
                            ]
                        ),
                    ]
                ),
            ]
        )
        svg = str(mapping.render())
        self.assertIn("<svg", svg)

    def test_render_pipeline_mapping(self):
        mapping = Mapping(
            nodes=[
                Storage(tensors=["A", "B"], component="MainMemory"),
                Pipeline(
                    nodes=[
                        Nested(
                            nodes=[
                                Storage(tensors=["A"], component="Buf"),
                                Temporal(rank_variable="m", tile_shape=1),
                                MappingCompute(einsum="E1", component="MAC"),
                            ]
                        ),
                        Nested(
                            nodes=[
                                Storage(tensors=["B"], component="Buf"),
                                Temporal(rank_variable="n", tile_shape=1),
                                MappingCompute(einsum="E2", component="MAC"),
                            ]
                        ),
                    ]
                ),
            ]
        )
        svg = str(mapping.render())
        self.assertIn("<svg", svg)


# ============================================================================
# Mapping render helpers
# ============================================================================


class TestMappingRenderHelpers(unittest.TestCase):
    def test_temporal_render_node_name(self):
        t = Temporal(rank_variable="m", tile_shape=4)
        name = t._render_node_name()
        self.assertIsInstance(name, str)
        self.assertGreater(len(name), 0)

    def test_temporal_render_node_label(self):
        t = Temporal(rank_variable="m", tile_shape=4)
        label = t._render_node_label()
        self.assertIsInstance(label, str)

    def test_storage_render_node_name(self):
        s = Storage(tensors=["A"], component="Mem")
        name = s._render_node_name()
        self.assertIsInstance(name, str)

    def test_compute_render_node_name(self):
        c = MappingCompute(einsum="E1", component="MAC")
        name = c._render_node_name()
        self.assertIsInstance(name, str)


# ============================================================================
# Rendering from YAML examples
# ============================================================================


class TestRenderFromYAML(unittest.TestCase):
    def test_full_spec_workload_render(self):
        yaml_path = EXAMPLES_DIR / "workloads" / "gpt3_6.7B.yaml"
        if not yaml_path.exists():
            self.skipTest("YAML file not found")
        spec = Spec.from_yaml(yaml_path)
        evaluated = spec._spec_eval_expressions()
        svg = str(evaluated.workload.render())
        self.assertIn("<svg", svg)
        # Should contain GPT einsum names (V, K, Q, QK, QK_softmax, AV, Z, FFA, FFB)
        for name in ["V", "K", "Q", "Z", "FFA", "FFB"]:
            self.assertIn(name, svg)

    def test_full_spec_mapping_render(self):
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
        svg = str(evaluated.mapping.render())
        self.assertIn("<svg", svg)


# ============================================================================
# Mapping _flatten helper
# ============================================================================


class TestMappingFlatten(unittest.TestCase):
    def test_flatten_simple(self):
        mapping = Mapping(
            nodes=[
                Storage(tensors=["A"], component="Mem"),
                Temporal(rank_variable="m", tile_shape=1),
                MappingCompute(einsum="E", component="MAC"),
            ]
        )
        flat = mapping._flatten()
        self.assertIsInstance(flat, list)
        # Root + Storage + Temporal + Compute
        self.assertEqual(len(flat), 4)

    def test_flatten_nested(self):
        mapping = Mapping(
            nodes=[
                Storage(tensors=["A"], component="Mem"),
                Sequential(
                    nodes=[
                        Nested(
                            nodes=[
                                Temporal(rank_variable="m", tile_shape=1),
                                MappingCompute(einsum="E", component="MAC"),
                            ]
                        ),
                    ]
                ),
            ]
        )
        flat = mapping._flatten()
        # Root + Storage + Sequential + Nested + Temporal + Compute
        self.assertEqual(len(flat), 6)

    def test_flatten_includes_all_types(self):
        mapping = Mapping(
            nodes=[
                Storage(tensors=["A"], component="Mem"),
                Temporal(rank_variable="m", tile_shape=1),
                MappingCompute(einsum="E", component="MAC"),
            ]
        )
        flat = mapping._flatten()
        types = {type(n).__name__ for n in flat}
        self.assertIn("Mapping", types)
        self.assertIn("Storage", types)
        self.assertIn("Temporal", types)
        self.assertIn("Compute", types)


if __name__ == "__main__":
    unittest.main()
