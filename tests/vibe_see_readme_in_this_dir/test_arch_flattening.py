"""
Tests for architecture flattening:
  - Arch._flatten (Hierarchical and Fork)
  - Spec._get_flattened_architecture
  - get_nodes_of_type
  - find
  - Duplicate name detection
  - Fork branch selection
"""

import unittest
from pathlib import Path

from accelforge.frontend.spec import Spec
from accelforge.frontend.arch import (
    Arch,
    Hierarchical,
    Fork,
    Memory,
    Compute,
    Container,
    Toll,
    Leaf,
    Branch,
)
from accelforge.frontend.workload import Workload

_REPO_ROOT = Path(__file__).parent.parent.parent
EXAMPLES_DIR = _REPO_ROOT / "examples"


def _simple_arch():
    """Simple 3-level architecture: MainMemory -> GlobalBuffer -> MAC."""
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


def _simple_workload():
    """Single matmul workload."""
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


def _fork_arch():
    """Arch with a fork: MainMemory -> GlobalBuffer -> Fork([ScalarUnit], [Container -> Register -> MAC])."""
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
            Fork(
                nodes=[
                    Compute(
                        name="ScalarUnit",
                        actions=[{"name": "compute", "energy": 0, "latency": 1}],
                        leak_power=0,
                        area=0,
                    ),
                ]
            ),
            Container(
                name="PE",
                spatial=[
                    {
                        "name": "reuse_input",
                        "fanout": 4,
                    },
                ],
            ),
            Memory(
                name="Register",
                size=8,
                actions=[
                    {"name": "read", "energy": 0, "latency": 0},
                    {"name": "write", "energy": 0, "latency": 0},
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


# ============================================================================
# Arch.find
# ============================================================================


class TestArchFind(unittest.TestCase):
    def test_find_memory(self):
        arch = _simple_arch()
        node = arch.find("MainMemory")
        self.assertIsInstance(node, Memory)
        self.assertEqual(node.name, "MainMemory")

    def test_find_compute(self):
        arch = _simple_arch()
        node = arch.find("MAC")
        self.assertIsInstance(node, Compute)
        self.assertEqual(node.name, "MAC")

    def test_find_nonexistent_raises(self):
        arch = _simple_arch()
        with self.assertRaises(ValueError):
            arch.find("NonexistentNode")

    def test_find_in_fork(self):
        arch = _fork_arch()
        node = arch.find("ScalarUnit")
        self.assertIsInstance(node, Compute)
        self.assertEqual(node.name, "ScalarUnit")

    def test_find_after_fork(self):
        arch = _fork_arch()
        node = arch.find("Register")
        self.assertIsInstance(node, Memory)

    def test_find_deep_compute(self):
        arch = _fork_arch()
        node = arch.find("MAC")
        self.assertIsInstance(node, Compute)


# ============================================================================
# Arch.get_nodes_of_type
# ============================================================================


class TestGetNodesOfType(unittest.TestCase):
    def test_get_all_memory(self):
        arch = _simple_arch()
        memories = list(arch.get_nodes_of_type(Memory))
        self.assertEqual(len(memories), 2)
        names = {m.name for m in memories}
        self.assertEqual(names, {"MainMemory", "GlobalBuffer"})

    def test_get_all_compute(self):
        arch = _simple_arch()
        computes = list(arch.get_nodes_of_type(Compute))
        self.assertEqual(len(computes), 1)
        self.assertEqual(computes[0].name, "MAC")

    def test_get_all_leaf(self):
        arch = _simple_arch()
        leaves = list(arch.get_nodes_of_type(Leaf))
        self.assertEqual(len(leaves), 3)

    def test_get_memory_in_fork_arch(self):
        arch = _fork_arch()
        memories = list(arch.get_nodes_of_type(Memory))
        names = {m.name for m in memories}
        self.assertIn("MainMemory", names)
        self.assertIn("GlobalBuffer", names)
        self.assertIn("Register", names)

    def test_get_compute_in_fork_arch(self):
        arch = _fork_arch()
        computes = list(arch.get_nodes_of_type(Compute))
        names = {c.name for c in computes}
        self.assertEqual(names, {"ScalarUnit", "MAC"})

    def test_get_fanout_in_fork_arch(self):
        arch = _fork_arch()
        fanouts = list(arch.get_nodes_of_type(Container))
        self.assertEqual(len(fanouts), 1)
        self.assertEqual(fanouts[0].name, "PE")


# ============================================================================
# Arch._flatten
# ============================================================================


class TestArchFlatten(unittest.TestCase):
    def test_flatten_simple(self):
        arch = _simple_arch()
        flat = arch._flatten("MAC")
        names = [n.name for n in flat]
        self.assertEqual(names, ["MainMemory", "GlobalBuffer", "MAC"])

    def test_flatten_returns_leaf_instances(self):
        arch = _simple_arch()
        flat = arch._flatten("MAC")
        for node in flat:
            self.assertIsInstance(node, Leaf)

    def test_flatten_ends_with_compute(self):
        arch = _simple_arch()
        flat = arch._flatten("MAC")
        self.assertIsInstance(flat[-1], Compute)

    def test_flatten_fork_mac_path(self):
        """Flattening to MAC should skip the Fork (ScalarUnit) branch."""
        arch = _fork_arch()
        flat = arch._flatten("MAC")
        names = [n.name for n in flat]
        self.assertIn("MainMemory", names)
        self.assertIn("GlobalBuffer", names)
        self.assertIn("MAC", names)
        # ScalarUnit is in the fork, should not be in the MAC path
        self.assertNotIn("ScalarUnit", names)

    def test_flatten_fork_scalar_path(self):
        """Flattening to ScalarUnit should go through the Fork branch."""
        arch = _fork_arch()
        flat = arch._flatten("ScalarUnit")
        names = [n.name for n in flat]
        self.assertIn("MainMemory", names)
        self.assertIn("GlobalBuffer", names)
        self.assertIn("ScalarUnit", names)
        # MAC is the main compute, should not be in ScalarUnit's path
        self.assertNotIn("MAC", names)

    def test_flatten_nonexistent_compute_returns_no_compute(self):
        """Flattening to a non-existent compute node returns a list without that node."""
        arch = _simple_arch()
        # The arch just collects non-matching leaves until it runs out of nodes.
        flat = arch._flatten("NonexistentCompute")
        names = [n.name for n in flat]
        self.assertNotIn("NonexistentCompute", names)


# ============================================================================
# Spec._get_flattened_architecture
# ============================================================================


class TestSpecGetFlattenedArchitecture(unittest.TestCase):
    def _make_spec(self, arch=None, workload=None):
        spec = Spec(
            arch=arch or _simple_arch(),
            workload=workload or _simple_workload(),
        )
        return spec._spec_eval_expressions()

    def test_returns_all_paths(self):
        spec = self._make_spec()
        result = spec._get_flattened_architecture()
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_single_compute_gives_one_path(self):
        spec = self._make_spec()
        result = spec._get_flattened_architecture()
        self.assertEqual(len(result), 1)
        names = [n.name for n in result[0]]
        self.assertIn("MAC", names)

    def test_path_ends_with_compute(self):
        spec = self._make_spec()
        result = spec._get_flattened_architecture()
        for path in result:
            self.assertIsInstance(path[-1], Compute)

    def test_specific_compute_node(self):
        """When compute_node is given, returns a single flat list (not list of lists)."""
        spec = self._make_spec()
        result = spec._get_flattened_architecture(compute_node="MAC")
        # Returns a single path (flat list of Leaf nodes), not list of lists
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[-1], Leaf)
        self.assertEqual(result[-1].name, "MAC")

    def test_unevaluated_spec_raises(self):
        spec = Spec(
            arch=_simple_arch(),
            workload=_simple_workload(),
        )
        with self.assertRaises(AssertionError):
            spec._get_flattened_architecture()

    def test_fork_arch_all_paths(self):
        spec = self._make_spec(arch=_fork_arch())
        result = spec._get_flattened_architecture()
        # Should have 2 paths: one to ScalarUnit and one to MAC
        self.assertEqual(len(result), 2)
        all_compute_names = {path[-1].name for path in result}
        self.assertEqual(all_compute_names, {"ScalarUnit", "MAC"})

    def test_fork_arch_specific_mac(self):
        """When compute_node is given for a fork arch, returns a single flat path."""
        spec = self._make_spec(arch=_fork_arch())
        result = spec._get_flattened_architecture(compute_node="MAC")
        # Returns a single path, not list of lists
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[-1], Leaf)
        self.assertEqual(result[-1].name, "MAC")


# ============================================================================
# Flattening from YAML examples
# ============================================================================


class TestFlattenFromYAML(unittest.TestCase):
    def test_simple_arch_flatten(self):
        arch_path = EXAMPLES_DIR / "arches" / "simple.yaml"
        wl_path = EXAMPLES_DIR / "workloads" / "matmuls.yaml"
        if not arch_path.exists() or not wl_path.exists():
            self.skipTest("YAML files not found")
        spec = Spec.from_yaml(arch_path, wl_path, jinja_parse_data={"N_EINSUMS": 1})
        evaluated = spec._spec_eval_expressions()
        result = evaluated._get_flattened_architecture()
        self.assertEqual(len(result), 1)
        names = [n.name for n in result[0]]
        self.assertEqual(names, ["MainMemory", "GlobalBuffer", "MAC"])

    def test_tpu_arch_flatten(self):
        """TPU arch requires einsum context for expressions like len(All) == 2."""
        arch_path = EXAMPLES_DIR / "arches" / "tpu_v4i.yaml"
        wl_path = EXAMPLES_DIR / "workloads" / "three_matmuls_annotated.yaml"
        if not arch_path.exists() or not wl_path.exists():
            self.skipTest("YAML files not found")
        spec = Spec.from_yaml(arch_path, wl_path)
        # Full evaluation with einsum provides needed context for expressions
        evaluated = spec._spec_eval_expressions(einsum_name="Matmul1")
        result = evaluated._get_flattened_architecture()
        # TPU has 2 computes: ScalarUnit and MAC
        self.assertEqual(len(result), 2)

    def test_tpu_mac_path_includes_register(self):
        arch_path = EXAMPLES_DIR / "arches" / "tpu_v4i.yaml"
        wl_path = EXAMPLES_DIR / "workloads" / "three_matmuls_annotated.yaml"
        if not arch_path.exists() or not wl_path.exists():
            self.skipTest("YAML files not found")
        spec = Spec.from_yaml(arch_path, wl_path)
        evaluated = spec._spec_eval_expressions(einsum_name="Matmul1")
        result = evaluated._get_flattened_architecture(compute_node="MAC")
        names = [n.name for n in result]
        self.assertIn("Register", names)
        self.assertIn("PE", names)
        self.assertEqual(names[-1], "MAC")

    def test_tpu_scalar_path_excludes_register(self):
        arch_path = EXAMPLES_DIR / "arches" / "tpu_v4i.yaml"
        wl_path = EXAMPLES_DIR / "workloads" / "three_matmuls_annotated.yaml"
        if not arch_path.exists() or not wl_path.exists():
            self.skipTest("YAML files not found")
        spec = Spec.from_yaml(arch_path, wl_path)
        evaluated = spec._spec_eval_expressions(einsum_name="Matmul1")
        result = evaluated._get_flattened_architecture(compute_node="ScalarUnit")
        names = [n.name for n in result]
        self.assertNotIn("Register", names)
        self.assertNotIn("PE", names)
        self.assertEqual(names[-1], "ScalarUnit")


# ============================================================================
# Duplicate name detection
# ============================================================================


class TestDuplicateNameDetection(unittest.TestCase):
    def test_duplicate_leaf_name_rejected_by_pydantic(self):
        """Architecture with duplicate leaf names is rejected during construction."""
        # Pydantic validation catches duplicate names, preventing construction
        with self.assertRaises(Exception):
            Arch(
                nodes=[
                    Memory(
                        name="Mem",
                        size=100,
                        actions=[
                            {"name": "read", "energy": 1, "latency": 0},
                            {"name": "write", "energy": 1, "latency": 0},
                        ],
                        leak_power=0,
                        area=0,
                    ),
                    Memory(
                        name="Mem",
                        size=200,
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
            )


if __name__ == "__main__":
    unittest.main()
