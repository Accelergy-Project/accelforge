import unittest
from pathlib import Path

from fastfusion.frontend.spec import Spec
from fastfusion.mapper import Metrics
from fastfusion.mapper.FFM.main import map_workload_to_arch


EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

M_SHAPE = 64
KN_SHAPE = 64


class ActionChecker(unittest.TestCase):
    def _check_memory_actions_exist(self, spec, memory_names, result):
        for einsum_name in spec.workload.einsum_names:
            for memory_name in memory_names:
                for memory_action in ["read", "write"]:
                    self.assertTrue(
                        f"{einsum_name}<SEP>action<SEP>{memory_name}<SEP>{memory_action}" in result.data.columns,
                        f"{einsum_name}<SEP>action<SEP>{memory_name}<SEP>{memory_action} "
                        f"not found in {result.data.columns}"
                    )


class TestMapper(ActionChecker, unittest.TestCase):
    def test_one_matmul(self):
        spec = Spec.from_yaml(
            EXAMPLES_DIR / "arches" / "simple.yaml",
            EXAMPLES_DIR / "workloads" / "matmuls.yaml",
            jinja_parse_data={"N_EINSUMS": 1, "M": 64, "KN": 64},
        )

        result = map_workload_to_arch(spec)
        self._check_memory_actions_exist(spec, ["MainMemory", "GlobalBuffer"], result)

    def test_two_matmuls(self):
        spec = Spec.from_yaml(
            EXAMPLES_DIR / "arches" / "simple.yaml",
            EXAMPLES_DIR / "workloads" / "matmuls.yaml",
            jinja_parse_data={"N_EINSUMS": 2, "M": 64, "KN": 64},
        )

        result = map_workload_to_arch(spec)
        self._check_memory_actions_exist(spec, ["MainMemory", "GlobalBuffer"], result)


class TestFanout(ActionChecker):
    FANOUT = 4
    """Need to sync this with YAMLs somehow."""

    def _run_with_arch(self, arch_fname: str, n_einsums=1):
        spec = Spec.from_yaml(
            EXAMPLES_DIR / "arches" / "fanout_variations" / arch_fname,
            EXAMPLES_DIR / "workloads" / "matmuls.yaml",
            jinja_parse_data={"N_EINSUMS": n_einsums, "M": 64, "KN": 64},
        )
        spec.mapper.ffm.metrics = Metrics.LATENCY

        result = map_workload_to_arch(spec)
        self._check_memory_actions_exist(spec, ["MainMemory", "GlobalBuffer"], result)
        self.assertEqual(
            result.data["Matmul0<SEP>Total<SEP>latency"].iloc[0],
            M_SHAPE*KN_SHAPE**2/self.FANOUT
        )
        return result


class TestMapperFanoutOneMatmul(TestFanout):
    def test_at_mac(self):
        self._run_with_arch("at_mac.yaml")

    def test_at_glb(self):
        self._run_with_arch("at_glb.yaml")

    def test_at_mac_with_fanout_node(self):
        self._run_with_arch("at_mac_with_fanout_node.yaml")

    def test_at_glb_with_fanout_node(self):
        self._run_with_arch("at_glb_with_fanout_node.yaml")


class TestMapperFanoutTwoMatmuls(TestFanout):
    def test_at_mac(self):
        self._run_with_arch("at_mac.yaml", n_einsums=2)

    def test_at_glb(self):
        self._run_with_arch("at_glb.yaml", n_einsums=2)

    def test_at_mac_with_fanout_node(self):
        self._run_with_arch("at_mac_with_fanout_node.yaml", n_einsums=2)

    def test_at_glb_with_fanout_node(self):
        self._run_with_arch("at_glb_with_fanout_node.yaml", n_einsums=2)


class TestMapperFanoutConstraints(TestFanout):
    def test_at_mac_constraints(self):
        self._run_with_arch("at_mac_with_constraints.yaml")