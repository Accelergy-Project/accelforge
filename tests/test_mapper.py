import unittest

import accelforge as af
from accelforge.frontend.spec import Spec
from accelforge.mapper import Metrics
from accelforge.mapper.FFM.main import map_workload_to_arch

try:
    from .paths import EXAMPLES_DIR
except ImportError:
    from paths import EXAMPLES_DIR

M_SHAPE = 64
KN_SHAPE = 64


class ActionChecker(unittest.TestCase):
    def _check_memory_actions_exist(self, spec, memory_names, result):
        for einsum_name in spec.workload.einsum_names:
            for memory_name in memory_names:
                for memory_action in ["read", "write"]:
                    k = f"{einsum_name}<SEP>action<SEP>{memory_name}<SEP>"
                    matched = [col for col in result.data.columns if col.startswith(k)]
                    matched = {x.split("<SEP>")[-1] for x in matched}
                    self.assertTrue(
                        memory_action in matched,
                        f"{einsum_name} {memory_name} {memory_action} not found in {result.data.columns}",
                    )


class TestMapper(ActionChecker, unittest.TestCase):
    def test_one_matmul(self):
        spec = Spec.from_yaml(
            af.examples.arches.simple,
            af.examples.workloads.matmuls,
            jinja_parse_data={"N_EINSUMS": 1, "M": 64, "KN": 64},
        )
        result = map_workload_to_arch(spec)
        self._check_memory_actions_exist(spec, ["MainMemory"], result)

    def test_two_matmuls(self):
        spec = Spec.from_yaml(
            af.examples.arches.simple,
            af.examples.workloads.matmuls,
            jinja_parse_data={"N_EINSUMS": 2, "M": 64, "KN": 64},
        )
        result = map_workload_to_arch(spec)
        self._check_memory_actions_exist(spec, ["MainMemory"], result)

    def test_mapper_return_many_mappings(self):
        spec = Spec.from_yaml(
            af.examples.arches.simple,
            af.examples.workloads.matmuls,
            jinja_parse_data={
                "N_EINSUMS": 1,
                "M": 64,
                "KN": 64,
                "MainMemoryEnergy": 10,
                "GlobalBufferLatency": 1,
            },
        )
        spec.mapper.metrics = Metrics.LATENCY | Metrics.ENERGY
        result = map_workload_to_arch(spec)
        self.assertTrue(
            result.data["Matmul0<SEP>energy<SEP>GlobalBuffer<SEP>W0<SEP>read"]
            .notna()
            .all(),
            "NaN found in mapper result",
        )
        self._check_memory_actions_exist(spec, ["MainMemory"], result)


class TestFanout(ActionChecker):
    FANOUT = 4
    """Need to sync this with YAMLs somehow."""

    def _run_with_arch(self, arch_fname: str, n_einsums=1):
        spec = Spec.from_yaml(
            EXAMPLES_DIR / "arches" / "fanout_variations" / arch_fname,
            EXAMPLES_DIR / "workloads" / "matmuls.yaml",
            jinja_parse_data={"N_EINSUMS": n_einsums, "M": 64, "KN": 64},
        )
        spec.mapper.metrics = Metrics.LATENCY

        result = map_workload_to_arch(spec)
        self._check_memory_actions_exist(spec, ["MainMemory", "GlobalBuffer"], result)
        latency = result.latency(per_einsum=True)
        self.assertIn(
            "Matmul0",
            latency,
            f"Matmul0 not in {latency}",
        )
        self.assertEqual(latency["Matmul0"], M_SHAPE * KN_SHAPE**2 / self.FANOUT)
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
