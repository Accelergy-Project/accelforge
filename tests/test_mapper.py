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

af.set_n_parallel_jobs(1)


class TestTransformer(unittest.TestCase):
    def test_bert(self):
        spec = Spec.from_yaml(
            af.examples.arches.simple,
            af.examples.workloads.bert,
        )
        spec.mapper.metrics = Metrics.ENERGY | Metrics.LATENCY
        spec.mapper.n_concurrent_threads = 2
        # spec.mapper._only_output_pmapping_with_index = {"Q": 1, "K": 1, "QK": 0}
        mappings = spec.map_workload_to_arch(einsum_names=["Q", "K", "QK"])

    def test_bert_heterogeneous(self):
        spec = Spec.from_yaml(
            af.examples.arches.simple_heterogeneous,
            af.examples.workloads.bert,
        )
        spec.mapper.metrics = Metrics.ENERGY | Metrics.LATENCY
        spec.mapper.n_concurrent_threads = 2
        mappings = spec.map_workload_to_arch(einsum_names=["Q", "K", "QK"])

    def test_tpuv4i_gpt3(self):
        spec = Spec.from_yaml(
            af.examples.arches.tpu_v4i,
            af.examples.workloads.gpt3_6_7B,
        )
        spec.mapper.metrics = Metrics.ENERGY | Metrics.LATENCY
        spec.mapper.n_concurrent_threads = 2
        import pickle
        # mappings = spec.map_workload_to_arch()

        # pmappings = af.mapper.FFM.make_pmappings(spec, einsum_names=["I", "Q", "K", "QK"])
        # with open("tmp.pkl", "wb") as f:
        #     pickle.dump(pmappings, f)

        with open("tmp.pkl", "rb") as f:
            pmappings = pickle.load(f)
        af.mapper.FFM.join_pmappings(pmappings, spec.mapper.metrics, require_all_einsums=False)

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


class TestMapperComprehensiveness(unittest.TestCase):
    def test_gpt3(self):
        for glb_size in [2, 4, 8]:
            spec = Spec.from_yaml(
                af.examples.arches.snowcat,
                af.examples.workloads.gpt3_6_7B,
            )
            spec.arch.find("GlobalBuffer").size = glb_size * 1024 * 1024 * 8
            spec.mapper.metrics = Metrics.ENERGY
            spec.arch["MainMemory"].tensors.keep = "~Intermediates"
            spec.arch["MainMemory"].tensors.may_keep = "All"
            spec.arch["GlobalBuffer"].tensors.keep = "~MainMemory"
            spec.arch["GlobalBuffer"].tensors.may_keep = "All"
            relaxed_num_accesses = result.energy()

            spec.arch["MainMemory"].tensors.keep = "All"
            result = spec.map_workload_to_arch()
            unfused_num_accesses = result.energy()

            self.assertGreaterEqual(
                unfused_num_accesses,
                relaxed_num_accesses,
                "more relaxed constraint led to worse ski-slope."
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
