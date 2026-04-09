import os
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
            result = spec.map_workload_to_arch()
            relaxed_n_accesses = result.energy()

            spec.arch["MainMemory"].tensors.keep = "All"
            result = spec.map_workload_to_arch()
            unfused_n_accesses = result.energy()

            self.assertGreaterEqual(
                unfused_n_accesses,
                relaxed_n_accesses,
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


class TestMinUsageIncompleteFormula(unittest.TestCase):
    """Regression test for a bug where try_best_if_none_reaches_min pruned
    tile shapes based on incomplete formulas, causing the mapper to miss
    optimal mappings. The bug manifested when a spatial dimension with
    min_usage could not be fully utilized for certain tile shapes (e.g.,
    only 49/64 rows), but those tile shapes gave the best energy. The
    premature pruning removed them because other tile shapes appeared to
    reach min_usage when evaluated with maximized unknown symbols."""

    def test_constrained_superset_of_unconstrained(self):
        """Adding an explicit loop_bounds constraint should restrict the
        search space. The unconstrained mapper must find a mapping at least
        as good as the constrained one."""
        import sys
        cim_dir = os.path.join(
            os.path.dirname(__file__), "..", "..",
            "accelforge-examples", "arches", "compute_in_memory"
        )
        cim_dir = os.path.abspath(cim_dir)
        if not os.path.exists(cim_dir):
            self.skipTest("accelforge-examples not found")
        sys.path.insert(0, cim_dir)
        from _load_spec import get_spec
        from accelforge.frontend.arch import Comparison

        spec_constrained = get_spec("wang_vlsi_2022", add_dummy_main_memory=True)
        sub_bank = spec_constrained.arch["SubBank"]
        for dim in sub_bank.spatial:
            if dim.name == "rows_ARRAY_ROWS":
                dim.loop_bounds = [
                    Comparison(expression="r0", operator="==", value=7),
                    Comparison(expression="s0", operator="==", value=7),
                    Comparison(expression="c0", operator="==", value=1),
                ]
        spec_constrained.mapper.metrics = Metrics.ENERGY
        result_constrained = map_workload_to_arch(spec_constrained)
        energy_constrained = result_constrained.energy()

        spec_unconstrained = get_spec("wang_vlsi_2022", add_dummy_main_memory=True)
        spec_unconstrained.mapper.metrics = Metrics.ENERGY
        result_unconstrained = map_workload_to_arch(spec_unconstrained)
        energy_unconstrained = result_unconstrained.energy()

        self.assertLessEqual(
            energy_unconstrained,
            energy_constrained * 1.01,  # 1% tolerance for Pareto approximation
            f"Unconstrained mapper ({energy_unconstrained:.6e}) found worse "
            f"mapping than constrained ({energy_constrained:.6e}). "
            f"The unconstrained search space is a superset.",
        )


class TestUntrackedSpatialNotPrunedByPareto(unittest.TestCase):
    """Regression test: when a spatial dimension's tile shape is not tracked by
    any Pareto objective in the alt_objectives pass, it was incorrectly
    collapsed during deduplication, losing optimal mappings that differed only
    in that spatial dimension.

    Specifically, setting min_usage=0 on a spatial dimension should yield
    same-or-better mappings than min_usage=1, since the search space is a
    superset. The bug caused the min_usage=0 case to lose the fully-utilized
    mapping because the Pareto pruning treated rows differing only in the
    untracked spatial stride as duplicates."""

    def test_relaxed_min_usage_finds_optimal(self):
        spec = Spec.from_yaml(
            os.path.join(EXAMPLES_DIR, "arches", "tpu_v4i.yaml"),
            os.path.join(EXAMPLES_DIR, "workloads", "gpt3_6.7B.yaml"),
            jinja_parse_data={"BATCH_SIZE": 1, "N_TOKENS": 1},
        )
        spec.mapper.metrics = Metrics.ENERGY | Metrics.LATENCY
        spec.mapper.max_pmapping_templates_per_einsum = 1

        # min_usage=1: forced full utilization
        spec.arch.find_spatial("Z").min_usage = 1
        spec.arch.find_spatial("reuse_input").min_usage = 1
        spec.arch.find_spatial("reuse_output").min_usage = 1
        spec.mapper._metric_aggregator = "prod"
        result_constrained = map_workload_to_arch(spec, einsum_names=["QK"])
        latency_constrained = min(
            result_constrained.latency(list_if_one_mapping=True)
        )

        # min_usage=0: superset of search space
        spec.arch.find_spatial("Z").min_usage = 0
        result_relaxed = map_workload_to_arch(spec, einsum_names=["QK"])
        latency_relaxed = min(result_relaxed.latency(list_if_one_mapping=True))

        self.assertLessEqual(
            latency_relaxed,
            latency_constrained * 1.01,
            f"Relaxed min_usage=0 latency ({latency_relaxed:.6e}) is worse than "
            f"min_usage=1 ({latency_constrained:.6e}). The relaxed search space "
            f"is a superset and should find same-or-better mappings.",
        )
