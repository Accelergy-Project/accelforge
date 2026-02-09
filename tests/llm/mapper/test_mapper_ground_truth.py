import json
import math
import unittest
from collections import defaultdict
from pathlib import Path

from accelforge.frontend.spec import Spec
from accelforge.mapper import Metrics
from accelforge.mapper.FFM.main import map_workload_to_arch
from accelforge.util.parallel import set_n_parallel_jobs

set_n_parallel_jobs(1)


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
GROUND_TRUTH_PATH = THIS_DIR / "ground_truth.json"


class TestMapperGroundTruth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ground_truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))

    def _assert_close(self, actual: float, expected: float):
        self.assertTrue(
            math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-12),
            msg=f"actual={actual}, expected={expected}",
        )

    def _assert_float_dict_close(self, actual: dict[str, float], expected: dict[str, float]):
        self.assertEqual(set(actual.keys()), set(expected.keys()))
        for key, expected_value in expected.items():
            self._assert_close(actual[key], expected_value)

    def test_mapper_metrics_against_ground_truth(self):
        for case in self.ground_truth["mapper_cases"]:
            with self.subTest(case=case["id"]):
                spec = Spec.from_yaml(
                    REPO_ROOT / case["arch"],
                    REPO_ROOT / case["workload"],
                    jinja_parse_data=case["jinja"],
                )
                spec.mapper.metrics = Metrics.ENERGY | Metrics.LATENCY | Metrics.ACTIONS

                mappings = map_workload_to_arch(
                    spec,
                    print_progress=False,
                    print_number_of_pmappings=False,
                )
                self.assertGreater(len(mappings.data), 0)
                mapping = mappings[0]
                row = mapping.data.iloc[0]
                expected = case["expected"]

                self._assert_close(mapping.energy(), expected["total_energy"])
                self._assert_close(mapping.latency(), expected["total_latency"])

                self._assert_float_dict_close(
                    {str(k): float(v) for k, v in mapping.energy(per_einsum=True).items()},
                    expected["energy_per_einsum"],
                )
                self._assert_float_dict_close(
                    {str(k): float(v) for k, v in mapping.latency(per_einsum=True).items()},
                    expected["latency_per_einsum"],
                )
                self._assert_float_dict_close(
                    {
                        f"{einsum}|{component}": float(value)
                        for (einsum, component), value in mapping.energy(
                            per_einsum=True, per_component=True
                        ).items()
                    },
                    expected["energy_per_einsum_component"],
                )
                self._assert_float_dict_close(
                    {
                        f"{einsum}|{component}": float(value)
                        for (einsum, component), value in mapping.latency(
                            per_einsum=True, per_component=True
                        ).items()
                    },
                    expected["latency_per_einsum_component"],
                )

                action_columns = sorted(
                    c for c in mapping.data.columns if "<SEP>action<SEP>" in c
                )
                action_counts = {column: float(row[column]) for column in action_columns}
                self._assert_float_dict_close(action_counts, expected["action_counts"])

                per_component = defaultdict(float)
                per_einsum_component_action = defaultdict(float)
                for column, value in action_counts.items():
                    einsum, _, component, _, action = column.split("<SEP>")
                    per_component[component] += value
                    per_einsum_component_action[f"{einsum}|{component}|{action}"] += value

                self._assert_float_dict_close(
                    dict(per_component), expected["action_counts_per_component"]
                )
                self._assert_float_dict_close(
                    dict(per_einsum_component_action),
                    expected["action_counts_per_einsum_component_action"],
                )

    def test_area_against_ground_truth(self):
        for case in self.ground_truth["area_cases"]:
            with self.subTest(case=case["id"]):
                spec = Spec.from_yaml(
                    REPO_ROOT / case["arch"],
                    REPO_ROOT / case["workload"],
                    jinja_parse_data=case["jinja"],
                )
                spec = spec.calculate_component_area_energy_latency_leak()
                expected = case["expected"]

                self._assert_close(float(spec.arch.total_area), expected["total_area"])
                self._assert_float_dict_close(
                    {
                        str(component): float(area)
                        for component, area in spec.arch.per_component_total_area.items()
                    },
                    expected["per_component_total_area"],
                )


if __name__ == "__main__":
    unittest.main()
