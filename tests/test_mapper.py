import unittest
from pathlib import Path

from fastfusion.frontend.spec import Spec
from fastfusion.mapper.FFM.main import map_workload_to_arch


EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


class TestMapper(unittest.TestCase):
    def test_one_matmul(self):
        spec = Spec.from_yaml(
            EXAMPLES_DIR / "arches" / "simple.arch.yaml",
            EXAMPLES_DIR / "workloads" / "matmuls.workload.yaml",
            jinja_parse_data={"N_EINSUMS": 1, "M": 64, "KN": 64},
        )

        result = map_workload_to_arch(spec)

        self._check_memory_actions_exist(spec, ["MainMemory", "GlobalBuffer"], result)

    def test_two_matmuls(self):
        spec = Spec.from_yaml(
            EXAMPLES_DIR / "arches" / "simple.arch.yaml",
            EXAMPLES_DIR / "workloads" / "matmuls.workload.yaml",
            jinja_parse_data={"N_EINSUMS": 2, "M": 64, "KN": 64},
        )

        result = map_workload_to_arch(spec)

        self._check_memory_actions_exist(spec, ["MainMemory", "GlobalBuffer"], result)

    def _check_memory_actions_exist(self, spec, memory_names, result):
        for einsum_name in spec.workload.einsum_names:
            for memory_name in memory_names:
                for memory_action in ["read", "write"]:
                    self.assertTrue(
                        f"{einsum_name}<SEP>action<SEP>{memory_name}<SEP>{memory_action}" in result.data.columns,
                        f"{einsum_name}<SEP>action<SEP>{memory_name}<SEP>{memory_action} "
                        f"not found in {result.data.columns}"
                    )