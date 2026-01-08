import unittest
from pathlib import Path

from fastfusion.frontend.spec import Spec
from fastfusion.mapper.FFM.main import map_workload_to_arch


EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


class TestModel(unittest.TestCase):
    def test_one_matmul(self):
        spec = Spec.from_yaml(
            EXAMPLES_DIR / "arches" / "simple.arch.yaml",
            EXAMPLES_DIR / "workloads" / "matmuls.workload.yaml",
            jinja_parse_data={"N_EINSUMS": 1, "M": 64, "KN": 64},
        )

        result = map_workload_to_arch(spec)

    def test_two_matmuls(self):
        spec = Spec.from_yaml(
            EXAMPLES_DIR / "arches" / "simple.arch.yaml",
            EXAMPLES_DIR / "workloads" / "matmuls.workload.yaml",
            jinja_parse_data={"N_EINSUMS": 2, "M": 64, "KN": 64},
        )

        result = map_workload_to_arch(spec)
