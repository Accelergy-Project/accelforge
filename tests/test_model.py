import unittest
from pathlib import Path

from fastfusion.frontend.spec import Spec
from fastfusion.model.main import evaluate_mapping


EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


class TestModel(unittest.TestCase):
    def test_one_matmul(self):
        spec = Spec.from_yaml(
            EXAMPLES_DIR / "arches" / "simple.arch.yaml",
            EXAMPLES_DIR / "workloads" / "matmuls.workload.yaml",
            EXAMPLES_DIR / "mappings" / "unfused_matmuls_to_simple.mapping.yaml",
            jinja_parse_data={"N_EINSUMS": 1, "M": 64, "KN": 64},
        )

        result = evaluate_mapping(spec)

    def test_two_matmuls(self):
        spec = Spec.from_yaml(
            EXAMPLES_DIR / "arches" / "simple.arch.yaml",
            EXAMPLES_DIR / "workloads" / "matmuls.workload.yaml",
            EXAMPLES_DIR / "mappings" / "unfused_matmuls_to_simple.mapping.yaml",
            jinja_parse_data={"N_EINSUMS": 2, "M": 64, "KN": 64},
        )

        result = evaluate_mapping(spec)