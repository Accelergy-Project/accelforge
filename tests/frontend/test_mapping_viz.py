"""
Uncommenting the lines with `graph.write_png` will generate PNG pictures of
LoopTrees so they can be inspected.

TODO: an automated way to check for correctness.
"""
import unittest
from pathlib import Path

from fastfusion.frontend.spec import Spec
from fastfusion.model.main import evaluate_mapping
from fastfusion.util.parallel import set_n_parallel_jobs

set_n_parallel_jobs(1)


EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"


class TestModel(unittest.TestCase):
    def test_default(self):
        spec = Spec.from_yaml(
            EXAMPLES_DIR / "arches" / "simple.arch.yaml",
            EXAMPLES_DIR / "workloads" / "matmuls.workload.yaml",
            EXAMPLES_DIR / "mappings" / "unfused_matmuls_to_simple.mapping.yaml",
            jinja_parse_data={"N_EINSUMS": 2, "M": 64, "KN": 64},
        )
        result = evaluate_mapping(spec)
        graph = result.data["Total<SEP>mapping"].iloc[0]().render_pydot()
        # graph.write_png("default.png")

    def test_without_reservations(self):
        spec = Spec.from_yaml(
            EXAMPLES_DIR / "arches" / "simple.arch.yaml",
            EXAMPLES_DIR / "workloads" / "matmuls.workload.yaml",
            EXAMPLES_DIR / "mappings" / "unfused_matmuls_to_simple.mapping.yaml",
            jinja_parse_data={"N_EINSUMS": 2, "M": 64, "KN": 64},
        )
        result = evaluate_mapping(spec)
        graph = result.data["Total<SEP>mapping"].iloc[0]().render_pydot(with_reservations=False)
        # graph.write_png("without_reservations.png")

    def test_without_stride(self):
        spec = Spec.from_yaml(
            EXAMPLES_DIR / "arches" / "simple.arch.yaml",
            EXAMPLES_DIR / "workloads" / "matmuls.workload.yaml",
            EXAMPLES_DIR / "mappings" / "unfused_matmuls_to_simple.mapping.yaml",
            jinja_parse_data={"N_EINSUMS": 2, "M": 64, "KN": 64},
        )
        result = evaluate_mapping(spec)
        graph = result.data["Total<SEP>mapping"].iloc[0]().render_pydot(with_tile_shape=False)
        # graph.write_png("without_stride.png")