import unittest

from accelforge.frontend.spec import Spec
from accelforge.model.main import evaluate_mapping
from accelforge.util.parallel import set_n_parallel_jobs
from accelforge.plotting.mappings import (
    plot_energy_comparison,
    plot_energy_breakdown,
    plot_action_breakdown,
)

set_n_parallel_jobs(1)

try:
    from .paths import EXAMPLES_DIR
except ImportError:
    from paths import EXAMPLES_DIR


class TestEnergyPlotting(unittest.TestCase):
    def test_comparison(self):
        spec = Spec.from_yaml(
            EXAMPLES_DIR / "arches" / "simple.yaml",
            EXAMPLES_DIR / "workloads" / "matmuls.yaml",
            EXAMPLES_DIR / "mappings" / "unfused_matmuls_to_simple.yaml",
            jinja_parse_data={"N_EINSUMS": 2, "M": 64, "KN": 64},
        )

        result = evaluate_mapping(spec)

        fig, ax = plot_energy_comparison([result])
        fig.tight_layout()
        fig.savefig("total.png", dpi=400, bbox_inches="tight")

    def test_breakdown(self):
        spec = Spec.from_yaml(
            EXAMPLES_DIR / "arches" / "simple.yaml",
            EXAMPLES_DIR / "workloads" / "matmuls.yaml",
            EXAMPLES_DIR / "mappings" / "unfused_matmuls_to_simple.yaml",
            jinja_parse_data={"N_EINSUMS": 2, "M": 64, "KN": 64},
        )

        result = evaluate_mapping(spec)

        fig, axes = plot_energy_breakdown([result, result], ["einsum", "component"])
        fig.tight_layout()
        fig.savefig("/tmp/fig.png", dpi=400, bbox_inches="tight")

    def test_breakdown_stacked(self):
        spec = Spec.from_yaml(
            EXAMPLES_DIR / "arches" / "simple.yaml",
            EXAMPLES_DIR / "workloads" / "matmuls.yaml",
            EXAMPLES_DIR / "mappings" / "unfused_matmuls_to_simple.yaml",
            jinja_parse_data={"N_EINSUMS": 2, "M": 64, "KN": 64},
        )

        result = evaluate_mapping(spec)

        fig, axes = plot_energy_breakdown(
            [result, result], ["einsum", "component"], ["action"]
        )
        fig.tight_layout()
        fig.savefig("/tmp/fig.png", dpi=400, bbox_inches="tight")


class TestActionPlotting(unittest.TestCase):
    def test_breakdown(self):
        spec = Spec.from_yaml(
            EXAMPLES_DIR / "arches" / "simple.yaml",
            EXAMPLES_DIR / "workloads" / "matmuls.yaml",
            EXAMPLES_DIR / "mappings" / "unfused_matmuls_to_simple.yaml",
            jinja_parse_data={"N_EINSUMS": 2, "M": 64, "KN": 64},
        )

        result = evaluate_mapping(spec)

        fig, axes = plot_action_breakdown([result, result], ["einsum", "component"])
        fig.tight_layout()
        fig.savefig("/tmp/fig.png", dpi=400, bbox_inches="tight")
