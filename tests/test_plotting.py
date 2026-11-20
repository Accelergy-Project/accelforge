import os
import unittest

from accelforge.frontend.spec import Spec
from accelforge.model.main import evaluate_mapping
from accelforge.util.parallel import set_n_parallel_jobs
from accelforge.plotting.roofline import plot_roofline, plot_roofline_3d
from accelforge.plotting.mappings import (
    plot_energy_comparison,
    plot_energy_breakdown,
    plot_action_breakdown,
)

set_n_parallel_jobs(1)

try:
    from .paths import EXAMPLES_DIR, CURRENT_DIR
except ImportError:
    from paths import EXAMPLES_DIR, CURRENT_DIR

SAVE_FIGS = os.environ.get("SAVE_FIGS", "").lower() in ("1", "true", "yes")


class TestEnergyPlotting(unittest.TestCase):
    def test_comparison(self):
        spec = Spec.from_yaml(
            EXAMPLES_DIR / "arches" / "simple.yaml",
            EXAMPLES_DIR / "workloads" / "basic" / "matmuls.yaml",
            EXAMPLES_DIR / "mappings" / "unfused_matmuls_to_simple.yaml",
            jinja_parse_data={"N_EINSUMS": 2, "M": 64, "KN": 64},
        )

        result = evaluate_mapping(spec)

        fig, ax = plot_energy_comparison([result])
        fig.tight_layout()
        if SAVE_FIGS:
            fig.savefig("total.png", dpi=400, bbox_inches="tight")

    def test_breakdown(self):
        spec = Spec.from_yaml(
            EXAMPLES_DIR / "arches" / "simple.yaml",
            EXAMPLES_DIR / "workloads" / "basic" / "matmuls.yaml",
            EXAMPLES_DIR / "mappings" / "unfused_matmuls_to_simple.yaml",
            jinja_parse_data={"N_EINSUMS": 2, "M": 64, "KN": 64},
        )

        result = evaluate_mapping(spec)

        fig, axes = plot_energy_breakdown([result, result], ["einsum", "component"])
        fig.tight_layout()
        if SAVE_FIGS:
            fig.savefig("/tmp/fig.png", dpi=400, bbox_inches="tight")

    def test_breakdown_stacked(self):
        spec = Spec.from_yaml(
            EXAMPLES_DIR / "arches" / "simple.yaml",
            EXAMPLES_DIR / "workloads" / "basic" / "matmuls.yaml",
            EXAMPLES_DIR / "mappings" / "unfused_matmuls_to_simple.yaml",
            jinja_parse_data={"N_EINSUMS": 2, "M": 64, "KN": 64},
        )

        result = evaluate_mapping(spec)

        fig, axes = plot_energy_breakdown(
            [result, result], ["einsum", "component"], ["action"]
        )
        fig.tight_layout()
        if SAVE_FIGS:
            fig.savefig("/tmp/fig.png", dpi=400, bbox_inches="tight")


class TestActionPlotting(unittest.TestCase):
    def test_breakdown(self):
        spec = Spec.from_yaml(
            EXAMPLES_DIR / "arches" / "simple.yaml",
            EXAMPLES_DIR / "workloads" / "basic" / "matmuls.yaml",
            EXAMPLES_DIR / "mappings" / "unfused_matmuls_to_simple.yaml",
            jinja_parse_data={"N_EINSUMS": 2, "M": 64, "KN": 64},
        )

        result = evaluate_mapping(spec)

        fig, axes = plot_action_breakdown([result, result], ["einsum", "component"])
        fig.tight_layout()
        if SAVE_FIGS:
            fig.savefig("/tmp/fig.png", dpi=400, bbox_inches="tight")


class TestRoofline(unittest.TestCase):
    def _spec(self, workload=True):
        files = [EXAMPLES_DIR / "arches" / "tpu_v4i.yaml"]
        if workload:
            files += [
                CURRENT_DIR / "input_files" / "decompress_matmul.workload.yaml",
                EXAMPLES_DIR / "workloads" / "_renames_default.yaml",
            ]
        return Spec.from_yaml(*files)

    def test_2d(self):
        sets = [["Matmul"], ["Decompress", "Matmul"], "Decompress"]
        fig, ax = plot_roofline(self._spec(), sets)
        if SAVE_FIGS:
            fig.savefig("/tmp/roofline.png", dpi=200, bbox_inches="tight")
        # Matmul has 16 x 4096 x 4096 computes and Decompress 4096 x 4096. Fused, the
        # 16b W stays on chip and only the 4b WC, 16b I, and 16b O move.
        n_computes = 16 * 4096 * 4096 + 4096 * 4096
        fused_bytes = 4096 * 4096 * 4 / 8 + 2 * 16 * 4096 * 2
        x, _ = ax.collections[1].get_offsets()[0]
        self.assertAlmostEqual(x, n_computes / fused_bytes)

    def test_3d(self):
        sets = [["Matmul"], ["Decompress", "Matmul"], "Decompress"]
        fig, ax = plot_roofline_3d(self._spec(), sets)
        if SAVE_FIGS:
            fig.savefig("/tmp/roofline_3d.png", dpi=200, bbox_inches="tight")

    def test_errors(self):
        with self.assertRaisesRegex(ValueError, "no workload"):
            plot_roofline(self._spec(workload=False), [["Matmul"]])
        with self.assertRaisesRegex(ValueError, "not in the workload"):
            plot_roofline(self._spec(), [["Matmul", "Missing"]])
        with self.assertRaisesRegex(ValueError, "not in the workload"):
            plot_roofline_3d(self._spec(), [["Missing"]])
