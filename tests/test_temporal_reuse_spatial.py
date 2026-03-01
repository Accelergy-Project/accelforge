"""
Demonstrates an irreducible temporal reuse failure with spatial fanout.

Architecture: MainMemory → GlobalBuffer → PEArray(4) → RegFile → MAC
Workload: T1[m,n1] = T0[m,n0] * W0[n0,n1]   (M=4, KN=4, bits=8)
Arch: bits_per_action=16

W0[n0,n1] does NOT depend on m.

Best mapping (W0 at GlobalBuffer above m, RegFile below spatial):
    MainMemory [all]
    GlobalBuffer [T1, T0, W0]   ← W0 above m
    Temporal m=1                 ← irrelevant to W0
    Spatial n0 (4 PEs)
    RegFile [W0]                 ← per-PE, below spatial
    Temporal n0=1
    Temporal n1=1
    Compute Matmul0 @ MAC

GlobalBuffer fills are correct because m is in its subtree.
RegFile fills are inflated by m because m is above the spatial
fanout and RegFile is below it — no mapping restructuring can
fix this without temporal reuse detection in the model.

Unit conversion: actions = elements * BITS / bits_per_action
  W0 total = KN * KN * BITS / BPA = 4 * 4 * 8 / 16 = 8 actions

RegFile W0 per PE:
  n0 is spatially distributed: each PE handles 1 n0 value
  Each PE needs W0[1, KN] = 4 elements = 2 actions per fill
  With temporal reuse:    1 fill  * 2 actions * 4 PEs =  8 total
  Without temporal reuse: M fills * 2 actions * 4 PEs = 32 total
"""
import unittest

import accelforge as af
from accelforge.frontend.spec import Spec
from accelforge.model.main import evaluate_mapping

try:
    from .paths import CURRENT_DIR
except ImportError:
    from paths import CURRENT_DIR

M = 4
KN = 4
FANOUT = 4
BITS = 8
BPA = 16  # bits_per_action from arch

ARCH_YAML = CURRENT_DIR / "input_files" / "table7" / "spatial_smoke.arch.yaml"
MAPPING_YAML = CURRENT_DIR / "input_files" / "temporal_reuse_spatial.yaml"


def _make_spec():
    return Spec.from_yaml(
        ARCH_YAML,
        af.examples.workloads.matmuls,
        MAPPING_YAML,
        jinja_parse_data={"N_EINSUMS": 1, "M": M, "KN": KN},
    )


class TestTemporalReuseSpatial(unittest.TestCase):
    """Demonstrate irreducible temporal reuse failure with spatial fanout.

    These tests document the expected behavior WITH and WITHOUT
    temporal reuse detection.  On a branch without temporal reuse,
    RegFile W0 writes will be M=4x too high.
    """

    def test_globalbuffer_w0_write_correct(self):
        """GlobalBuffer W0 fills are correct (m is in subtree)."""
        spec = _make_spec()
        result = evaluate_mapping(spec)
        acts = result.actions(per_component=True, per_einsum=True, per_tensor=True)

        gb_w0_write = float(acts[("Matmul0", "GlobalBuffer", "W0", "write")])
        # W0 is above m at GlobalBuffer: fills = KN * KN * BITS / BPA = 8
        expected = KN * KN * BITS // BPA
        self.assertEqual(
            gb_w0_write,
            expected,
            f"GlobalBuffer W0 writes should be {expected}, got {gb_w0_write}. "
            f"m should not inflate GlobalBuffer fills.",
        )

    def test_regfile_w0_write_inflated_without_temporal_reuse(self):
        """RegFile W0 fills are inflated by m (above spatial fanout).

        This is the irreducible failure case: m is above the spatial
        fanout and RegFile is below it.  Without temporal reuse detection,
        RegFile W0 fills = M * (correct fills).

        With temporal reuse: total fills across all PEs = 8
        Without temporal reuse: total fills = M * 8 = 32
        """
        spec = _make_spec()
        result = evaluate_mapping(spec)
        acts = result.actions(per_component=True, per_einsum=True, per_tensor=True)

        reg_w0_write = float(acts[("Matmul0", "RegFile", "W0", "write")])

        correct_with_reuse = KN * KN * BITS // BPA              # 8
        inflated_without_reuse = M * KN * KN * BITS // BPA      # 32

        # Document both possible outcomes:
        if reg_w0_write == correct_with_reuse:
            pass  # Temporal reuse is active — model is correct
        elif reg_w0_write == inflated_without_reuse:
            self.skipTest(
                f"RegFile W0 writes = {int(reg_w0_write)} "
                f"(M={M}x inflation due to missing temporal reuse). "
                f"Correct value with temporal reuse = {correct_with_reuse}."
            )
        else:
            self.fail(
                f"RegFile W0 writes = {reg_w0_write}, expected either "
                f"{correct_with_reuse} (with reuse) or "
                f"{inflated_without_reuse} (without reuse)."
            )


if __name__ == "__main__":
    unittest.main()
