"""
Minimal test for structural temporal reuse via mapping.

Verifies that placing a buffer's Storage node ABOVE an irrelevant
temporal loop prevents that loop from inflating parent fills.

Architecture: simple (MainMemory -> GlobalBuffer -> MAC)
Workload: Single matmul T1[m,n1] = T0[m,n0] * W0[n0,n1]  (M=4, KN=4)
          bits_per_value = 8

Mapping (W0 and T1 both at GlobalBuffer, above m):
    Storage [W0, T0, T1] @ MainMemory
    Storage [T1, W0] @ GlobalBuffer     <- both tensors above m
    Temporal m=1                         <- m is IRRELEVANT to W0[n0,n1]
    Temporal n0=1
    Temporal n1=1
    Compute Matmul0 @ MAC

Because W0 is stored at GlobalBuffer ABOVE the m loop, the m loop
is processed as part of GlobalBuffer's child subtree.  The model
computes W0 fills from tile occupancy (not multiplied by m).

Action counts are in bits (elements * bits_per_value).
W0 shape = [n0, n1] = [4, 4] = 16 elements = 128 bits.

Expected:
    GlobalBuffer W0 write (fill from MainMemory) = 1 * 128 = 128
    GlobalBuffer W0 read  (consumed by compute)  = M*n0*n1 * 8 = 512
    MainMemory   W0 read  (to fill GlobalBuffer)  = 1 * 128 = 128
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
BITS = 8  # bits_per_value from workload

MAPPING_YAML = CURRENT_DIR / "input_files" / "temporal_reuse_minimal.yaml"


def _make_spec():
    spec = Spec.from_yaml(
        af.examples.arches.simple,
        af.examples.workloads.matmuls,
        MAPPING_YAML,
        jinja_parse_data={"N_EINSUMS": 1, "M": M, "KN": KN},
    )
    return spec


class TestTemporalReuseMinimal(unittest.TestCase):
    """Verify structural temporal reuse: W0 parent fill happens once, not M times."""

    def test_globalbuffer_w0_write(self):
        spec = _make_spec()
        result = evaluate_mapping(spec)
        acts = result.actions(per_component=True, per_einsum=True, per_tensor=True)

        gb_w0_write = float(acts[("Matmul0", "GlobalBuffer", "W0", "write")])
        # W0 is above m, so fills = 1 * KN*KN * BITS = 128
        expected = 1 * KN * KN * BITS
        self.assertEqual(
            gb_w0_write,
            expected,
            f"GlobalBuffer W0 writes should be {expected} (one fill of "
            f"{KN*KN} elements * {BITS} bits), got {gb_w0_write}.",
        )

    def test_globalbuffer_w0_read_unchanged(self):
        spec = _make_spec()
        result = evaluate_mapping(spec)
        acts = result.actions(per_component=True, per_einsum=True, per_tensor=True)

        gb_w0_read = float(acts[("Matmul0", "GlobalBuffer", "W0", "read")])
        # Reads are NOT affected — every compute reads W0
        expected = M * KN * KN * BITS
        self.assertEqual(
            gb_w0_read,
            expected,
            f"GlobalBuffer W0 reads should be {expected}, got {gb_w0_read}",
        )

    def test_mainmemory_w0_read(self):
        spec = _make_spec()
        result = evaluate_mapping(spec)
        acts = result.actions(per_component=True, per_einsum=True, per_tensor=True)

        mm_w0_read = float(acts[("Matmul0", "MainMemory", "W0", "read")])
        # MainMemory reads to fill GlobalBuffer: should be ONE fill
        expected = 1 * KN * KN * BITS
        self.assertEqual(
            mm_w0_read,
            expected,
            f"MainMemory W0 reads should be {expected} (one fill), "
            f"got {mm_w0_read}.",
        )


if __name__ == "__main__":
    unittest.main()
