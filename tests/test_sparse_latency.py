"""Sparse-adjusted latency model tests (Phase 9/11: Latency).

Tests validate that sparse optimizations correctly affect latency:
- Gating: gated reads still consume port bandwidth (cycles consumed, energy saved)
- Skipping: skipped reads do NOT consume bandwidth (both cycles AND energy saved)
- Metadata reads/writes also consume bandwidth

Latency model uses per-tensor MAX at Reg (dedicated register ports) and
aggregated SUM at Buffer/BackingStorage (shared bandwidth).

Phase 11 fixes:
- Removed skipped_first subtraction (fills count towards bandwidth)
- Metadata bandwidth in data-word equivalents (not packed SRAM accesses)
- metadata_word_bits=14 for CSR (was 7)

With arch_latency.yaml bandwidth model (d=0.1015625, fig1):

| Level          | Dense     | Bitmask (gating) | Coord list (skipping) |
|----------------|-----------|-------------------|-----------------------|
| MAC            | 2,097,152 | 21,632            | 21,632                |
| Reg (per-t max)| 2,113,536 | 2,113,536         | 38,016                |
| Buffer         | 2,097,152 | 220,599           | 295,152               |
| Total          | 2,113,536 | 2,113,536         | 295,152               |

Dense/Bitmask Reg = 2,113,536 (includes Z fills: 2,097,152 + 16,384).
Coord list total = 295,152 (Buffer bottleneck with metadata bandwidth).

Uses arch_latency.yaml with bandwidth-based total_latency expressions.
"""

import os
import unittest

from accelforge.frontend.spec import Spec
from accelforge.model.main import evaluate_mapping

FIG1_DIR = os.path.join(os.path.dirname(__file__), "input_files", "fig1")


def _load_latency(*extra_yamls):
    """Load fig1 with arch_latency.yaml and optional sparse config."""
    files = [
        os.path.join(FIG1_DIR, "arch_latency.yaml"),
        os.path.join(FIG1_DIR, "workload.yaml"),
        os.path.join(FIG1_DIR, "mapping.yaml"),
    ]
    for f in extra_yamls:
        files.append(os.path.join(FIG1_DIR, f))
    spec = Spec.from_yaml(*files)
    return evaluate_mapping(spec)


def _get_total_latency(result):
    """Get total latency from the raw DataFrame."""
    return float(result.data["Total<SEP>latency"].iloc[0])


def _get_component_latency(result, component):
    """Get per-component latency from the raw DataFrame."""
    suffix = f"latency<SEP>{component}"
    for col in result.data.columns:
        if col.endswith(suffix):
            return float(result.data[col].iloc[0])
    return 0.0


# ===========================================================================
# Dense baseline latency
# ===========================================================================


class TestDenseLatency(unittest.TestCase):
    """Dense latency (no sparse) — regression test with bandwidth model."""

    @classmethod
    def setUpClass(cls):
        cls.result = _load_latency()

    def test_mac_cycles(self):
        """Dense MAC = 2,097,152 cycles (one compute per cycle)."""
        mac = _get_component_latency(self.result, "MAC")
        self.assertEqual(int(mac), 2_097_152)

    def test_reg_cycles(self):
        """Dense Reg = 2,113,536 (per-tensor max: Z write = 2,097,152 + 16,384 fills)."""
        reg = _get_component_latency(self.result, "Reg")
        self.assertEqual(int(reg), 2_113_536)

    def test_total_latency_is_reg(self):
        """Dense total = Reg (Reg is bottleneck)."""
        total = _get_total_latency(self.result)
        reg = _get_component_latency(self.result, "Reg")
        self.assertEqual(int(total), int(reg))


# ===========================================================================
# Bitmask (gating) latency
# ===========================================================================


class TestBitmaskLatency(unittest.TestCase):
    """Bitmask (gating) latency: Reg bottleneck unchanged from dense.

    With gating, gated reads still consume port bandwidth. Reg sees the
    full dense read/write traffic because gated operations still read
    all operands. Only MAC is reduced via compute_latency_ratio.
    """

    @classmethod
    def setUpClass(cls):
        cls.result = _load_latency("sparse_bitmask_latency.yaml")

    def test_mac_cycles(self):
        """Bitmask MAC ~ 21,632 cycles (post-4b scaled compute).

        Dense 2,097,152 * compute_latency_ratio ≈ 21,632.
        """
        mac = _get_component_latency(self.result, "MAC")
        self.assertAlmostEqual(mac, 21_632, delta=2)

    def test_reg_equals_dense(self):
        """Bitmask Reg = Dense Reg (gated reads consume full BW).

        Gating → gated deltas added back → latency = dense.
        Per-tensor max: Z write = 2,097,152 + 16,384 fills = 2,113,536.
        """
        reg = _get_component_latency(self.result, "Reg")
        self.assertEqual(int(reg), 2_113_536)

    def test_total_is_reg(self):
        """Bitmask total = Reg (Reg is bottleneck, same as dense)."""
        total = _get_total_latency(self.result)
        self.assertEqual(int(total), 2_113_536)


# ===========================================================================
# Coord list (skipping) latency
# ===========================================================================


class TestCoordListLatency(unittest.TestCase):
    """Coord list (skipping) latency: Buffer bottleneck.

    With skipping, skipped reads do NOT consume bandwidth. Reg traffic
    drops dramatically because skipped MAC operations don't read A/B/Z
    from Reg.
    """

    @classmethod
    def setUpClass(cls):
        cls.result = _load_latency("sparse_coord_list_latency.yaml")

    def test_mac_cycles(self):
        """Coord list MAC ~ 21,632 cycles (same post-4b compute count).

        Same SAF probabilities → same compute_latency_ratio → same MAC.
        """
        mac = _get_component_latency(self.result, "MAC")
        self.assertAlmostEqual(mac, 21_632, delta=2)

    def test_reg_much_less_than_dense(self):
        """Coord list Reg << Dense Reg (skipped reads don't consume BW).

        Skipping eliminates most tensor reads at Reg. Per-tensor max tracks
        Z writes = 21,632 post-sparse + 16,384 fills = 38,016.
        """
        reg = _get_component_latency(self.result, "Reg")
        self.assertAlmostEqual(reg, 38_016, delta=100)

    def test_buffer_is_bottleneck(self):
        """Buffer becomes bottleneck when Reg traffic drops."""
        reg = _get_component_latency(self.result, "Reg")
        buf = _get_component_latency(self.result, "Buffer")
        self.assertGreater(buf, reg)

    def test_total_cycles(self):
        """Coord list total latency ~ 295K (Buffer bottleneck with metadata BW)."""
        total = _get_total_latency(self.result)
        self.assertAlmostEqual(total, 295_152, delta=1000)


# ===========================================================================
# Cross-format comparisons
# ===========================================================================


class TestGatingVsSkippingLatency(unittest.TestCase):
    """Cross-format latency comparisons: gating vs skipping."""

    @classmethod
    def setUpClass(cls):
        cls.dense = _load_latency()
        cls.bitmask = _load_latency("sparse_bitmask_latency.yaml")
        cls.coord_list = _load_latency("sparse_coord_list_latency.yaml")

    def test_bitmask_reg_much_higher_than_coord_list(self):
        """Bitmask Reg >> coord_list Reg.

        Gating keeps full dense BW at Reg; skipping eliminates most of it.
        Per-tensor max: Bitmask 2,097,152 vs Coord list ~21,632.
        """
        bm_reg = _get_component_latency(self.bitmask, "Reg")
        cl_reg = _get_component_latency(self.coord_list, "Reg")
        self.assertGreater(bm_reg, cl_reg * 10)

    def test_coord_list_total_much_lower_than_bitmask(self):
        """Coord list total << bitmask total.

        Skipping removes Reg bottleneck → total drops to Buffer bandwidth.
        """
        bm_total = _get_total_latency(self.bitmask)
        cl_total = _get_total_latency(self.coord_list)
        self.assertLess(cl_total, bm_total / 2)

    def test_bitmask_total_equals_dense(self):
        """Bitmask total = Dense total (Reg bandwidth unchanged by gating)."""
        dense_total = _get_total_latency(self.dense)
        bm_total = _get_total_latency(self.bitmask)
        self.assertEqual(int(bm_total), int(dense_total))

    def test_coord_list_less_than_dense(self):
        """Coord list total << Dense total (skipping reduces bandwidth)."""
        dense_total = _get_total_latency(self.dense)
        cl_total = _get_total_latency(self.coord_list)
        self.assertLess(cl_total, dense_total / 5)

    def test_mac_same_for_both(self):
        """MAC cycles identical for bitmask and coord_list.

        Same SAF probabilities → same compute_latency_ratio.
        """
        bm_mac = _get_component_latency(self.bitmask, "MAC")
        cl_mac = _get_component_latency(self.coord_list, "MAC")
        self.assertAlmostEqual(bm_mac, cl_mac, delta=2)


if __name__ == "__main__":
    unittest.main()
