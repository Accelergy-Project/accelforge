"""Sparse energy model tests (Phase 8).

Tests validate that sparse-specific actions (gated_read, metadata_read,
gated_compute, etc.) produce correct energy when arch YAML declares
those actions with realistic ERT values.

Uses arch_energy.yaml which has Sparseloop-like ERT values and declares
sparse action names (gated_read, metadata_read, gated_compute, etc.).

Energy = action_count * energy_per_action (from arch YAML).
"""

import os
import unittest

from accelforge.frontend.spec import Spec
from accelforge.model.main import evaluate_mapping

FIG1_DIR = os.path.join(os.path.dirname(__file__), "input_files", "fig1")

# ERT values from arch_energy.yaml
ERT = {
    "MAC_compute": 0.5608,
    "MAC_gated_compute": 0.03642,
    "Reg_read": 0.49,
    "Reg_write": 0.49,
    "Buffer_read": 0.42568,
    "Buffer_write": 0.58331,
    "Buffer_gated_read": 1e-5,
    "Buffer_metadata_read": 0.7383,
    "Buffer_metadata_write": 1.42366,
    "Buffer_gated_metadata_read": 2e-5,
    "BackingStorage_read": 32.2859,
    "BackingStorage_write": 26.065,
    "BackingStorage_metadata_read": 14.0361,
}


def _load_energy(*extra_yamls):
    """Load fig1 with arch_energy.yaml and optional sparse config."""
    files = [
        os.path.join(FIG1_DIR, "arch_energy.yaml"),
        os.path.join(FIG1_DIR, "workload.yaml"),
        os.path.join(FIG1_DIR, "mapping.yaml"),
    ]
    for f in extra_yamls:
        files.append(os.path.join(FIG1_DIR, f))
    spec = Spec.from_yaml(*files)
    return evaluate_mapping(spec)


def _get_action(result, component, tensor, action):
    """Get a standard per-tensor action count."""
    actions = result.actions(per_component=True, per_tensor=True)
    key = (component, tensor, action)
    return int(actions.get(key, 0))


def _get_energy(result, component, tensor, action):
    """Get a standard per-tensor energy value."""
    energy = result.energy(per_component=True, per_tensor=True, per_action=True)
    key = (component, tensor, action)
    return float(energy.get(key, 0))


def _get_sparse_col(result, col_type, component, action):
    """Get a sparse action count or energy from the raw DataFrame columns.

    Sparse actions (metadata_read, gated_read, etc.) don't have a tensor
    dimension, so they appear as 2-part keys in the DataFrame.
    """
    # Columns are like: SpMSpM<SEP>action<SEP>Buffer<SEP>gated_read
    # or: SpMSpM<SEP>energy<SEP>Buffer<SEP>metadata_read
    suffix = f"{component}<SEP>{action}"
    for col in result.data.columns:
        if col_type in col and col.endswith(suffix):
            val = result.data[col].iloc[0]
            return float(val)
    return 0.0


def _get_sparse_action(result, component, action):
    """Get a sparse-specific action count (no tensor dimension)."""
    return _get_sparse_col(result, "action", component, action)


def _get_sparse_energy(result, component, action):
    """Get a sparse-specific energy value (no tensor dimension)."""
    return _get_sparse_col(result, "energy", component, action)


def _total_energy(result):
    """Get total energy from the raw DataFrame."""
    return float(result.data["Total<SEP>energy"].iloc[0])


# ===========================================================================
# Dense baseline with realistic ERT
# ===========================================================================


class TestDenseEnergyBaseline(unittest.TestCase):
    """Dense (no sparse) energy with realistic ERT values — regression test."""

    @classmethod
    def setUpClass(cls):
        cls.result = _load_energy()

    def test_mac_compute_energy(self):
        """MAC compute energy = 2,097,152 * 0.5608."""
        expected = 2_097_152 * ERT["MAC_compute"]
        actual = _get_energy(self.result, "MAC", "None", "compute")
        self.assertAlmostEqual(actual, expected, places=2)

    def test_buffer_read_energy(self):
        """Buffer total read energy = (2,097,152 + 2,097,152) * 0.42568."""
        # Buffer reads A and B, each 2,097,152
        expected = (2_097_152 + 2_097_152) * ERT["Buffer_read"]
        a_read = _get_energy(self.result, "Buffer", "A", "read")
        b_read = _get_energy(self.result, "Buffer", "B", "read")
        self.assertAlmostEqual(a_read + b_read, expected, places=2)

    def test_backing_storage_read_energy(self):
        """BackingStorage read energy: A=2,097,152 + B=16,384 reads."""
        a_read = _get_energy(self.result, "BackingStorage", "A", "read")
        b_read = _get_energy(self.result, "BackingStorage", "B", "read")
        expected = (2_097_152 + 16_384) * ERT["BackingStorage_read"]
        self.assertAlmostEqual(a_read + b_read, expected, places=2)

    def test_no_sparse_actions(self):
        """Dense has no sparse-specific actions."""
        self.assertEqual(_get_sparse_action(self.result, "Buffer", "gated_read"), 0)
        self.assertEqual(_get_sparse_action(self.result, "Buffer", "metadata_read"), 0)
        self.assertEqual(_get_sparse_action(self.result, "MAC", "gated_compute"), 0)


# ===========================================================================
# Bitmask energy
# ===========================================================================


class TestBitmaskEnergy(unittest.TestCase):
    """Bitmask format energy with realistic ERT values."""

    @classmethod
    def setUpClass(cls):
        cls.result = _load_energy("sparse_bitmask_energy.yaml")

    # --- MAC ---

    def test_mac_effectual_compute_energy(self):
        """Bitmask MAC effectual compute energy = 223 * 0.5608."""
        computes = _get_action(self.result, "MAC", "None", "compute")
        self.assertEqual(computes, 223)
        expected = computes * ERT["MAC_compute"]
        actual = _get_energy(self.result, "MAC", "None", "compute")
        self.assertAlmostEqual(actual, expected, places=2)

    def test_mac_gated_compute_energy(self):
        """Bitmask MAC gated compute energy = 21,409 * 0.03642."""
        gated = _get_sparse_action(self.result, "MAC", "gated_compute")
        self.assertEqual(int(gated), 21_409)
        expected = gated * ERT["MAC_gated_compute"]
        actual = _get_sparse_energy(self.result, "MAC", "gated_compute")
        self.assertAlmostEqual(actual, expected, places=2)

    # --- Buffer ---

    def test_buffer_gated_read_action_count(self):
        """Bitmask Buffer gated reads = 382,720 (SAF delta for A+B)."""
        gated = _get_sparse_action(self.result, "Buffer", "gated_read")
        self.assertEqual(int(gated), 382_720)

    def test_buffer_gated_read_energy(self):
        """Bitmask Buffer gated_read energy = 382,720 * 1e-5."""
        expected = 382_720 * ERT["Buffer_gated_read"]
        actual = _get_sparse_energy(self.result, "Buffer", "gated_read")
        self.assertAlmostEqual(actual, expected, places=4)

    def test_buffer_metadata_read_action_count(self):
        """Bitmask Buffer metadata_read count = 15,214."""
        count = _get_sparse_action(self.result, "Buffer", "metadata_read")
        self.assertEqual(int(count), 15_214)

    def test_buffer_metadata_read_energy(self):
        """Bitmask Buffer metadata_read energy = 15,214 * 0.7383."""
        count = 15_214
        expected = count * ERT["Buffer_metadata_read"]
        actual = _get_sparse_energy(self.result, "Buffer", "metadata_read")
        self.assertAlmostEqual(actual, expected, places=2)

    def test_buffer_metadata_write_energy(self):
        """Bitmask Buffer metadata_write energy = 7,667 * 1.42366."""
        count = _get_sparse_action(self.result, "Buffer", "metadata_write")
        self.assertEqual(int(count), 7_667)
        expected = count * ERT["Buffer_metadata_write"]
        actual = _get_sparse_energy(self.result, "Buffer", "metadata_write")
        self.assertAlmostEqual(actual, expected, places=2)

    # --- BackingStorage ---

    def test_backing_storage_metadata_read_energy(self):
        """Bitmask BackingStorage metadata_read energy = 7,667 * 14.0361."""
        count = _get_sparse_action(self.result, "BackingStorage", "metadata_read")
        self.assertEqual(int(count), 7_667)
        expected = count * ERT["BackingStorage_metadata_read"]
        actual = _get_sparse_energy(self.result, "BackingStorage", "metadata_read")
        self.assertAlmostEqual(actual, expected, places=2)


# ===========================================================================
# Coord list energy
# ===========================================================================


class TestCoordListEnergy(unittest.TestCase):
    """Coord list format energy with realistic ERT values."""

    @classmethod
    def setUpClass(cls):
        cls.result = _load_energy("sparse_coord_list_energy.yaml")

    # --- MAC (skipping = zero energy for skipped computes) ---

    def test_mac_effectual_compute_energy(self):
        """Coord list MAC effectual compute energy = 223 * 0.5608."""
        computes = _get_action(self.result, "MAC", "None", "compute")
        self.assertEqual(computes, 223)
        expected = computes * ERT["MAC_compute"]
        actual = _get_energy(self.result, "MAC", "None", "compute")
        self.assertAlmostEqual(actual, expected, places=2)

    def test_mac_no_gated_compute(self):
        """Coord list uses skipping, not gating — no gated_compute action."""
        gated = _get_sparse_action(self.result, "MAC", "gated_compute")
        self.assertEqual(gated, 0)

    # --- Buffer (no gated_read for coord_list — it uses skipping) ---

    def test_buffer_no_gated_read(self):
        """Coord list uses skipping at Buffer — no gated_read emitted."""
        gated = _get_sparse_action(self.result, "Buffer", "gated_read")
        self.assertEqual(gated, 0)

    def test_buffer_metadata_read_count(self):
        """Coord list Buffer metadata_read = 10,816."""
        count = _get_sparse_action(self.result, "Buffer", "metadata_read")
        self.assertEqual(int(count), 10_816)

    def test_buffer_metadata_write_count(self):
        """Coord list Buffer metadata_write = 53,664."""
        count = _get_sparse_action(self.result, "Buffer", "metadata_write")
        self.assertEqual(int(count), 53_664)

    def test_buffer_metadata_write_energy(self):
        """Coord list Buffer metadata_write energy = 53,664 * 1.42366."""
        count = 53_664
        expected = count * ERT["Buffer_metadata_write"]
        actual = _get_sparse_energy(self.result, "Buffer", "metadata_write")
        self.assertAlmostEqual(actual, expected, places=2)

    # --- BackingStorage ---

    def test_backing_storage_metadata_read_energy(self):
        """Coord list BackingStorage metadata_read energy = 53,664 * 14.0361."""
        count = _get_sparse_action(self.result, "BackingStorage", "metadata_read")
        self.assertEqual(int(count), 53_664)
        expected = count * ERT["BackingStorage_metadata_read"]
        actual = _get_sparse_energy(self.result, "BackingStorage", "metadata_read")
        self.assertAlmostEqual(actual, expected, places=2)


# ===========================================================================
# Cross-format energy comparisons
# ===========================================================================


class TestEnergyComparisons(unittest.TestCase):
    """Total energy ordering: coord_list > bitmask (CP metadata overhead)."""

    @classmethod
    def setUpClass(cls):
        cls.dense = _load_energy()
        cls.bitmask = _load_energy("sparse_bitmask_energy.yaml")
        cls.coord_list = _load_energy("sparse_coord_list_energy.yaml")

    def test_bitmask_less_than_dense(self):
        """Bitmask total energy < dense total energy."""
        self.assertLess(_total_energy(self.bitmask), _total_energy(self.dense))

    def test_coord_list_less_than_dense(self):
        """Coord list total energy < dense total energy."""
        self.assertLess(_total_energy(self.coord_list), _total_energy(self.dense))

    def test_coord_list_more_than_bitmask(self):
        """Coord list total energy > bitmask total energy.

        CP/CSR metadata has more bits per coordinate (7 vs 1) and more
        metadata accesses, resulting in higher metadata energy overhead.
        """
        self.assertGreater(
            _total_energy(self.coord_list), _total_energy(self.bitmask)
        )

    def test_bitmask_metadata_energy_less_than_coord_list(self):
        """Bitmask metadata energy < coord list metadata energy.

        Bitmask: 1 bit per element, packed 28:1 → fewer physical accesses.
        CSR: 7 bits per coordinate, packed 4:1 → more physical accesses.
        """
        bm_meta = (
            _get_sparse_energy(self.bitmask, "Buffer", "metadata_read")
            + _get_sparse_energy(self.bitmask, "Buffer", "metadata_write")
            + _get_sparse_energy(self.bitmask, "BackingStorage", "metadata_read")
        )
        cl_meta = (
            _get_sparse_energy(self.coord_list, "Buffer", "metadata_read")
            + _get_sparse_energy(self.coord_list, "Buffer", "metadata_write")
            + _get_sparse_energy(self.coord_list, "BackingStorage", "metadata_read")
        )
        self.assertLess(bm_meta, cl_meta)


if __name__ == "__main__":
    unittest.main()
