"""Per-rank format model integration tests (Phase 9).

Validates per-rank format capacity and access counts against
ARTIFACT_EVALUATION reference values for fig1 bitmask and coord_list
configurations.

These tests verify the format_capacity, format_reads, and format_fills
DataFrame columns produced by the per-rank computation in
sparse_adjustment.py.
"""

import os
import unittest

from accelforge.frontend.spec import Spec
from accelforge.model.main import evaluate_mapping

FIG1_DIR = os.path.join(os.path.dirname(__file__), "input_files", "fig1")


def _load(*extra_yamls):
    """Load fig1 with optional sparse config."""
    files = [
        os.path.join(FIG1_DIR, "arch_energy.yaml"),
        os.path.join(FIG1_DIR, "workload.yaml"),
        os.path.join(FIG1_DIR, "mapping.yaml"),
    ]
    for f in extra_yamls:
        files.append(os.path.join(FIG1_DIR, f))
    spec = Spec.from_yaml(*files)
    return evaluate_mapping(spec)


def _col(result, col_name):
    """Get a column value from the raw DataFrame."""
    for c in result.data.columns:
        if c.endswith(col_name):
            return result.data[c].iloc[0]
    raise KeyError(f"Column ending with {col_name!r} not found")


# ===========================================================================
# Bitmask per-rank format capacity (ARTIFACT_EVALUATION §2.3)
# ===========================================================================


class TestBitmaskFormatCapacity(unittest.TestCase):
    """Per-rank format capacity for bitmask (UOP+B) format."""

    @classmethod
    def setUpClass(cls):
        cls.result = _load("sparse_bitmask_energy.yaml")

    # --- Buffer A: bitmask -> 1 rank (B, K=128) ---

    def test_buffer_a_rank0_metadata(self):
        """Buffer A rank0 (B): metadata = 128 (one bit per element)."""
        val = _col(self.result, "format_capacity<SEP>Buffer<SEP>A<SEP>rank0<SEP>metadata")
        self.assertEqual(val, 128)

    def test_buffer_a_rank0_payload(self):
        """Buffer A rank0 (B): payload = 0."""
        val = _col(self.result, "format_capacity<SEP>Buffer<SEP>A<SEP>rank0<SEP>payload")
        self.assertEqual(val, 0)

    # --- Buffer B: bitmask -> 1 rank (B, K=128) ---

    def test_buffer_b_rank0_metadata(self):
        """Buffer B rank0 (B): metadata = 128."""
        val = _col(self.result, "format_capacity<SEP>Buffer<SEP>B<SEP>rank0<SEP>metadata")
        self.assertEqual(val, 128)

    def test_buffer_b_rank0_payload(self):
        """Buffer B rank0 (B): payload = 0."""
        val = _col(self.result, "format_capacity<SEP>Buffer<SEP>B<SEP>rank0<SEP>payload")
        self.assertEqual(val, 0)

    # --- BackingStorage A: bitmask -> 2 ranks (UOP M=128, B K=128) ---

    def test_bs_a_rank0_metadata(self):
        """BackingStorage A rank0 (UOP): metadata = 0."""
        val = _col(self.result, "format_capacity<SEP>BackingStorage<SEP>A<SEP>rank0<SEP>metadata")
        self.assertEqual(val, 0)

    def test_bs_a_rank0_payload(self):
        """BackingStorage A rank0 (UOP): payload = 129 (128+1 offset pairs)."""
        val = _col(self.result, "format_capacity<SEP>BackingStorage<SEP>A<SEP>rank0<SEP>payload")
        self.assertEqual(val, 129)

    def test_bs_a_rank1_metadata(self):
        """BackingStorage A rank1 (B): metadata = 16384 (128*128 bitmask)."""
        val = _col(self.result, "format_capacity<SEP>BackingStorage<SEP>A<SEP>rank1<SEP>metadata")
        self.assertEqual(val, 16384)

    def test_bs_a_rank1_payload(self):
        """BackingStorage A rank1 (B): payload = 0."""
        val = _col(self.result, "format_capacity<SEP>BackingStorage<SEP>A<SEP>rank1<SEP>payload")
        self.assertEqual(val, 0)

    # --- BackingStorage B: bitmask -> 2 ranks (UOP N=128, B K=128) ---

    def test_bs_b_rank0_payload(self):
        """BackingStorage B rank0 (UOP): payload = 129."""
        val = _col(self.result, "format_capacity<SEP>BackingStorage<SEP>B<SEP>rank0<SEP>payload")
        self.assertEqual(val, 129)

    def test_bs_b_rank1_metadata(self):
        """BackingStorage B rank1 (B): metadata = 16384."""
        val = _col(self.result, "format_capacity<SEP>BackingStorage<SEP>B<SEP>rank1<SEP>metadata")
        self.assertEqual(val, 16384)


# ===========================================================================
# Bitmask per-rank format access counts (ARTIFACT_EVALUATION §2.4)
# ===========================================================================


class TestBitmaskFormatAccessCounts(unittest.TestCase):
    """Per-rank format access counts for bitmask format."""

    @classmethod
    def setUpClass(cls):
        cls.result = _load("sparse_bitmask_energy.yaml")

    # --- Buffer A reads ---

    def test_buffer_a_rank0_metadata_reads(self):
        """Buffer A rank0 (B) metadata reads = 2,097,152."""
        val = _col(self.result, "format_reads<SEP>Buffer<SEP>A<SEP>rank0<SEP>metadata")
        self.assertEqual(val, 2097152)

    def test_buffer_a_rank0_payload_reads(self):
        """Buffer A rank0 (B) payload reads = 0."""
        val = _col(self.result, "format_reads<SEP>Buffer<SEP>A<SEP>rank0<SEP>payload")
        self.assertEqual(val, 0)

    # --- Buffer B reads ---

    def test_buffer_b_rank0_metadata_reads(self):
        """Buffer B rank0 (B) metadata reads = 2,097,152."""
        val = _col(self.result, "format_reads<SEP>Buffer<SEP>B<SEP>rank0<SEP>metadata")
        self.assertEqual(val, 2097152)

    # --- BackingStorage A reads ---

    def test_bs_a_rank0_payload_reads(self):
        """BackingStorage A rank0 (UOP) payload reads = 16,512."""
        val = _col(self.result, "format_reads<SEP>BackingStorage<SEP>A<SEP>rank0<SEP>payload")
        self.assertEqual(val, 16512)

    def test_bs_a_rank0_metadata_reads(self):
        """BackingStorage A rank0 (UOP) metadata reads = 0."""
        val = _col(self.result, "format_reads<SEP>BackingStorage<SEP>A<SEP>rank0<SEP>metadata")
        self.assertEqual(val, 0)

    def test_bs_a_rank1_metadata_reads(self):
        """BackingStorage A rank1 (B) metadata reads = 2,097,152."""
        val = _col(self.result, "format_reads<SEP>BackingStorage<SEP>A<SEP>rank1<SEP>metadata")
        self.assertEqual(val, 2097152)

    # --- BackingStorage B reads ---

    def test_bs_b_rank0_payload_reads(self):
        """BackingStorage B rank0 (UOP) payload reads = 129."""
        val = _col(self.result, "format_reads<SEP>BackingStorage<SEP>B<SEP>rank0<SEP>payload")
        self.assertEqual(val, 129)

    def test_bs_b_rank1_metadata_reads(self):
        """BackingStorage B rank1 (B) metadata reads = 16,384."""
        val = _col(self.result, "format_reads<SEP>BackingStorage<SEP>B<SEP>rank1<SEP>metadata")
        self.assertEqual(val, 16384)

    # --- Buffer A fills ---

    def test_buffer_a_rank0_metadata_fills(self):
        """Buffer A rank0 (B) metadata fills = 2,097,152."""
        val = _col(self.result, "format_fills<SEP>Buffer<SEP>A<SEP>rank0<SEP>metadata")
        self.assertEqual(val, 2097152)

    # --- Buffer B fills ---

    def test_buffer_b_rank0_metadata_fills(self):
        """Buffer B rank0 (B) metadata fills = 16,384."""
        val = _col(self.result, "format_fills<SEP>Buffer<SEP>B<SEP>rank0<SEP>metadata")
        self.assertEqual(val, 16384)


# ===========================================================================
# Coord list (CSR) per-rank format capacity
# ===========================================================================


class TestCoordListFormatCapacity(unittest.TestCase):
    """Per-rank format capacity for coord list (CSR = UOP+CP) format."""

    @classmethod
    def setUpClass(cls):
        cls.result = _load("sparse_coord_list_energy.yaml")

    # --- Buffer A: CSR -> 1 rank (CP, K=128) ---

    def test_buffer_a_rank0_metadata(self):
        """Buffer A rank0 (CP): metadata = 13 (ceil(ennz) coordinates)."""
        val = _col(self.result, "format_capacity<SEP>Buffer<SEP>A<SEP>rank0<SEP>metadata")
        self.assertEqual(val, 13)

    def test_buffer_a_rank0_payload(self):
        """Buffer A rank0 (CP): payload = 0."""
        val = _col(self.result, "format_capacity<SEP>Buffer<SEP>A<SEP>rank0<SEP>payload")
        self.assertEqual(val, 0)

    # --- Buffer B: CSR -> 1 rank (CP, K=128) ---

    def test_buffer_b_rank0_metadata(self):
        """Buffer B rank0 (CP): metadata = 13."""
        val = _col(self.result, "format_capacity<SEP>Buffer<SEP>B<SEP>rank0<SEP>metadata")
        self.assertEqual(val, 13)

    # --- BackingStorage A: CSR -> 2 ranks (UOP M=128, CP K=128) ---

    def test_bs_a_rank0_payload(self):
        """BackingStorage A rank0 (UOP): payload = 129."""
        val = _col(self.result, "format_capacity<SEP>BackingStorage<SEP>A<SEP>rank0<SEP>payload")
        self.assertEqual(val, 129)

    def test_bs_a_rank1_metadata(self):
        """BackingStorage A rank1 (CP): metadata = 1664 (ceil(ennz)*fibers)."""
        val = _col(self.result, "format_capacity<SEP>BackingStorage<SEP>A<SEP>rank1<SEP>metadata")
        self.assertEqual(val, 1664)


# ===========================================================================
# Coord list (CSR) per-rank format access counts
# ===========================================================================


class TestCoordListFormatAccessCounts(unittest.TestCase):
    """Per-rank format access counts for coord list (CSR) format."""

    @classmethod
    def setUpClass(cls):
        cls.result = _load("sparse_coord_list_energy.yaml")

    # --- Buffer A reads ---

    def test_buffer_a_rank0_metadata_reads(self):
        """Buffer A rank0 (CP) metadata reads = 212,992."""
        val = _col(self.result, "format_reads<SEP>Buffer<SEP>A<SEP>rank0<SEP>metadata")
        self.assertEqual(val, 212992)

    # --- BackingStorage A reads ---

    def test_bs_a_rank0_payload_reads(self):
        """BackingStorage A rank0 (UOP) payload reads = 16,512."""
        val = _col(self.result, "format_reads<SEP>BackingStorage<SEP>A<SEP>rank0<SEP>payload")
        self.assertEqual(val, 16512)

    def test_bs_a_rank1_metadata_reads(self):
        """BackingStorage A rank1 (CP) metadata reads = 212,992."""
        val = _col(self.result, "format_reads<SEP>BackingStorage<SEP>A<SEP>rank1<SEP>metadata")
        self.assertEqual(val, 212992)


# ===========================================================================
# Dense has no per-rank columns
# ===========================================================================


class TestDenseNoPerRankColumns(unittest.TestCase):
    """Dense (no sparse) should not have per-rank format columns."""

    @classmethod
    def setUpClass(cls):
        cls.result = _load()

    def test_no_format_columns(self):
        """Dense result has no format_capacity/format_reads/format_fills columns."""
        format_cols = [
            c for c in self.result.data.columns if "format_capacity" in c
        ]
        self.assertEqual(len(format_cols), 0)


# ===========================================================================
# Bitmask vs Coord List capacity comparisons
# ===========================================================================


class TestCapacityComparisons(unittest.TestCase):
    """Bitmask vs coord list per-rank capacity differences."""

    @classmethod
    def setUpClass(cls):
        cls.bitmask = _load("sparse_bitmask_energy.yaml")
        cls.coord_list = _load("sparse_coord_list_energy.yaml")

    def test_buffer_a_bitmask_larger_than_coord_list(self):
        """Bitmask Buffer A metadata capacity (128) > CSR Buffer A (13).

        Bitmask stores one bit per position regardless of density.
        CSR stores one coordinate per nonzero.
        """
        bm = _col(self.bitmask, "format_capacity<SEP>Buffer<SEP>A<SEP>rank0<SEP>metadata")
        cl = _col(self.coord_list, "format_capacity<SEP>Buffer<SEP>A<SEP>rank0<SEP>metadata")
        self.assertGreater(bm, cl)

    def test_bs_a_rank1_bitmask_larger_than_coord_list(self):
        """BackingStorage A rank1: bitmask metadata (16384) > CSR (1664).

        At full tensor level, bitmask = M*K = 16384.
        CSR CP = 128 * ceil(ennz) = 128 * 13 = 1664.
        """
        bm = _col(self.bitmask, "format_capacity<SEP>BackingStorage<SEP>A<SEP>rank1<SEP>metadata")
        cl = _col(self.coord_list, "format_capacity<SEP>BackingStorage<SEP>A<SEP>rank1<SEP>metadata")
        self.assertGreater(bm, cl)

    def test_uop_payload_same_for_both(self):
        """UOP payload at BackingStorage is format-independent (always 129)."""
        bm = _col(self.bitmask, "format_capacity<SEP>BackingStorage<SEP>A<SEP>rank0<SEP>payload")
        cl = _col(self.coord_list, "format_capacity<SEP>BackingStorage<SEP>A<SEP>rank0<SEP>payload")
        self.assertEqual(bm, cl)
        self.assertEqual(bm, 129)


# ===========================================================================
# Energy pipeline unchanged (metadata action counts same as before)
# ===========================================================================


class TestEnergyPipelinePerRank(unittest.TestCase):
    """Per-rank model determines metadata_read/write action counts."""

    @classmethod
    def setUpClass(cls):
        cls.bitmask = _load("sparse_bitmask_energy.yaml")
        cls.coord_list = _load("sparse_coord_list_energy.yaml")

    def test_bitmask_buffer_metadata_read(self):
        """Bitmask Buffer metadata_read = 15,214 (actual only, gating split).

        Per-rank model: bitmask at Buffer has 1 non-trivial dim (k=128),
        format=["B"]. For gating, only effectual iterations are charged at
        full metadata_read rate. Gated iterations at gated_metadata_read.
        """
        for c in self.bitmask.data.columns:
            if "action" in c and c.endswith("Buffer<SEP>metadata_read"):
                self.assertEqual(int(self.bitmask.data[c].iloc[0]), 15214)
                return
        self.fail("metadata_read column not found")

    def test_bitmask_buffer_metadata_write(self):
        """Bitmask Buffer metadata_write = 75,485 (per-rank: density-independent mask).

        A fills=2,097,152 → packed 74,899. B fills=16,384 → packed 586.
        """
        for c in self.bitmask.data.columns:
            if "action" in c and c.endswith("Buffer<SEP>metadata_write"):
                self.assertEqual(int(self.bitmask.data[c].iloc[0]), 75485)
                return
        self.fail("metadata_write column not found")

    def test_coord_list_buffer_metadata_read(self):
        """Coord list Buffer metadata_read = 21,632 (per-rank: CP scales with ennz)."""
        for c in self.coord_list.data.columns:
            if "action" in c and c.endswith("Buffer<SEP>metadata_read"):
                self.assertEqual(int(self.coord_list.data[c].iloc[0]), 21632)
                return
        self.fail("metadata_read column not found")


if __name__ == "__main__":
    unittest.main()
