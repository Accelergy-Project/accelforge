"""Tests for Phase 2: Sparse format occupancy models and auto-expansion.

Validation tests sourced from ARTIFACT_EVALUATION.md fig1 reference outputs
and Lab 4 Part 4 storage capacity sweep.
"""

import unittest

from accelforge.model.sparse_formats import (
    UOP,
    CP,
    Bitmask,
    RLE,
    expand_format,
    compute_format_occupancy,
)
from accelforge.model.density_model import HypergeometricDensityModel


# ---------------------------------------------------------------------------
# Individual format model tests
# ---------------------------------------------------------------------------


class TestUOP(unittest.TestCase):
    """UOP: metadata=0, payload=fibers*(shape+1)."""

    def test_backing_storage_a_rank1(self):
        """Fig1: UOP, fibers=1, shape=128 -> (0, 129)."""
        occ = UOP().get_occupancy(fibers=1, fiber_shape=128)
        self.assertEqual(occ.metadata_units, 0)
        self.assertEqual(occ.payload_units, 129)

    def test_backing_storage_b_rank2(self):
        """Fig1: UOP, fibers=1, shape=128 -> (0, 129)."""
        occ = UOP().get_occupancy(fibers=1, fiber_shape=128)
        self.assertEqual(occ.metadata_units, 0)
        self.assertEqual(occ.payload_units, 129)

    def test_backing_storage_b_rank1(self):
        """Fig1: UOP, fibers=128, shape=128 -> (0, 16512)."""
        occ = UOP().get_occupancy(fibers=128, fiber_shape=128)
        self.assertEqual(occ.metadata_units, 0)
        self.assertEqual(occ.payload_units, 128 * 129)

    def test_lab4_uop_outer(self):
        """Lab 4: UOP outer rank for A[M=8,K=8] -> payload = 1*(8+1) = 9."""
        occ = UOP().get_occupancy(fibers=1, fiber_shape=8)
        self.assertEqual(occ.payload_units, 9)
        self.assertEqual(occ.metadata_units, 0)


class TestCP(unittest.TestCase):
    """CP: metadata=fibers*ceil(expected_nnz), payload=0."""

    def test_buffer_a_coord_list(self):
        """Fig1 coord_list: CP, 1 fiber, shape=128, d=0.1015625 -> (13, 0)."""
        model = HypergeometricDensityModel(0.1015625, 16384)
        ennz = model.expected_occupancy(128)  # 13.0
        occ = CP().get_occupancy(fibers=1, fiber_shape=128, expected_nnz_per_fiber=ennz)
        self.assertEqual(occ.metadata_units, 13)
        self.assertEqual(occ.payload_units, 0)

    def test_backing_storage_a_rank0_coord_list(self):
        """Fig1 coord_list: CP, 128 fibers, shape=128, expected_nnz=13 -> (1664, 0)."""
        model = HypergeometricDensityModel(0.1015625, 16384)
        ennz = model.expected_occupancy(128)  # 13.0
        occ = CP().get_occupancy(
            fibers=128, fiber_shape=128, expected_nnz_per_fiber=ennz
        )
        self.assertEqual(occ.metadata_units, 1664)
        self.assertEqual(occ.payload_units, 0)

    def test_lab4_cp_d02(self):
        """Lab 4 d=0.2: CP inner, 8 fibers, expected_nnz = 8*13/64 = 1.625 -> 8*2 = 16."""
        model = HypergeometricDensityModel(0.2, 64)
        ennz = model.expected_occupancy(8)  # 8 * 13/64 = 1.625
        occ = CP().get_occupancy(fibers=8, fiber_shape=8, expected_nnz_per_fiber=ennz)
        self.assertEqual(occ.metadata_units, 16)

    def test_lab4_cp_d10(self):
        """Lab 4 d=1.0: CP inner, 8 fibers, expected_nnz = 8.0 -> 8*8 = 64."""
        model = HypergeometricDensityModel(1.0, 64)
        ennz = model.expected_occupancy(8)  # 8.0
        occ = CP().get_occupancy(fibers=8, fiber_shape=8, expected_nnz_per_fiber=ennz)
        self.assertEqual(occ.metadata_units, 64)


class TestBitmask(unittest.TestCase):
    """Bitmask: metadata=fibers*fiber_shape, payload=0."""

    def test_buffer_a(self):
        """Fig1 bitmask: B, 1 fiber, shape=128 -> (128, 0)."""
        occ = Bitmask().get_occupancy(fibers=1, fiber_shape=128)
        self.assertEqual(occ.metadata_units, 128)
        self.assertEqual(occ.payload_units, 0)

    def test_backing_storage_a_rank0(self):
        """Fig1 bitmask: B, 128 fibers, shape=128 -> (16384, 0)."""
        occ = Bitmask().get_occupancy(fibers=128, fiber_shape=128)
        self.assertEqual(occ.metadata_units, 16384)
        self.assertEqual(occ.payload_units, 0)

class TestRLE(unittest.TestCase):
    """RLE: metadata=fibers*expected_nnz (NO ceil), payload=0."""

    def test_fractional_metadata(self):
        """RLE does NOT ceil -- keeps fractional value."""
        occ = RLE().get_occupancy(fibers=8, fiber_shape=8, expected_nnz_per_fiber=1.625)
        self.assertAlmostEqual(occ.metadata_units, 8 * 1.625)
        self.assertEqual(occ.payload_units, 0)

# ---------------------------------------------------------------------------
# Auto-expansion tests
# ---------------------------------------------------------------------------


class TestExpandFormat(unittest.TestCase):
    """Test user-friendly format name -> per-rank primitive expansion."""

    def test_bitmask_2_ranks(self):
        self.assertEqual(expand_format("bitmask", 2), ["UOP", "B"])

    def test_bitmask_3_ranks(self):
        self.assertEqual(expand_format("bitmask", 3), ["UOP", "UOP", "B"])

    def test_bitmask_1_rank(self):
        self.assertEqual(expand_format("bitmask", 1), ["B"])

    def test_csr_2_ranks(self):
        self.assertEqual(expand_format("csr", 2), ["UOP", "CP"])

    def test_csr_3_ranks(self):
        self.assertEqual(expand_format("csr", 3), ["UOP", "UOP", "CP"])

    def test_csr_1_rank(self):
        self.assertEqual(expand_format("csr", 1), ["CP"])

    def test_coo_2_ranks(self):
        self.assertEqual(expand_format("coo", 2), ["CP", "CP"])

    def test_coo_3_ranks(self):
        self.assertEqual(expand_format("coo", 3), ["CP", "CP", "CP"])

    def test_coo_1_rank(self):
        self.assertEqual(expand_format("coo", 1), ["CP"])

    def test_rle_2_ranks(self):
        self.assertEqual(expand_format("rle", 2), ["UOP", "RLE"])

    def test_rle_3_ranks(self):
        self.assertEqual(expand_format("rle", 3), ["UOP", "UOP", "RLE"])

    def test_case_insensitive(self):
        self.assertEqual(expand_format("CSR", 2), ["UOP", "CP"])
        self.assertEqual(expand_format("Bitmask", 2), ["UOP", "B"])
        self.assertEqual(expand_format("COO", 3), ["CP", "CP", "CP"])

    def test_b_alias(self):
        """'b' is an alias for 'bitmask'."""
        self.assertEqual(expand_format("b", 2), ["UOP", "B"])

    def test_unknown_format_raises(self):
        with self.assertRaises(ValueError):
            expand_format("unknown", 2)

    def test_zero_ranks_raises(self):
        with self.assertRaises(ValueError):
            expand_format("csr", 0)


# ---------------------------------------------------------------------------
# Multi-rank format occupancy tests
# ---------------------------------------------------------------------------


class TestComputeFormatOccupancy(unittest.TestCase):
    """Test total format occupancy across multiple ranks."""

    def test_lab4_uop_cp_d02(self):
        """UOP+CP, M=K=8, d=0.2.

        With UOP empty fiber filtering, prob_empty(8)≈0.144 at d=0.2,
        so effective UOP payload < 9.  Total ≈ 21.4.
        """
        _, total = compute_format_occupancy(
            rank_formats=["UOP", "CP"],
            dimension_sizes=[8, 8],
            density=0.2,
            tensor_size=64,
        )
        self.assertAlmostEqual(total, 21.403, places=2)

    def test_lab4_uop_cp_d04(self):
        """UOP+CP, M=K=8, d=0.4.  prob_empty(8)≈0.011, total≈40.5."""
        _, total = compute_format_occupancy(
            rank_formats=["UOP", "CP"],
            dimension_sizes=[8, 8],
            density=0.4,
            tensor_size=64,
        )
        self.assertAlmostEqual(total, 40.547, places=2)

    def test_lab4_uop_cp_d06(self):
        """UOP+CP, M=K=8, d=0.6.  prob_empty(8)≈0.0002, total≈49.0."""
        _, total = compute_format_occupancy(
            rank_formats=["UOP", "CP"],
            dimension_sizes=[8, 8],
            density=0.6,
            tensor_size=64,
        )
        self.assertAlmostEqual(total, 48.988, places=2)

    def test_lab4_uop_cp_d08(self):
        """UOP+CP, M=K=8, d=0.8.  prob_empty(8)≈0, total≈65.0."""
        _, total = compute_format_occupancy(
            rank_formats=["UOP", "CP"],
            dimension_sizes=[8, 8],
            density=0.8,
            tensor_size=64,
        )
        self.assertAlmostEqual(total, 65.0, places=2)

    def test_lab4_uop_cp_d10(self):
        """UOP+CP, M=K=8, d=1.0 -> 73 total (exact, no filtering at d=1)."""
        _, total = compute_format_occupancy(
            rank_formats=["UOP", "CP"],
            dimension_sizes=[8, 8],
            density=1.0,
            tensor_size=64,
        )
        self.assertEqual(total, 73)

    def test_fig1_bitmask_backing_storage_a(self):
        """Fig1: UOP+B for A at BackingStorage, M=K=128.

        With UOP filtering, prob_empty(128)≈1e-6 → negligible change.
        Rank 0 (UOP): (0, ~129). Rank 1 (B): (~16384, 0).
        """
        occs, total = compute_format_occupancy(
            rank_formats=["UOP", "B"],
            dimension_sizes=[128, 128],
            density=0.1015625,
            tensor_size=16384,
        )
        self.assertEqual(occs[0].metadata_units, 0)
        self.assertAlmostEqual(occs[0].payload_units, 129, places=2)
        self.assertAlmostEqual(occs[1].metadata_units, 16384, places=1)
        self.assertEqual(occs[1].payload_units, 0)
        self.assertAlmostEqual(total, 16513, places=1)

    def test_fig1_bitmask_backing_storage_b(self):
        """Fig1: UOP+UOP+B for B at BackingStorage, K=N=128.

        With UOP filtering at d=0.1, the outer UOP(128) has prob_empty≈1e-6
        and the inner UOP(128) has fibers slightly filtered.
        """
        occs, total = compute_format_occupancy(
            rank_formats=["UOP", "UOP", "B"],
            dimension_sizes=[128, 128, 1],
            density=0.1015625,
            tensor_size=16384,
        )
        self.assertEqual(occs[0].metadata_units, 0)
        self.assertAlmostEqual(occs[0].payload_units, 129, places=2)
        self.assertEqual(occs[1].metadata_units, 0)
        self.assertAlmostEqual(occs[1].payload_units, 16512, places=1)
        # Innermost B rank: fibers≈128*128, shape=1
        self.assertAlmostEqual(occs[2].metadata_units, 16384, places=1)
        self.assertEqual(occs[2].payload_units, 0)

    def test_fig1_csr_backing_storage_a(self):
        """Fig1 coord_list: UOP+CP for A at BackingStorage, M=K=128.

        UOP filtering has negligible effect at d=0.1, tile=128.
        """
        occs, total = compute_format_occupancy(
            rank_formats=["UOP", "CP"],
            dimension_sizes=[128, 128],
            density=0.1015625,
            tensor_size=16384,
        )
        self.assertEqual(occs[0].metadata_units, 0)
        self.assertAlmostEqual(occs[0].payload_units, 129, places=2)
        self.assertAlmostEqual(occs[1].metadata_units, 1664, places=1)
        self.assertEqual(occs[1].payload_units, 0)
        self.assertAlmostEqual(total, 1793, places=1)

    def test_density_zero_csr(self):
        """d=0 with CSR: all fibers are empty, UOP filters them all out.

        With UOP empty fiber filtering, prob_empty(8)=1.0 at d=0,
        so effective_fibers=0 and UOP payload=0.
        """
        occs, total = compute_format_occupancy(
            rank_formats=["UOP", "CP"],
            dimension_sizes=[8, 8],
            density=0.0,
            tensor_size=64,
        )
        self.assertEqual(occs[0].payload_units, 0)  # UOP: all fibers empty
        self.assertEqual(occs[1].metadata_units, 0)  # CP: no nonzeros
        self.assertEqual(total, 0)

    def test_mismatched_lengths_raises(self):
        """rank_formats and dimension_sizes must match length."""
        with self.assertRaises(ValueError):
            compute_format_occupancy(["UOP", "CP"], [8], 0.5, 64)

    def test_single_rank_bitmask(self):
        """Single-rank bitmask (like Buffer A in fig1)."""
        occs, total = compute_format_occupancy(
            rank_formats=["B"],
            dimension_sizes=[128],
            density=0.1015625,
            tensor_size=16384,
        )
        self.assertEqual(occs[0].metadata_units, 128)
        self.assertEqual(total, 128)


class TestFlattenedDimensionOccupancy(unittest.TestCase):
    """Occupancy with flattened dimensions (fiber_shape = product of dims)."""

    def test_rle_occupancy_flattened_fiber(self):
        """RLE with flattened fiber_shape = C*R = 24, density=0.5 -> ennz=12."""
        rle = RLE()
        occ = rle.get_occupancy(fibers=1, fiber_shape=24, expected_nnz_per_fiber=12.0)
        self.assertAlmostEqual(occ.metadata_units, 12.0)

    def test_uop_occupancy_flattened_fiber(self):
        """UOP with flattened fiber_shape = S*F = 96."""
        uop = UOP()
        occ = uop.get_occupancy(fibers=1, fiber_shape=96)
        self.assertEqual(occ.payload_units, 97)  # 1 * (96 + 1)

    def test_bitmask_occupancy_flattened_fiber(self):
        """Bitmask with flattened fiber_shape = C*R = 24."""
        bm = Bitmask()
        occ = bm.get_occupancy(fibers=1, fiber_shape=24)
        self.assertEqual(occ.metadata_units, 24)

    def test_cp_occupancy_flattened_fiber(self):
        """CP with flattened fiber_shape = 96, density=0.1 -> ennz=10."""
        cp = CP()
        occ = cp.get_occupancy(fibers=1, fiber_shape=96, expected_nnz_per_fiber=9.6)
        self.assertEqual(occ.metadata_units, 10)  # ceil(9.6) = 10

    def test_multirank_with_flattened_sizes(self):
        """UOP+RLE with dimension_sizes derived from flattened ranks."""
        # Simulating flattened: rank0=[S,F]->96, rank1=[C]->64
        occs, total = compute_format_occupancy(
            rank_formats=["UOP", "RLE"],
            dimension_sizes=[96, 64],
            density=0.5,
            tensor_size=6144,  # 96 * 64
        )
        # UOP: (0, 97)
        self.assertEqual(occs[0].metadata_units, 0)
        self.assertEqual(occs[0].payload_units, 97)
        # RLE: fibers=96, ennz_per_fiber=32 -> metadata=96*32=3072
        self.assertGreater(occs[1].metadata_units, 0)


if __name__ == "__main__":
    unittest.main()
