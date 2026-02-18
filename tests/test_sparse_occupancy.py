"""Tests for Phase 4: Compressed format impact on occupancy and accesses.

Validation tests sourced from ARTIFACT_EVALUATION.md fig1 reference outputs
and Lab 4 Part 4 storage capacity sweep.
"""

import math
import unittest

from accelforge.model.sparse import (
    SparseOccupancy,
    FormatAccessCounts,
    compute_sparse_occupancy,
    compute_format_access_counts,
)
from accelforge.model.density_model import HypergeometricDensityModel


# ---------------------------------------------------------------------------
# Sparse occupancy tests
# ---------------------------------------------------------------------------


class TestSparseDataOccupancy(unittest.TestCase):
    """Test data occupancy (expected NNZ) at each level."""

    def test_buffer_a_bitmask(self):
        """Fig1: Buffer A, d=0.1015625, tile=128 -> 13 elements."""
        occ = compute_sparse_occupancy(
            density=0.1015625,
            tensor_size=16384,
            tile_shape=128,
            bits_per_value=8,
        )
        self.assertEqual(occ.data_elements, 13)

    def test_buffer_b_bitmask(self):
        """Fig1: Buffer B, d=0.1015625, tile=128 -> 13 elements."""
        occ = compute_sparse_occupancy(
            density=0.1015625,
            tensor_size=16384,
            tile_shape=128,
            bits_per_value=8,
        )
        self.assertEqual(occ.data_elements, 13)

    def test_backing_storage_a(self):
        """Fig1: BackingStorage A, tile=16384 -> 1664 elements."""
        occ = compute_sparse_occupancy(
            density=0.1015625,
            tensor_size=16384,
            tile_shape=16384,
            bits_per_value=8,
        )
        self.assertEqual(occ.data_elements, 1664)

    def test_dense_tensor_z(self):
        """Dense tensor (Z, d=1.0) -> data_elements = tile_shape."""
        occ = compute_sparse_occupancy(
            density=1.0,
            tensor_size=16384,
            tile_shape=16384,
            bits_per_value=8,
        )
        self.assertEqual(occ.data_elements, 16384)
        self.assertEqual(occ.data_bits, 16384 * 8)

    def test_lab4_data_d02(self):
        """Lab 4 Part 4: d=0.2, N=64, tile=64 -> 13 elements."""
        occ = compute_sparse_occupancy(
            density=0.2, tensor_size=64, tile_shape=64, bits_per_value=8,
        )
        self.assertEqual(occ.data_elements, 13)

    def test_lab4_data_d10(self):
        """Lab 4 Part 4: d=1.0, N=64, tile=64 -> 64 elements."""
        occ = compute_sparse_occupancy(
            density=1.0, tensor_size=64, tile_shape=64, bits_per_value=8,
        )
        self.assertEqual(occ.data_elements, 64)


class TestSparseFormatOccupancy(unittest.TestCase):
    """Test format (metadata+payload) occupancy at each level."""

    def test_buffer_a_bitmask(self):
        """Fig1: Buffer A, B format, fiber=128 -> Rank0: (128, 0)."""
        occ = compute_sparse_occupancy(
            density=0.1015625,
            tensor_size=16384,
            tile_shape=128,
            bits_per_value=8,
            rank_formats=["B"],
            dimension_sizes=[128],
        )
        self.assertEqual(len(occ.rank_occupancies), 1)
        self.assertEqual(occ.rank_occupancies[0].metadata_units, 128)
        self.assertEqual(occ.rank_occupancies[0].payload_units, 0)

    def test_buffer_a_cp(self):
        """Fig1 coord_list: Buffer A, CP format -> Rank0: (13, 0)."""
        occ = compute_sparse_occupancy(
            density=0.1015625,
            tensor_size=16384,
            tile_shape=128,
            bits_per_value=8,
            rank_formats=["CP"],
            dimension_sizes=[128],
        )
        self.assertEqual(occ.rank_occupancies[0].metadata_units, 13)

    def test_backing_storage_a_bitmask(self):
        """Fig1: BackingStorage A, UOP+B -> Rank1:(0,129), Rank0:(16384,0)."""
        occ = compute_sparse_occupancy(
            density=0.1015625,
            tensor_size=16384,
            tile_shape=16384,
            bits_per_value=8,
            rank_formats=["UOP", "B"],
            dimension_sizes=[128, 128],
        )
        self.assertEqual(occ.rank_occupancies[0].metadata_units, 0)
        self.assertEqual(occ.rank_occupancies[0].payload_units, 129)
        self.assertEqual(occ.rank_occupancies[1].metadata_units, 16384)
        self.assertEqual(occ.rank_occupancies[1].payload_units, 0)

    def test_backing_storage_a_format_bits(self):
        """Fig1: Format bits = format_units * bits_per_value (default word size)."""
        occ = compute_sparse_occupancy(
            density=0.1015625,
            tensor_size=16384,
            tile_shape=16384,
            bits_per_value=8,
            rank_formats=["UOP", "B"],
            dimension_sizes=[128, 128],
        )
        # format_units = 129 + 16384 = 16513
        self.assertAlmostEqual(occ.format_units, 16513)
        # format_bits = 16513 * 8 (default bits_per_value for both metadata and payload)
        self.assertAlmostEqual(occ.format_bits, 16513 * 8)

    def test_total_bits(self):
        """Total = data_bits + format_bits."""
        occ = compute_sparse_occupancy(
            density=0.1015625,
            tensor_size=16384,
            tile_shape=128,
            bits_per_value=8,
            rank_formats=["B"],
            dimension_sizes=[128],
        )
        # data = 13 * 8 = 104, format = 128 * 8 = 1024
        self.assertAlmostEqual(occ.total_bits, 104 + 1024)

    def test_no_format_dense(self):
        """No format -> format occupancy is 0."""
        occ = compute_sparse_occupancy(
            density=1.0,
            tensor_size=16384,
            tile_shape=16384,
            bits_per_value=8,
        )
        self.assertAlmostEqual(occ.format_units, 0)
        self.assertAlmostEqual(occ.format_bits, 0)
        self.assertEqual(occ.data_elements, 16384)

    def test_custom_metadata_word_bits(self):
        """Custom metadata_word_bits (like coordinate_list.yaml: 14-bit coords)."""
        occ = compute_sparse_occupancy(
            density=0.1015625,
            tensor_size=16384,
            tile_shape=128,
            bits_per_value=8,
            rank_formats=["CP"],
            dimension_sizes=[128],
            metadata_word_bits=[14],
        )
        # CP metadata = 13 units. At 14 bits each: 13 * 14 = 182 bits
        self.assertAlmostEqual(occ.format_bits, 13 * 14)


class TestLab4StorageSweep(unittest.TestCase):
    """Lab 4 Part 4: Storage capacity sweep with UOP+CP, M=K=8."""

    def _check(self, density, expected_data, expected_format):
        occ = compute_sparse_occupancy(
            density=density,
            tensor_size=64,
            tile_shape=64,
            bits_per_value=8,
            rank_formats=["UOP", "CP"],
            dimension_sizes=[8, 8],
        )
        self.assertEqual(occ.data_elements, expected_data,
                         f"d={density}: data_elements")
        self.assertAlmostEqual(occ.format_units, expected_format,
                               msg=f"d={density}: format_units")

    def test_d02(self):
        """d=0.2: data=13, format=25."""
        self._check(0.2, 13, 25)

    def test_d04(self):
        """d=0.4: data=26, format=41."""
        self._check(0.4, 26, 41)

    def test_d06(self):
        """d=0.6: data=39, format=49."""
        self._check(0.6, 39, 49)

    def test_d08(self):
        """d=0.8: data=52, format=65."""
        self._check(0.8, 52, 65)

    def test_d10(self):
        """d=1.0: data=64, format=73."""
        self._check(1.0, 64, 73)

    def test_breakeven(self):
        """Compression beneficial below ~d=0.4: total < 64 only when d<0.4."""
        occ_02 = compute_sparse_occupancy(
            density=0.2, tensor_size=64, tile_shape=64,
            bits_per_value=8, rank_formats=["UOP", "CP"], dimension_sizes=[8, 8],
        )
        occ_04 = compute_sparse_occupancy(
            density=0.4, tensor_size=64, tile_shape=64,
            bits_per_value=8, rank_formats=["UOP", "CP"], dimension_sizes=[8, 8],
        )
        uncompressed = 64  # dense elements
        # d=0.2: total = 13 + 25 = 38 < 64 -> beneficial
        self.assertLess(occ_02.data_elements + occ_02.format_units, uncompressed)
        # d=0.4: total = 26 + 41 = 67 > 64 -> not beneficial
        self.assertGreater(occ_04.data_elements + occ_04.format_units, uncompressed)


# ---------------------------------------------------------------------------
# Format access count tests
# ---------------------------------------------------------------------------


class TestFormatAccessCounts(unittest.TestCase):
    """Test format (metadata) access counts scaled by algorithmic ratios."""

    def test_buffer_a_bitmask_reads(self):
        """Fig1: Buffer A bitmask, alg_reads=2,097,152, tile=128.
        B metadata=128 per tile. read_ratio=16384 tile reads.
        Format md reads = ceil(128 * 16384) = 2,097,152."""
        fac = compute_format_access_counts(
            rank_formats=["B"],
            dimension_sizes=[128],
            density=0.1015625,
            tensor_size=16384,
            tile_shape=128,
            algorithmic_reads=2097152,
            algorithmic_fills=2097152,
        )
        self.assertEqual(fac.rank_metadata_reads[0], 2097152)
        self.assertEqual(fac.rank_payload_reads[0], 0)

    def test_buffer_a_bitmask_fills(self):
        """Fig1: Buffer A format fills = 2,097,152 (same as reads for A)."""
        fac = compute_format_access_counts(
            rank_formats=["B"],
            dimension_sizes=[128],
            density=0.1015625,
            tensor_size=16384,
            tile_shape=128,
            algorithmic_reads=2097152,
            algorithmic_fills=2097152,
        )
        self.assertEqual(fac.rank_metadata_fills[0], 2097152)

    def test_backing_storage_a_bitmask_reads(self):
        """Fig1: BackingStorage A, UOP+B, alg_reads=2,097,152, tile=16384.
        read_ratio = 128. Rank1 UOP payload=129: reads=ceil(129*128)=16512.
        Rank0 B metadata=16384: reads=ceil(16384*128)=2,097,152."""
        fac = compute_format_access_counts(
            rank_formats=["UOP", "B"],
            dimension_sizes=[128, 128],
            density=0.1015625,
            tensor_size=16384,
            tile_shape=16384,
            algorithmic_reads=2097152,
            algorithmic_fills=16384,
        )
        # Rank 1 (UOP): payload reads = 16512
        self.assertEqual(fac.rank_payload_reads[0], 16512)
        self.assertEqual(fac.rank_metadata_reads[0], 0)
        # Rank 0 (B): metadata reads = 2,097,152
        self.assertEqual(fac.rank_metadata_reads[1], 2097152)
        self.assertEqual(fac.rank_payload_reads[1], 0)

    def test_backing_storage_b_bitmask_reads(self):
        """Fig1: BackingStorage B, UOP+UOP+B, alg_reads=16384, tile=16384.
        read_ratio = 1. Rank2 UOP: payload=129, reads=129.
        Rank1 UOP: payload=16512, reads=16512.
        Rank0 B: metadata depends on innermost dimension."""
        fac = compute_format_access_counts(
            rank_formats=["UOP", "UOP", "B"],
            dimension_sizes=[128, 128, 1],
            density=0.1015625,
            tensor_size=16384,
            tile_shape=16384,
            algorithmic_reads=16384,
            algorithmic_fills=16384,
        )
        # read_ratio = 16384/16384 = 1
        # Rank 2 (UOP): payload=129, reads=ceil(129*1)=129
        self.assertEqual(fac.rank_payload_reads[0], 129)
        # Rank 1 (UOP): payload=16512, reads=ceil(16512*1)=16512
        self.assertEqual(fac.rank_payload_reads[1], 16512)

    def test_total_reads(self):
        """Total reads = sum of all rank metadata + payload reads."""
        fac = compute_format_access_counts(
            rank_formats=["UOP", "B"],
            dimension_sizes=[128, 128],
            density=0.1015625,
            tensor_size=16384,
            tile_shape=16384,
            algorithmic_reads=2097152,
            algorithmic_fills=16384,
        )
        self.assertEqual(fac.total_reads, 16512 + 2097152)

    def test_zero_reads(self):
        """Zero algorithmic reads -> zero format reads."""
        fac = compute_format_access_counts(
            rank_formats=["UOP", "CP"],
            dimension_sizes=[8, 8],
            density=0.5,
            tensor_size=64,
            tile_shape=64,
            algorithmic_reads=0,
            algorithmic_fills=100,
        )
        self.assertEqual(fac.total_reads, 0)

    def test_dense_no_format(self):
        """Dense tensor (no format) -> use empty format list."""
        fac = compute_format_access_counts(
            rank_formats=[],
            dimension_sizes=[],
            density=1.0,
            tensor_size=16384,
            tile_shape=16384,
            algorithmic_reads=2097152,
            algorithmic_fills=16384,
        )
        self.assertEqual(fac.total_reads, 0)
        self.assertEqual(fac.total_fills, 0)


class TestSparseOccupancyDataclass(unittest.TestCase):
    """Test SparseOccupancy dataclass properties."""

    def test_total_bits(self):
        occ = SparseOccupancy(
            data_elements=13,
            data_bits=104,
            format_units=128,
            format_bits=1024,
            rank_occupancies=[],
        )
        self.assertEqual(occ.total_bits, 1128)


class TestFormatAccessCountsDataclass(unittest.TestCase):
    """Test FormatAccessCounts dataclass properties."""

    def test_totals(self):
        fac = FormatAccessCounts(
            rank_metadata_reads=[100, 200],
            rank_payload_reads=[50, 0],
            rank_metadata_fills=[10, 20],
            rank_payload_fills=[5, 0],
        )
        self.assertEqual(fac.total_metadata_reads, 300)
        self.assertEqual(fac.total_payload_reads, 50)
        self.assertEqual(fac.total_reads, 350)
        self.assertEqual(fac.total_metadata_fills, 30)
        self.assertEqual(fac.total_payload_fills, 5)
        self.assertEqual(fac.total_fills, 35)


if __name__ == "__main__":
    unittest.main()
