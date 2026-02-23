"""Tests for sparse frontend specification parsing.

Tests auto-expansion logic, query API, and edge cases for
RepresentationFormat, SparseOptimizations, and RankFormat.
YAML file loading tests removed — covered by reproduction tests.
"""

import unittest

from accelforge.frontend.sparse import (
    RankFormat,
    RepresentationFormat,
    ActionOptimization,
    ComputeOptimization,
    SparseTarget,
    SparseOptimizations,
)


class TestRepresentationFormat(unittest.TestCase):
    """Test RepresentationFormat auto-expansion."""

    def test_simplified_csr(self):
        """format: csr auto-expands to UOP+CP for 2 ranks."""
        rf = RepresentationFormat(name="A", format="csr")
        ranks = rf.get_rank_formats(num_ranks=2)
        self.assertEqual(len(ranks), 2)
        self.assertEqual(ranks[0].format, "UOP")
        self.assertEqual(ranks[1].format, "CP")

    def test_simplified_bitmask(self):
        """format: bitmask auto-expands to UOP+B for 2 ranks."""
        rf = RepresentationFormat(name="A", format="bitmask")
        ranks = rf.get_rank_formats(num_ranks=2)
        self.assertEqual(len(ranks), 2)
        self.assertEqual(ranks[0].format, "UOP")
        self.assertEqual(ranks[1].format, "B")

    def test_simplified_coo_3_ranks(self):
        """format: coo auto-expands to CP+CP+CP for 3 ranks."""
        rf = RepresentationFormat(name="B", format="coo")
        ranks = rf.get_rank_formats(num_ranks=3)
        self.assertEqual(len(ranks), 3)
        for r in ranks:
            self.assertEqual(r.format, "CP")

    def test_explicit_overrides_format(self):
        """When both format and ranks given, ranks takes precedence."""
        rf = RepresentationFormat(
            name="A",
            format="csr",
            ranks=[RankFormat(format="CP")],
        )
        ranks = rf.get_rank_formats()
        self.assertEqual(len(ranks), 1)
        self.assertEqual(ranks[0].format, "CP")

    def test_no_format_no_ranks(self):
        """No format or ranks -> empty list."""
        rf = RepresentationFormat(name="A")
        ranks = rf.get_rank_formats()
        self.assertEqual(ranks, [])

    def test_format_requires_num_ranks(self):
        """Auto-expand without num_ranks raises."""
        rf = RepresentationFormat(name="A", format="csr")
        with self.assertRaises(ValueError):
            rf.get_rank_formats()


class TestSparseOptimizations(unittest.TestCase):
    """Test query API: has_format, get_formats_for, get_action_optimizations_for."""

    def test_empty_default(self):
        so = SparseOptimizations()
        self.assertEqual(len(so.targets), 0)
        self.assertFalse(so.has_format("Buffer", "A"))

    def test_has_format(self):
        so = SparseOptimizations(
            targets=[
                SparseTarget(
                    target="BackingStorage",
                    representation_format=[
                        RepresentationFormat(name="A", format="bitmask"),
                    ],
                ),
            ]
        )
        self.assertTrue(so.has_format("BackingStorage", "A"))
        self.assertFalse(so.has_format("BackingStorage", "B"))

    def test_get_formats_for(self):
        so = SparseOptimizations(
            targets=[
                SparseTarget(target="Buffer", representation_format=[
                    RepresentationFormat(name="A", format="bitmask"),
                    RepresentationFormat(name="B", format="csr"),
                ]),
            ]
        )
        a_fmts = so.get_formats_for("Buffer", "A")
        self.assertEqual(len(a_fmts), 1)
        self.assertEqual(a_fmts[0].format, "bitmask")
        z_fmts = so.get_formats_for("Buffer", "Z")
        self.assertEqual(len(z_fmts), 0)

    def test_get_action_optimizations_for(self):
        so = SparseOptimizations(
            targets=[
                SparseTarget(target="Buffer", action_optimization=[
                    ActionOptimization(kind="skipping", target="A", condition_on=["B"]),
                    ActionOptimization(kind="skipping", target="B", condition_on=["A"]),
                ]),
            ]
        )
        safs = so.get_action_optimizations_for("Buffer")
        self.assertEqual(len(safs), 2)

    def test_get_compute_optimizations_for(self):
        so = SparseOptimizations(
            targets=[
                SparseTarget(target="MAC", compute_optimization=[
                    ComputeOptimization(kind="gating", target="Z", condition_on=["A", "B"]),
                ]),
            ]
        )
        cops = so.get_compute_optimizations_for("MAC")
        self.assertEqual(len(cops), 1)

    def test_duplicate_targets_merged(self):
        """Multiple entries for same target are logically merged by helpers."""
        so = SparseOptimizations(
            targets=[
                SparseTarget(target="Buffer", representation_format=[
                    RepresentationFormat(name="A", format="csr"),
                ]),
                SparseTarget(target="Buffer", action_optimization=[
                    ActionOptimization(kind="skipping", target="A", condition_on=["B"]),
                ]),
            ]
        )
        fmts = so.get_formats_for("Buffer", "A")
        self.assertEqual(len(fmts), 1)
        safs = so.get_action_optimizations_for("Buffer")
        self.assertEqual(len(safs), 1)


class TestRankFormat(unittest.TestCase):
    """Test RankFormat parsing and flattened_rank_ids."""

    def test_with_word_bits(self):
        rf = RankFormat(format="CP", metadata_word_bits=14, payload_word_bits=0)
        self.assertEqual(rf.metadata_word_bits, 14)

    def test_flattened_rank_ids_parse(self):
        rf = RankFormat(
            format="UOP",
            payload_word_bits=0,
            flattened_rank_ids=[["S", "F"]],
        )
        self.assertEqual(rf.flattened_rank_ids, [["S", "F"]])

    def test_explicit_ranks_with_flattened_ids(self):
        rf = RepresentationFormat(
            name="Inputs",
            ranks=[
                RankFormat(format="UOP", payload_word_bits=4, flattened_rank_ids=[["R"]]),
                RankFormat(format="RLE", metadata_word_bits=4, flattened_rank_ids=[["C"]]),
            ],
        )
        ranks = rf.get_rank_formats()
        self.assertEqual(len(ranks), 2)
        self.assertEqual(ranks[0].flattened_rank_ids, [["R"]])
        self.assertEqual(ranks[1].flattened_rank_ids, [["C"]])


if __name__ == "__main__":
    unittest.main()
