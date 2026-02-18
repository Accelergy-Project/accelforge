"""Tests for Phase 3: Sparse frontend specification parsing.

Tests YAML parsing of sparse_optimizations, auto-expansion, and round-trip.
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
    """Test RepresentationFormat parsing and auto-expansion."""

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

    def test_explicit_ranks(self):
        """Explicit per-rank specification."""
        rf = RepresentationFormat(
            name="A",
            ranks=[
                RankFormat(format="UOP", payload_word_bits=0),
                RankFormat(format="B"),
            ],
        )
        ranks = rf.get_rank_formats()
        self.assertEqual(len(ranks), 2)
        self.assertEqual(ranks[0].format, "UOP")
        self.assertEqual(ranks[0].payload_word_bits, 0)
        self.assertEqual(ranks[1].format, "B")
        self.assertIsNone(ranks[1].metadata_word_bits)

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


class TestActionOptimization(unittest.TestCase):
    """Test ActionOptimization parsing."""

    def test_gating(self):
        ao = ActionOptimization(kind="gating", target="A", condition_on=["B"])
        self.assertEqual(ao.kind, "gating")
        self.assertEqual(ao.target, "A")
        self.assertEqual(ao.condition_on, ["B"])

    def test_skipping_multi_condition(self):
        ao = ActionOptimization(
            kind="skipping", target="Z", condition_on=["A", "B"]
        )
        self.assertEqual(ao.kind, "skipping")
        self.assertEqual(len(ao.condition_on), 2)

    def test_position_skipping(self):
        ao = ActionOptimization(
            kind="position_skipping", target="A", condition_on=["B"]
        )
        self.assertEqual(ao.kind, "position_skipping")


class TestComputeOptimization(unittest.TestCase):
    """Test ComputeOptimization parsing."""

    def test_gating(self):
        co = ComputeOptimization(kind="gating", target="Z", condition_on=["A", "B"])
        self.assertEqual(co.kind, "gating")
        self.assertEqual(co.target, "Z")
        self.assertEqual(co.condition_on, ["A", "B"])


class TestSparseTarget(unittest.TestCase):
    """Test SparseTarget structure."""

    def test_buffer_gating(self):
        """Lab 4 buffer_gating.yaml pattern."""
        st = SparseTarget(
            target="Buffer",
            action_optimization=[
                ActionOptimization(kind="gating", target="Z", condition_on=["A", "B"]),
            ],
        )
        self.assertEqual(st.target, "Buffer")
        self.assertEqual(len(st.action_optimization), 1)
        self.assertEqual(len(st.representation_format), 0)
        self.assertEqual(len(st.compute_optimization), 0)

    def test_buffer_with_format_and_saf(self):
        """Buffer with both representation format and SAF."""
        st = SparseTarget(
            target="Buffer",
            representation_format=[
                RepresentationFormat(name="A", format="bitmask"),
                RepresentationFormat(name="B", format="csr"),
            ],
            action_optimization=[
                ActionOptimization(kind="skipping", target="A", condition_on=["B"]),
                ActionOptimization(kind="skipping", target="B", condition_on=["A"]),
            ],
        )
        self.assertEqual(len(st.representation_format), 2)
        self.assertEqual(len(st.action_optimization), 2)

    def test_mac_compute_optimization(self):
        st = SparseTarget(
            target="MAC",
            compute_optimization=[
                ComputeOptimization(
                    kind="gating", target="Z", condition_on=["A", "B"]
                ),
            ],
        )
        self.assertEqual(len(st.compute_optimization), 1)


class TestSparseOptimizations(unittest.TestCase):
    """Test top-level SparseOptimizations."""

    def test_empty_default(self):
        """Default is empty (dense model)."""
        so = SparseOptimizations()
        self.assertEqual(len(so.targets), 0)
        self.assertFalse(so.has_format("Buffer", "A"))

    def test_fig1_bitmask_pattern(self):
        """fig1 bitmask.yaml pattern: format at BackingStorage+Buffer, gating."""
        so = SparseOptimizations(
            targets=[
                SparseTarget(
                    target="BackingStorage",
                    representation_format=[
                        RepresentationFormat(name="A", format="bitmask"),
                        RepresentationFormat(name="B", format="bitmask"),
                    ],
                ),
                SparseTarget(
                    target="Buffer",
                    representation_format=[
                        RepresentationFormat(name="A", format="bitmask"),
                        RepresentationFormat(name="B", format="bitmask"),
                    ],
                    action_optimization=[
                        ActionOptimization(
                            kind="gating", target="A", condition_on=["B"]
                        ),
                        ActionOptimization(
                            kind="gating", target="B", condition_on=["A"]
                        ),
                    ],
                ),
                SparseTarget(
                    target="Reg",
                    action_optimization=[
                        ActionOptimization(
                            kind="gating", target="Z", condition_on=["A", "B"]
                        ),
                    ],
                ),
            ]
        )
        self.assertEqual(len(so.targets), 3)
        self.assertTrue(so.has_format("BackingStorage", "A"))
        self.assertTrue(so.has_format("Buffer", "A"))
        self.assertFalse(so.has_format("Reg", "Z"))

    def test_get_targets_for(self):
        so = SparseOptimizations(
            targets=[
                SparseTarget(target="Buffer", representation_format=[
                    RepresentationFormat(name="A", format="csr"),
                ]),
                SparseTarget(target="Buffer", action_optimization=[
                    ActionOptimization(kind="skipping", target="A", condition_on=["B"]),
                ]),
                SparseTarget(target="MAC"),
            ]
        )
        buffer_targets = so.get_targets_for("Buffer")
        self.assertEqual(len(buffer_targets), 2)
        mac_targets = so.get_targets_for("MAC")
        self.assertEqual(len(mac_targets), 1)

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
        b_fmts = so.get_formats_for("Buffer", "B")
        self.assertEqual(len(b_fmts), 1)
        self.assertEqual(b_fmts[0].format, "csr")
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
        self.assertEqual(safs[0].target, "A")
        self.assertEqual(safs[1].target, "B")

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
        self.assertEqual(cops[0].target, "Z")

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


class TestSpecIntegration(unittest.TestCase):
    """Test that sparse_optimizations is part of Spec."""

    def test_spec_has_sparse_optimizations(self):
        from accelforge.frontend.spec import Spec

        spec = Spec()
        self.assertIsNotNone(spec.sparse_optimizations)
        self.assertEqual(len(spec.sparse_optimizations.targets), 0)

    def test_spec_with_sparse(self):
        from accelforge.frontend.spec import Spec

        spec = Spec(
            sparse_optimizations=SparseOptimizations(
                targets=[
                    SparseTarget(
                        target="Buffer",
                        representation_format=[
                            RepresentationFormat(name="A", format="bitmask"),
                        ],
                    ),
                ]
            )
        )
        self.assertTrue(spec.sparse_optimizations.has_format("Buffer", "A"))


class TestRankFormat(unittest.TestCase):
    """Test RankFormat parsing."""

    def test_simple(self):
        rf = RankFormat(format="UOP")
        self.assertEqual(rf.format, "UOP")
        self.assertIsNone(rf.metadata_word_bits)
        self.assertIsNone(rf.payload_word_bits)

    def test_with_word_bits(self):
        rf = RankFormat(format="CP", metadata_word_bits=14, payload_word_bits=0)
        self.assertEqual(rf.format, "CP")
        self.assertEqual(rf.metadata_word_bits, 14)
        self.assertEqual(rf.payload_word_bits, 0)


if __name__ == "__main__":
    unittest.main()
