"""Tests for sparse_adjustment.py — Phase 7 integration of sparse pipeline.

Tests cover:
  - No-op when no sparse optimizations
  - Format compression reduces fills (total_reads_to_parent)
  - SAF reduces child reads/writes
  - SAF propagation to compute
  - Compute classification
  - Action count recomputation
  - Fig1-derived validation scenarios
"""

import unittest
from dataclasses import dataclass, field
from unittest.mock import MagicMock
from typing import Any

from accelforge.model._looptree.reuse.symbolic import (
    SymbolicAnalysisOutput,
    BuffetStats,
)
from accelforge.model._looptree.reuse.symbolic.symbolic import (
    Compute,
    ComputeStats,
)
from accelforge.model._looptree.types import Buffet

from accelforge.frontend.sparse import (
    SparseOptimizations,
    SparseTarget,
    RepresentationFormat,
    ActionOptimization,
    ComputeOptimization,
)

from accelforge.model.sparse_adjustment import (
    apply_sparse_adjustments,
    _recompute_action_counts,
    _get_tensor_size,
)


def make_mock_job(einsum_name="E0"):
    """Create a minimal mock Job."""
    job = MagicMock()
    job.einsum_name = einsum_name
    return job


def make_mock_spec(
    sparse_opts=None,
    tensor_accesses=None,
    arch_components=None,
    rank_sizes=None,
):
    """Create a minimal mock Spec.

    Parameters
    ----------
    sparse_opts : SparseOptimizations or None
    tensor_accesses : list of dicts with keys: name, density, output, bits_per_value
    arch_components : dict of component_name -> dict(bits_per_value_scale, read_bpa, write_bpa)
    rank_sizes : dict of rank_name -> size (for Einsum.rank_sizes)
    """
    spec = MagicMock()

    if sparse_opts is None:
        sparse_opts = SparseOptimizations()
    spec.sparse_optimizations = sparse_opts

    # Build tensor access mocks
    ta_mocks = []
    if tensor_accesses:
        for ta_info in tensor_accesses:
            ta = MagicMock()
            ta.name = ta_info["name"]
            ta.density = ta_info.get("density", None)
            ta.output = ta_info.get("output", False)
            ta.bits_per_value = ta_info.get("bits_per_value", 8)
            ta.projection = ta_info.get("projection", {})
            ta_mocks.append(ta)

    einsum = MagicMock()
    einsum.tensor_accesses = ta_mocks
    einsum.rank_sizes = rank_sizes or {}

    spec.workload.einsums = {"E0": einsum}

    # Build arch components
    if arch_components:
        def find_component(name):
            if name not in arch_components:
                raise ValueError(f"Component {name} not found")
            info = arch_components[name]

            comp = MagicMock()
            comp.name = name

            # bits_per_value_scale: dict of tensor -> scale
            bpv_scale = info.get("bits_per_value_scale", {})
            comp.bits_per_value_scale = bpv_scale

            # Actions
            read_action = MagicMock()
            read_action.bits_per_action = info.get("read_bpa", 8)
            write_action = MagicMock()
            write_action.bits_per_action = info.get("write_bpa", 8)

            comp.actions = {"read": read_action, "write": write_action}

            # Type checks
            from accelforge.frontend import arch
            comp.__class__ = arch.Memory

            return comp

        spec.arch.find = find_component

    return spec


class TestNoSparseOptimizations(unittest.TestCase):
    """When no sparse optimizations are specified, apply_sparse_adjustments is a no-op."""

    def test_no_targets_is_noop(self):
        """Empty sparse_optimizations should not modify any stats."""
        reuse = SymbolicAnalysisOutput()
        buffet = Buffet("A", "E0", "Buffer")
        stats = BuffetStats()
        stats.total_reads_to_parent = 1000
        stats.total_write_actions = 500
        stats.total_read_actions = 200
        reuse.buffet_stats[buffet] = stats

        spec = make_mock_spec()
        job = make_mock_job()

        apply_sparse_adjustments(reuse, spec, job)

        self.assertEqual(reuse.buffet_stats[buffet].total_reads_to_parent, 1000)
        self.assertEqual(reuse.buffet_stats[buffet].total_write_actions, 500)
        self.assertEqual(reuse.buffet_stats[buffet].total_read_actions, 200)


class TestFormatCompression(unittest.TestCase):
    """Format compression reduces total_reads_to_parent (fills) by density."""

    def test_compression_reduces_fills(self):
        """Buffer A with bitmask format at density 0.1015625 → fills compressed."""
        density = 0.1015625  # 13/128

        sparse_opts = SparseOptimizations(
            targets=[
                SparseTarget(
                    target="Buffer",
                    representation_format=[
                        RepresentationFormat(name="A", format="bitmask"),
                    ],
                )
            ]
        )

        reuse = SymbolicAnalysisOutput()
        buffet_a = Buffet("A", "E0", "Buffer")
        stats_a = BuffetStats()
        stats_a.total_reads_to_parent = 2_097_152
        stats_a.max_per_parent_reads_to_parent = 2_097_152
        stats_a.max_occupancy = 128 * 8  # 128 elements × 8 bits
        reuse.buffet_stats[buffet_a] = stats_a

        # Need arch component for action recomputation
        arch_comps = {
            "Buffer": {
                "bits_per_value_scale": {"A": 1},
                "read_bpa": 8,
                "write_bpa": 8,
            }
        }

        spec = make_mock_spec(
            sparse_opts=sparse_opts,
            tensor_accesses=[
                {"name": "A", "density": density, "output": False, "bits_per_value": 8},
            ],
            arch_components=arch_comps,
        )
        job = make_mock_job()

        apply_sparse_adjustments(reuse, spec, job)

        # apply_format_compression(2097152, 0.1015625)
        # = 2097152 - floor(2097152 * 0.8984375) = 2097152 - 1884160 = 212992
        self.assertEqual(stats_a.total_reads_to_parent, 212_992)
        self.assertEqual(stats_a.max_per_parent_reads_to_parent, 212_992)

    def test_compression_reduces_occupancy(self):
        """max_occupancy is also compressed."""
        density = 0.5
        sparse_opts = SparseOptimizations(
            targets=[
                SparseTarget(
                    target="Buffer",
                    representation_format=[
                        RepresentationFormat(name="A", format="bitmask"),
                    ],
                )
            ]
        )

        reuse = SymbolicAnalysisOutput()
        buffet_a = Buffet("A", "E0", "Buffer")
        stats_a = BuffetStats()
        stats_a.total_reads_to_parent = 1000
        stats_a.max_occupancy = 200
        reuse.buffet_stats[buffet_a] = stats_a

        arch_comps = {
            "Buffer": {
                "bits_per_value_scale": {"A": 1},
                "read_bpa": 8,
                "write_bpa": 8,
            }
        }
        spec = make_mock_spec(
            sparse_opts=sparse_opts,
            tensor_accesses=[
                {"name": "A", "density": density, "output": False, "bits_per_value": 8},
            ],
            arch_components=arch_comps,
        )
        job = make_mock_job()

        apply_sparse_adjustments(reuse, spec, job)

        # apply_format_compression(200, 0.5) = 200 - floor(200*0.5) = 100
        self.assertEqual(stats_a.max_occupancy, 100)

    def test_output_tensor_compresses_drains(self):
        """Output tensor with format → both fills and drains compressed."""
        density = 0.5
        sparse_opts = SparseOptimizations(
            targets=[
                SparseTarget(
                    target="Buffer",
                    representation_format=[
                        RepresentationFormat(name="Z", format="csr"),
                    ],
                )
            ]
        )

        reuse = SymbolicAnalysisOutput()
        buffet_z = Buffet("Z", "E0", "Buffer")
        stats_z = BuffetStats()
        stats_z.total_reads_to_parent = 1000
        stats_z.total_writes_to_parent = 1000
        stats_z.max_occupancy = 100
        reuse.buffet_stats[buffet_z] = stats_z

        arch_comps = {
            "Buffer": {
                "bits_per_value_scale": {"Z": 1},
                "read_bpa": 8,
                "write_bpa": 8,
            }
        }
        spec = make_mock_spec(
            sparse_opts=sparse_opts,
            tensor_accesses=[
                {"name": "Z", "density": density, "output": True, "bits_per_value": 8},
            ],
            arch_components=arch_comps,
        )
        job = make_mock_job()

        apply_sparse_adjustments(reuse, spec, job)

        self.assertEqual(stats_z.total_reads_to_parent, 500)
        self.assertEqual(stats_z.total_writes_to_parent, 500)

    def test_dense_tensor_no_compression(self):
        """Tensor with density=1.0 is not compressed even with format."""
        sparse_opts = SparseOptimizations(
            targets=[
                SparseTarget(
                    target="Buffer",
                    representation_format=[
                        RepresentationFormat(name="A", format="bitmask"),
                    ],
                )
            ]
        )

        reuse = SymbolicAnalysisOutput()
        buffet_a = Buffet("A", "E0", "Buffer")
        stats_a = BuffetStats()
        stats_a.total_reads_to_parent = 1000
        reuse.buffet_stats[buffet_a] = stats_a

        arch_comps = {
            "Buffer": {
                "bits_per_value_scale": {"A": 1},
                "read_bpa": 8,
                "write_bpa": 8,
            }
        }
        spec = make_mock_spec(
            sparse_opts=sparse_opts,
            tensor_accesses=[
                {"name": "A", "density": 1.0, "output": False, "bits_per_value": 8},
            ],
            arch_components=arch_comps,
        )
        job = make_mock_job()

        apply_sparse_adjustments(reuse, spec, job)

        self.assertEqual(stats_a.total_reads_to_parent, 1000)

    def test_no_format_no_compression(self):
        """Tensor without representation format is not compressed."""
        sparse_opts = SparseOptimizations(
            targets=[
                SparseTarget(
                    target="Buffer",
                    # No representation_format
                )
            ]
        )

        reuse = SymbolicAnalysisOutput()
        buffet_a = Buffet("A", "E0", "Buffer")
        stats_a = BuffetStats()
        stats_a.total_reads_to_parent = 1000
        reuse.buffet_stats[buffet_a] = stats_a

        arch_comps = {
            "Buffer": {
                "bits_per_value_scale": {"A": 1},
                "read_bpa": 8,
                "write_bpa": 8,
            }
        }
        spec = make_mock_spec(
            sparse_opts=sparse_opts,
            tensor_accesses=[
                {"name": "A", "density": 0.5, "output": False, "bits_per_value": 8},
            ],
            arch_components=arch_comps,
        )
        job = make_mock_job()

        apply_sparse_adjustments(reuse, spec, job)

        self.assertEqual(stats_a.total_reads_to_parent, 1000)


class TestSAF(unittest.TestCase):
    """SAF reduces child reads/writes via action optimization."""

    def _make_two_level_reuse(self, parent_level="Buffer", child_level="Reg"):
        """Create a reuse with parent buffet having a child buffet for same tensor."""
        reuse = SymbolicAnalysisOutput()

        # Child buffet (inserted first = deeper level)
        child_buffet = Buffet("Z", "E0", child_level)
        child_stats = BuffetStats()
        child_stats.total_reads_to_parent = 2_080_768  # Fig1 Reg Z reads
        child_stats.max_per_parent_reads_to_parent = 2_080_768
        child_stats.total_writes_to_parent = 16_384
        child_stats.max_per_parent_writes_to_parent = 16_384
        reuse.buffet_stats[child_buffet] = child_stats

        # Parent buffet (inserted after = shallower level)
        parent_buffet = Buffet("Z", "E0", parent_level)
        parent_stats = BuffetStats()
        parent_stats.total_reads_to_parent = 16_384
        parent_stats.max_per_parent_reads_to_parent = 16_384
        reuse.buffet_stats[parent_buffet] = parent_stats

        return reuse, child_stats, parent_stats

    def test_saf_reduces_child_reads(self):
        """SAF at Buffer for Z (gating on A,B) reduces child's reads."""
        density_a = 0.1015625
        density_b = 0.1015625
        # P(effectual) = dA * dB = 0.010319...
        # optimization_prob = 1 - 0.010319... = 0.989680...

        sparse_opts = SparseOptimizations(
            targets=[
                SparseTarget(
                    target="Buffer",
                    action_optimization=[
                        ActionOptimization(
                            kind="gating",
                            target="Z",
                            condition_on=["A", "B"],
                        ),
                    ],
                )
            ]
        )

        reuse, child_stats, _ = self._make_two_level_reuse()

        # Also need A and B buffets for tile shape lookup
        buffet_a = Buffet("A", "E0", "Buffer")
        stats_a = BuffetStats()
        stats_a.max_occupancy = 128 * 8  # 128 elements × 8 bits
        reuse.buffet_stats[buffet_a] = stats_a

        buffet_b = Buffet("B", "E0", "Buffer")
        stats_b = BuffetStats()
        stats_b.max_occupancy = 128 * 8
        reuse.buffet_stats[buffet_b] = stats_b

        arch_comps = {
            "Buffer": {
                "bits_per_value_scale": {"Z": 1, "A": 1, "B": 1},
                "read_bpa": 8,
                "write_bpa": 8,
            },
            "Reg": {
                "bits_per_value_scale": {"Z": 1},
                "read_bpa": 8,
                "write_bpa": 8,
            },
        }

        spec = make_mock_spec(
            sparse_opts=sparse_opts,
            tensor_accesses=[
                {"name": "A", "density": density_a, "output": False, "bits_per_value": 8},
                {"name": "B", "density": density_b, "output": False, "bits_per_value": 8},
                {"name": "Z", "density": None, "output": True, "bits_per_value": 8},
            ],
            arch_components=arch_comps,
            rank_sizes={"M": 128, "K": 128, "N": 128},
        )
        job = make_mock_job()

        apply_sparse_adjustments(reuse, spec, job)

        # Z is output (read-write) → reads use ceil rounding
        # optimization_prob ≈ 0.98968505859375
        # ceil(2080768 * 0.98968505859375) = ceil(2059304.53...) = 2059305
        # actual = 2080768 - 2059305 = 21463
        self.assertEqual(child_stats.total_reads_to_parent, 21_463)

    def test_saf_reduces_output_writes(self):
        """SAF on output tensor also reduces child's writes (updates)."""
        sparse_opts = SparseOptimizations(
            targets=[
                SparseTarget(
                    target="Buffer",
                    action_optimization=[
                        ActionOptimization(
                            kind="gating",
                            target="Z",
                            condition_on=["A"],
                        ),
                    ],
                )
            ]
        )

        reuse, child_stats, _ = self._make_two_level_reuse()

        buffet_a = Buffet("A", "E0", "Buffer")
        stats_a = BuffetStats()
        stats_a.max_occupancy = 8  # 1 element × 8 bits (scalar)
        reuse.buffet_stats[buffet_a] = stats_a

        arch_comps = {
            "Buffer": {
                "bits_per_value_scale": {"Z": 1, "A": 1},
                "read_bpa": 8,
                "write_bpa": 8,
            },
            "Reg": {
                "bits_per_value_scale": {"Z": 1},
                "read_bpa": 8,
                "write_bpa": 8,
            },
        }

        spec = make_mock_spec(
            sparse_opts=sparse_opts,
            tensor_accesses=[
                {"name": "A", "density": 0.5, "output": False, "bits_per_value": 8},
                {"name": "Z", "density": None, "output": True, "bits_per_value": 8},
            ],
            arch_components=arch_comps,
        )
        job = make_mock_job()

        original_writes = child_stats.total_writes_to_parent

        apply_sparse_adjustments(reuse, spec, job)

        # prob = 1 - 0.5 = 0.5
        # floor(16384 * 0.5) = 8192
        # actual = 16384 - 8192 = 8192
        self.assertEqual(child_stats.total_writes_to_parent, 8192)

    def test_saf_no_child_is_noop(self):
        """SAF at Buffer for input tensor A (no child) doesn't crash."""
        sparse_opts = SparseOptimizations(
            targets=[
                SparseTarget(
                    target="Buffer",
                    representation_format=[
                        RepresentationFormat(name="A", format="bitmask"),
                    ],
                    action_optimization=[
                        ActionOptimization(
                            kind="gating",
                            target="A",
                            condition_on=["B"],
                        ),
                    ],
                )
            ]
        )

        reuse = SymbolicAnalysisOutput()
        # Only Buffer A, no child for A
        buffet_a = Buffet("A", "E0", "Buffer")
        stats_a = BuffetStats()
        stats_a.total_reads_to_parent = 2_097_152
        stats_a.max_occupancy = 128 * 8
        reuse.buffet_stats[buffet_a] = stats_a

        # B buffet for condition_on tile shape
        buffet_b = Buffet("B", "E0", "Buffer")
        stats_b = BuffetStats()
        stats_b.max_occupancy = 128 * 8
        reuse.buffet_stats[buffet_b] = stats_b

        arch_comps = {
            "Buffer": {
                "bits_per_value_scale": {"A": 1, "B": 1},
                "read_bpa": 8,
                "write_bpa": 8,
            }
        }

        spec = make_mock_spec(
            sparse_opts=sparse_opts,
            tensor_accesses=[
                {"name": "A", "density": 0.1015625, "output": False, "bits_per_value": 8},
                {"name": "B", "density": 0.1015625, "output": False, "bits_per_value": 8},
            ],
            arch_components=arch_comps,
        )
        job = make_mock_job()

        # Should not raise even though A has no child
        apply_sparse_adjustments(reuse, spec, job)

        # Fills are compressed (format compression)
        self.assertEqual(stats_a.total_reads_to_parent, 212_992)


class TestSAFPropagationToCompute(unittest.TestCase):
    """SAF probabilities propagate to reduce compute operations."""

    def test_single_saf_propagation(self):
        """Single SAF (A gated on B) propagates to reduce compute."""
        sparse_opts = SparseOptimizations(
            targets=[
                SparseTarget(
                    target="Buffer",
                    action_optimization=[
                        ActionOptimization(
                            kind="gating",
                            target="A",
                            condition_on=["B"],
                        ),
                    ],
                )
            ]
        )

        reuse = SymbolicAnalysisOutput()
        # Buffets
        buffet_a = Buffet("A", "E0", "Buffer")
        stats_a = BuffetStats()
        stats_a.max_occupancy = 8  # scalar
        reuse.buffet_stats[buffet_a] = stats_a

        buffet_b = Buffet("B", "E0", "Buffer")
        stats_b = BuffetStats()
        stats_b.max_occupancy = 8
        reuse.buffet_stats[buffet_b] = stats_b

        # Compute
        compute_key = Compute("E0", "MAC")
        compute_stats = ComputeStats(total_ops=2_097_152, max_per_unit_ops=2_097_152)
        reuse.compute_stats[compute_key] = compute_stats

        arch_comps = {
            "Buffer": {
                "bits_per_value_scale": {"A": 1, "B": 1},
                "read_bpa": 8,
                "write_bpa": 8,
            }
        }

        spec = make_mock_spec(
            sparse_opts=sparse_opts,
            tensor_accesses=[
                {"name": "A", "density": 0.1015625, "output": False, "bits_per_value": 8},
                {"name": "B", "density": 0.1015625, "output": False, "bits_per_value": 8},
            ],
            arch_components=arch_comps,
        )
        job = make_mock_job()

        apply_sparse_adjustments(reuse, spec, job)

        # prob = 1 - density_B = 1 - 0.1015625 = 0.8984375
        # propagate_saf_reduction(2097152, 0.8984375)
        # = 2097152 - floor(2097152 * 0.8984375) = 2097152 - 1884160 = 212992
        self.assertEqual(compute_stats.total_ops, 212_992)

    def test_two_saf_cascading_propagation(self):
        """Two SAFs (A gated on B, B gated on A) cascade to reduce compute."""
        sparse_opts = SparseOptimizations(
            targets=[
                SparseTarget(
                    target="Buffer",
                    action_optimization=[
                        ActionOptimization(
                            kind="gating",
                            target="A",
                            condition_on=["B"],
                        ),
                        ActionOptimization(
                            kind="gating",
                            target="B",
                            condition_on=["A"],
                        ),
                    ],
                )
            ]
        )

        reuse = SymbolicAnalysisOutput()
        buffet_a = Buffet("A", "E0", "Buffer")
        stats_a = BuffetStats()
        stats_a.max_occupancy = 8
        reuse.buffet_stats[buffet_a] = stats_a

        buffet_b = Buffet("B", "E0", "Buffer")
        stats_b = BuffetStats()
        stats_b.max_occupancy = 8
        reuse.buffet_stats[buffet_b] = stats_b

        compute_key = Compute("E0", "MAC")
        compute_stats = ComputeStats(total_ops=2_097_152, max_per_unit_ops=2_097_152)
        reuse.compute_stats[compute_key] = compute_stats

        arch_comps = {
            "Buffer": {
                "bits_per_value_scale": {"A": 1, "B": 1},
                "read_bpa": 8,
                "write_bpa": 8,
            }
        }

        spec = make_mock_spec(
            sparse_opts=sparse_opts,
            tensor_accesses=[
                {"name": "A", "density": 0.1015625, "output": False, "bits_per_value": 8},
                {"name": "B", "density": 0.1015625, "output": False, "bits_per_value": 8},
            ],
            arch_components=arch_comps,
        )
        job = make_mock_job()

        apply_sparse_adjustments(reuse, spec, job)

        # First SAF: 2097152 → 212992 (prob = 0.8984375)
        # Second SAF: 212992 → 21632 (prob = 0.8984375)
        self.assertEqual(compute_stats.total_ops, 21_632)


class TestComputeClassification(unittest.TestCase):
    """Compute classification replaces total_ops with effectual computes."""

    def test_gating_classification(self):
        """With compute gating, total_ops = random_compute only."""
        sparse_opts = SparseOptimizations(
            targets=[
                SparseTarget(
                    target="MAC",
                    compute_optimization=[
                        ComputeOptimization(
                            kind="gating",
                            target="Z",
                            condition_on=["A", "B"],
                        ),
                    ],
                )
            ]
        )

        reuse = SymbolicAnalysisOutput()
        compute_key = Compute("E0", "MAC")
        # Pre-propagated: total_ops already reduced by SAF to 21632
        compute_stats = ComputeStats(total_ops=21_632, max_per_unit_ops=21_632)
        reuse.compute_stats[compute_key] = compute_stats

        spec = make_mock_spec(
            sparse_opts=sparse_opts,
            tensor_accesses=[
                {"name": "A", "density": 0.1015625, "output": False, "bits_per_value": 8},
                {"name": "B", "density": 0.1015625, "output": False, "bits_per_value": 8},
                {"name": "Z", "density": None, "output": True, "bits_per_value": 8},
            ],
        )
        job = make_mock_job()

        apply_sparse_adjustments(reuse, spec, job)

        # classify_compute(21632, [0.1015625, 0.1015625], "gating")
        # effectual = effectual_operations(21632, 0.1015625, 0.1015625)
        # = 21632 - floor(21632 * (1 - 0.1015625*0.1015625))
        # This gives a smaller number. But for fig1, compute classification
        # at the MAC happens AFTER propagation, so total_ops is already
        # the propagated count. The classification then further splits it.
        # For this test we just verify total_ops is reduced.
        self.assertLess(compute_stats.total_ops, 21_632)

    def test_lab4_gating(self):
        """Lab4: 512 computes, d_A=0.25, d_B=0.5 → 64 effectual."""
        sparse_opts = SparseOptimizations(
            targets=[
                SparseTarget(
                    target="MAC",
                    compute_optimization=[
                        ComputeOptimization(
                            kind="gating",
                            target="Z",
                            condition_on=["A", "B"],
                        ),
                    ],
                )
            ]
        )

        reuse = SymbolicAnalysisOutput()
        compute_key = Compute("E0", "MAC")
        compute_stats = ComputeStats(total_ops=512, max_per_unit_ops=512)
        reuse.compute_stats[compute_key] = compute_stats

        spec = make_mock_spec(
            sparse_opts=sparse_opts,
            tensor_accesses=[
                {"name": "A", "density": 0.25, "output": False, "bits_per_value": 8},
                {"name": "B", "density": 0.5, "output": False, "bits_per_value": 8},
                {"name": "Z", "density": None, "output": True, "bits_per_value": 8},
            ],
        )
        job = make_mock_job()

        apply_sparse_adjustments(reuse, spec, job)

        # effectual_operations(512, 0.25, 0.5) = 512 - floor(512 * (1 - 0.125)) = 64
        self.assertEqual(compute_stats.total_ops, 64)


class TestActionRecomputation(unittest.TestCase):
    """Action counts are correctly recomputed after element count modifications."""

    def test_write_actions_from_fills(self):
        """write_actions = total_reads_to_parent * write_scale."""
        reuse = SymbolicAnalysisOutput()
        buffet = Buffet("A", "E0", "Buffer")
        stats = BuffetStats()
        stats.total_reads_to_parent = 500  # After compression
        stats.max_per_parent_reads_to_parent = 500
        # Set stale action counts (pre-compression values)
        stats.total_write_actions = 9999
        stats.total_read_actions = 9999
        reuse.buffet_stats[buffet] = stats

        arch_comps = {
            "Buffer": {
                "bits_per_value_scale": {"A": 1},
                "read_bpa": 8,
                "write_bpa": 8,
            }
        }
        spec = make_mock_spec(
            tensor_accesses=[
                {"name": "A", "density": 0.5, "output": False, "bits_per_value": 8},
            ],
            arch_components=arch_comps,
        )
        job = make_mock_job()

        _recompute_action_counts(reuse, spec, job, set())

        # write_scale = 8 / 8 = 1, read_scale = 8 / 8 = 1
        # write_actions = total_reads_to_parent * write_scale = 500 * 1 = 500
        # read_actions = total_writes_to_parent * read_scale = 0 (input tensor)
        self.assertEqual(stats.total_write_actions, 500)
        self.assertEqual(stats.total_read_actions, 0)

    def test_read_actions_from_child(self):
        """read_actions includes child.total_reads_to_parent * read_scale."""
        reuse = SymbolicAnalysisOutput()

        # Child (deeper level)
        child_buffet = Buffet("A", "E0", "Reg")
        child_stats = BuffetStats()
        child_stats.total_reads_to_parent = 200
        child_stats.max_per_parent_reads_to_parent = 200
        reuse.buffet_stats[child_buffet] = child_stats

        # Parent
        parent_buffet = Buffet("A", "E0", "Buffer")
        parent_stats = BuffetStats()
        parent_stats.total_reads_to_parent = 100
        parent_stats.total_write_actions = 9999  # stale
        parent_stats.total_read_actions = 9999
        reuse.buffet_stats[parent_buffet] = parent_stats

        arch_comps = {
            "Buffer": {
                "bits_per_value_scale": {"A": 1},
                "read_bpa": 8,
                "write_bpa": 8,
            },
            "Reg": {
                "bits_per_value_scale": {"A": 1},
                "read_bpa": 8,
                "write_bpa": 8,
            },
        }
        spec = make_mock_spec(
            tensor_accesses=[
                {"name": "A", "density": 0.5, "output": False, "bits_per_value": 8},
            ],
            arch_components=arch_comps,
        )
        job = make_mock_job()

        _recompute_action_counts(reuse, spec, job, set())

        # Buffer read_actions = child.total_reads_to_parent * read_scale = 200 * 1 = 200
        # Buffer write_actions = parent.total_reads_to_parent * write_scale = 100 * 1 = 100
        self.assertEqual(parent_stats.total_read_actions, 200)
        self.assertEqual(parent_stats.total_write_actions, 100)


class TestEndToEnd(unittest.TestCase):
    """Combined compression + SAF + compute scenarios."""

    def test_fig1_buffer_a_compression(self):
        """Fig1 Buffer A: compression reduces fills from 2097152 to 212992."""
        density = 0.1015625

        sparse_opts = SparseOptimizations(
            targets=[
                SparseTarget(
                    target="Buffer",
                    representation_format=[
                        RepresentationFormat(name="A", format="bitmask"),
                    ],
                    action_optimization=[
                        ActionOptimization(
                            kind="gating",
                            target="A",
                            condition_on=["B"],
                        ),
                    ],
                ),
                SparseTarget(
                    target="BackingStorage",
                    representation_format=[
                        RepresentationFormat(name="A", format="bitmask"),
                    ],
                ),
            ]
        )

        reuse = SymbolicAnalysisOutput()

        # Buffer A (no child for A at this level)
        buffet_a_buf = Buffet("A", "E0", "Buffer")
        stats_a_buf = BuffetStats()
        stats_a_buf.total_reads_to_parent = 2_097_152
        stats_a_buf.max_per_parent_reads_to_parent = 2_097_152
        stats_a_buf.max_occupancy = 128 * 8
        reuse.buffet_stats[buffet_a_buf] = stats_a_buf

        # BackingStorage A
        buffet_a_bs = Buffet("A", "E0", "BackingStorage")
        stats_a_bs = BuffetStats()
        stats_a_bs.total_reads_to_parent = 0  # top level
        stats_a_bs.max_occupancy = 16384 * 8
        reuse.buffet_stats[buffet_a_bs] = stats_a_bs

        # B buffet for condition_on
        buffet_b = Buffet("B", "E0", "Buffer")
        stats_b = BuffetStats()
        stats_b.max_occupancy = 128 * 8
        reuse.buffet_stats[buffet_b] = stats_b

        arch_comps = {
            "Buffer": {
                "bits_per_value_scale": {"A": 1, "B": 1},
                "read_bpa": 8,
                "write_bpa": 8,
            },
            "BackingStorage": {
                "bits_per_value_scale": {"A": 1},
                "read_bpa": 8,
                "write_bpa": 8,
            },
        }

        spec = make_mock_spec(
            sparse_opts=sparse_opts,
            tensor_accesses=[
                {"name": "A", "density": density, "output": False, "bits_per_value": 8},
                {"name": "B", "density": density, "output": False, "bits_per_value": 8},
            ],
            arch_components=arch_comps,
        )
        job = make_mock_job()

        apply_sparse_adjustments(reuse, spec, job)

        # Buffer A fills compressed: 2097152 → 212992
        self.assertEqual(stats_a_buf.total_reads_to_parent, 212_992)

        # BackingStorage A: write_actions = Buffer A's fills → reflected in
        # BackingStorage's read_actions from child (Buffer A)
        # After recomputation:
        # BackingStorage read_actions = child (Buffer A).total_reads_to_parent * read_scale
        # = 212992 * (8/8) = 212992
        self.assertEqual(stats_a_bs.total_read_actions, 212_992)


class TestGetTensorSize(unittest.TestCase):
    """_get_tensor_size extracts correct size from einsum rank_sizes."""

    def test_basic_tensor_size(self):
        einsum = MagicMock()
        ta = MagicMock()
        ta.name = "A"
        ta.projection = {"M": "m", "K": "k"}
        einsum.tensor_accesses = [ta]
        einsum.rank_sizes = {"M": 128, "K": 128, "N": 128}

        size = _get_tensor_size(einsum, "A")
        self.assertEqual(size, 128 * 128)

    def test_unknown_tensor_returns_1(self):
        einsum = MagicMock()
        einsum.tensor_accesses = []
        size = _get_tensor_size(einsum, "X")
        self.assertEqual(size, 1)


# ===========================================================================
# Missing tests per IMPLEMENTATION_PLAN.md Phase 7
# ===========================================================================


class TestCoordinateListSkipping(unittest.TestCase):
    """Coordinate_list (skipping) variant tests.

    Per IMPLEMENTATION_PLAN.md Phase 7: mirror bitmask/gating tests with
    skipping SAF. The numeric results are identical (skipping vs gating
    only affects labeling), but we verify the pipeline handles both kinds.
    """

    def test_skipping_reduces_child_reads(self):
        """SAF with kind=skipping reduces child reads identically to gating."""
        sparse_opts = SparseOptimizations(
            targets=[
                SparseTarget(
                    target="Buffer",
                    action_optimization=[
                        ActionOptimization(
                            kind="skipping",
                            target="Z",
                            condition_on=["A", "B"],
                        ),
                    ],
                )
            ]
        )

        reuse = SymbolicAnalysisOutput()

        # Child (Reg Z)
        child_buffet = Buffet("Z", "E0", "Reg")
        child_stats = BuffetStats()
        child_stats.total_reads_to_parent = 2_080_768
        child_stats.max_per_parent_reads_to_parent = 2_080_768
        child_stats.total_writes_to_parent = 16_384
        child_stats.max_per_parent_writes_to_parent = 16_384
        reuse.buffet_stats[child_buffet] = child_stats

        # Parent (Buffer Z)
        parent_buffet = Buffet("Z", "E0", "Buffer")
        parent_stats = BuffetStats()
        parent_stats.total_reads_to_parent = 16_384
        reuse.buffet_stats[parent_buffet] = parent_stats

        # Condition-on tensors
        for name in ("A", "B"):
            b = Buffet(name, "E0", "Buffer")
            s = BuffetStats()
            s.max_occupancy = 128 * 8
            reuse.buffet_stats[b] = s

        arch_comps = {
            "Buffer": {
                "bits_per_value_scale": {"Z": 1, "A": 1, "B": 1},
                "read_bpa": 8,
                "write_bpa": 8,
            },
            "Reg": {
                "bits_per_value_scale": {"Z": 1},
                "read_bpa": 8,
                "write_bpa": 8,
            },
        }

        spec = make_mock_spec(
            sparse_opts=sparse_opts,
            tensor_accesses=[
                {"name": "A", "density": 0.1015625, "output": False, "bits_per_value": 8},
                {"name": "B", "density": 0.1015625, "output": False, "bits_per_value": 8},
                {"name": "Z", "density": None, "output": True, "bits_per_value": 8},
            ],
            arch_components=arch_comps,
            rank_sizes={"M": 128, "K": 128, "N": 128},
        )
        job = make_mock_job()

        apply_sparse_adjustments(reuse, spec, job)

        # Same result as gating: ceil rounding for read-write
        self.assertEqual(child_stats.total_reads_to_parent, 21_463)

    def test_skipping_propagates_to_compute(self):
        """Skipping SAF propagates to compute identically to gating."""
        sparse_opts = SparseOptimizations(
            targets=[
                SparseTarget(
                    target="Buffer",
                    action_optimization=[
                        ActionOptimization(
                            kind="skipping",
                            target="A",
                            condition_on=["B"],
                        ),
                        ActionOptimization(
                            kind="skipping",
                            target="B",
                            condition_on=["A"],
                        ),
                    ],
                )
            ]
        )

        reuse = SymbolicAnalysisOutput()
        for name in ("A", "B"):
            b = Buffet(name, "E0", "Buffer")
            s = BuffetStats()
            s.max_occupancy = 8  # scalar
            reuse.buffet_stats[b] = s

        from accelforge.model._looptree.reuse.symbolic.symbolic import Compute, ComputeStats
        compute_key = Compute("E0", "MAC")
        compute_stats = ComputeStats(total_ops=2_097_152, max_per_unit_ops=2_097_152)
        reuse.compute_stats[compute_key] = compute_stats

        arch_comps = {
            "Buffer": {
                "bits_per_value_scale": {"A": 1, "B": 1},
                "read_bpa": 8,
                "write_bpa": 8,
            }
        }

        spec = make_mock_spec(
            sparse_opts=sparse_opts,
            tensor_accesses=[
                {"name": "A", "density": 0.1015625, "output": False, "bits_per_value": 8},
                {"name": "B", "density": 0.1015625, "output": False, "bits_per_value": 8},
            ],
            arch_components=arch_comps,
        )
        job = make_mock_job()

        apply_sparse_adjustments(reuse, spec, job)

        # Two cascading SAFs: 2097152 → 212992 → 21632
        self.assertEqual(compute_stats.total_ops, 21_632)

    def test_coordinate_list_compression_at_backing_storage(self):
        """BackingStorage A with CSR format: fills compressed by density."""
        sparse_opts = SparseOptimizations(
            targets=[
                SparseTarget(
                    target="BackingStorage",
                    representation_format=[
                        RepresentationFormat(name="A", format="csr"),
                    ],
                ),
            ]
        )

        reuse = SymbolicAnalysisOutput()
        buffet = Buffet("A", "E0", "BackingStorage")
        stats = BuffetStats()
        stats.total_reads_to_parent = 2_097_152
        stats.max_per_parent_reads_to_parent = 2_097_152
        stats.max_occupancy = 16384 * 8
        reuse.buffet_stats[buffet] = stats

        arch_comps = {
            "BackingStorage": {
                "bits_per_value_scale": {"A": 1},
                "read_bpa": 8,
                "write_bpa": 8,
            }
        }
        spec = make_mock_spec(
            sparse_opts=sparse_opts,
            tensor_accesses=[
                {"name": "A", "density": 0.1015625, "output": False, "bits_per_value": 8},
            ],
            arch_components=arch_comps,
        )
        job = make_mock_job()

        apply_sparse_adjustments(reuse, spec, job)

        self.assertEqual(stats.total_reads_to_parent, 212_992)


class TestBackingStorageBReads(unittest.TestCase):
    """BackingStorage B reads with format compression.

    Per IMPLEMENTATION_PLAN.md Phase 7: verify BackingStorage B actual reads.
    fig1: BackingStorage B, alg_reads=16384, d=0.1015625 → 1664 after compression.
    """

    def test_backing_storage_b_compression(self):
        """BackingStorage B: 16384 fills → 1664 after compression."""
        sparse_opts = SparseOptimizations(
            targets=[
                SparseTarget(
                    target="BackingStorage",
                    representation_format=[
                        RepresentationFormat(name="B", format="bitmask"),
                    ],
                ),
            ]
        )

        reuse = SymbolicAnalysisOutput()
        buffet = Buffet("B", "E0", "BackingStorage")
        stats = BuffetStats()
        stats.total_reads_to_parent = 16_384
        stats.max_per_parent_reads_to_parent = 16_384
        stats.max_occupancy = 16384 * 8
        reuse.buffet_stats[buffet] = stats

        arch_comps = {
            "BackingStorage": {
                "bits_per_value_scale": {"B": 1},
                "read_bpa": 8,
                "write_bpa": 8,
            }
        }
        spec = make_mock_spec(
            sparse_opts=sparse_opts,
            tensor_accesses=[
                {"name": "B", "density": 0.1015625, "output": False, "bits_per_value": 8},
            ],
            arch_components=arch_comps,
        )
        job = make_mock_job()

        apply_sparse_adjustments(reuse, spec, job)

        # apply_format_compression(16384, 0.1015625) = 16384 - floor(16384*0.8984375) = 1664
        self.assertEqual(stats.total_reads_to_parent, 1_664)


class TestDenseRegressionNoSparse(unittest.TestCase):
    """Dense (no sparse_opts) should leave everything unchanged.

    IMPLEMENTATION_PLAN.md Phase 7: "Dense (no sparse_opts) unchanged" regression test.
    """

    def test_dense_all_counts_unchanged(self):
        """No sparse targets → all buffet/compute stats unchanged."""
        reuse = SymbolicAnalysisOutput()

        # Multiple buffets
        for tensor, level in [("A", "Buffer"), ("B", "Buffer"), ("Z", "Reg")]:
            buffet = Buffet(tensor, "E0", level)
            stats = BuffetStats()
            stats.total_reads_to_parent = 1000
            stats.total_writes_to_parent = 500
            stats.total_read_actions = 200
            stats.total_write_actions = 300
            stats.max_occupancy = 100
            reuse.buffet_stats[buffet] = stats

        from accelforge.model._looptree.reuse.symbolic.symbolic import Compute, ComputeStats
        compute_key = Compute("E0", "MAC")
        compute_stats = ComputeStats(total_ops=2000, max_per_unit_ops=2000)
        reuse.compute_stats[compute_key] = compute_stats

        # Empty sparse optimizations
        spec = make_mock_spec(
            sparse_opts=SparseOptimizations(),
            tensor_accesses=[
                {"name": "A", "density": 0.5, "output": False, "bits_per_value": 8},
                {"name": "B", "density": 0.5, "output": False, "bits_per_value": 8},
                {"name": "Z", "density": None, "output": True, "bits_per_value": 8},
            ],
        )
        job = make_mock_job()

        apply_sparse_adjustments(reuse, spec, job)

        # Everything should be unchanged
        for buffet, stats in reuse.buffet_stats.items():
            self.assertEqual(stats.total_reads_to_parent, 1000,
                             f"{buffet} reads changed")
            self.assertEqual(stats.total_writes_to_parent, 500,
                             f"{buffet} writes changed")
            self.assertEqual(stats.total_read_actions, 200,
                             f"{buffet} read_actions changed")
            self.assertEqual(stats.total_write_actions, 300,
                             f"{buffet} write_actions changed")
        self.assertEqual(compute_stats.total_ops, 2000)


class TestGatedSkippedCounts(unittest.TestCase):
    """Verify that SAF produces correct gated/skipped counts.

    IMPLEMENTATION_PLAN.md Phase 7: verify gated/skipped count values.
    Since sparse_adjustment modifies total_reads_to_parent (the actual value),
    the gated/skipped count = original - actual.
    """

    def test_buffer_a_gated_count(self):
        """Buffer A bitmask gating: format compression + SAF gives actual=21632."""
        sparse_opts = SparseOptimizations(
            targets=[
                SparseTarget(
                    target="Buffer",
                    representation_format=[
                        RepresentationFormat(name="A", format="bitmask"),
                    ],
                    action_optimization=[
                        ActionOptimization(
                            kind="gating", target="A", condition_on=["B"],
                        ),
                    ],
                ),
            ]
        )

        reuse = SymbolicAnalysisOutput()

        # Child: Reg A (to receive format compression + SAF reduction)
        child = Buffet("A", "E0", "Reg")
        child_stats = BuffetStats()
        child_stats.total_reads_to_parent = 2_097_152  # Algorithmic
        child_stats.max_per_parent_reads_to_parent = 2_097_152
        reuse.buffet_stats[child] = child_stats

        # Parent: Buffer A (has bitmask format)
        parent = Buffet("A", "E0", "Buffer")
        parent_stats = BuffetStats()
        parent_stats.total_reads_to_parent = 2_097_152
        parent_stats.max_per_parent_reads_to_parent = 2_097_152
        parent_stats.max_occupancy = 128 * 8
        reuse.buffet_stats[parent] = parent_stats

        # B for condition_on
        b = Buffet("B", "E0", "Buffer")
        bs = BuffetStats()
        bs.max_occupancy = 8  # scalar tile
        reuse.buffet_stats[b] = bs

        arch_comps = {
            "Buffer": {
                "bits_per_value_scale": {"A": 1, "B": 1},
                "read_bpa": 8,
                "write_bpa": 8,
            },
            "Reg": {
                "bits_per_value_scale": {"A": 1},
                "read_bpa": 8,
                "write_bpa": 8,
            },
        }
        spec = make_mock_spec(
            sparse_opts=sparse_opts,
            tensor_accesses=[
                {"name": "A", "density": 0.1015625, "output": False, "bits_per_value": 8},
                {"name": "B", "density": 0.1015625, "output": False, "bits_per_value": 8},
            ],
            arch_components=arch_comps,
        )
        job = make_mock_job()

        original_child_reads = child_stats.total_reads_to_parent

        apply_sparse_adjustments(reuse, spec, job)

        actual = child_stats.total_reads_to_parent
        gated = original_child_reads - actual
        # Phase 2: format compression on child reads
        #   child.total_reads_to_parent: 2,097,152 → 212,992 (× density_A)
        # Phase 3-4a: SAF (A gated on B), per-element prob = 0.8984375
        #   floor(212,992 × 0.8984375) = 191,360
        #   actual = 212,992 - 191,360 = 21,632
        self.assertEqual(actual, 21_632)
        self.assertEqual(gated, 2_097_152 - 21_632)

    def test_reg_z_gated_reads_and_updates(self):
        """Reg Z gating: verify both reads (ceil) and updates (floor) gated counts.
        actual_reads=21463, gated_reads=2059305
        actual_updates=21632, gated_updates=2075520"""
        sparse_opts = SparseOptimizations(
            targets=[
                SparseTarget(
                    target="Buffer",
                    action_optimization=[
                        ActionOptimization(
                            kind="gating", target="Z", condition_on=["A", "B"],
                        ),
                    ],
                ),
            ]
        )

        reuse = SymbolicAnalysisOutput()

        child = Buffet("Z", "E0", "Reg")
        child_stats = BuffetStats()
        child_stats.total_reads_to_parent = 2_080_768
        child_stats.max_per_parent_reads_to_parent = 2_080_768
        child_stats.total_writes_to_parent = 2_097_152
        child_stats.max_per_parent_writes_to_parent = 2_097_152
        reuse.buffet_stats[child] = child_stats

        parent = Buffet("Z", "E0", "Buffer")
        parent_stats = BuffetStats()
        parent_stats.total_reads_to_parent = 16_384
        reuse.buffet_stats[parent] = parent_stats

        for name in ("A", "B"):
            b = Buffet(name, "E0", "Buffer")
            s = BuffetStats()
            s.max_occupancy = 8  # scalar
            reuse.buffet_stats[b] = s

        arch_comps = {
            "Buffer": {
                "bits_per_value_scale": {"Z": 1, "A": 1, "B": 1},
                "read_bpa": 8,
                "write_bpa": 8,
            },
            "Reg": {
                "bits_per_value_scale": {"Z": 1},
                "read_bpa": 8,
                "write_bpa": 8,
            },
        }
        spec = make_mock_spec(
            sparse_opts=sparse_opts,
            tensor_accesses=[
                {"name": "A", "density": 0.1015625, "output": False, "bits_per_value": 8},
                {"name": "B", "density": 0.1015625, "output": False, "bits_per_value": 8},
                {"name": "Z", "density": None, "output": True, "bits_per_value": 8},
            ],
            arch_components=arch_comps,
            rank_sizes={"M": 128, "K": 128, "N": 128},
        )
        job = make_mock_job()

        orig_reads = child_stats.total_reads_to_parent
        orig_updates = child_stats.total_writes_to_parent

        apply_sparse_adjustments(reuse, spec, job)

        actual_reads = child_stats.total_reads_to_parent
        actual_updates = child_stats.total_writes_to_parent
        gated_reads = orig_reads - actual_reads
        gated_updates = orig_updates - actual_updates

        self.assertEqual(actual_reads, 21_463)
        self.assertEqual(gated_reads, 2_059_305)
        self.assertEqual(actual_updates, 21_632)
        self.assertEqual(gated_updates, 2_075_520)


if __name__ == "__main__":
    unittest.main()
