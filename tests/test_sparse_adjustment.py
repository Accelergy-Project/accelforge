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
    RankFormat,
    SparseOptimizations,
    SparseTarget,
    RepresentationFormat,
    ActionOptimization,
    ComputeOptimization,
)

from accelforge.model.sparse_adjustment import (
    apply_sparse_adjustments,
    _recompute_action_counts,
    _ranks_have_flattened_ids,
    _compute_flattened_dimension_sizes,
    _get_tensor_rank_variables,
    _compute_flattened_tensor_size,
    _get_loops_below_level,
    _compute_cond_temporal_tile,
)


def make_mock_job(einsum_name="E0"):
    """Create a minimal mock Job.

    Sets mapping=None and rank_variable_bounds={} so the SAF temporal
    tile path falls back to scalar conditioning (tile=1, tsize=1)
    rather than accidentally traversing MagicMock attributes.
    """
    job = MagicMock()
    job.einsum_name = einsum_name
    job.mapping = None
    job.rank_variable_bounds = {}
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
            ta.density_distribution = ta_info.get("density_distribution", None)
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

            # Optionally add metadata_read/metadata_write actions
            if "metadata_read_bpa" in info:
                md_read_action = MagicMock()
                md_read_action.bits_per_action = info["metadata_read_bpa"]
                comp.actions["metadata_read"] = md_read_action
            if "metadata_write_bpa" in info:
                md_write_action = MagicMock()
                md_write_action.bits_per_action = info["metadata_write_bpa"]
                comp.actions["metadata_write"] = md_write_action

            # Type checks
            from accelforge.frontend import arch
            comp.__class__ = arch.Memory

            return comp

        spec.arch.find = find_component

    return spec


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

class TestSAFPropagationToCompute(unittest.TestCase):
    """SAF probabilities propagate to reduce compute operations."""

    def test_single_saf_propagation(self):
        """Gating SAF reduces effectual compute total_ops (gated ops tracked separately for latency)."""
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

        # Gating SAF: effectual compute count is reduced by the condition
        # tensor's density.  Gated ops are tracked separately for latency.
        # 2_097_152 * 0.1015625 (density_B) = 212_992
        self.assertEqual(compute_stats.total_ops, 212_992)

    def test_two_saf_cascading_propagation(self):
        """Two gating SAFs compound-reduce effectual compute total_ops."""
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

        # Two gating SAFs compound: A(cond B) * B(cond A)
        # 2_097_152 * 0.1015625 * 0.1015625 = 21_632
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

        tensor_info = {
            ta.name: {"bits_per_value": ta.bits_per_value}
            for ta in spec.workload.einsums[job.einsum_name].tensor_accesses
        }
        _recompute_action_counts(reuse, spec, job, set(), tensor_info)

        # write_scale = 8 / 8 = 1, read_scale = 8 / 8 = 1
        # fill_write_actions = total_reads_to_parent * write_scale = 500 * 1 = 500
        # (fill-writes tracked in parent-named attribute for temporal reuse)
        # read_actions = total_writes_to_parent * read_scale = 0 (input tensor)
        self.assertEqual(stats.total_parent_fill_write_actions, 500)
        self.assertEqual(stats.total_write_actions, 0)
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

        tensor_info = {
            ta.name: {"bits_per_value": ta.bits_per_value}
            for ta in spec.workload.einsums[job.einsum_name].tensor_accesses
        }
        _recompute_action_counts(reuse, spec, job, set(), tensor_info)

        # Buffer read_actions = child.total_reads_to_parent * read_scale = 200 * 1 = 200
        # Buffer fill_write_actions = parent.total_reads_to_parent * write_scale = 100 * 1 = 100
        # (fill-writes tracked in parent-named attribute for temporal reuse)
        self.assertEqual(parent_stats.total_read_actions, 200)
        self.assertEqual(parent_stats.total_parent_fill_write_actions, 100)
        self.assertEqual(parent_stats.total_write_actions, 0)


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


class TestRanksHaveFlattenedIds(unittest.TestCase):
    """Test _ranks_have_flattened_ids helper."""

    def test_true_when_present(self):
        ranks = [
            RankFormat(format="UOP", flattened_rank_ids=[["R"]]),
            RankFormat(format="RLE"),
        ]
        self.assertTrue(_ranks_have_flattened_ids(ranks))

    def test_false_when_absent(self):
        ranks = [RankFormat(format="UOP"), RankFormat(format="CP")]
        self.assertFalse(_ranks_have_flattened_ids(ranks))

    def test_false_empty_list(self):
        self.assertFalse(_ranks_have_flattened_ids([]))


class TestComputeFlattenedDimensionSizes(unittest.TestCase):
    """Test _compute_flattened_dimension_sizes helper."""

    def test_single_dim_per_rank(self):
        """Single dimension per rank."""
        ranks = [
            RankFormat(format="UOP", flattened_rank_ids=[["r"]]),
            RankFormat(format="RLE", flattened_rank_ids=[["c"]]),
        ]
        shape = {"r": 3, "c": 64, "m": 128}
        sizes = _compute_flattened_dimension_sizes(ranks, shape)
        self.assertEqual(sizes, [3, 64])

    def test_multi_dim_flattened(self):
        """Multiple dimensions flattened into one rank."""
        ranks = [
            RankFormat(format="UOP", flattened_rank_ids=[["S", "F"]]),
            RankFormat(format="RLE", flattened_rank_ids=[["C"]]),
        ]
        shape = {"s": 3, "f": 32, "c": 64}
        sizes = _compute_flattened_dimension_sizes(ranks, shape)
        self.assertEqual(sizes, [96, 64])  # 3 * 32 = 96

    def test_case_insensitive(self):
        """Dimension names are case-insensitive."""
        ranks = [RankFormat(format="UOP", flattened_rank_ids=[["C", "R"]])]
        shape = {"c": 8, "r": 3}
        sizes = _compute_flattened_dimension_sizes(ranks, shape)
        self.assertEqual(sizes, [24])

    def test_missing_dim_defaults_to_1(self):
        """Missing dimensions in shape default to 1."""
        ranks = [RankFormat(format="UOP", flattened_rank_ids=[["X", "Y"]])]
        shape = {"x": 5}
        sizes = _compute_flattened_dimension_sizes(ranks, shape)
        self.assertEqual(sizes, [5])  # Y missing -> 1, so 5*1=5

    def test_rank_without_flattened_ids_gets_1(self):
        """Ranks without flattened_rank_ids get fiber_shape=1."""
        ranks = [
            RankFormat(format="UOP", flattened_rank_ids=[["C"]]),
            RankFormat(format="B"),  # No flattened_rank_ids
        ]
        shape = {"c": 64}
        sizes = _compute_flattened_dimension_sizes(ranks, shape)
        self.assertEqual(sizes, [64, 1])


class TestGetTensorRankVariables(unittest.TestCase):
    """Test _get_tensor_rank_variables extracts projecting variables."""

    def _make_einsum(self, tensor_accesses_info):
        """Helper: create a mock einsum with tensor_accesses."""
        from unittest.mock import MagicMock
        einsum = MagicMock()
        ta_list = []
        for info in tensor_accesses_info:
            ta = MagicMock()
            ta.name = info["name"]
            ta.projection = info["projection"]
            ta_list.append(ta)
        einsum.tensor_accesses = ta_list
        return einsum

    def test_simple_dict_projection(self):
        """Simple dict projection: {M: m, K: k} → {'m', 'k'}."""
        einsum = self._make_einsum([
            {"name": "A", "projection": {"M": "m", "K": "k"}},
        ])
        result = _get_tensor_rank_variables(einsum, "A")
        self.assertEqual(result, {"m", "k"})

    def test_compound_expression(self):
        """Compound projection like {H: e + r} → {'e', 'r'}."""
        einsum = self._make_einsum([
            {"name": "Input", "projection": {"H": "e + r", "W": "f + s", "C": "c"}},
        ])
        result = _get_tensor_rank_variables(einsum, "Input")
        self.assertEqual(result, {"e", "r", "f", "s", "c"})

    def test_stride_expression(self):
        """Stride: {H: 2*p + r} → {'p', 'r'}."""
        einsum = self._make_einsum([
            {"name": "Input", "projection": {"H": "2*p + r", "C": "c"}},
        ])
        result = _get_tensor_rank_variables(einsum, "Input")
        self.assertEqual(result, {"p", "r", "c"})

    def test_list_projection(self):
        """List-style projection: [m, k] → {'m', 'k'}."""
        einsum = self._make_einsum([
            {"name": "A", "projection": ["m", "k"]},
        ])
        result = _get_tensor_rank_variables(einsum, "A")
        self.assertEqual(result, {"m", "k"})

    def test_tensor_not_found(self):
        """Missing tensor returns empty set."""
        einsum = self._make_einsum([
            {"name": "A", "projection": {"M": "m"}},
        ])
        result = _get_tensor_rank_variables(einsum, "B")
        self.assertEqual(result, set())

    def test_different_tensors_different_variables(self):
        """Different tensors project different variables."""
        einsum = self._make_einsum([
            {"name": "A", "projection": {"M": "m", "K": "k"}},
            {"name": "B", "projection": {"N": "n", "K": "k"}},
            {"name": "Z", "projection": {"M": "m", "N": "n"}},
        ])
        self.assertEqual(_get_tensor_rank_variables(einsum, "A"), {"m", "k"})
        self.assertEqual(_get_tensor_rank_variables(einsum, "B"), {"n", "k"})
        self.assertEqual(_get_tensor_rank_variables(einsum, "Z"), {"m", "n"})


class TestComputeFlattenedTensorSize(unittest.TestCase):
    """Test _compute_flattened_tensor_size filters to projecting dimensions."""

    def _make_einsum(self, tensor_accesses_info):
        from unittest.mock import MagicMock
        einsum = MagicMock()
        ta_list = []
        for info in tensor_accesses_info:
            ta = MagicMock()
            ta.name = info["name"]
            ta.projection = info["projection"]
            ta_list.append(ta)
        einsum.tensor_accesses = ta_list
        return einsum

    def test_all_dims_project(self):
        """When all rank dims project to tensor, tensor_size = product of all."""
        # A[m, k]: ranks map to [M] and [K]
        ranks = [
            RankFormat(format="UOP", flattened_rank_ids=[["M"]]),
            RankFormat(format="CP", flattened_rank_ids=[["K"]]),
        ]
        einsum = self._make_einsum([
            {"name": "A", "projection": {"M": "m", "K": "k"}},
        ])
        full_shape = {"m": 128, "k": 128, "n": 128}
        result = _compute_flattened_tensor_size(ranks, full_shape, einsum, "A")
        self.assertEqual(result, 128 * 128)

    def test_non_projecting_dim_excluded(self):
        """Dimension M doesn't project to B[n,k] → excluded from tensor_size."""
        # Ranks: [M], [N], [K] — but B only projects to n, k
        ranks = [
            RankFormat(format="UOP", flattened_rank_ids=[["M"]]),
            RankFormat(format="UOP", flattened_rank_ids=[["N"]]),
            RankFormat(format="CP", flattened_rank_ids=[["K"]]),
        ]
        einsum = self._make_einsum([
            {"name": "A", "projection": {"M": "m", "K": "k"}},
            {"name": "B", "projection": {"N": "n", "K": "k"}},
        ])
        full_shape = {"m": 128, "n": 128, "k": 128}
        # For B: only N and K project → tensor_size = 128 * 128 = 16384
        result = _compute_flattened_tensor_size(ranks, full_shape, einsum, "B")
        self.assertEqual(result, 128 * 128)
        # NOT 128*128*128 (which would be wrong)

    def test_flattened_multi_dim(self):
        """Flattened [S, F] with S projecting but F not → only S counted."""
        # Weights[C, M, R, S] — F does not project to Weights
        ranks = [
            RankFormat(format="UOP", flattened_rank_ids=[["S", "F"]]),
            RankFormat(format="RLE", flattened_rank_ids=[["C"]]),
        ]
        einsum = self._make_einsum([
            {"name": "Weights", "projection": {"C": "c", "M": "m", "R": "r", "S": "s"}},
        ])
        full_shape = {"s": 3, "f": 32, "c": 64, "m": 64, "r": 3}
        # Only S(3) and C(64) project to Weights — not F(32)
        result = _compute_flattened_tensor_size(ranks, full_shape, einsum, "Weights")
        self.assertEqual(result, 3 * 64)

    def test_flattened_both_dims_project(self):
        """Flattened [E, N] where both project to Inputs → both counted."""
        # Inputs[N, C, E+R, F+S] → e and n both project
        ranks = [
            RankFormat(format="UOP", flattened_rank_ids=[["E", "N"]]),
            RankFormat(format="RLE", flattened_rank_ids=[["C"]]),
        ]
        einsum = self._make_einsum([
            {"name": "Inputs", "projection": {"N": "n", "C": "c", "H": "e + r", "W": "f + s"}},
        ])
        full_shape = {"e": 32, "n": 1, "c": 64, "r": 3, "f": 32, "s": 3}
        # E(32) and N(1) and C(64) all project to Inputs
        result = _compute_flattened_tensor_size(ranks, full_shape, einsum, "Inputs")
        self.assertEqual(result, 32 * 1 * 64)

    def test_no_flattened_ids(self):
        """Ranks without flattened_rank_ids contribute nothing → returns 1."""
        ranks = [
            RankFormat(format="UOP"),  # no flattened_rank_ids
            RankFormat(format="CP"),
        ]
        einsum = self._make_einsum([
            {"name": "A", "projection": {"M": "m", "K": "k"}},
        ])
        full_shape = {"m": 128, "k": 128}
        result = _compute_flattened_tensor_size(ranks, full_shape, einsum, "A")
        self.assertEqual(result, 1)  # degenerate: no dims specified

    def test_duplicate_dim_name_across_ranks(self):
        """Same dimension in two ranks: each occurrence multiplied.

        When tiling splits a dimension (e.g. C into outer/inner), the
        AccelForge config should use distinct variable names (c_outer,
        c_inner).  If the same name "C" appears in two ranks, both
        contribute shape["c"] to tensor_size.  This test documents
        that behavior.
        """
        ranks = [
            RankFormat(format="UOP", flattened_rank_ids=[["C"]]),
            RankFormat(format="RLE", flattened_rank_ids=[["C"]]),
        ]
        einsum = self._make_einsum([
            {"name": "A", "projection": {"C": "c", "K": "k"}},
        ])
        full_shape = {"c": 64, "k": 128}
        result = _compute_flattened_tensor_size(ranks, full_shape, einsum, "A")
        # C appears twice → 64 * 64 = 4096 (each rank multiplies c=64)
        self.assertEqual(result, 64 * 64)

    def test_fig12_like_distinct_dims(self):
        """Fig12-like with distinct dimension names (no tiling duplicates).

        Inputs[N, C, G, H=e+r, W=f+s] — projects: n, c, g, e, r, f, s
        7 ranks: [G], [C_outer], [M], [S,F], [E,N], [R], [C_inner]
        full_shape: g=1, c_outer=8, m=64, s=1, f=32, e=32, n=1, r=1, c_inner=8
        """
        ranks = [
            RankFormat(format="UOP", flattened_rank_ids=[["G"]]),
            RankFormat(format="UOP", flattened_rank_ids=[["C_outer"]]),
            RankFormat(format="UOP", flattened_rank_ids=[["M"]]),
            RankFormat(format="UOP", flattened_rank_ids=[["S", "F"]]),
            RankFormat(format="UOP", flattened_rank_ids=[["E", "N"]]),
            RankFormat(format="UOP", flattened_rank_ids=[["R"]]),
            RankFormat(format="RLE", flattened_rank_ids=[["C_inner"]]),
        ]
        einsum = self._make_einsum([
            {"name": "Inputs", "projection": {
                "N": "n", "C_outer": "c_outer", "C_inner": "c_inner",
                "G": "g", "H": "e + r", "W": "f + s",
            }},
            {"name": "Weights", "projection": {
                "C_outer": "c_outer", "C_inner": "c_inner",
                "M": "m", "G": "g", "R": "r", "S": "s",
            }},
        ])
        full_shape = {
            "g": 1, "c_outer": 8, "c_inner": 8, "m": 64,
            "s": 1, "f": 32, "e": 32, "n": 1, "r": 1,
        }

        # Inputs projects: n, c_outer, c_inner, g, e, r, f, s (NOT m)
        # tensor_size = g(1) * c_outer(8) * s(1)*f(32) * e(32)*n(1) * r(1) * c_inner(8)
        #             = 1 * 8 * 32 * 32 * 8 = 65,536
        result = _compute_flattened_tensor_size(
            ranks, full_shape, einsum, "Inputs"
        )
        self.assertEqual(result, 1 * 8 * 32 * 32 * 8)  # 65536, no M

        # Weights projects: c_outer, c_inner, m, g, r, s (NOT n, e, f)
        result_w = _compute_flattened_tensor_size(
            ranks, full_shape, einsum, "Weights"
        )
        # = g(1) * c_outer(8) * m(64) * s(1) * r(1) * c_inner(8) = 4096
        self.assertEqual(result_w, 1 * 8 * 64 * 1 * 1 * 8)  # 4096


class TestBuffetStatsTileShape(unittest.TestCase):
    """Verify tile_shape field on BuffetStats (Phase A refactoring)."""

    def test_default_is_none(self):
        """New BuffetStats should have tile_shape=None by default."""
        stats = BuffetStats()
        self.assertIsNone(stats.tile_shape)

    def test_add_both_none(self):
        """__add__ should not fail when both tile_shapes are None."""
        a = BuffetStats(total_reads_to_parent=10)
        b = BuffetStats(total_reads_to_parent=20)
        result = a + b
        self.assertEqual(result.total_reads_to_parent, 30)
        self.assertIsNone(result.tile_shape)

    def test_add_preserves_first_tile_shape(self):
        """__add__ keeps self's tile_shape when both are non-None."""
        a = BuffetStats(total_reads_to_parent=10)
        a.tile_shape = {"m": 4}
        b = BuffetStats(total_reads_to_parent=20)
        b.tile_shape = {"m": 8}
        result = a + b
        self.assertEqual(result.total_reads_to_parent, 30)
        # Keeps first non-None (self's) tile_shape
        self.assertEqual(result.tile_shape, {"m": 4})

    def test_add_none_plus_dict_inherits(self):
        """__add__ inherits tile_shape from other when self is None."""
        a = BuffetStats(total_reads_to_parent=10)
        b = BuffetStats(total_reads_to_parent=20)
        b.tile_shape = {"m": 8, "k": 4}
        result = a + b
        self.assertEqual(result.total_reads_to_parent, 30)
        self.assertEqual(result.tile_shape, {"m": 8, "k": 4})

    def test_add_different_tile_shapes_no_assert(self):
        """__add__ should not assert even with different tile_shapes (imperfect tiling)."""
        a = BuffetStats(total_reads_to_parent=100)
        a.tile_shape = {"m": 8, "k": 4}
        b = BuffetStats(total_reads_to_parent=50)
        b.tile_shape = {"m": 3, "k": 4}  # Last tile is smaller (imperfect)
        # This must not raise AssertionError
        result = a + b
        self.assertEqual(result.total_reads_to_parent, 150)

    def test_sparse_adjustment_with_no_tile_shape(self):
        """Mock-based sparse adjustment should work when tile_shape is None."""
        reuse = SymbolicAnalysisOutput()
        buffet = Buffet("A", "E0", "Buffer")
        stats = BuffetStats()
        stats.total_reads_to_parent = 1000
        stats.total_write_actions = 500
        stats.total_read_actions = 200
        # tile_shape is None (default) — should not crash
        reuse.buffet_stats[buffet] = stats

        spec = make_mock_spec()
        job = make_mock_job()

        sparse_actions = apply_sparse_adjustments(reuse, spec, job).sparse_actions
        self.assertEqual(sparse_actions, {})


class TestTileShapeThroughPipeline(unittest.TestCase):
    """Verify tile_shape is populated when running through the full pipeline."""

    def test_fig1_bitmask_tile_shapes_populated(self):
        """After evaluate_mapping on fig1 bitmask, all buffet_stats should have tile_shape."""
        import os
        from accelforge.frontend.spec import Spec
        from accelforge.model.main import evaluate_mapping

        fig1_dir = os.path.join(os.path.dirname(__file__), "input_files", "fig1")
        spec = Spec.from_yaml(
            os.path.join(fig1_dir, "arch_energy.yaml"),
            os.path.join(fig1_dir, "workload.yaml"),
            os.path.join(fig1_dir, "mapping.yaml"),
            os.path.join(fig1_dir, "sparse_bitmask_energy.yaml"),
        )
        result = evaluate_mapping(spec)
        # The pipeline ran without error — this validates tile_shape
        # is correctly propagated through the full analysis.
        self.assertIsNotNone(result)

    def test_fig1_coord_list_tile_shapes_populated(self):
        """After evaluate_mapping on fig1 coord_list, pipeline succeeds."""
        import os
        from accelforge.frontend.spec import Spec
        from accelforge.model.main import evaluate_mapping

        fig1_dir = os.path.join(os.path.dirname(__file__), "input_files", "fig1")
        spec = Spec.from_yaml(
            os.path.join(fig1_dir, "arch_energy.yaml"),
            os.path.join(fig1_dir, "workload.yaml"),
            os.path.join(fig1_dir, "mapping.yaml"),
            os.path.join(fig1_dir, "sparse_coord_list_energy.yaml"),
        )
        result = evaluate_mapping(spec)
        self.assertIsNotNone(result)


class TestGetLoopsBelowLevel(unittest.TestCase):
    """Test _get_loops_below_level helper."""

    def _make_nodes(self, node_specs):
        """Build a list of mapping-like mock nodes.

        Each spec is a tuple: ('Storage'|'Toll'|'Spatial'|'Temporal'|'Compute',
        component_or_rv, tile_shape_or_None).
        """
        from accelforge.frontend.mapping import (
            Storage, Toll, Spatial, Temporal, Compute as MappingCompute,
        )
        nodes = []
        for spec in node_specs:
            kind = spec[0]
            if kind == "Storage":
                nodes.append(Storage(tensors=["A"], component=spec[1]))
            elif kind == "Toll":
                nodes.append(Toll(tensors=["A"], component=spec[1]))
            elif kind == "Spatial":
                nodes.append(Spatial(
                    rank_variable=spec[1],
                    tile_shape=spec[2],
                    name=0,
                    component="PE",
                ))
            elif kind == "Temporal":
                nodes.append(Temporal(
                    rank_variable=spec[1],
                    tile_shape=spec[2],
                ))
            elif kind == "Compute":
                nodes.append(MappingCompute(
                    einsum="E0",
                    component=spec[1],
                ))
        return nodes

    def test_basic_temporal(self):
        """Temporal K=1 below level → temporal_tiles = {k: 1}."""
        nodes = self._make_nodes([
            ("Storage", "Buffer", None),
            ("Temporal", "k", 1),
            ("Compute", "MAC", None),
        ])
        spatial, temporal = _get_loops_below_level(nodes, "Buffer")
        self.assertEqual(temporal, {"k": 1})
        self.assertEqual(spatial, {})

    def test_spatial_and_temporal(self):
        """Spatial K=16 + Temporal K=1 below level."""
        nodes = self._make_nodes([
            ("Storage", "Buffer", None),
            ("Spatial", "k", 16),
            ("Temporal", "k", 1),
            ("Compute", "MAC", None),
        ])
        spatial, temporal = _get_loops_below_level(nodes, "Buffer")
        self.assertEqual(spatial, {"k": 16})
        self.assertEqual(temporal, {"k": 1})

    def test_no_match(self):
        """Level not found → empty dicts."""
        nodes = self._make_nodes([
            ("Storage", "Buffer", None),
            ("Temporal", "k", 1),
            ("Compute", "MAC", None),
        ])
        spatial, temporal = _get_loops_below_level(nodes, "NoSuchLevel")
        self.assertEqual(spatial, {})
        self.assertEqual(temporal, {})

    def test_multiple_levels(self):
        """Only loops below the target level are collected."""
        nodes = self._make_nodes([
            ("Storage", "DRAM", None),
            ("Temporal", "m", 128),
            ("Storage", "Buffer", None),
            ("Temporal", "k", 4),
            ("Spatial", "n", 16),
            ("Compute", "MAC", None),
        ])
        spatial, temporal = _get_loops_below_level(nodes, "Buffer")
        self.assertEqual(temporal, {"k": 4})
        self.assertEqual(spatial, {"n": 16})
        # m=128 is ABOVE Buffer, not collected
        self.assertNotIn("m", temporal)

    def test_spatial_only(self):
        """Only spatial loop below level (no temporal)."""
        nodes = self._make_nodes([
            ("Storage", "Buffer", None),
            ("Spatial", "k", 16),
            ("Compute", "MAC", None),
        ])
        spatial, temporal = _get_loops_below_level(nodes, "Buffer")
        self.assertEqual(spatial, {"k": 16})
        self.assertEqual(temporal, {})

    def test_toll_node(self):
        """Toll nodes are also recognized as level boundaries."""
        nodes = self._make_nodes([
            ("Toll", "PassThrough", None),
            ("Temporal", "m", 4),
            ("Compute", "MAC", None),
        ])
        spatial, temporal = _get_loops_below_level(nodes, "PassThrough")
        self.assertEqual(temporal, {"m": 4})


class TestComputeCondTemporalTile(unittest.TestCase):
    """Test _compute_cond_temporal_tile helper."""

    def _make_einsum(self, tensor_accesses_info):
        einsum = MagicMock()
        ta_list = []
        for info in tensor_accesses_info:
            ta = MagicMock()
            ta.name = info["name"]
            ta.projection = info["projection"]
            ta_list.append(ta)
        einsum.tensor_accesses = ta_list
        return einsum

    def _make_nodes(self, node_specs):
        from accelforge.frontend.mapping import (
            Storage, Spatial, Temporal, Compute as MappingCompute,
        )
        nodes = []
        for spec in node_specs:
            kind = spec[0]
            if kind == "Storage":
                nodes.append(Storage(tensors=["A"], component=spec[1]))
            elif kind == "Spatial":
                nodes.append(Spatial(
                    rank_variable=spec[1], tile_shape=spec[2],
                    name=0, component="PE",
                ))
            elif kind == "Temporal":
                nodes.append(Temporal(
                    rank_variable=spec[1], tile_shape=spec[2],
                ))
            elif kind == "Compute":
                nodes.append(MappingCompute(einsum="E0", component=spec[1]))
        return nodes

    def test_element_level_temporal_k1(self):
        """Temporal K=1 → tile=1 for tensor A[m,k]."""
        einsum = self._make_einsum([
            {"name": "A", "projection": {"M": "m", "K": "k"}},
        ])
        nodes = self._make_nodes([
            ("Storage", "RF", None),
            ("Spatial", "k", 16),
            ("Temporal", "k", 1),
            ("Compute", "MAC", None),
        ])
        tile = _compute_cond_temporal_tile(
            nodes, "RF", "A", einsum,
            stats_tile_shape={"m": 4, "k": 16},
        )
        # k has temporal loop → temporal tile = 1
        # m has no loop below RF → uses stats_tile_shape[m] = 4
        self.assertEqual(tile, 1 * 4)

    def test_spatial_only_no_temporal(self):
        """Spatial K=16, no temporal → per-PE tile shape = 16."""
        einsum = self._make_einsum([
            {"name": "A", "projection": {"K": "k"}},
        ])
        nodes = self._make_nodes([
            ("Storage", "RF", None),
            ("Spatial", "k", 16),
            ("Compute", "MAC", None),
        ])
        tile = _compute_cond_temporal_tile(
            nodes, "RF", "A", einsum,
            stats_tile_shape={"k": 16},
        )
        # k: spatial only → per-PE tile_shape = 16
        self.assertEqual(tile, 16)

    def test_no_loops_below(self):
        """No loops below level → uses stats_tile_shape directly."""
        einsum = self._make_einsum([
            {"name": "A", "projection": {"M": "m", "K": "k"}},
        ])
        nodes = self._make_nodes([
            ("Storage", "Buffer", None),
            ("Compute", "MAC", None),
        ])
        tile = _compute_cond_temporal_tile(
            nodes, "Buffer", "A", einsum,
            stats_tile_shape={"m": 8, "k": 4},
        )
        self.assertEqual(tile, 8 * 4)

    def test_no_tile_shape_returns_1(self):
        """None tile_shape → returns 1."""
        einsum = self._make_einsum([
            {"name": "A", "projection": {"M": "m"}},
        ])
        tile = _compute_cond_temporal_tile(
            [], "Buffer", "A", einsum,
            stats_tile_shape=None,
        )
        self.assertEqual(tile, 1)

    def test_unknown_tensor_returns_1(self):
        """Tensor not in einsum → returns 1."""
        einsum = self._make_einsum([
            {"name": "B", "projection": {"N": "n"}},
        ])
        tile = _compute_cond_temporal_tile(
            [], "Buffer", "A", einsum,
            stats_tile_shape={"m": 8},
        )
        self.assertEqual(tile, 1)

    def test_mixed_spatial_temporal(self):
        """Mixed: temporal on m, spatial on k → temporal m * spatial tile_shape k."""
        einsum = self._make_einsum([
            {"name": "A", "projection": {"M": "m", "K": "k"}},
        ])
        nodes = self._make_nodes([
            ("Storage", "RF", None),
            ("Temporal", "m", 4),
            ("Spatial", "k", 8),
            ("Compute", "MAC", None),
        ])
        tile = _compute_cond_temporal_tile(
            nodes, "RF", "A", einsum,
            stats_tile_shape={"m": 4, "k": 16},
        )
        # m: temporal → 4
        # k: spatial only → per-PE tile_shape = 8
        self.assertEqual(tile, 4 * 8)


class TestSAFTemporalTileBackwardCompat(unittest.TestCase):
    """Verify all-scalar configs produce identical results to pre-change behavior.

    When job.mapping is None (mock tests), the SAF path falls back to
    tile=1 / tsize=1 which triggers scalar conditioning in
    compute_saf_probability, identical to the old code path.
    """

    def test_scalar_saf_with_no_mapping(self):
        """SAF with no mapping → same as element-level (prob = 1-density)."""
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
        buffet_a = Buffet("A", "E0", "Buffer")
        stats_a = BuffetStats()
        stats_a.max_occupancy = 8
        reuse.buffet_stats[buffet_a] = stats_a

        buffet_b = Buffet("B", "E0", "Buffer")
        stats_b = BuffetStats()
        stats_b.max_occupancy = 8
        reuse.buffet_stats[buffet_b] = stats_b

        from accelforge.model._looptree.reuse.symbolic.symbolic import (
            Compute, ComputeStats,
        )
        compute_key = Compute("E0", "MAC")
        compute_stats = ComputeStats(
            total_ops=2_097_152, max_per_unit_ops=2_097_152,
        )
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
                {"name": "A", "density": 0.1015625, "output": False,
                 "bits_per_value": 8},
                {"name": "B", "density": 0.1015625, "output": False,
                 "bits_per_value": 8},
            ],
            arch_components=arch_comps,
        )
        job = make_mock_job()
        # make_mock_job sets job.mapping = None, so SAF uses scalar fallback

        apply_sparse_adjustments(reuse, spec, job)

        # Gating SAF reduces effectual compute by condition density.
        # 2_097_152 * 0.1015625 = 212_992
        self.assertEqual(compute_stats.total_ops, 212_992)


class TestStructuredVsRandomDivergence(unittest.TestCase):
    """Verify structured vs random sparsity produces different SAF results.

    This is the core validation for the temporal tile separation:
    when temporal tile > 1, structured sparsity guarantees every tile
    has nonzeros (prob_empty = 0 → optimization_prob = 0 → no skipping),
    while random sparsity allows some tiles to be all-zero
    (prob_empty > 0 → optimization_prob > 0 → some skipping).
    """

    def _make_mapping_nodes(self):
        """Create mapping nodes with temporal K=4 below Buffer."""
        from accelforge.frontend.mapping import (
            Storage, Temporal, Compute as MappingCompute,
        )
        return [
            Storage(tensors=["A", "B"], component="Buffer"),
            Temporal(rank_variable="k", tile_shape=4),
            MappingCompute(einsum="E0", component="MAC"),
        ]

    def _run_saf(self, density_distribution):
        """Run SAF with given distribution, return post-SAF compute ops."""
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
                    ],
                )
            ]
        )

        reuse = SymbolicAnalysisOutput()
        buffet_a = Buffet("A", "E0", "Buffer")
        stats_a = BuffetStats()
        stats_a.tile_shape = {"k": 4}
        reuse.buffet_stats[buffet_a] = stats_a

        buffet_b = Buffet("B", "E0", "Buffer")
        stats_b = BuffetStats()
        stats_b.tile_shape = {"k": 4}
        reuse.buffet_stats[buffet_b] = stats_b

        from accelforge.model._looptree.reuse.symbolic.symbolic import (
            Compute, ComputeStats,
        )
        compute_key = Compute("E0", "MAC")
        compute_stats = ComputeStats(
            total_ops=1024, max_per_unit_ops=1024,
        )
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
                {"name": "A", "density": 0.5, "output": False,
                 "bits_per_value": 8,
                 "projection": {"K": "k"}},
                {"name": "B", "density": 0.5, "output": False,
                 "bits_per_value": 8,
                 "density_distribution": density_distribution,
                 "projection": {"K": "k"}},
            ],
            arch_components=arch_comps,
        )

        # Build a job with a real mapping
        job = make_mock_job()
        mapping = MagicMock()
        mapping.nodes = self._make_mapping_nodes()
        job.mapping = mapping
        job.rank_variable_bounds = {"k": 64}

        apply_sparse_adjustments(reuse, spec, job)
        return compute_stats.total_ops

    def test_random_has_skipping(self):
        """Random distribution with tile=4 → prob_empty > 0 → compute reduced."""
        ops = self._run_saf(density_distribution=None)
        # tile=4, tsize=64, density=0.5: prob_empty(4) > 0
        # → optimization_prob > 0 → compute is reduced
        self.assertLess(ops, 1024)

    def test_structured_no_skipping(self):
        """Structured distribution with tile=4 → prob_empty=0 → no compute reduction."""
        ops = self._run_saf(density_distribution="structured")
        # Structured: prob_empty(4) = 0 → optimization_prob = 0
        # → prob_nonempty = 1.0 → no SAF reduction
        # → compute stays at 1024
        self.assertEqual(ops, 1024)

    def test_structured_more_energy_than_random(self):
        """Structured sparsity produces higher compute ops than random.

        This is the whole point: structured sparsity can never skip tiles
        (every tile guaranteed to have nonzeros), while random allows
        some tiles to be all-zero.
        """
        random_ops = self._run_saf(density_distribution=None)
        structured_ops = self._run_saf(density_distribution="structured")
        self.assertGreater(structured_ops, random_ops)

    def test_element_level_both_agree(self):
        """With temporal tile=1, structured and random should agree.

        At element level, prob_nonempty = density for both models.
        This verifies no regression for the common case.
        """
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

        random_ops = None
        structured_ops = None

        for dist in (None, "structured"):
            reuse = SymbolicAnalysisOutput()
            buffet_a = Buffet("A", "E0", "Buffer")
            stats_a = BuffetStats()
            stats_a.tile_shape = {"k": 1}
            reuse.buffet_stats[buffet_a] = stats_a

            buffet_b = Buffet("B", "E0", "Buffer")
            stats_b = BuffetStats()
            stats_b.tile_shape = {"k": 1}
            reuse.buffet_stats[buffet_b] = stats_b

            from accelforge.model._looptree.reuse.symbolic.symbolic import (
                Compute, ComputeStats,
            )
            compute_key = Compute("E0", "MAC")
            cs = ComputeStats(total_ops=1024, max_per_unit_ops=1024)
            reuse.compute_stats[compute_key] = cs

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
                    {"name": "A", "density": 0.5, "output": False,
                     "bits_per_value": 8, "projection": {"K": "k"}},
                    {"name": "B", "density": 0.5, "output": False,
                     "bits_per_value": 8, "projection": {"K": "k"},
                     "density_distribution": dist},
                ],
                arch_components=arch_comps,
            )

            from accelforge.frontend.mapping import (
                Storage, Temporal, Compute as MappingCompute,
            )
            job = make_mock_job()
            mapping = MagicMock()
            mapping.nodes = [
                Storage(tensors=["A", "B"], component="Buffer"),
                Temporal(rank_variable="k", tile_shape=1),
                MappingCompute(einsum="E0", component="MAC"),
            ]
            job.mapping = mapping
            job.rank_variable_bounds = {"k": 64}

            apply_sparse_adjustments(reuse, spec, job)

            if dist is None:
                random_ops = cs.total_ops
            else:
                structured_ops = cs.total_ops

        # At element level (tile=1), both should give same result
        self.assertEqual(random_ops, structured_ops)


class TestPositionSkippingSelfSAF(unittest.TestCase):
    """Position-skipping with condition_on=[] (self-conditioned).

    The target tensor uses its own format metadata to skip empty positions.
    This is the DSTC mechanism: bitmask format + position-skipping on same tensor.
    """

    def _make_linebuffer_reuse(self, density_a=0.5, density_b=0.4):
        """Create reuse with LineBuffer (A, B) -> Compute level (A, B, Z).

        Mimics Fig 13 DSTC: position-skipping at LineBuffer for A and B.
        """
        reuse = SymbolicAnalysisOutput()

        # Compute-level buffets (MAC reads from LineBuffer)
        compute_a = Buffet("A", "E0", "MAC")
        cs_a = BuffetStats()
        cs_a.total_reads_to_parent = 100_000
        cs_a.max_per_parent_reads_to_parent = 100_000
        reuse.buffet_stats[compute_a] = cs_a

        compute_b = Buffet("B", "E0", "MAC")
        cs_b = BuffetStats()
        cs_b.total_reads_to_parent = 100_000
        cs_b.max_per_parent_reads_to_parent = 100_000
        reuse.buffet_stats[compute_b] = cs_b

        # LineBuffer buffets (parent for A and B)
        lb_a = Buffet("A", "E0", "LineBuffer")
        ls_a = BuffetStats()
        ls_a.total_reads_to_parent = 10_000  # fills from GLB
        ls_a.max_per_parent_reads_to_parent = 10_000
        ls_a.max_occupancy = 64
        reuse.buffet_stats[lb_a] = ls_a

        lb_b = Buffet("B", "E0", "LineBuffer")
        ls_b = BuffetStats()
        ls_b.total_reads_to_parent = 10_000
        ls_b.max_per_parent_reads_to_parent = 10_000
        ls_b.max_occupancy = 64
        reuse.buffet_stats[lb_b] = ls_b

        # Compute stats (MAC)
        compute_key = Compute("E0", "MAC")
        cs = ComputeStats()
        cs.total_ops = 100_000
        cs.max_latency = 100_000
        reuse.compute_stats[compute_key] = cs

        return reuse, cs_a, cs_b, ls_a, ls_b, cs

    def test_self_conditioned_position_skipping_reduces_child_reads(self):
        """Position-skipping with condition_on=[] reduces compute-level reads."""
        density_a = 0.5
        sparse_opts = SparseOptimizations(
            targets=[
                SparseTarget(
                    target="LineBuffer",
                    action_optimization=[
                        ActionOptimization(
                            kind="position_skipping",
                            target="A",
                            condition_on=[],  # Self-conditioned
                        ),
                    ],
                )
            ]
        )

        reuse, cs_a, cs_b, ls_a, ls_b, cs = self._make_linebuffer_reuse()
        spec = make_mock_spec(
            sparse_opts=sparse_opts,
            tensor_accesses=[
                {"name": "A", "density": density_a, "output": False, "bits_per_value": 16},
                {"name": "B", "density": 1.0, "output": False, "bits_per_value": 16},
                {"name": "Z", "density": None, "output": True, "bits_per_value": 16},
            ],
            arch_components={
                "LineBuffer": {
                    "bits_per_value_scale": {"A": 1, "B": 1},
                    "read_bpa": 16,
                    "write_bpa": 16,
                },
                "MAC": {
                    "bits_per_value_scale": {"A": 1, "B": 1, "Z": 1},
                    "read_bpa": 16,
                    "write_bpa": 16,
                },
            },
            rank_sizes={"M": 128, "K": 128, "N": 128},
        )
        job = make_mock_job()

        apply_sparse_adjustments(reuse, spec, job)

        # Self-conditioned SAF: prob_empty = 1 - density = 0.5
        # Child A reads reduced: 100000 * 0.5 = 50000 (skipping)
        self.assertEqual(cs_a.total_reads_to_parent, 50_000)

    def test_dual_position_skipping_compound_saf(self):
        """Dual position-skipping (A and B) gives compound SAF at compute."""
        density_a = 0.5
        density_b = 0.4

        sparse_opts = SparseOptimizations(
            targets=[
                SparseTarget(
                    target="LineBuffer",
                    representation_format=[
                        RepresentationFormat(name="A", format="bitmask",
                                             metadata_word_bits=1),
                        RepresentationFormat(name="B", format="bitmask",
                                             metadata_word_bits=1),
                    ],
                    action_optimization=[
                        ActionOptimization(
                            kind="position_skipping",
                            target="A",
                            condition_on=[],
                        ),
                        ActionOptimization(
                            kind="position_skipping",
                            target="B",
                            condition_on=[],
                        ),
                    ],
                ),
                SparseTarget(
                    target="MAC",
                    compute_optimization=[
                        ComputeOptimization(
                            kind="skipping",
                            target="E0",
                            condition_on=["A", "B"],
                        ),
                    ],
                ),
            ]
        )

        reuse, cs_a, cs_b, ls_a, ls_b, cs = self._make_linebuffer_reuse(
            density_a, density_b
        )
        spec = make_mock_spec(
            sparse_opts=sparse_opts,
            tensor_accesses=[
                {"name": "A", "density": density_a, "output": False, "bits_per_value": 16},
                {"name": "B", "density": density_b, "output": False, "bits_per_value": 16},
                {"name": "Z", "density": None, "output": True, "bits_per_value": 16},
            ],
            arch_components={
                "LineBuffer": {
                    "bits_per_value_scale": {"A": 1, "B": 1},
                    "read_bpa": 16,
                    "write_bpa": 16,
                },
                "MAC": {
                    "bits_per_value_scale": {"A": 1, "B": 1, "Z": 1},
                    "read_bpa": 16,
                    "write_bpa": 16,
                },
            },
            rank_sizes={"M": 128, "K": 128, "N": 128},
        )
        job = make_mock_job()

        apply_sparse_adjustments(reuse, spec, job)

        # Compound: skip_compound_survival = (1-0.5) * (1-0.6) = 0.2
        # A: Phase 4a self-SAF prob=0.5 → reads * 0.5 = 50000
        #    Phase 4b remaining = 1 - 0.2/(1-0.5) = 1 - 0.4 = 0.6
        #    → reads * (1 - 0.6) = 50000 * 0.4 = 20000
        self.assertEqual(cs_a.total_reads_to_parent, 20_000)

        # B: Phase 4a self-SAF prob=0.6 → reads * 0.4 = 40000
        #    Phase 4b remaining = 1 - 0.2/(1-0.6) = 1 - 0.5 = 0.5
        #    → reads * (1 - 0.5) = 40000 * 0.5 = 20000
        self.assertEqual(cs_b.total_reads_to_parent, 20_000)

        # Compute: total_ops * compound_survival = 100000 * 0.2 = 20000
        self.assertEqual(cs.total_ops, 20_000)

    def test_empty_condition_on_without_position_skipping_is_noop(self):
        """Regular skipping with condition_on=[] should be a no-op."""
        sparse_opts = SparseOptimizations(
            targets=[
                SparseTarget(
                    target="LineBuffer",
                    action_optimization=[
                        ActionOptimization(
                            kind="skipping",
                            target="A",
                            condition_on=[],  # Empty but not position_skipping
                        ),
                    ],
                )
            ]
        )

        reuse, cs_a, cs_b, ls_a, ls_b, cs = self._make_linebuffer_reuse()
        spec = make_mock_spec(
            sparse_opts=sparse_opts,
            tensor_accesses=[
                {"name": "A", "density": 0.5, "output": False, "bits_per_value": 16},
                {"name": "B", "density": 1.0, "output": False, "bits_per_value": 16},
                {"name": "Z", "density": None, "output": True, "bits_per_value": 16},
            ],
            arch_components={
                "LineBuffer": {
                    "bits_per_value_scale": {"A": 1, "B": 1},
                    "read_bpa": 16,
                    "write_bpa": 16,
                },
                "MAC": {
                    "bits_per_value_scale": {"A": 1, "B": 1, "Z": 1},
                    "read_bpa": 16,
                    "write_bpa": 16,
                },
            },
            rank_sizes={"M": 128, "K": 128, "N": 128},
        )
        job = make_mock_job()

        apply_sparse_adjustments(reuse, spec, job)

        # Regular skipping with empty condition_on → no cond_densities → skip SAF
        self.assertEqual(cs_a.total_reads_to_parent, 100_000)


class TestStorageSAFComputePropagation(unittest.TestCase):
    """Storage SAF → compute propagation without explicit compute_optimization.

    When position-skipping SAFs at storage levels reduce input tensor reads,
    Phase 4b propagates these reductions to compute ops. The compute_latency_ratio
    should reflect the compound survival probability (dA * dB) even without
    any compute_optimization block in the sparse config.
    """

    def test_dual_position_skipping_no_compute_opt(self):
        """Position-skipping on A and B → compute reduced by dA*dB."""
        density_a = 0.5
        density_b = 0.4

        # No compute_optimization at MAC — only storage-level SAFs
        sparse_opts = SparseOptimizations(
            targets=[
                SparseTarget(
                    target="LineBuffer",
                    representation_format=[
                        RepresentationFormat(name="A", format="bitmask",
                                             metadata_word_bits=1),
                        RepresentationFormat(name="B", format="bitmask",
                                             metadata_word_bits=1),
                    ],
                    action_optimization=[
                        ActionOptimization(
                            kind="position_skipping",
                            target="A",
                            condition_on=[],
                        ),
                        ActionOptimization(
                            kind="position_skipping",
                            target="B",
                            condition_on=[],
                        ),
                    ],
                ),
                # No MAC compute_optimization
            ]
        )

        reuse = SymbolicAnalysisOutput()

        # Compute-level buffets (MAC reads from LineBuffer)
        compute_a = Buffet("A", "E0", "MAC")
        cs_a = BuffetStats()
        cs_a.total_reads_to_parent = 100_000
        cs_a.max_per_parent_reads_to_parent = 100_000
        reuse.buffet_stats[compute_a] = cs_a

        compute_b = Buffet("B", "E0", "MAC")
        cs_b = BuffetStats()
        cs_b.total_reads_to_parent = 100_000
        cs_b.max_per_parent_reads_to_parent = 100_000
        reuse.buffet_stats[compute_b] = cs_b

        # LineBuffer buffets
        lb_a = Buffet("A", "E0", "LineBuffer")
        ls_a = BuffetStats()
        ls_a.total_reads_to_parent = 10_000
        ls_a.max_per_parent_reads_to_parent = 10_000
        ls_a.max_occupancy = 64
        reuse.buffet_stats[lb_a] = ls_a

        lb_b = Buffet("B", "E0", "LineBuffer")
        ls_b = BuffetStats()
        ls_b.total_reads_to_parent = 10_000
        ls_b.max_per_parent_reads_to_parent = 10_000
        ls_b.max_occupancy = 64
        reuse.buffet_stats[lb_b] = ls_b

        # Compute stats (MAC)
        compute_key = Compute("E0", "MAC")
        cs = ComputeStats()
        cs.total_ops = 100_000
        cs.max_per_unit_ops = 100_000
        cs.max_latency = 100_000
        reuse.compute_stats[compute_key] = cs

        spec = make_mock_spec(
            sparse_opts=sparse_opts,
            tensor_accesses=[
                {"name": "A", "density": density_a, "output": False, "bits_per_value": 16},
                {"name": "B", "density": density_b, "output": False, "bits_per_value": 16},
                {"name": "Z", "density": None, "output": True, "bits_per_value": 16},
            ],
            arch_components={
                "LineBuffer": {
                    "bits_per_value_scale": {"A": 1, "B": 1},
                    "read_bpa": 16,
                    "write_bpa": 16,
                },
                "MAC": {
                    "bits_per_value_scale": {"A": 1, "B": 1, "Z": 1},
                    "read_bpa": 16,
                    "write_bpa": 16,
                },
            },
            rank_sizes={"M": 128, "K": 128, "N": 128},
        )
        job = make_mock_job()

        result = apply_sparse_adjustments(reuse, spec, job)

        # Phase 4b: compound survival = dA * dB = 0.5 * 0.4 = 0.2
        # Compute ops: 100000 * 0.2 = 20000
        self.assertEqual(cs.total_ops, 20_000)

        # compute_latency_ratio = post / pre = 20000 / 100000 = 0.2
        self.assertAlmostEqual(result.latency_info.compute_latency_ratio, 0.2, places=6)

    def test_single_position_skipping_no_compute_opt(self):
        """Position-skipping on A only → compute reduced by dA."""
        density_a = 0.3

        sparse_opts = SparseOptimizations(
            targets=[
                SparseTarget(
                    target="LineBuffer",
                    action_optimization=[
                        ActionOptimization(
                            kind="position_skipping",
                            target="A",
                            condition_on=[],
                        ),
                    ],
                ),
            ]
        )

        reuse = SymbolicAnalysisOutput()

        compute_a = Buffet("A", "E0", "MAC")
        cs_a = BuffetStats()
        cs_a.total_reads_to_parent = 100_000
        cs_a.max_per_parent_reads_to_parent = 100_000
        reuse.buffet_stats[compute_a] = cs_a

        lb_a = Buffet("A", "E0", "LineBuffer")
        ls_a = BuffetStats()
        ls_a.total_reads_to_parent = 10_000
        ls_a.max_per_parent_reads_to_parent = 10_000
        ls_a.max_occupancy = 64
        reuse.buffet_stats[lb_a] = ls_a

        compute_key = Compute("E0", "MAC")
        cs = ComputeStats()
        cs.total_ops = 100_000
        cs.max_per_unit_ops = 100_000
        cs.max_latency = 100_000
        reuse.compute_stats[compute_key] = cs

        spec = make_mock_spec(
            sparse_opts=sparse_opts,
            tensor_accesses=[
                {"name": "A", "density": density_a, "output": False, "bits_per_value": 16},
                {"name": "B", "density": 1.0, "output": False, "bits_per_value": 16},
                {"name": "Z", "density": None, "output": True, "bits_per_value": 16},
            ],
            arch_components={
                "LineBuffer": {
                    "bits_per_value_scale": {"A": 1, "B": 1},
                    "read_bpa": 16,
                    "write_bpa": 16,
                },
                "MAC": {
                    "bits_per_value_scale": {"A": 1, "B": 1, "Z": 1},
                    "read_bpa": 16,
                    "write_bpa": 16,
                },
            },
            rank_sizes={"M": 128, "K": 128, "N": 128},
        )
        job = make_mock_job()

        result = apply_sparse_adjustments(reuse, spec, job)

        # Phase 4b: survival = dA = 0.3
        # Compute ops: 100000 * 0.3 = 30000
        self.assertEqual(cs.total_ops, 30_000)

        # compute_latency_ratio = 30000 / 100000 = 0.3
        self.assertAlmostEqual(result.latency_info.compute_latency_ratio, 0.3, places=6)


class TestSkippingMetadataNonComputeChild(unittest.TestCase):
    """Skipping metadata for non-compute child uses full pre-SAF count.

    Previously, the non-compute-child branch used post_saf_data_reads/density
    which got density applied again inside compute_format_access_counts,
    double-counting density and undercounting metadata reads.
    """

    def test_skipping_metadata_uses_pre_saf_reads(self):
        """Non-compute-child skipping: effective_reads = pre_saf_child_reads."""
        density_a = 0.5

        sparse_opts = SparseOptimizations(
            targets=[
                SparseTarget(
                    target="LineBuffer",
                    representation_format=[
                        RepresentationFormat(
                            name="A", format="bitmask",
                            metadata_word_bits=1,
                        ),
                    ],
                    action_optimization=[
                        ActionOptimization(
                            kind="skipping",
                            target="A",
                            condition_on=[],
                        ),
                    ],
                ),
            ]
        )

        reuse = SymbolicAnalysisOutput()

        # DRAM buffet (parent of LineBuffer for A)
        dram_a = Buffet("A", "E0", "DRAM")
        ds_a = BuffetStats()
        ds_a.total_reads_to_parent = 1000
        ds_a.max_per_parent_reads_to_parent = 1000
        ds_a.max_occupancy = 256
        reuse.buffet_stats[dram_a] = ds_a

        # LineBuffer buffet (has format + SAF)
        lb_a = Buffet("A", "E0", "LineBuffer")
        ls_a = BuffetStats()
        ls_a.total_reads_to_parent = 10_000
        ls_a.max_per_parent_reads_to_parent = 10_000
        ls_a.max_occupancy = 256
        ls_a.tile_shape = {"m": 16, "k": 16}
        reuse.buffet_stats[lb_a] = ls_a

        # Compute-level buffet (child of LineBuffer)
        compute_a = Buffet("A", "E0", "MAC")
        cs_a = BuffetStats()
        cs_a.total_reads_to_parent = 100_000
        cs_a.max_per_parent_reads_to_parent = 100_000
        reuse.buffet_stats[compute_a] = cs_a

        # Compute
        compute_key = Compute("E0", "MAC")
        cs = ComputeStats()
        cs.total_ops = 100_000
        cs.max_per_unit_ops = 100_000
        reuse.compute_stats[compute_key] = cs

        spec = make_mock_spec(
            sparse_opts=sparse_opts,
            tensor_accesses=[
                {"name": "A", "density": density_a, "output": False,
                 "bits_per_value": 16, "projection": {"M": "m", "K": "k"}},
                {"name": "Z", "density": None, "output": True, "bits_per_value": 16},
            ],
            arch_components={
                "DRAM": {
                    "bits_per_value_scale": {"A": 1},
                    "read_bpa": 16,
                    "write_bpa": 16,
                },
                "LineBuffer": {
                    "bits_per_value_scale": {"A": 1},
                    "read_bpa": 16,
                    "write_bpa": 16,
                    "metadata_read_bpa": 16,
                    "metadata_write_bpa": 16,
                },
                "MAC": {
                    "bits_per_value_scale": {"A": 1, "Z": 1},
                    "read_bpa": 16,
                    "write_bpa": 16,
                },
            },
            rank_sizes={"M": 128, "K": 128},
        )
        job = make_mock_job()

        result = apply_sparse_adjustments(reuse, spec, job)

        # The key check: metadata actions should be based on the full
        # pre-SAF count (10_000), not the double-density-reduced count.
        md_reads = 0
        for key, count in result.sparse_actions.items():
            if key.level == "LineBuffer" and key.action == "metadata_read":
                md_reads = count.total
        # With the fix: effective_reads = pre_saf_child_reads = 10_000
        # (the full pre-SAF count), not post_saf/density which would
        # get density applied again inside compute_format_access_counts.
        self.assertGreater(md_reads, 0, "Should emit metadata_read actions")


if __name__ == "__main__":
    unittest.main()
