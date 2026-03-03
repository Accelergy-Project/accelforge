"""Tests for Phases 5-6: SAF probability, format compression, local SAF,
SAF propagation, and compute classification.

Validation tests sourced from ARTIFACT_EVALUATION.md and IMPLEMENTATION_PLAN.md.
"""

import unittest

from accelforge.model.sparse_pipeline import (
    compute_saf_probability,
    apply_format_compression,
    apply_local_saf_reads,
    propagate_saf_reduction,
    compute_nested_saf_effective_prob,
    classify_compute,
    ComputeClassification,
    compute_operand_states,
    _round6,
)


# ---------------------------------------------------------------------------
# Phase 5: SAF probability
# ---------------------------------------------------------------------------


class TestSAFProbability(unittest.TestCase):
    """Test SAF optimization probability computation."""

    def test_buffer_a_gated_on_b_scalar(self):
        """Fig1: Buffer A gated on B, d=0.1015625, scalar.
        P(B=0) = 1 - 0.1015625 = 0.8984375."""
        prob = compute_saf_probability([0.1015625])
        self.assertAlmostEqual(prob, 0.8984375)

    def test_buffer_b_gated_on_a_scalar(self):
        """Fig1: Buffer B gated on A, d=0.1015625, scalar.
        Symmetric: same probability."""
        prob = compute_saf_probability([0.1015625])
        self.assertAlmostEqual(prob, 0.8984375)

    def test_reg_z_gated_on_ab_scalar(self):
        """Fig1: Reg Z gated on [A, B], d_A=d_B=0.1015625, scalar.
        P(at least one zero) = 1 - dA*dB = 0.98968505859375."""
        prob = compute_saf_probability([0.1015625, 0.1015625])
        self.assertAlmostEqual(prob, 0.98968505859375, places=12)

    def test_verify_from_action_ratio(self):
        """Verify SAF prob matches fig1 action ratio: 191360/212992."""
        prob = compute_saf_probability([0.1015625])
        self.assertAlmostEqual(prob, 191360 / 212992)

    def test_single_condition_high_density(self):
        """d=0.9: prob = 1 - 0.9 = 0.1."""
        prob = compute_saf_probability([0.9])
        self.assertAlmostEqual(prob, 0.1)

    def test_condition_all_zero(self):
        """d=0.0: condition tensor is all zeros -> prob = 1.0."""
        prob = compute_saf_probability([0.0])
        self.assertAlmostEqual(prob, 1.0)

    def test_condition_all_dense(self):
        """d=1.0: condition tensor is fully dense -> prob = 0.0."""
        prob = compute_saf_probability([1.0])
        self.assertAlmostEqual(prob, 0.0)

    def test_multi_condition_mixed(self):
        """d_A=0.5, d_B=0.25: prob = 1 - 0.5*0.25 = 0.875."""
        prob = compute_saf_probability([0.5, 0.25])
        self.assertAlmostEqual(prob, 0.875)

    def test_hypergeometric_tiled(self):
        """Tiled SAF: tile=128, tensor=16384, d=0.1015625.
        P(tile empty) = hypergeometric P(k=0 | N=16384, r=1664, n=128).
        This should be very close to zero (almost impossible for 128
        elements to all be zero when density is 10%)."""
        prob = compute_saf_probability(
            [0.1015625],
            condition_on_tile_shapes=[128],
            condition_on_tensor_sizes=[16384],
        )
        # P(tile nonempty) is very high, so prob is very low
        self.assertGreater(prob, 0.0)
        self.assertLess(prob, 0.01)

    def test_scalar_same_as_no_tile(self):
        """tile=1 should give same result as no tile specified."""
        prob_no_tile = compute_saf_probability([0.5])
        prob_tile_1 = compute_saf_probability(
            [0.5],
            condition_on_tile_shapes=[1],
            condition_on_tensor_sizes=[100],
        )
        self.assertAlmostEqual(prob_no_tile, prob_tile_1)


# ---------------------------------------------------------------------------
# Phase 5: Format compression (Phase 4a)
# ---------------------------------------------------------------------------


class TestFormatCompression(unittest.TestCase):
    """Test Phase 4a: compressed format impact on data accesses."""

    def test_buffer_a_reads(self):
        """Fig1: Buffer A, alg=2097152, d=0.1015625 -> random=212992.
        sparsity=0.8984375. floor(2097152*0.8984375)=1884160.
        random = 2097152 - 1884160 = 212992."""
        result = apply_format_compression(2097152, 0.1015625)
        self.assertEqual(result, 212992)

    def test_buffer_a_fills(self):
        """Fig1: Buffer A fills also reduced: 2097152 -> 212992."""
        result = apply_format_compression(2097152, 0.1015625)
        self.assertEqual(result, 212992)

    def test_density_one_no_reduction(self):
        """d=1.0: no sparsity -> no reduction."""
        result = apply_format_compression(1000, 1.0)
        self.assertEqual(result, 1000)

    def test_density_zero_all_removed(self):
        """d=0.0: all elements zero -> 0 random accesses."""
        result = apply_format_compression(1000, 0.0)
        self.assertEqual(result, 0)

    def test_small_example_floor(self):
        """7 accesses, d=0.3: sparsity=0.7. floor(7*0.7)=floor(4.9)=4.
        random = 7 - 4 = 3."""
        result = apply_format_compression(7, 0.3)
        self.assertEqual(result, 3)

    def test_exact_density(self):
        """100 accesses, d=0.25: sparsity=0.75. floor(100*0.75)=75.
        random = 25."""
        result = apply_format_compression(100, 0.25)
        self.assertEqual(result, 25)

    def test_zero_accesses(self):
        """0 accesses -> 0."""
        result = apply_format_compression(0, 0.5)
        self.assertEqual(result, 0)


# ---------------------------------------------------------------------------
# Phase 5: Local SAF (Phase 4b)
# ---------------------------------------------------------------------------


class TestLocalSAFReads(unittest.TestCase):
    """Test Phase 4b: local SAF on reads."""

    def test_buffer_a_read_only(self):
        """Fig1: Buffer A, random=212992, p=0.8984375, read-only.
        gated = floor(212992 * 0.8984375) = 191360.
        actual = 212992 - 191360 = 21632."""
        actual, gated = apply_local_saf_reads(212992, 0.8984375, is_read_write=False)
        self.assertEqual(actual, 21632)
        self.assertEqual(gated, 191360)

    def test_reg_z_read_write(self):
        """Fig1: Reg Z, random=2080768, p=0.98968505859375, read-write.
        gated = ceil(2080768 * 0.98968505859375) = 2059305.
        actual = 2080768 - 2059305 = 21463."""
        actual, gated = apply_local_saf_reads(
            2080768, 0.98968505859375, is_read_write=True
        )
        self.assertEqual(actual, 21463)
        self.assertEqual(gated, 2059305)

    def test_rounding_asymmetry_read_only(self):
        """7 reads, p=0.3, read-only: gated = floor(7*0.3) = floor(2.1) = 2.
        actual = 5."""
        actual, gated = apply_local_saf_reads(7, 0.3, is_read_write=False)
        self.assertEqual(gated, 2)
        self.assertEqual(actual, 5)

    def test_rounding_asymmetry_read_write(self):
        """7 reads, p=0.3, read-write: gated = ceil(7*0.3) = ceil(2.1) = 3.
        actual = 4."""
        actual, gated = apply_local_saf_reads(7, 0.3, is_read_write=True)
        self.assertEqual(gated, 3)
        self.assertEqual(actual, 4)

    def test_zero_prob(self):
        """p=0: no gating/skipping."""
        actual, gated = apply_local_saf_reads(1000, 0.0)
        self.assertEqual(actual, 1000)
        self.assertEqual(gated, 0)

    def test_full_prob(self):
        """p=1.0: all gated, read-only."""
        actual, gated = apply_local_saf_reads(1000, 1.0, is_read_write=False)
        self.assertEqual(actual, 0)
        self.assertEqual(gated, 1000)

    def test_zero_reads(self):
        """0 reads -> 0."""
        actual, gated = apply_local_saf_reads(0, 0.5)
        self.assertEqual(actual, 0)
        self.assertEqual(gated, 0)


class TestLocalSAFUpdates(unittest.TestCase):
    """Test Phase 4b: local SAF on updates."""

    def test_reg_z_updates(self):
        """Fig1: Reg Z updates, random=2097152, p=0.98968505859375.
        gated = floor(2097152 * 0.98968505859375) = 2075520.
        actual = 2097152 - 2075520 = 21632."""
        actual, gated = apply_local_saf_reads(2097152, 0.98968505859375)
        self.assertEqual(actual, 21632)
        self.assertEqual(gated, 2075520)

    def test_rounding_difference_from_reads(self):
        """Reads vs updates rounding asymmetry.
        Reg Z: actual_reads=21463, actual_updates=21632.
        Difference = 169, solely from floor vs ceil."""
        actual_reads, _ = apply_local_saf_reads(
            2080768, 0.98968505859375, is_read_write=True
        )
        actual_updates, _ = apply_local_saf_reads(
            2097152, 0.98968505859375
        )
        self.assertEqual(actual_updates - actual_reads, 169)

    def test_updates_use_floor(self):
        """7 updates, p=0.3: gated = floor(2.1) = 2, actual = 5.
        Same as read-only reads (both use floor)."""
        actual, gated = apply_local_saf_reads(7, 0.3)
        self.assertEqual(gated, 2)
        self.assertEqual(actual, 5)

    def test_zero_prob(self):
        """p=0: no reduction."""
        actual, gated = apply_local_saf_reads(1000, 0.0)
        self.assertEqual(actual, 1000)
        self.assertEqual(gated, 0)


# ---------------------------------------------------------------------------
# Phase 6: SAF propagation
# ---------------------------------------------------------------------------


class TestSAFPropagation(unittest.TestCase):
    """Test Phase 4b: top-down SAF propagation."""

    def test_single_saf_to_compute(self):
        """Fig1: A SAF p=0.8984375, propagation to compute level.
        remaining = 2097152 - floor(2097152 * 0.8984375) = 212992."""
        result = propagate_saf_reduction(2097152, 0.8984375)
        self.assertEqual(result, 212992)

    def test_two_sequential_safs(self):
        """Fig1: A then B SAFs, both p=0.8984375.
        After A: 2097152 -> 212992. After B: 212992 -> 21632."""
        after_a = propagate_saf_reduction(2097152, 0.8984375)
        self.assertEqual(after_a, 212992)
        after_b = propagate_saf_reduction(after_a, 0.8984375)
        self.assertEqual(after_b, 21632)

    def test_propagation_matches_effectual_ops(self):
        """Two sequential propagations should give ~= effectual_operations.
        2097152 -> 212992 -> 21632 = round(2097152 * 0.1015625^2) = 21632."""
        from accelforge.model.density_model import effectual_operations
        after_a = propagate_saf_reduction(2097152, 0.8984375)
        after_b = propagate_saf_reduction(after_a, 0.8984375)
        expected = effectual_operations(2097152, 0.1015625, 0.1015625)
        self.assertEqual(after_b, expected)

    def test_zero_prob_no_change(self):
        """p=0: no propagation."""
        result = propagate_saf_reduction(1000, 0.0)
        self.assertEqual(result, 1000)

    def test_full_prob(self):
        """p=1.0: all removed."""
        result = propagate_saf_reduction(1000, 1.0)
        self.assertEqual(result, 0)

    def test_zero_count(self):
        """Count=0: stays 0."""
        result = propagate_saf_reduction(0, 0.5)
        self.assertEqual(result, 0)


class TestNestedSAF(unittest.TestCase):
    """Test nested SAF effective probability computation."""

    def test_basic_nesting(self):
        """outer=0.5, local=0.8 -> effective = 1 - (1-0.8)/(1-0.5) = 0.6."""
        eff = compute_nested_saf_effective_prob(0.8, 0.5)
        self.assertAlmostEqual(eff, 0.6)

    def test_no_outer(self):
        """outer=0.0: effective = local."""
        eff = compute_nested_saf_effective_prob(0.7, 0.0)
        self.assertAlmostEqual(eff, 0.7)

    def test_outer_equals_local(self):
        """outer=local: effective = 0.0 (outer handles everything)."""
        eff = compute_nested_saf_effective_prob(0.5, 0.5)
        self.assertAlmostEqual(eff, 0.0)

    def test_outer_full(self):
        """outer=1.0: effective = 0.0 (outer already catches all)."""
        eff = compute_nested_saf_effective_prob(0.8, 1.0)
        self.assertAlmostEqual(eff, 0.0)

    def test_small_local_large_outer(self):
        """outer=0.9, local=0.95 -> effective = 1 - (0.05)/(0.1) = 0.5."""
        eff = compute_nested_saf_effective_prob(0.95, 0.9)
        self.assertAlmostEqual(eff, 0.5)


# ---------------------------------------------------------------------------
# Phase 6: Compute classification
# ---------------------------------------------------------------------------


class TestComputeClassification(unittest.TestCase):
    """Test Phase 5: 3-state compute classification."""

    def test_fig1_no_optimization(self):
        """Fig1 without compute optimization: all random."""
        cc = classify_compute(2097152, [0.1015625, 0.1015625])
        self.assertEqual(cc.random_compute, 2097152)
        self.assertEqual(cc.gated_compute, 0)
        self.assertEqual(cc.skipped_compute, 0)
        self.assertEqual(cc.total, 2097152)

    def test_fig1_with_gating(self):
        """Fig1 with gating: random=21632, gated=2075520."""
        cc = classify_compute(
            2097152, [0.1015625, 0.1015625], compute_optimization_kind="gating"
        )
        self.assertEqual(cc.random_compute, 21632)
        self.assertEqual(cc.gated_compute, 2097152 - 21632)
        self.assertEqual(cc.skipped_compute, 0)
        self.assertEqual(cc.total, 2097152)

    def test_fig1_with_skipping_no_metadata(self):
        """Fig1 with skipping but no metadata: 9-state model has no NE states.

        Without metadata (default), all elements exist (ENZ or EZ).
        Skipping requires NE (not-exist) states to identify absent elements.
        So with has_metadata=[False, False], no skipping occurs.
        """
        cc = classify_compute(
            2097152, [0.1015625, 0.1015625], compute_optimization_kind="skipping"
        )
        # No metadata → no NE → no skipping; all are random
        self.assertEqual(cc.random_compute, 2097152)
        self.assertEqual(cc.skipped_compute, 0)
        self.assertEqual(cc.total, 2097152)

    def test_fig1_with_skipping_with_metadata(self):
        """Fig1 with skipping and metadata: NE states enable skipping.

        With metadata on both operands (compressed format), absent elements
        are NE (not-exist). Skipping filters NE combinations.
        """
        cc = classify_compute(
            2097152, [0.1015625, 0.1015625],
            compute_optimization_kind="skipping",
            operand_has_metadata=[True, True],
        )
        self.assertEqual(cc.random_compute, 21633)
        self.assertGreater(cc.skipped_compute, 0)
        self.assertEqual(cc.gated_compute, 0)
        self.assertEqual(cc.total, 2097152)

    def test_lab4_part1_gating(self):
        """Lab 4 Part 1+2: total=512, d=[0.25, 0.5], gating.
        random = round(512 * 0.25 * 0.5) = 64.
        gated = 512 - 64 = 448."""
        cc = classify_compute(512, [0.25, 0.5], compute_optimization_kind="gating")
        self.assertEqual(cc.random_compute, 64)
        self.assertEqual(cc.gated_compute, 448)
        self.assertEqual(cc.skipped_compute, 0)

    def test_lab4_part1_skipping_no_metadata(self):
        """Lab 4 Part 3: skipping without metadata → no skipping (9-state model)."""
        cc = classify_compute(512, [0.25, 0.5], compute_optimization_kind="skipping")
        # No metadata → no NE → no skipping
        self.assertEqual(cc.random_compute, 512)
        self.assertEqual(cc.skipped_compute, 0)

    def test_lab4_part1_skipping_with_metadata(self):
        """Lab 4 Part 3: skipping with metadata → NE states enable skipping."""
        cc = classify_compute(
            512, [0.25, 0.5],
            compute_optimization_kind="skipping",
            operand_has_metadata=[True, True],
        )
        self.assertGreater(cc.skipped_compute, 0)
        self.assertEqual(cc.gated_compute, 0)
        self.assertEqual(cc.total, 512)

    def test_one_zero_operand_gating(self):
        """One operand at d=0: all ineffectual -> random=0, gated=1000."""
        cc = classify_compute(1000, [0.5, 0.0], compute_optimization_kind="gating")
        self.assertEqual(cc.random_compute, 0)
        self.assertEqual(cc.gated_compute, 1000)

    def test_unknown_kind_raises(self):
        """Unknown optimization kind should raise ValueError."""
        with self.assertRaises(ValueError):
            classify_compute(100, [0.5], compute_optimization_kind="bogus")


class TestComputeClassificationOperandStates(unittest.TestCase):
    """Test 3-state operand classification (ENZ/EZ/NE).

    IMPLEMENTATION_PLAN.md Phase 6:
    - Dense operands (no metadata) -> P(ENZ)=d, P(EZ)=1-d, P(NE)=0
    - Compressed operands (with metadata) -> P(ENZ)=d, P(EZ)=0, P(NE)=1-d
    """

    def test_dense_operands_gating_states(self):
        """Dense (no metadata): d=0.3. With gating:
        effectual = floor(100 * 0.3) = 30 (assuming single operand).
        gated = 100 - 30 = 70 (these were EZ cases)."""
        cc = classify_compute(100, [0.3], compute_optimization_kind="gating")
        self.assertEqual(cc.random_compute, 30)
        self.assertEqual(cc.gated_compute, 70)
        # No skipped because gating doesn't skip
        self.assertEqual(cc.skipped_compute, 0)

    def test_dense_operands_no_ne(self):
        """Dense single operand with skipping: uses simple product model.

        Single-operand path is backward-compatible (not 9-state).
        """
        cc = classify_compute(100, [0.3], compute_optimization_kind="skipping")
        self.assertEqual(cc.random_compute, 30)
        self.assertEqual(cc.skipped_compute, 70)

    def test_two_dense_operands_gating(self):
        """Two dense operands, gating: 9-state pessimistic floor rounding.

        p_enz_enz = 0.3 * 0.4 = 0.12; p_gated = 1 - 0.12 = 0.88
        gated = floor(1000 * 0.88) = 879 (floor of 879.999... due to FP)
        random = 1000 - 879 = 121
        """
        cc = classify_compute(1000, [0.3, 0.4], compute_optimization_kind="gating")
        self.assertEqual(cc.random_compute, 121)
        self.assertEqual(cc.gated_compute, 879)

    def test_no_compute_opt_all_random(self):
        """fig1: no compute_optimization → all remaining computes are random."""
        cc = classify_compute(21632, [0.1015625, 0.1015625])
        self.assertEqual(cc.random_compute, 21632)
        self.assertEqual(cc.gated_compute, 0)
        self.assertEqual(cc.skipped_compute, 0)


# ---------------------------------------------------------------------------
# 9-state compute model tests
# ---------------------------------------------------------------------------


class TestRound6(unittest.TestCase):
    """Test _round6 precision helper."""

    def test_exact(self):
        self.assertEqual(_round6(0.5), 0.5)

    def test_rounding(self):
        self.assertEqual(_round6(0.10156250001), 0.101563)

    def test_zero(self):
        self.assertEqual(_round6(0.0), 0.0)

    def test_one(self):
        self.assertEqual(_round6(1.0), 1.0)


class TestOperandStates(unittest.TestCase):
    """Test compute_operand_states per-operand state probabilities."""

    def test_dense_no_metadata(self):
        """No metadata: P(ENZ)=d, P(EZ)=1-d, P(NE)=0."""
        s = compute_operand_states(0.3, has_metadata=False)
        self.assertAlmostEqual(s.p_enz, 0.3)
        self.assertAlmostEqual(s.p_ez, 0.7)
        self.assertEqual(s.p_ne, 0.0)

    def test_compressed_with_metadata(self):
        """With metadata: P(ENZ)=d, P(EZ)=0, P(NE)=1-d."""
        s = compute_operand_states(0.3, has_metadata=True)
        self.assertAlmostEqual(s.p_enz, 0.3)
        self.assertEqual(s.p_ez, 0.0)
        self.assertAlmostEqual(s.p_ne, 0.7)

    def test_density_zero_no_metadata(self):
        """d=0, no metadata: all EZ."""
        s = compute_operand_states(0.0, has_metadata=False)
        self.assertEqual(s.p_enz, 0.0)
        self.assertEqual(s.p_ez, 1.0)
        self.assertEqual(s.p_ne, 0.0)

    def test_density_zero_with_metadata(self):
        """d=0, with metadata: all NE."""
        s = compute_operand_states(0.0, has_metadata=True)
        self.assertEqual(s.p_enz, 0.0)
        self.assertEqual(s.p_ez, 0.0)
        self.assertEqual(s.p_ne, 1.0)

    def test_density_one_no_metadata(self):
        """d=1, no metadata: all ENZ."""
        s = compute_operand_states(1.0, has_metadata=False)
        self.assertEqual(s.p_enz, 1.0)
        self.assertEqual(s.p_ez, 0.0)
        self.assertEqual(s.p_ne, 0.0)

    def test_density_one_with_metadata(self):
        """d=1, with metadata: all ENZ, P(NE)=0."""
        s = compute_operand_states(1.0, has_metadata=True)
        self.assertEqual(s.p_enz, 1.0)
        self.assertEqual(s.p_ez, 0.0)
        self.assertEqual(s.p_ne, 0.0)

    def test_round6_applied(self):
        """Density should be rounded to 6 decimals."""
        s = compute_operand_states(0.1015625, has_metadata=True)
        self.assertEqual(s.p_enz, 0.101562)  # _round6(0.1015625) — banker's rounding
        self.assertAlmostEqual(s.p_ne, 1.0 - 0.101562)


class TestNineStateCompute(unittest.TestCase):
    """Test 9-state compute classification model."""

    def test_both_metadata_gating(self):
        """Both operands have metadata, gating.

        With metadata: NE terms exist but gating maps them to gated.
        Result should be same as gating without metadata for joint
        (ENZ,ENZ) → random, everything else → gated.
        """
        cc = classify_compute(
            1000, [0.5, 0.5], "gating",
            operand_has_metadata=[True, True],
        )
        self.assertEqual(cc.random_compute + cc.gated_compute
                         + cc.skipped_compute + cc.nonexistent_compute, 1000)
        # ENZ×ENZ = 0.25 → random ≈ 250
        # NE×NE = 0.25 → nonexistent
        self.assertEqual(cc.random_compute, 250)
        self.assertEqual(cc.nonexistent_compute, 250)
        self.assertEqual(cc.gated_compute, 500)
        self.assertEqual(cc.skipped_compute, 0)

    def test_both_metadata_skipping(self):
        """Both operands have metadata, skipping.

        ENZ×ENZ → random, ENZ×NE/NE×ENZ → skipped,
        NE×NE → nonexistent.
        No EZ states (both have metadata).
        """
        cc = classify_compute(
            1000, [0.5, 0.5], "skipping",
            operand_has_metadata=[True, True],
        )
        self.assertEqual(cc.total, 1000)
        # ENZ×ENZ = 0.25 → random
        self.assertEqual(cc.random_compute, 250)
        # ENZ×NE + NE×ENZ = 0.25 + 0.25 = 0.5 → skipped
        self.assertEqual(cc.skipped_compute, 500)
        # NE×NE = 0.25 → nonexistent
        self.assertEqual(cc.nonexistent_compute, 250)
        self.assertEqual(cc.gated_compute, 0)

    def test_mixed_metadata_skipping(self):
        """One operand with metadata, one without.

        Op0: d=0.5, has_metadata=True → ENZ=0.5, EZ=0, NE=0.5
        Op1: d=0.5, has_metadata=False → ENZ=0.5, EZ=0.5, NE=0

        Joint:
          (ENZ,ENZ)=0.25 → random
          (ENZ,EZ)=0.25 → random (skipping, EZ is random)
          (NE,ENZ)=0.25 → skipped
          (NE,EZ)=0.25 → skipped
          NE×NE = 0 → nonexistent
        """
        cc = classify_compute(
            1000, [0.5, 0.5], "skipping",
            operand_has_metadata=[True, False],
        )
        self.assertEqual(cc.total, 1000)
        self.assertEqual(cc.random_compute, 500)  # 0.25 + 0.25
        self.assertEqual(cc.skipped_compute, 500)  # 0.25 + 0.25
        self.assertEqual(cc.nonexistent_compute, 0)

    def test_nonexistent_compute_field(self):
        """Verify nonexistent_compute included in total."""
        cc = ComputeClassification(
            random_compute=100, gated_compute=50,
            skipped_compute=25, nonexistent_compute=25,
        )
        self.assertEqual(cc.total, 200)

    def test_nonexistent_compute_default(self):
        """Backward compat: nonexistent_compute defaults to 0."""
        cc = ComputeClassification(
            random_compute=100, gated_compute=50, skipped_compute=25,
        )
        self.assertEqual(cc.nonexistent_compute, 0)
        self.assertEqual(cc.total, 175)

    def test_no_optimization_includes_nonexistent(self):
        """No compute optimization: all random, nonexistent=0."""
        cc = classify_compute(1000, [0.5, 0.5])
        self.assertEqual(cc.random_compute, 1000)
        self.assertEqual(cc.nonexistent_compute, 0)

    def test_density_zero_both_metadata_gating(self):
        """d=0 on both with metadata: all NE×NE = nonexistent."""
        cc = classify_compute(
            1000, [0.0, 0.0], "gating",
            operand_has_metadata=[True, True],
        )
        self.assertEqual(cc.nonexistent_compute, 1000)
        self.assertEqual(cc.random_compute, 0)
        self.assertEqual(cc.gated_compute, 0)

    def test_density_one_both_metadata_skipping(self):
        """d=1 on both: all ENZ×ENZ = random."""
        cc = classify_compute(
            1000, [1.0, 1.0], "skipping",
            operand_has_metadata=[True, True],
        )
        self.assertEqual(cc.random_compute, 1000)
        self.assertEqual(cc.skipped_compute, 0)
        self.assertEqual(cc.nonexistent_compute, 0)


if __name__ == "__main__":
    unittest.main()
