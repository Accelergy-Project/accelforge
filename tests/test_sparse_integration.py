"""End-to-end integration tests for the sparse pipeline.

Loads full YAML specs (arch + workload + mapping + sparse_opts), runs
evaluate_mapping, and compares action counts against expected values.

Per IMPLEMENTATION_PLAN.md Phase 7 validation tests.

Loop order: n (outer) → m → k (inner).
A is below both N and M loops → A fills = N*M*K (no N-reuse).
B is below N, above M → B fills = N*K (reused across M).

Sparseloop reference values from ARTIFACT_EVALUATION.md fig1 d=0.1015625.
Where AccelForge differs from Sparseloop, the test documents why.
"""

import os
import unittest

from accelforge.frontend.spec import Spec
from accelforge.model.main import evaluate_mapping

FIG1_DIR = os.path.join(os.path.dirname(__file__), "input_files", "fig1")


def _load_fig1(*extra_yamls):
    """Load fig1 base spec with optional extra YAML files (e.g. sparse)."""
    files = [
        os.path.join(FIG1_DIR, "arch.yaml"),
        os.path.join(FIG1_DIR, "workload.yaml"),
        os.path.join(FIG1_DIR, "mapping.yaml"),
    ]
    for f in extra_yamls:
        files.append(os.path.join(FIG1_DIR, f))
    spec = Spec.from_yaml(*files)
    return evaluate_mapping(spec)


def _get_action(result, component, tensor, action):
    """Get a specific action count from the result."""
    actions = result.actions(per_component=True, per_tensor=True)
    key = (component, tensor, action)
    return int(actions.get(key, 0))


# ===========================================================================
# Dense baseline — no sparse optimizations
# ===========================================================================


class TestDenseBaseline(unittest.TestCase):
    """Dense (no sparse_opts) baseline — regression test.

    Sparseloop reference: fig1 dense stats.
    M = K = N = 128. Total computes = 128^3 = 2,097,152.
    """

    @classmethod
    def setUpClass(cls):
        cls.result = _load_fig1()

    # --- Buffer reads (child demand) ---

    def test_buffer_a_reads(self):
        """Dense: Buffer A reads = M*N*K = 2,097,152."""
        self.assertEqual(_get_action(self.result, "Buffer", "A", "read"), 2_097_152)

    def test_buffer_b_reads(self):
        """Dense: Buffer B reads = M*N*K = 2,097,152."""
        self.assertEqual(_get_action(self.result, "Buffer", "B", "read"), 2_097_152)

    # --- Buffer writes (fills from BackingStorage) ---

    def test_buffer_a_writes(self):
        """Dense: Buffer A writes (fills) = N*M*K = 2,097,152.

        A is below both N and M loops → re-filled every (n,m) pair.
        Sparseloop reference: 2,097,152. Matches.
        """
        self.assertEqual(_get_action(self.result, "Buffer", "A", "write"), 2_097_152)

    def test_buffer_b_writes(self):
        """Dense: Buffer B writes (fills) = N*K = 16,384.

        B is below N, above M → reused across M=128 iterations.
        Sparseloop reference: 16,384. Matches.
        """
        self.assertEqual(_get_action(self.result, "Buffer", "B", "write"), 16_384)

    # --- MAC ---

    def test_mac_computes(self):
        """Dense: MAC computes = M*N*K = 2,097,152."""
        self.assertEqual(_get_action(self.result, "MAC", "None", "compute"), 2_097_152)

    # --- BackingStorage ---

    def test_backing_storage_a_reads(self):
        """Dense: BackingStorage A reads = N*M*K = 2,097,152.

        A below both N and M → no reuse. Sparseloop reference: 2,097,152. Matches.
        """
        self.assertEqual(_get_action(self.result, "BackingStorage", "A", "read"), 2_097_152)

    def test_backing_storage_b_reads(self):
        """Dense: BackingStorage B reads = N*K = 16,384.

        B below N, reused across M. Sparseloop reference: 16,384. Matches.
        """
        self.assertEqual(_get_action(self.result, "BackingStorage", "B", "read"), 16_384)

    def test_backing_storage_z_writes(self):
        """Dense: BackingStorage Z writes = M*N = 16,384."""
        self.assertEqual(_get_action(self.result, "BackingStorage", "Z", "write"), 16_384)

    # --- Reg (zero-cost pass-through for A/B, accumulator for Z) ---

    def test_reg_z_reads(self):
        """Dense: Reg Z reads = M*N*(K-1) = 2,080,768.

        First accumulation along k doesn't read old Z (initialized to 0).
        Matches Sparseloop reference: 2,080,768.
        """
        self.assertEqual(_get_action(self.result, "Reg", "Z", "read"), 2_080_768)

    def test_reg_z_writes(self):
        """Dense: Reg Z writes = M*N*K = 2,097,152.

        Sparseloop reference: 2,097,152 (updates). Matches.
        """
        self.assertEqual(_get_action(self.result, "Reg", "Z", "write"), 2_097_152)

    def test_regpassthrough_a_reads(self):
        """Dense: RegPassthrough A reads = M*N*K = 2,097,152 (pass-through)."""
        self.assertEqual(_get_action(self.result, "RegPassthrough", "A", "read"), 2_097_152)

    def test_regpassthrough_a_writes(self):
        """Dense: RegPassthrough A writes = M*N*K = 2,097,152 (fills from Buffer A)."""
        self.assertEqual(_get_action(self.result, "RegPassthrough", "A", "write"), 2_097_152)

    def test_regpassthrough_b_reads(self):
        """Dense: RegPassthrough B reads = M*N*K = 2,097,152 (pass-through)."""
        self.assertEqual(_get_action(self.result, "RegPassthrough", "B", "read"), 2_097_152)

    def test_regpassthrough_b_writes(self):
        """Dense: RegPassthrough B writes = M*N*K = 2,097,152 (fills from Buffer B)."""
        self.assertEqual(_get_action(self.result, "RegPassthrough", "B", "write"), 2_097_152)


# ===========================================================================
# Bitmask sparse — gating SAF
# ===========================================================================


class TestBitmaskSparse(unittest.TestCase):
    """Bitmask format (gating SAF) — fig1 d_A=d_B=0.1015625.

    Sparseloop reference values from ARTIFACT_EVALUATION.md §2.
    """

    @classmethod
    def setUpClass(cls):
        cls.result = _load_fig1("sparse_bitmask.yaml")

    # --- Buffer reads (post-compression + post-SAF) ---

    def test_buffer_a_reads(self):
        """Bitmask Buffer A reads = 21,632.

        Phase 2 format compression: 2,097,152 * d_A = 212,992
        Phase 4b SAF (gating on B): 212,992 - floor(212,992 * 0.8984375) = 21,632
        Sparseloop reference: 21,632. Matches.
        """
        self.assertEqual(_get_action(self.result, "Buffer", "A", "read"), 21_632)

    def test_buffer_b_reads(self):
        """Bitmask Buffer B reads = 21,632 (symmetric to A).

        Sparseloop reference: 21,632. Matches.
        """
        self.assertEqual(_get_action(self.result, "Buffer", "B", "read"), 21_632)

    # --- Buffer writes (compressed fills) ---

    def test_buffer_a_writes(self):
        """Bitmask Buffer A writes (fills) = N*M*K * d_A = 212,992.

        A below both N and M loops → fills = 2,097,152.
        Compressed by density: 2,097,152 - floor(2,097,152 * 0.8984375) = 212,992.
        Sparseloop reference: 212,992. Matches.
        """
        self.assertEqual(_get_action(self.result, "Buffer", "A", "write"), 212_992)

    def test_buffer_b_writes(self):
        """Bitmask Buffer B writes (fills) = N*K * d_B = 1,664.

        B fills = 16,384. Compressed: 16,384 - floor(16,384 * 0.8984375) = 1,664.
        Sparseloop reference: 1,664. Matches.
        """
        self.assertEqual(_get_action(self.result, "Buffer", "B", "write"), 1_664)

    # --- BackingStorage ---

    def test_backing_storage_a_reads(self):
        """Bitmask BackingStorage A reads = N*M*K * d_A = 212,992.

        Sparseloop reference: 212,992. Matches.
        """
        self.assertEqual(_get_action(self.result, "BackingStorage", "A", "read"), 212_992)

    def test_backing_storage_b_reads(self):
        """Bitmask BackingStorage B reads = N*K * d_B = 1,664.

        B below N, above M — filled N times, reused across M.
        Sparseloop reference: 1,664. Matches.
        """
        self.assertEqual(_get_action(self.result, "BackingStorage", "B", "read"), 1_664)

    def test_backing_storage_z_writes(self):
        """Bitmask BackingStorage Z writes = M*N = 16,384 (Z is dense output).

        Sparseloop reference: 16,384. Matches.
        """
        self.assertEqual(_get_action(self.result, "BackingStorage", "Z", "write"), 16_384)

    # --- MAC ---

    def test_mac_computes(self):
        """Bitmask MAC computes = 21,632.

        Two cascading SAFs propagate to compute:
        SAF(A on B): 2,097,152 - floor(2,097,152 * 0.8984375) = 212,992
        SAF(B on A): 212,992 - floor(212,992 * 0.8984375) = 21,632

        Sparseloop reference: 21,633 (3-state compute remainder ±1).
        AccelForge uses round(total * d_A * d_B) = 21,632.
        """
        self.assertEqual(_get_action(self.result, "MAC", "None", "compute"), 21_632)

    # --- Reg Z (accumulator) ---

    def test_reg_z_reads(self):
        """Bitmask Reg Z reads = 21,463.

        Z SAF at Reg (gating on [A,B]): prob = 1 - d_A*d_B = 0.98968...
        Base = M*N*(K-1) = 2,080,768 (first-k read skipped).
        apply_local_saf_reads(2,080,768, 0.98968..., is_read_write=True) → 21,463.
        Matches Sparseloop reference: 21,463.
        """
        self.assertEqual(_get_action(self.result, "Reg", "Z", "read"), 21_463)

    def test_reg_z_writes(self):
        """Bitmask Reg Z writes = 21,632.

        Sparseloop reference: 21,632 (updates). Matches.
        """
        self.assertEqual(_get_action(self.result, "Reg", "Z", "write"), 21_632)

    # --- Reg A/B (sparse fills from Buffer) ---

    def test_regpassthrough_a_reads(self):
        """Bitmask RegPassthrough A reads = 2,097,152.

        RegPassthrough A is zero-cost (SAF child-buffet support).
        Reads are algorithmic (not reduced by format compression or SAF).
        """
        self.assertEqual(_get_action(self.result, "RegPassthrough", "A", "read"), 2_097_152)

    def test_regpassthrough_a_writes(self):
        """Bitmask RegPassthrough A writes = 21,632.

        Fills from Buffer A to RegPassthrough = Buffer A reads (post-SAF) = 21,632.
        """
        self.assertEqual(_get_action(self.result, "RegPassthrough", "A", "write"), 21_632)

    def test_regpassthrough_b_reads(self):
        """Bitmask RegPassthrough B reads = 2,097,152 (pass-through, not reduced)."""
        self.assertEqual(_get_action(self.result, "RegPassthrough", "B", "read"), 2_097_152)

    def test_regpassthrough_b_writes(self):
        """Bitmask RegPassthrough B writes = 21,632 (fills from Buffer B post-SAF)."""
        self.assertEqual(_get_action(self.result, "RegPassthrough", "B", "write"), 21_632)


# ===========================================================================
# Coordinate list sparse — skipping SAF
# ===========================================================================


class TestCoordListSparse(unittest.TestCase):
    """Coordinate list format (skipping SAF) — fig1 d=0.1015625.

    All actual action counts match bitmask exactly. Gating vs skipping
    only affects energy labeling (gated accesses cost ~0 energy, skipped
    accesses cost exactly 0) and latency (gated consume BW, skipped don't).

    Sparseloop reference values from ARTIFACT_EVALUATION.md §3.
    """

    @classmethod
    def setUpClass(cls):
        cls.result = _load_fig1("sparse_coord_list.yaml")

    # --- Buffer reads ---

    def test_buffer_a_reads(self):
        """Coord_list Buffer A reads = 21,632 (same as bitmask).

        Sparseloop reference: 21,632. Matches.
        """
        self.assertEqual(_get_action(self.result, "Buffer", "A", "read"), 21_632)

    def test_buffer_b_reads(self):
        """Coord_list Buffer B reads = 21,632.

        Sparseloop reference: 21,632. Matches.
        """
        self.assertEqual(_get_action(self.result, "Buffer", "B", "read"), 21_632)

    # --- Buffer writes (compressed fills) ---

    def test_buffer_a_writes(self):
        """Coord_list Buffer A writes = 212,992 (same as bitmask).

        Sparseloop reference: 212,992. Matches.
        """
        self.assertEqual(_get_action(self.result, "Buffer", "A", "write"), 212_992)

    def test_buffer_b_writes(self):
        """Coord_list Buffer B writes = 1,664 (same as bitmask).

        Sparseloop reference: 1,664. Matches.
        """
        self.assertEqual(_get_action(self.result, "Buffer", "B", "write"), 1_664)

    # --- BackingStorage ---

    def test_backing_storage_a_reads(self):
        """Coord_list BackingStorage A reads = 212,992.

        Sparseloop reference: 212,992. Matches.
        """
        self.assertEqual(_get_action(self.result, "BackingStorage", "A", "read"), 212_992)

    def test_backing_storage_b_reads(self):
        """Coord_list BackingStorage B reads = 1,664.

        Sparseloop reference: 1,664. Matches.
        """
        self.assertEqual(_get_action(self.result, "BackingStorage", "B", "read"), 1_664)

    def test_backing_storage_z_writes(self):
        """Coord_list BackingStorage Z writes = 16,384 (Z is dense).

        Sparseloop reference: 16,384. Matches.
        """
        self.assertEqual(_get_action(self.result, "BackingStorage", "Z", "write"), 16_384)

    # --- MAC ---

    def test_mac_computes(self):
        """Coord_list MAC computes = 21,632.

        Sparseloop reference: 21,633 (±1 from 3-state remainder).
        """
        self.assertEqual(_get_action(self.result, "MAC", "None", "compute"), 21_632)

    # --- Reg Z ---

    def test_reg_z_reads(self):
        """Coord_list Reg Z reads = 21,463.

        Base = M*N*(K-1) = 2,080,768 (first-k read skipped).
        SAF same as bitmask → 21,463. Matches Sparseloop reference.
        """
        self.assertEqual(_get_action(self.result, "Reg", "Z", "read"), 21_463)

    def test_reg_z_writes(self):
        """Coord_list Reg Z writes = 21,632.

        Sparseloop reference: 21,632. Matches.
        """
        self.assertEqual(_get_action(self.result, "Reg", "Z", "write"), 21_632)

    # --- Reg A/B ---

    def test_regpassthrough_a_writes(self):
        """Coord_list RegPassthrough A writes = 21,632 (fills from Buffer A post-SAF)."""
        self.assertEqual(_get_action(self.result, "RegPassthrough", "A", "write"), 21_632)

    def test_regpassthrough_b_writes(self):
        """Coord_list RegPassthrough B writes = 21,632 (fills from Buffer B post-SAF)."""
        self.assertEqual(_get_action(self.result, "RegPassthrough", "B", "write"), 21_632)


# ===========================================================================
# Cross-format consistency checks
# ===========================================================================


class TestBitmaskCoordListParity(unittest.TestCase):
    """All actual action counts must match between bitmask and coord_list.

    ARTIFACT_EVALUATION.md §3: "All actual counts are identical — the SAF
    probability determines the split, and it's the same for both formats.
    Only the classification (gated vs skipped) changes."
    """

    @classmethod
    def setUpClass(cls):
        cls.bitmask = _load_fig1("sparse_bitmask.yaml")
        cls.coord_list = _load_fig1("sparse_coord_list.yaml")

    def _check_parity(self, component, tensor, action):
        bm = _get_action(self.bitmask, component, tensor, action)
        cl = _get_action(self.coord_list, component, tensor, action)
        self.assertEqual(
            bm, cl,
            f"Parity mismatch: ({component}, {tensor}, {action}): "
            f"bitmask={bm}, coord_list={cl}",
        )

    def test_buffer_a_reads_parity(self):
        self._check_parity("Buffer", "A", "read")

    def test_buffer_b_reads_parity(self):
        self._check_parity("Buffer", "B", "read")

    def test_buffer_a_writes_parity(self):
        self._check_parity("Buffer", "A", "write")

    def test_buffer_b_writes_parity(self):
        self._check_parity("Buffer", "B", "write")

    def test_backing_storage_a_reads_parity(self):
        self._check_parity("BackingStorage", "A", "read")

    def test_backing_storage_b_reads_parity(self):
        self._check_parity("BackingStorage", "B", "read")

    def test_backing_storage_z_writes_parity(self):
        self._check_parity("BackingStorage", "Z", "write")

    def test_mac_computes_parity(self):
        self._check_parity("MAC", "None", "compute")

    def test_reg_z_reads_parity(self):
        self._check_parity("Reg", "Z", "read")

    def test_reg_z_writes_parity(self):
        self._check_parity("Reg", "Z", "write")


# ===========================================================================
# Energy comparison
# ===========================================================================


class TestSparseReducesEnergy(unittest.TestCase):
    """Sparse optimizations reduce total energy compared to dense."""

    @classmethod
    def setUpClass(cls):
        cls.dense = _load_fig1()
        cls.bitmask = _load_fig1("sparse_bitmask.yaml")
        cls.coord_list = _load_fig1("sparse_coord_list.yaml")

    def test_bitmask_less_than_dense(self):
        """Bitmask total energy < dense total energy."""
        dense_energy = float(self.dense.energy())
        bitmask_energy = float(self.bitmask.energy())
        self.assertLess(bitmask_energy, dense_energy)

    def test_coord_list_less_than_dense(self):
        """Coord_list total energy < dense total energy."""
        dense_energy = float(self.dense.energy())
        coord_energy = float(self.coord_list.energy())
        self.assertLess(coord_energy, dense_energy)


if __name__ == "__main__":
    unittest.main()
