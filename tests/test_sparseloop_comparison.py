"""Density sweep regression tests matching micro22-sparseloop-artifact Fig.1.

Tests verify AccelForge reproduces the key trends from Sparseloop's Fig.1:
- Bitmask cycles constant across all densities (gating never saves cycles)
- CoordList cycles scale roughly linearly with density
- Speed ratio (coord_list/bitmask) increases monotonically
- Energy crossover: CoordList cheaper at low density, more expensive at high density
- Bitmask energy increases monotonically with density

AccelForge uses an analytical hypergeometric model vs Sparseloop's simulation,
so absolute values may differ by 10-20%. Trend directions must match exactly.

Uses arch_latency.yaml for cycles, arch_energy.yaml for energy. Both use
the fig1 128x128x128 SpMSpM workload with variable density.
"""

import os
import tempfile
import unittest

import yaml

from accelforge.frontend.spec import Spec
from accelforge.model.main import evaluate_mapping

FIG1_DIR = os.path.join(os.path.dirname(__file__), "input_files", "fig1")

DENSITIES = [0.01, 0.02, 0.04, 0.08, 0.1, 0.2, 0.4, 0.8]


def _make_workload_yaml(density):
    """Generate a workload YAML dict with the given density for A and B."""
    return {
        "workload": {
            "iteration_space_shape": {
                "m": "0 <= m < 128",
                "n": "0 <= n < 128",
                "k": "0 <= k < 128",
            },
            "bits_per_value": {"All": 8},
            "einsums": [
                {
                    "name": "SpMSpM",
                    "tensor_accesses": [
                        {"name": "A", "projection": ["m", "k"], "density": density},
                        {"name": "B", "projection": ["n", "k"], "density": density},
                        {"name": "Z", "projection": ["m", "n"], "output": True},
                    ],
                }
            ],
        }
    }


def _run_config(density, arch_yaml, sparse_yaml):
    """Run a single config and return (total_cycles, total_energy)."""
    workload = _make_workload_yaml(density)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        yaml.dump(workload, f)
        wf = f.name
    try:
        spec = Spec.from_yaml(
            os.path.join(FIG1_DIR, arch_yaml),
            wf,
            os.path.join(FIG1_DIR, "mapping.yaml"),
            os.path.join(FIG1_DIR, sparse_yaml),
        )
        result = evaluate_mapping(spec)
        cycles = float(result.data["Total<SEP>latency"].iloc[0])
        energy = float(result.data["Total<SEP>energy"].iloc[0])
        return cycles, energy
    finally:
        os.unlink(wf)


class TestFig1DensitySweep(unittest.TestCase):
    """Cross-density sweep matching micro22-sparseloop-artifact Fig.1."""

    @classmethod
    def setUpClass(cls):
        cls.bitmask_cycles = []
        cls.coord_list_cycles = []
        cls.bitmask_energy = []
        cls.coord_list_energy = []

        for d in DENSITIES:
            bm_c, _ = _run_config(d, "arch_latency.yaml", "sparse_bitmask_latency.yaml")
            cl_c, _ = _run_config(d, "arch_latency.yaml", "sparse_coord_list_latency.yaml")
            # Energy uses energy arch (no total_latency expressions but has energy ERTs)
            _, bm_e = _run_config(d, "arch_energy.yaml", "sparse_bitmask_energy.yaml")
            _, cl_e = _run_config(d, "arch_energy.yaml", "sparse_coord_list_energy.yaml")

            cls.bitmask_cycles.append(bm_c)
            cls.coord_list_cycles.append(cl_c)
            cls.bitmask_energy.append(bm_e)
            cls.coord_list_energy.append(cl_e)

    def test_bitmask_cycles_constant(self):
        """Bitmask cycles = 2,113,536 at all densities.

        Gating never saves cycles — gated reads still consume port bandwidth.
        Includes Z fills (16,384) in Reg bandwidth.
        """
        for i, d in enumerate(DENSITIES):
            with self.subTest(density=d):
                self.assertEqual(
                    int(self.bitmask_cycles[i]),
                    2_113_536,
                    f"Bitmask cycles at d={d} should be 2,113,536",
                )

    def test_coord_list_cycles_increase_with_density(self):
        """CoordList cycles increase monotonically with density."""
        for i in range(len(DENSITIES) - 1):
            with self.subTest(
                d_low=DENSITIES[i], d_high=DENSITIES[i + 1]
            ):
                self.assertLess(
                    self.coord_list_cycles[i],
                    self.coord_list_cycles[i + 1],
                    f"CoordList cycles should increase from d={DENSITIES[i]} to d={DENSITIES[i+1]}",
                )

    def test_coord_list_cycles_roughly_linear(self):
        """CoordList cycles scale roughly linearly with density.

        Check that doubling density approximately doubles cycles (within 2x).
        """
        # Compare d=0.1 vs d=0.2 (indices 4 and 5)
        ratio = self.coord_list_cycles[5] / self.coord_list_cycles[4]
        self.assertAlmostEqual(ratio, 2.0, delta=0.5)

    def test_speed_ratio_increases_with_density(self):
        """coord_list/bitmask cycle ratio increases monotonically with density."""
        ratios = [
            cl / bm
            for cl, bm in zip(self.coord_list_cycles, self.bitmask_cycles)
        ]
        for i in range(len(ratios) - 1):
            with self.subTest(d_low=DENSITIES[i], d_high=DENSITIES[i + 1]):
                self.assertLess(
                    ratios[i],
                    ratios[i + 1],
                    f"Speed ratio should increase from d={DENSITIES[i]} to d={DENSITIES[i+1]}",
                )

    def test_coord_list_faster_at_low_density(self):
        """CoordList much faster than bitmask at low density (d<=0.1)."""
        for i in range(5):  # d=0.01 through d=0.1
            with self.subTest(density=DENSITIES[i]):
                self.assertLess(
                    self.coord_list_cycles[i],
                    self.bitmask_cycles[i],
                    f"CoordList should be faster than bitmask at d={DENSITIES[i]}",
                )

    def test_energy_crossover(self):
        """Energy crossover: bitmask > coord_list at low d, reversed at high d.

        Per-rank model: bitmask metadata is density-independent (always reads
        full mask), so metadata overhead is large at low density. Coord list
        (CSR) metadata scales with ennz (density-dependent), so it's cheaper
        at low density but grows faster. Crossover at d ~ 0.05.
        This matches Sparseloop's crossover behavior.
        """
        # At low density (d <= 0.04): bitmask more expensive
        for i, d in enumerate(DENSITIES):
            if d > 0.04:
                break
            with self.subTest(density=d, regime="low"):
                self.assertGreater(
                    self.bitmask_energy[i],
                    self.coord_list_energy[i],
                    f"Bitmask should be more expensive than CoordList at d={d}",
                )
        # At high density (d >= 0.1): coord_list more expensive
        for i, d in enumerate(DENSITIES):
            if d < 0.1:
                continue
            with self.subTest(density=d, regime="high"):
                self.assertGreater(
                    self.coord_list_energy[i],
                    self.bitmask_energy[i],
                    f"CoordList should be more expensive than bitmask at d={d}",
                )

    def test_bitmask_energy_increases_with_density(self):
        """Bitmask energy increases monotonically with density."""
        for i in range(len(DENSITIES) - 1):
            with self.subTest(
                d_low=DENSITIES[i], d_high=DENSITIES[i + 1]
            ):
                self.assertLess(
                    self.bitmask_energy[i],
                    self.bitmask_energy[i + 1],
                    f"Bitmask energy should increase from d={DENSITIES[i]} to d={DENSITIES[i+1]}",
                )

    def test_coord_list_energy_increases_with_density(self):
        """CoordList energy increases monotonically with density."""
        for i in range(len(DENSITIES) - 1):
            with self.subTest(
                d_low=DENSITIES[i], d_high=DENSITIES[i + 1]
            ):
                self.assertLess(
                    self.coord_list_energy[i],
                    self.coord_list_energy[i + 1],
                    f"CoordList energy should increase from d={DENSITIES[i]} to d={DENSITIES[i+1]}",
                )

    def test_energy_ratio_increases_with_density(self):
        """Energy ratio (coord_list/bitmask) increases monotonically with density.

        Per-rank model: bitmask metadata is constant (density-independent),
        while coord_list metadata scales with density. So the CL/BM ratio
        increases from < 1 at low density to > 1 at high density.
        """
        ratios = [
            cl / bm
            for cl, bm in zip(self.coord_list_energy, self.bitmask_energy)
        ]
        for i in range(len(ratios) - 1):
            with self.subTest(d_low=DENSITIES[i], d_high=DENSITIES[i + 1]):
                self.assertLess(
                    ratios[i],
                    ratios[i + 1],
                    f"CL/BM ratio should increase from d={DENSITIES[i]} to d={DENSITIES[i+1]}",
                )


class TestFig1AbsoluteValues(unittest.TestCase):
    """Spot-check absolute values at specific densities.

    AccelForge values differ from Sparseloop (analytical vs simulation model),
    but should be within reasonable bounds.
    """

    @classmethod
    def setUpClass(cls):
        # d=0.1 reference point
        cls.bm_cycles, cls.bm_energy = _run_config(
            0.1, "arch_latency.yaml", "sparse_bitmask_latency.yaml"
        )
        _, cls.bm_energy = _run_config(
            0.1, "arch_energy.yaml", "sparse_bitmask_energy.yaml"
        )
        cls.cl_cycles, cls.cl_energy = _run_config(
            0.1, "arch_latency.yaml", "sparse_coord_list_latency.yaml"
        )
        _, cls.cl_energy = _run_config(
            0.1, "arch_energy.yaml", "sparse_coord_list_energy.yaml"
        )

    def test_bitmask_cycles_at_d01(self):
        """Bitmask cycles = 2,113,536 at d=0.1 (includes Z fills at Reg)."""
        self.assertEqual(int(self.bm_cycles), 2_113_536)

    def test_coord_list_cycles_at_d01(self):
        """CoordList cycles ~ 290,614 at d=0.1 (Buffer bottleneck with metadata BW)."""
        self.assertAlmostEqual(self.cl_cycles, 290_614, delta=5000)

    def test_speed_ratio_at_d01(self):
        """Speed ratio (CL/BM) ~ 0.14 at d=0.1.

        Matches Sparseloop's 0.14 closely after Phase 11 fixes
        (skipped_first removal + metadata bandwidth conversion).
        """
        ratio = self.cl_cycles / self.bm_cycles
        self.assertAlmostEqual(ratio, 0.14, delta=0.02)


if __name__ == "__main__":
    unittest.main()
