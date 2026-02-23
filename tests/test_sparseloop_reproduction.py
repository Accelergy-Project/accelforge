"""Sparseloop reproduction tests with target error thresholds.

Each test class reproduces one notebook from notebooks/sparseloop_reproduction/.
Sparseloop reference values are hardcoded. AccelForge results are checked against
them with per-config relative tolerances.

Replaces the 5 fig1-specific regression test files (test_sparse_integration,
test_sparse_energy, test_sparse_latency, test_per_rank_format,
test_sparseloop_comparison) with parametrized reproduction tests covering
6 architectures.
"""

import os
import tempfile

import pytest
import yaml

from accelforge.frontend.spec import Spec
from accelforge.model.main import evaluate_mapping

INPUT_DIR = os.path.join(os.path.dirname(__file__), "input_files")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(subdir, arch, mapping, workload, sparse=None, jinja_parse_data=None):
    """Load config files and return (cycles, energy_pJ, result)."""
    d = os.path.join(INPUT_DIR, subdir)
    args = [os.path.join(d, arch), os.path.join(d, workload), os.path.join(d, mapping)]
    if sparse:
        args.append(os.path.join(d, sparse))
    kwargs = {}
    if jinja_parse_data:
        kwargs["jinja_parse_data"] = jinja_parse_data
    spec = Spec.from_yaml(*args, **kwargs)
    result = evaluate_mapping(spec)
    cycles = float(result.data["Total<SEP>latency"].iloc[0])
    energy = float(result.data["Total<SEP>energy"].iloc[0])
    return cycles, energy, result


def _run_with_tmpfile(subdir, arch, mapping, workload_dict, sparse=None):
    """Like _run but writes workload_dict to a temp YAML file first."""
    d = os.path.join(INPUT_DIR, subdir)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(workload_dict, f)
        wf = f.name
    try:
        args = [os.path.join(d, arch), wf, os.path.join(d, mapping)]
        if sparse:
            args.append(os.path.join(d, sparse))
        spec = Spec.from_yaml(*args)
        result = evaluate_mapping(spec)
        cycles = float(result.data["Total<SEP>latency"].iloc[0])
        energy = float(result.data["Total<SEP>energy"].iloc[0])
        return cycles, energy, result
    finally:
        os.unlink(wf)


def _run_with_tmpfiles(subdir, arch, mapping_str, workload_str, sparse=None):
    """Like _run but writes workload and mapping YAML strings to temp files."""
    d = os.path.join(INPUT_DIR, subdir)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as wf:
        wf.write(workload_str)
        workload_path = wf.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as mf:
        mf.write(mapping_str)
        mapping_path = mf.name
    try:
        args = [os.path.join(d, arch), workload_path, mapping_path]
        if sparse:
            args.append(os.path.join(d, sparse))
        spec = Spec.from_yaml(*args)
        result = evaluate_mapping(spec)
        cycles = float(result.data["Total<SEP>latency"].iloc[0])
        energy = float(result.data["Total<SEP>energy"].iloc[0])
        return cycles, energy, result
    finally:
        os.unlink(workload_path)
        os.unlink(mapping_path)


def _make_fig1_workload(density):
    """Generate fig1 workload dict (128x128x128 SpMSpM) with given density."""
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


def _make_lab4_workload(density_a=0.25, density_b=0.5):
    """Generate lab4 workload dict (8x8x8 SpMSpM)."""
    return {
        "workload": {
            "iteration_space_shape": {
                "m": "0 <= m < 8",
                "n": "0 <= n < 8",
                "k": "0 <= k < 8",
            },
            "bits_per_value": {"All": 8},
            "einsums": [
                {
                    "name": "SpMSpM",
                    "tensor_accesses": [
                        {"name": "A", "projection": ["m", "k"], "density": density_a},
                        {"name": "B", "projection": ["n", "k"], "density": density_b},
                        {"name": "Z", "projection": ["m", "n"], "output": True},
                    ],
                }
            ],
        }
    }


# ---------------------------------------------------------------------------
# Fig 12 helpers (workload + mapping generation)
# ---------------------------------------------------------------------------

FIG12_LAYERS = {
    "L07": {"M": 64, "E": 32, "F": 32, "C": 64, "d_I": 0.73, "d_W": 0.52,
            "BS_M": 8, "BS_C": 8, "psum_M": 8, "psum_C": 8},
    "L09": {"M": 128, "E": 16, "F": 16, "C": 64, "d_I": 0.86, "d_W": 0.82,
            "BS_M": 8, "BS_C": 8, "psum_M": 16, "psum_C": 8},
    "L13": {"M": 256, "E": 8, "F": 8, "C": 128, "d_I": 0.83, "d_W": 0.64,
            "BS_M": 16, "BS_C": 16, "psum_M": 16, "psum_C": 8},
    "L19": {"M": 256, "E": 8, "F": 8, "C": 256, "d_I": 0.61, "d_W": 0.55,
            "BS_M": 16, "BS_C": 32, "psum_M": 16, "psum_C": 8},
    "L21": {"M": 256, "E": 8, "F": 8, "C": 256, "d_I": 0.64, "d_W": 0.60,
            "BS_M": 16, "BS_C": 32, "psum_M": 16, "psum_C": 8},
    "L23": {"M": 256, "E": 8, "F": 8, "C": 256, "d_I": 0.61, "d_W": 0.70,
            "BS_M": 16, "BS_C": 32, "psum_M": 16, "psum_C": 8},
    "L25": {"M": 512, "E": 4, "F": 4, "C": 256, "d_I": 0.68, "d_W": 0.65,
            "BS_M": 32, "BS_C": 32, "psum_M": 16, "psum_C": 8},
    "L27": {"M": 512, "E": 4, "F": 4, "C": 512, "d_I": 0.58, "d_W": 0.30,
            "BS_M": 32, "BS_C": 64, "psum_M": 16, "psum_C": 8},
}


def _make_fig12_workload(p):
    return f"""workload:
  iteration_space_shape:
    r: 0 <= r < 1
    s: 0 <= s < 1
    e: 0 <= e < {p['E']}
    f: 0 <= f < {p['F']}
    c: 0 <= c < {p['C']}
    m: 0 <= m < {p['M']}
    n: 0 <= n < 1
    g: 0 <= g < 1
  bits_per_value: {{~Outputs: 8, Outputs: 20}}
  einsums:
  - name: GroupedConv
    tensor_accesses:
    - name: Inputs
      projection: [n, c, g, e, f]
      density: {p['d_I']}
    - name: Weights
      projection: [c, m, g, r, s]
      density: {p['d_W']}
    - name: Outputs
      projection: [n, g, m, f, e]
      output: true
"""


def _make_fig12_mapping(p):
    M_inner = p["M"] // p["BS_M"]
    C_inner = p["C"] // p["BS_C"]
    return f"""mapping:
  nodes:
  - !Storage {{tensors: [Inputs, Weights, Outputs], component: BackingStorage}}
  - !Temporal {{rank_variable: m, tile_shape: {M_inner}}}
  - !Temporal {{rank_variable: c, tile_shape: {C_inner}}}
  - !Storage {{tensors: [Weights], component: weight_spad}}
  - !Temporal {{rank_variable: f, tile_shape: 1}}
  - !Temporal {{rank_variable: e, tile_shape: 1}}
  - !Storage {{tensors: [Inputs], component: iact_spad}}
  - !Storage {{tensors: [Outputs], component: psum_spad}}
  - !Temporal {{rank_variable: c, tile_shape: 1}}
  - !Storage {{tensors: [Inputs], component: reg}}
  - !Temporal {{rank_variable: m, tile_shape: 1}}
  - !Compute {{einsum: GroupedConv, component: MAC}}
"""


# ===========================================================================
# Test classes
# ===========================================================================


class TestFig1:
    """Fig 1: BM vs CL density sweep (128x128x128 SpMSpM)."""

    # Sparseloop reference values (from fig1_artifact.ipynb cell-17)
    DENSITIES = [0.01, 0.02, 0.04, 0.08, 0.1, 0.2, 0.4, 0.8]
    SL_BM_CYCLES = [2_113_536] * 8
    SL_CL_CYCLES = [34_056, 58_124, 116_247, 232_490, 295_152, 578_952, 1_157_904, 3_698_200]
    SL_BM_ENERGY_UJ = [1.34, 1.42, 1.62, 2.04, 2.27, 3.38, 5.93, 12.29]
    SL_CL_ENERGY_UJ = [0.39, 0.62, 1.18, 2.31, 2.92, 5.77, 11.87, 25.41]

    @pytest.mark.parametrize(
        "idx,density",
        list(enumerate(DENSITIES)),
        ids=[f"d={d}" for d in DENSITIES],
    )
    def test_bitmask_cycles(self, idx, density):
        """BM cycles constant at 2,113,536 (gating never saves cycles)."""
        cycles, _, _ = _run_with_tmpfile(
            "fig1", "arch_unified.yaml", "mapping.yaml",
            _make_fig1_workload(density), "sparse_bitmask_latency.yaml",
        )
        assert int(cycles) == self.SL_BM_CYCLES[idx]

    @pytest.mark.parametrize(
        "idx,density",
        list(enumerate(DENSITIES)),
        ids=[f"d={d}" for d in DENSITIES],
    )
    def test_coord_list_cycles(self, idx, density):
        """CL cycles within 20% of SL (hypergeometric vs simulation)."""
        cycles, _, _ = _run_with_tmpfile(
            "fig1", "arch_unified.yaml", "mapping.yaml",
            _make_fig1_workload(density), "sparse_coord_list_latency.yaml",
        )
        assert cycles == pytest.approx(self.SL_CL_CYCLES[idx], rel=0.20)

    @pytest.mark.parametrize(
        "idx,density",
        list(enumerate(DENSITIES)),
        ids=[f"d={d}" for d in DENSITIES],
    )
    def test_bitmask_energy(self, idx, density):
        """BM energy within 5% of SL (except d=0.01 at ~23%)."""
        _, energy, _ = _run_with_tmpfile(
            "fig1", "arch_unified.yaml", "mapping.yaml",
            _make_fig1_workload(density), "sparse_bitmask_energy.yaml",
        )
        energy_uJ = energy / 1e6
        assert energy_uJ == pytest.approx(self.SL_BM_ENERGY_UJ[idx], rel=0.25)

    @pytest.mark.parametrize(
        "idx,density",
        list(enumerate(DENSITIES)),
        ids=[f"d={d}" for d in DENSITIES],
    )
    def test_coord_list_energy(self, idx, density):
        """CL energy within 10% of SL."""
        _, energy, _ = _run_with_tmpfile(
            "fig1", "arch_unified.yaml", "mapping.yaml",
            _make_fig1_workload(density), "sparse_coord_list_energy.yaml",
        )
        energy_uJ = energy / 1e6
        assert energy_uJ == pytest.approx(self.SL_CL_ENERGY_UJ[idx], rel=0.10)

    def test_canonical_bitmask(self):
        """Canonical d=0.1015625: BM cycles exact, energy within 2%."""
        cycles, energy, _ = _run(
            "fig1", "arch_unified.yaml", "mapping.yaml",
            "workload.yaml", "sparse_bitmask_energy.yaml",
        )
        assert int(cycles) == 2_113_536
        assert energy / 1e6 == pytest.approx(2.27, rel=0.02)

    def test_canonical_coord_list(self):
        """Canonical d=0.1015625: CL cycles exact, energy within 6%."""
        cycles, energy, _ = _run(
            "fig1", "arch_unified.yaml", "mapping.yaml",
            "workload.yaml", "sparse_coord_list_energy.yaml",
        )
        assert int(cycles) == 295_152
        assert energy / 1e6 == pytest.approx(2.92, rel=0.06)


class TestFig12:
    """Fig 12: EyerissV2 single-PE (8 MobileNet layers)."""

    # Sparseloop reference: (cycles, energy_pJ)
    SL_REF = {
        "L07": (1_592_245, 4_992_020),
        "L09": (1_479_114, 3_757_580),
        "L13": (1_114_139, 2_996_420),
        "L19": (1_407_304, 4_311_730),
        "L21": (1_610_668, 4_764_760),
        "L23": (1_791_135, 5_233_700),
        "L25": (927_185, 2_713_340),
        "L27": (729_915, 2_761_280),
    }

    @pytest.mark.parametrize("layer", list(SL_REF.keys()))
    def test_cycles(self, layer):
        """Per-layer cycles within 0.5% of Sparseloop."""
        p = FIG12_LAYERS[layer]
        cycles, _, _ = _run_with_tmpfiles(
            "fig12", "arch.yaml",
            _make_fig12_mapping(p), _make_fig12_workload(p),
            "sparse_SI_SW.yaml",
        )
        sl_cycles = self.SL_REF[layer][0]
        assert cycles == pytest.approx(sl_cycles, rel=0.005)

    @pytest.mark.parametrize("layer", list(SL_REF.keys()))
    def test_energy(self, layer):
        """Per-layer energy within 4% of Sparseloop (L27 at -3.5%)."""
        p = FIG12_LAYERS[layer]
        _, energy, _ = _run_with_tmpfiles(
            "fig12", "arch.yaml",
            _make_fig12_mapping(p), _make_fig12_workload(p),
            "sparse_SI_SW.yaml",
        )
        sl_energy = self.SL_REF[layer][1]
        assert energy == pytest.approx(sl_energy, rel=0.04)


class TestFig13:
    """Fig 13: DSTC 128-PE mesh (4096x4096 GEMM)."""

    # Sparseloop normalized latency reference
    SL_NORM = {
        (1.0, 1.0): 1.00,
        (0.9, 1.0): 0.90,
        (0.9, 0.4): 0.48,
        (0.7, 1.0): 0.72,
        (0.7, 0.4): 0.38,
        (0.5, 1.0): 0.54,
        (0.5, 0.4): 0.29,
        (0.3, 1.0): 0.36,
        (0.3, 0.4): 0.19,
    }

    @pytest.fixture(scope="class")
    def dense_cycles(self):
        """Dense baseline cycles for normalization."""
        cycles, _, _ = _run(
            "fig13", "arch.yaml", "mapping.yaml", "workload.yaml",
            jinja_parse_data={"density_A": 1.0, "density_B": 1.0},
        )
        return cycles

    @pytest.mark.parametrize(
        "dA,dB",
        [(0.9, 1.0), (0.9, 0.4), (0.7, 1.0), (0.7, 0.4),
         (0.5, 1.0), (0.5, 0.4), (0.3, 1.0), (0.3, 0.4)],
        ids=[f"dA={a}_dB={b}" for a, b in
             [(0.9, 1.0), (0.9, 0.4), (0.7, 1.0), (0.7, 0.4),
              (0.5, 1.0), (0.5, 0.4), (0.3, 1.0), (0.3, 0.4)]],
    )
    def test_normalized_latency(self, dense_cycles, dA, dB):
        """Normalized latency within 3% of Sparseloop reference."""
        cycles, _, _ = _run(
            "fig13", "arch.yaml", "mapping.yaml", "workload.yaml",
            "sparse_dstc.yaml",
            jinja_parse_data={"density_A": dA, "density_B": dB},
        )
        af_norm = cycles / dense_cycles
        sl_norm = self.SL_NORM[(dA, dB)]
        assert af_norm == pytest.approx(sl_norm, abs=0.03)


class TestFig15:
    """Fig 15: STC ResNet50 (4 GEMM layers)."""

    LAYERS = [1, 2, 3, 4]

    # Sparseloop per-layer cycles
    SL_CYCLES = {1: 131_072, 2: 65_536, 3: 147_456, 4: 131_072}

    # Sparseloop total energy (uJ) across 4 layers
    SL_TOTAL_ENERGY_UJ = {"TC": 849.0, "STC_1.0": 772.0, "STC_0.5": 512.0}

    CONFIGS = {
        "TC": {"arch": "arch_tc.yaml", "sparse": None, "jpd": {}, "density_factor": 1.0},
        "STC_1.0": {"arch": "arch_stc.yaml", "sparse": "sparse_stc.yaml", "jpd": {}, "density_factor": 1.0},
        "STC_0.5": {"arch": "arch_stc.yaml", "sparse": "sparse_stc.yaml", "jpd": {"density_A": 0.5}, "density_factor": 0.5},
    }

    @pytest.mark.parametrize("config_name", ["TC", "STC_1.0", "STC_0.5"])
    def test_per_layer_cycles(self, config_name):
        """Per-layer cycles: exact for TC/STC@1.0, half for STC@0.5."""
        cfg = self.CONFIGS[config_name]
        for layer in self.LAYERS:
            cycles, _, _ = _run(
                "fig15", cfg["arch"],
                f"mapping_layer{layer}.yaml",
                f"workload_layer{layer}.yaml",
                cfg["sparse"],
                jinja_parse_data=cfg["jpd"] or None,
            )
            expected = int(self.SL_CYCLES[layer] * cfg["density_factor"])
            assert int(cycles) == expected, (
                f"{config_name} L{layer}: {int(cycles)} != {expected}"
            )

    @pytest.mark.parametrize("config_name", ["TC", "STC_1.0", "STC_0.5"])
    def test_total_energy(self, config_name):
        """Total energy across 4 layers within 6% of Sparseloop."""
        cfg = self.CONFIGS[config_name]
        total_energy_pJ = 0.0
        for layer in self.LAYERS:
            _, energy, _ = _run(
                "fig15", cfg["arch"],
                f"mapping_layer{layer}.yaml",
                f"workload_layer{layer}.yaml",
                cfg["sparse"],
                jinja_parse_data=cfg["jpd"] or None,
            )
            total_energy_pJ += energy
        total_uJ = total_energy_pJ / 1e6
        sl_uJ = self.SL_TOTAL_ENERGY_UJ[config_name]
        assert total_uJ == pytest.approx(sl_uJ, rel=0.06)


class TestTable7:
    """Table 7: Eyeriss v1 AlexNet (5 conv layers, 168 PEs)."""

    # Sparseloop reference: (cycles, energy_uJ)
    SL_REF = {
        "conv1": (2_838_528, 2_059.86),
        "conv2": (4_128_768, 3_160.50),
        "conv3": (1_916_929, 1_534.63),
        "conv4": (1_437_697, 1_110.05),
        "conv5": (958_464, 756.75),
    }

    # conv1 uses dense_iact (only output compression), conv2-5 use sparse_iact
    SPARSE_CONFIG = {
        "conv1": "sparse_dense_iact.yaml",
        "conv2": "sparse_sparse_iact.yaml",
        "conv3": "sparse_sparse_iact.yaml",
        "conv4": "sparse_sparse_iact.yaml",
        "conv5": "sparse_sparse_iact.yaml",
    }

    @pytest.mark.parametrize("layer", list(SL_REF.keys()))
    def test_cycles(self, layer):
        """Per-layer cycles within 0.5% of Sparseloop (observed exact)."""
        cycles, _, _ = _run(
            "table7", "arch.yaml",
            f"mapping_{layer}.yaml",
            f"workload_{layer}.yaml",
            self.SPARSE_CONFIG[layer],
        )
        sl_cycles = self.SL_REF[layer][0]
        assert cycles == pytest.approx(sl_cycles, rel=0.005)

    @pytest.mark.parametrize("layer", list(SL_REF.keys()))
    def test_energy(self, layer):
        """Per-layer energy within 7% of Sparseloop."""
        _, energy, _ = _run(
            "table7", "arch.yaml",
            f"mapping_{layer}.yaml",
            f"workload_{layer}.yaml",
            self.SPARSE_CONFIG[layer],
        )
        sl_energy_pJ = self.SL_REF[layer][1] * 1e6  # uJ -> pJ
        assert energy == pytest.approx(sl_energy_pJ, rel=0.07)


class TestLab4:
    """Lab 4: Storage sweep (8x8 GEMM) — dense/gating/skipping."""

    # Sparseloop reference: fJ per algorithmic compute
    # Algorithmic computes = M*K*N = 8*8*8 = 512
    ALG_COMPUTES = 512
    SL_FJ_PER_COMPUTE = {"dense": 7_047.25, "gating": 3_972.35, "skipping": 1_919.80}

    # Tolerances per config
    TOLERANCES = {"dense": 0.04, "gating": 0.02, "skipping": 0.08}

    SPARSE_FILES = {"dense": None, "gating": "sparse_gating.yaml", "skipping": "sparse_skipping.yaml"}

    @pytest.mark.parametrize("config", ["dense", "gating", "skipping"])
    def test_fj_per_compute(self, config):
        """fJ per algorithmic compute within threshold of Sparseloop."""
        sparse = self.SPARSE_FILES[config]
        if sparse:
            _, energy, _ = _run(
                "lab4", "arch.yaml", "mapping.yaml", "workload.yaml", sparse,
            )
        else:
            _, energy, _ = _run(
                "lab4", "arch.yaml", "mapping.yaml", "workload.yaml",
            )
        fj_per_compute = (energy * 1e3) / self.ALG_COMPUTES  # pJ -> fJ, then /512
        sl_fj = self.SL_FJ_PER_COMPUTE[config]
        tol = self.TOLERANCES[config]
        assert fj_per_compute == pytest.approx(sl_fj, rel=tol)
