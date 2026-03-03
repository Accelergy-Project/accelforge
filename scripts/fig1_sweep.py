#!/usr/bin/env python3
"""Density sweep reproducing micro22-sparseloop-artifact Fig.1.

Runs bitmask (gating) and coord_list (skipping) configurations across 8
densities using the fig1 128x128x128 SpMSpM workload, then plots:
  1. Normalized speed (coord_list/bitmask cycles) vs density
  2. Normalized energy (coord_list/bitmask energy) vs density

Comparable to Sparseloop's parse_and_plot.py from the artifact.

Usage:
    python scripts/fig1_sweep.py [--output-dir DIR]
"""

import argparse
import os
import sys
import tempfile

import yaml

# Add accelforge to path if running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from accelforge.frontend.spec import Spec
from accelforge.model.main import evaluate_mapping

FIG1_DIR = os.path.join(
    os.path.dirname(__file__), "..", "tests", "input_files", "fig1"
)

DENSITIES = [0.01, 0.02, 0.04, 0.08, 0.1, 0.2, 0.4, 0.8]

# Sparseloop ground truth from micro22-sparseloop-artifact
SPARSELOOP_BM_CYCLES = [2113536] * 8
SPARSELOOP_CL_CYCLES = [34056, 58124, 116247, 232490, 295152, 578952, 1157904, 3698200]
SPARSELOOP_BM_ENERGY_UJ = [1.34, 1.42, 1.62, 2.04, 2.27, 3.38, 5.93, 12.29]
SPARSELOOP_CL_ENERGY_UJ = [0.39, 0.62, 1.18, 2.31, 2.92, 5.77, 11.87, 25.41]


def make_workload_yaml(density):
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


def run_config(density, arch_yaml, sparse_yaml):
    workload = make_workload_yaml(density)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
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


def main():
    parser = argparse.ArgumentParser(description="Fig.1 density sweep")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to save output PNG files (default: current directory)",
    )
    args = parser.parse_args()

    print("Running Fig.1 density sweep...")
    print(f"{'Density':>8} | {'BM cycles':>12} | {'CL cycles':>12} | "
          f"{'BM energy':>12} | {'CL energy':>12} | {'Speed':>8} | {'Energy':>8}")
    print("-" * 90)

    bm_cycles, cl_cycles = [], []
    bm_energy, cl_energy = [], []

    for d in DENSITIES:
        bm_c, _ = run_config(d, "arch_latency.yaml", "sparse_bitmask_latency.yaml")
        cl_c, _ = run_config(d, "arch_latency.yaml", "sparse_coord_list_latency.yaml")
        _, bm_e = run_config(d, "arch_energy.yaml", "sparse_bitmask_energy.yaml")
        _, cl_e = run_config(d, "arch_energy.yaml", "sparse_coord_list_energy.yaml")

        bm_cycles.append(bm_c)
        cl_cycles.append(cl_c)
        bm_energy.append(bm_e)
        cl_energy.append(cl_e)

        sr = cl_c / bm_c if bm_c > 0 else 0
        er = cl_e / bm_e if bm_e > 0 else 0
        print(f"{d:8.2f} | {bm_c:12.0f} | {cl_c:12.0f} | "
              f"{bm_e:12.2f} | {cl_e:12.2f} | {sr:8.4f} | {er:8.4f}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed — skipping plot generation.")
        print("Install with: pip install matplotlib")
        return

    speed_ratios_af = [cl / bm for cl, bm in zip(cl_cycles, bm_cycles)]
    energy_ratios_af = [cl / bm for cl, bm in zip(cl_energy, bm_energy)]
    speed_ratios_sl = [
        cl / bm for cl, bm in zip(SPARSELOOP_CL_CYCLES, SPARSELOOP_BM_CYCLES)
    ]
    energy_ratios_sl = [
        cl / bm
        for cl, bm in zip(SPARSELOOP_CL_ENERGY_UJ, SPARSELOOP_BM_ENERGY_UJ)
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Speed ratio plot
    ax1.plot(DENSITIES, speed_ratios_af, "o-", label="AccelForge", color="tab:blue")
    ax1.plot(DENSITIES, speed_ratios_sl, "s--", label="Sparseloop", color="tab:orange")
    ax1.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
    ax1.set_xlabel("Density")
    ax1.set_ylabel("Normalized Speed (CoordList / Bitmask)")
    ax1.set_title("Fig.1a: Speed Ratio vs Density")
    ax1.set_xscale("log")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Energy ratio plot
    ax2.plot(DENSITIES, energy_ratios_af, "o-", label="AccelForge", color="tab:blue")
    ax2.plot(DENSITIES, energy_ratios_sl, "s--", label="Sparseloop", color="tab:orange")
    ax2.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
    ax2.set_xlabel("Density")
    ax2.set_ylabel("Normalized Energy (CoordList / Bitmask)")
    ax2.set_title("Fig.1b: Energy Ratio vs Density")
    ax2.set_xscale("log")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(args.output_dir, "fig1_density_sweep.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nFigure saved to {out_path}")


if __name__ == "__main__":
    main()
