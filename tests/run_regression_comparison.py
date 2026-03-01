#!/usr/bin/env python3
"""
Regression comparison: main branch reference vs current sparsity-support branch.

Two modes:
  1. Fast mode (default): Compare regression_reference_from_main.json against
     regression_reference.json (current branch's cached reference). Then validate
     a few quick test cases by re-running them.
  2. Full mode (--full): Re-run all 32 test configurations against current code.

Usage:
    python tests/run_regression_comparison.py          # fast mode
    python tests/run_regression_comparison.py --full    # full re-run mode
"""

import json
import sys
import time
import traceback
from numbers import Number
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import accelforge as af
from accelforge.frontend.spec import Spec
from accelforge.mapper import Metrics

MAIN_REF_PATH = Path(__file__).parent / "regression_reference_from_main.json"
CURRENT_REF_PATH = Path(__file__).parent / "regression_reference.json"


def cast(d):
    if isinstance(d, dict):
        return {str(k): cast(v) for k, v in d.items()}
    if isinstance(d, list):
        return [cast(v) for v in d]
    if isinstance(d, Number):
        return float(d)
    return d


def _run(arch, workload, fused, jinja_parse_data=None, print_progress=False):
    spec = Spec.from_yaml(
        arch,
        workload,
        jinja_parse_data=jinja_parse_data,
    )
    spec.mapper.metrics = Metrics.ENERGY
    spec.mapper.max_fused_loops = 1
    if not fused:
        for node in spec.arch.nodes:
            if isinstance(node, af.arch.Memory):
                node.tensors.keep = "All"
                break
    mappings = spec.map_workload_to_arch(print_progress=print_progress)
    m = mappings[0]
    return cast(
        {
            "energy": float(m.energy()),
            "latency": float(m.latency()),
            "energy_per_component": m.energy(
                per_component=True, per_einsum=True, per_action=True
            ),
            "latency_per_component": m.latency(per_component=True, per_einsum=True),
            "actions": m.actions(per_component=True, per_einsum=True, per_tensor=True),
            "n_mappings": int(len(mappings)),
        }
    )


def parse_key(key):
    """Parse key like 'simple|matmuls|KN=64,M=64,N_EINSUMS=2|fused'."""
    parts = key.split("|")
    arch_name = parts[0]
    workload_name = parts[1]
    jinja_str = parts[2] if len(parts) > 2 else ""
    fusion_mode = parts[3] if len(parts) > 3 else ""

    arch_map = {
        "simple": af.examples.arches.simple,
        "eyeriss": af.examples.arches.eyeriss,
        "simba": af.examples.arches.simba,
        "tpu_v4i": af.examples.arches.tpu_v4i,
    }
    workload_map = {
        "matmuls": af.examples.workloads.matmuls,
        "three_matmuls_annotated": af.examples.workloads.three_matmuls_annotated,
        "gpt3_175B": af.examples.workloads.gpt3_175B,
        "gpt3_175B_kv_cache": af.examples.workloads.gpt3_175B_kv_cache,
        "gpt3_6.7B": af.examples.workloads.gpt3_6_7B,
        "gpt3_6.7B_kv_cache": af.examples.workloads.gpt3_6_7B_kv_cache,
    }

    jinja_parse_data = None
    if jinja_str:
        jinja_parse_data = {}
        for pair in jinja_str.split(","):
            k, v = pair.split("=")
            try:
                jinja_parse_data[k] = int(v)
            except ValueError:
                jinja_parse_data[k] = v

    fused = fusion_mode == "fused"
    return arch_map[arch_name], workload_map[workload_name], jinja_parse_data, fused


def pct_diff(ref_val, cur_val):
    if ref_val == 0 and cur_val == 0:
        return 0.0
    if ref_val == 0:
        return float("inf")
    return ((cur_val - ref_val) / abs(ref_val)) * 100.0


def compare_dicts(ref_dict, cur_dict):
    diffs = []
    all_keys = set(list(ref_dict.keys()) + list(cur_dict.keys()))
    for k in sorted(all_keys):
        ref_val = ref_dict.get(k, None)
        cur_val = cur_dict.get(k, None)
        if ref_val is None:
            diffs.append((k, "NEW in current", None, cur_val))
        elif cur_val is None:
            diffs.append((k, "MISSING in current", ref_val, None))
        elif abs(ref_val - cur_val) > 1e-6:
            pct = pct_diff(ref_val, cur_val)
            diffs.append((k, f"{pct:+.4f}%", ref_val, cur_val))
    return diffs


def compare_test_entry(ref, cur):
    """Compare a single test entry. Returns dict of differences or empty dict if matching."""
    test_diffs = {}

    if abs(ref["energy"] - cur["energy"]) > 1e-6:
        pct = pct_diff(ref["energy"], cur["energy"])
        test_diffs["energy"] = {"ref": ref["energy"], "cur": cur["energy"], "pct": pct}

    if abs(ref["latency"] - cur["latency"]) > 1e-6:
        pct = pct_diff(ref["latency"], cur["latency"])
        test_diffs["latency"] = {"ref": ref["latency"], "cur": cur["latency"], "pct": pct}

    if ref["n_mappings"] != cur["n_mappings"]:
        test_diffs["n_mappings"] = {"ref": ref["n_mappings"], "cur": cur["n_mappings"]}

    for sub in ["energy_per_component", "latency_per_component", "actions"]:
        sub_diffs = compare_dicts(ref.get(sub, {}), cur.get(sub, {}))
        if sub_diffs:
            test_diffs[sub] = sub_diffs

    return test_diffs


def print_summary(total, matching, differing, errors, diff_details):
    print("\n" + "=" * 100)
    print("REGRESSION COMPARISON SUMMARY")
    print("=" * 100)
    print(f"Total tests:    {total}")
    print(f"Matching:       {matching}")
    print(f"Differing:      {differing}")
    print(f"Errors:         {errors}")
    print()

    if diff_details:
        print("DETAILED DIFFERENCES:")
        print("-" * 100)
        for key, diffs in sorted(diff_details.items()):
            print(f"\n  {key}:")
            if "error" in diffs:
                print(f"    ERROR: {diffs['error']}")
                continue

            if "energy" in diffs:
                d = diffs["energy"]
                print(f"    energy:     ref={d['ref']:<25.1f}  cur={d['cur']:<25.1f}  change={d['pct']:+.4f}%")
            if "latency" in diffs:
                d = diffs["latency"]
                print(f"    latency:    ref={d['ref']:<25.1f}  cur={d['cur']:<25.1f}  change={d['pct']:+.4f}%")
            if "n_mappings" in diffs:
                d = diffs["n_mappings"]
                print(f"    n_mappings: ref={d['ref']}  cur={d['cur']}")

            for sub in ["energy_per_component", "latency_per_component", "actions"]:
                if sub in diffs:
                    n_sub_diffs = len(diffs[sub])
                    n_new = sum(1 for e in diffs[sub] if e[1] == "NEW in current")
                    n_missing = sum(1 for e in diffs[sub] if e[1] == "MISSING in current")
                    n_changed = n_sub_diffs - n_new - n_missing
                    print(f"    {sub}: {n_changed} changed, {n_new} new, {n_missing} missing")
                    for entry in diffs[sub][:5]:
                        k, change, ref_val, cur_val = entry
                        if ref_val is not None and cur_val is not None:
                            print(f"      {k}: {ref_val} -> {cur_val} ({change})")
                        elif ref_val is None:
                            print(f"      {k}: {change} (value={cur_val})")
                        else:
                            print(f"      {k}: {change} (was {ref_val})")
                    if n_sub_diffs > 5:
                        print(f"      ... and {n_sub_diffs - 5} more")

    if diff_details:
        print("\n" + "=" * 100)
        print("AGGREGATE ENERGY/LATENCY CHANGES:")
        print(f"{'Test Key':<65} {'Energy %':>12} {'Latency %':>12}")
        print("-" * 100)
        for key in sorted(diff_details.keys()):
            diffs = diff_details[key]
            if "error" in diffs:
                print(f"{key:<65} {'ERROR':>12} {'ERROR':>12}")
                continue
            e_pct = f"{diffs['energy']['pct']:+.4f}%" if "energy" in diffs else "match"
            l_pct = f"{diffs['latency']['pct']:+.4f}%" if "latency" in diffs else "match"
            print(f"{key:<65} {e_pct:>12} {l_pct:>12}")


def fast_mode():
    """Compare the two JSON files directly."""
    print("MODE: Fast comparison (JSON-to-JSON)")
    print(f"Main reference:    {MAIN_REF_PATH}")
    print(f"Current reference: {CURRENT_REF_PATH}")

    with open(MAIN_REF_PATH) as f:
        main_ref = json.load(f)
    with open(CURRENT_REF_PATH) as f:
        current_ref = json.load(f)

    print(f"\nMain reference:    {len(main_ref)} entries")
    print(f"Current reference: {len(current_ref)} entries")

    # Check for key differences
    main_keys = set(main_ref.keys())
    current_keys = set(current_ref.keys())
    only_in_main = main_keys - current_keys
    only_in_current = current_keys - main_keys
    if only_in_main:
        print(f"\nKeys only in main reference: {only_in_main}")
    if only_in_current:
        print(f"\nKeys only in current reference: {only_in_current}")

    # Compare shared keys
    shared_keys = main_keys & current_keys
    total = len(shared_keys)
    matching = 0
    differing = 0
    diff_details = {}

    for key in sorted(shared_keys):
        diffs = compare_test_entry(main_ref[key], current_ref[key])
        if diffs:
            differing += 1
            diff_details[key] = diffs
        else:
            matching += 1

    errors = len(only_in_main) + len(only_in_current)
    print_summary(total, matching, differing, errors, diff_details)

    # Now validate a few quick cases by re-running on current code
    print("\n" + "=" * 100)
    print("VALIDATION: Re-running quick test cases to confirm current reference is accurate...")
    print("=" * 100)

    af.set_n_parallel_jobs(1)

    # Pick the fastest cases (small workloads)
    quick_keys = [k for k in sorted(shared_keys) if "matmuls|KN=64" in k]
    validation_ok = True
    for key in quick_keys:
        print(f"\n  Validating: {key} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            arch, workload, jinja_parse_data, fused = parse_key(key)
            cur = _run(arch, workload, fused, jinja_parse_data=jinja_parse_data)
            elapsed = time.time() - t0

            cur_ref = current_ref[key]
            # Check if the fresh run matches the current reference
            if abs(cur["energy"] - cur_ref["energy"]) > 1e-6 or abs(cur["latency"] - cur_ref["latency"]) > 1e-6:
                print(f"MISMATCH! ({elapsed:.1f}s)")
                print(f"    Fresh energy:  {cur['energy']}")
                print(f"    Cached energy: {cur_ref['energy']}")
                print(f"    Fresh latency:  {cur['latency']}")
                print(f"    Cached latency: {cur_ref['latency']}")
                validation_ok = False
            else:
                print(f"OK ({elapsed:.1f}s)")
        except Exception as e:
            print(f"ERROR: {e}")
            validation_ok = False

    if validation_ok:
        print("\nValidation: All quick cases confirm current reference is accurate.")
    else:
        print("\nValidation: WARNING - Some cases don't match the cached reference!")
        print("The current regression_reference.json may be stale. Consider regenerating.")

    return 0 if differing == 0 and errors == 0 else 1


def full_mode():
    """Re-run all 32 test cases and compare against main reference."""
    print("MODE: Full re-run comparison")

    with open(MAIN_REF_PATH) as f:
        main_ref = json.load(f)

    af.set_n_parallel_jobs(1)

    total = len(main_ref)
    matching = 0
    differing = 0
    errors = 0
    diff_details = {}

    for idx, key in enumerate(sorted(main_ref.keys()), 1):
        ref = main_ref[key]
        print(f"\n[{idx}/{total}] Running: {key} ...", flush=True)
        t0 = time.time()
        try:
            arch, workload, jinja_parse_data, fused = parse_key(key)
            cur = _run(arch, workload, fused, jinja_parse_data=jinja_parse_data)
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            errors += 1
            diff_details[key] = {"error": str(e)}
            continue

        elapsed = time.time() - t0
        diffs = compare_test_entry(ref, cur)
        if diffs:
            differing += 1
            diff_details[key] = diffs
            print(f"  DIFFERS ({elapsed:.1f}s)")
        else:
            matching += 1
            print(f"  MATCH ({elapsed:.1f}s)")

    print_summary(total, matching, differing, errors, diff_details)
    return 0 if differing == 0 and errors == 0 else 1


def main():
    if "--full" in sys.argv:
        return full_mode()
    else:
        return fast_mode()


if __name__ == "__main__":
    sys.exit(main())
