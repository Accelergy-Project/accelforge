"""
Regression tests for the FFM mapper.

    python tests/test_regression.py            # (re)generate both reference jsons
    pytest tests/test_regression.py -v         # compare against reference
"""

import json
import math
from numbers import Number
import os
import re
import unittest
from pathlib import Path

import accelforge as af
from accelforge.frontend.spec import Spec
from accelforge.mapper import Metrics

JSON_PATH = Path(__file__).parent / "regression_reference.json"
HWCOMPONENTS_JSON_PATH = Path(__file__).parent / "hwcomponents_expected.json"

REGRESSION_CASES = {
    af.examples.arches.simple: [
        {
            "workload": af.examples.workloads.matmuls,
            "jinja_parse_data": {"N_EINSUMS": 2, "M": 64, "KN": 64},
        },
        {"workload": af.examples.workloads.three_matmuls_annotated},
        {"workload": af.examples.workloads.gpt3_175B},
        {"workload": af.examples.workloads.gpt3_175B_kv_cache},
        # {"workload": af.examples.workloads.gpt3_6_7B},
        # {"workload": af.examples.workloads.gpt3_6_7B_kv_cache},
    ],
    af.examples.arches.eyeriss: [
        {
            "workload": af.examples.workloads.matmuls,
            "jinja_parse_data": {"N_EINSUMS": 2, "M": 64, "KN": 64},
        },
        {"workload": af.examples.workloads.three_matmuls_annotated},
        # {"workload": af.examples.workloads.gpt3_175B},
        # {"workload": af.examples.workloads.gpt3_175B_kv_cache},
        {"workload": af.examples.workloads.gpt3_6_7B},
        {"workload": af.examples.workloads.gpt3_6_7B_kv_cache},
    ],
    af.examples.arches.simba: [
        {
            "workload": af.examples.workloads.matmuls,
            "jinja_parse_data": {"N_EINSUMS": 2, "M": 64, "KN": 64},
        },
        {"workload": af.examples.workloads.three_matmuls_annotated},
        # {"workload": af.examples.workloads.gpt3_175B},
        # {"workload": af.examples.workloads.gpt3_175B_kv_cache},
        {"workload": af.examples.workloads.gpt3_6_7B},
        {"workload": af.examples.workloads.gpt3_6_7B_kv_cache},
    ],
    af.examples.arches.tpu_v4i: [
        {
            "workload": af.examples.workloads.matmuls,
            "jinja_parse_data": {"N_EINSUMS": 2, "M": 64, "KN": 64},
        },
        {"workload": af.examples.workloads.three_matmuls_annotated},
        {"workload": af.examples.workloads.gpt3_175B},
        {"workload": af.examples.workloads.gpt3_175B_kv_cache},
        # {"workload": af.examples.workloads.gpt3_6_7B},
        # {"workload": af.examples.workloads.gpt3_6_7B_kv_cache},
    ],
}


def cast(d):
    if isinstance(d, dict):
        return {str(k): cast(v) for k, v in d.items()}
    if isinstance(d, list):
        return [cast(v) for v in d]
    if isinstance(d, Number):
        return float(d)
    return d


def _key(arch, workload, fused):
    jinja = workload.get("jinja_parse_data", {})
    j = ",".join(f"{k}={v}" for k, v in sorted(jinja.items()))
    return (
        f"{Path(arch).stem}|{Path(workload['workload']).stem}"
        f"|{j}|{'fused' if fused else 'unfused'}"
    )


def _run(arch, workload, fused, print_progress: bool = True):
    spec = Spec.from_yaml(
        arch,
        workload["workload"],
        jinja_parse_data=workload.get("jinja_parse_data"),
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


def _cases(fusion_choices=(False, True)):
    for arch, workloads in REGRESSION_CASES.items():
        for workload in workloads:
            for fused in fusion_choices:
                yield _key(arch, workload, fused), arch, workload, fused


def generate(fusion_choices=(False, True)):
    PARALLEL_GENERATE = False
    from accelforge.util import parallel, delayed, get_n_parallel_jobs

    jobs = {
        key: delayed(_run)(arch, workload, fused, print_progress=not PARALLEL_GENERATE)
        for key, arch, workload, fused in _cases(fusion_choices)
    }
    n_jobs = 1 if not PARALLEL_GENERATE else get_n_parallel_jobs()
    results = parallel(jobs, pbar="Generating regression reference", n_jobs=n_jobs)
    with open(JSON_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {len(results)} results to {JSON_PATH}")


class TestFFMRegression(unittest.TestCase):
    _ref = None

    @classmethod
    def setUpClass(cls):
        af.set_n_parallel_jobs(os.cpu_count(), print_message=True)
        assert JSON_PATH.exists(), f"No reference json. Run: python {__file__}"
        with open(JSON_PATH) as f:
            cls._ref = json.load(f)

    def _check(self, key, arch, workload, fused):
        if key not in self._ref:
            self.skipTest(f"not in json: {key}")
        ref = self._ref[key]
        cur = _run(arch, workload, fused)
        self.assertEqual(
            cur["n_mappings"],
            ref["n_mappings"],
            msg=f"n_mappings for {key}: ref={ref['n_mappings']}, cur={cur['n_mappings']}",
        )
        self.assertTrue(
            math.isclose(cur["energy"], ref["energy"], rel_tol=0.01),
            msg=f"energy for {key}: ref={ref['energy']}, cur={cur['energy']}",
        )
        self.assertTrue(
            math.isclose(cur["latency"], ref["latency"], rel_tol=0.01),
            msg=f"latency for {key}: ref={ref['latency']}, cur={cur['latency']}",
        )
        failures = []
        for s in ["energy_per_component", "latency_per_component", "actions"]:
            for c in ref[s]:
                if c not in cur[s]:
                    failures.append(
                        f"{s} {c}: missing in current (keys: {sorted(cur[s].keys())})"
                    )
                    continue
                if not math.isclose(cur[s][c], ref[s][c], rel_tol=0.01):
                    ratio = cur[s][c] / ref[s][c] if ref[s][c] != 0 else float("inf")
                    failures.append(
                        f"{s} {c}: ref={ref[s][c]:.6e} cur={cur[s][c]:.6e} "
                        f"ratio={ratio:.4f}"
                    )
        if failures:
            self.fail(
                f"Regression failures for {key} "
                f"(n_mappings ref={ref['n_mappings']} cur={cur['n_mappings']}, "
                f"energy ref={ref['energy']:.6e} cur={cur['energy']:.6e}, "
                f"latency ref={ref['latency']:.6e} cur={cur['latency']:.6e}):\n"
                + "\n".join(f"  {f}" for f in failures)
            )


for _k, _a, _w, _f in _cases():
    _name = "test_" + re.sub(r"[^a-zA-Z0-9]", "_", _k)

    def _t(_k=_k, _a=_a, _w=_w, _f=_f):
        def t(self):
            self._check(_k, _a, _w, _f)

        return t

    setattr(TestFFMRegression, _name, _t())


class TestHWComponentsConsistency(unittest.TestCase):
    """
    Checks that hwcomponents models produce expected per-component area, energy, leak
    power, and latency.

    If these fail, then the other regression tests will likely fail as well.
    """

    _expected = None
    _results = {}

    @classmethod
    def setUpClass(cls):
        assert (
            HWCOMPONENTS_JSON_PATH.exists()
        ), f"No hwcomponents reference json at {HWCOMPONENTS_JSON_PATH}"
        with open(HWCOMPONENTS_JSON_PATH) as f:
            cls._expected = json.load(f)

        arches = {
            "eyeriss": af.examples.arches.eyeriss,
            "simba": af.examples.arches.simba,
            "simple": af.examples.arches.simple,
            "tpu_v4i": af.examples.arches.tpu_v4i,
        }
        for name, arch_path in arches.items():
            spec = Spec.from_yaml(
                arch_path,
                af.examples.workloads.matmuls,
                jinja_parse_data={"N_EINSUMS": 2, "M": 64, "KN": 64},
            )
            spec = spec.calculate_component_area_energy_latency_leak(
                einsum_name="Matmul0"
            )
            components = {}
            for node in spec.arch.nodes:
                if not isinstance(node, (af.arch.Memory, af.arch.Compute)):
                    continue
                comp = {}
                if node.area is not None:
                    comp["area"] = float(node.area)
                if node.leak_power is not None:
                    comp["leak_power"] = float(node.leak_power)
                actions = {}
                for a in node.actions:
                    act = {}
                    if a.energy is not None:
                        act["energy"] = float(a.energy)
                    if a.latency is not None:
                        act["latency"] = float(a.latency)
                    if act:
                        actions[a.name] = act
                if actions:
                    comp["actions"] = actions
                if comp:
                    components[node.name] = comp
            cls._results[name] = components

    def _check_arch(self, name):
        expected = self._expected[name]
        actual = self._results[name]
        for comp_name, exp_comp in expected.items():
            self.assertIn(comp_name, actual, f"{name}: missing component {comp_name}")
            act_comp = actual[comp_name]
            for field in ["area", "leak_power"]:
                if field in exp_comp:
                    exp_val = exp_comp[field]
                    act_val = act_comp.get(field)
                    self.assertIsNotNone(act_val, f"{name}.{comp_name}.{field} missing")
                    delta = max(abs(exp_val) * 1e-6, 1e-15)
                    self.assertAlmostEqual(
                        act_val,
                        exp_val,
                        delta=delta,
                        msg=f"{name}.{comp_name}.{field}: {act_val} != {exp_val}",
                    )
            for act_name, exp_act in exp_comp.get("actions", {}).items():
                self.assertIn(
                    act_name,
                    act_comp.get("actions", {}),
                    f"{name}.{comp_name}: missing action {act_name}",
                )
                act_act = act_comp["actions"][act_name]
                for field, exp_val in exp_act.items():
                    act_val = act_act.get(field)
                    self.assertIsNotNone(
                        act_val, f"{name}.{comp_name}.{act_name}.{field} missing"
                    )
                    delta = max(abs(exp_val) * 1e-6, 1e-15)
                    self.assertAlmostEqual(
                        act_val,
                        exp_val,
                        delta=delta,
                        msg=f"{name}.{comp_name}.{act_name}.{field}: {act_val} != {exp_val}",
                    )


for _arch_name in ["eyeriss", "simba", "simple", "tpu_v4i"]:
    _test_name = f"test_{_arch_name}"

    def _make_test(_name=_arch_name):
        def t(self):
            self._check_arch(_name)

        return t

    setattr(TestHWComponentsConsistency, _test_name, _make_test())


def generate_hwcomponents():
    arches = {
        "eyeriss": af.examples.arches.eyeriss,
        "simba": af.examples.arches.simba,
        "simple": af.examples.arches.simple,
        "tpu_v4i": af.examples.arches.tpu_v4i,
    }
    results = {}
    for name, arch_path in arches.items():
        spec = Spec.from_yaml(
            arch_path,
            af.examples.workloads.matmuls,
            jinja_parse_data={"N_EINSUMS": 2, "M": 64, "KN": 64},
        )
        spec = spec.calculate_component_area_energy_latency_leak(einsum_name="Matmul0")
        components = {}
        for node in spec.arch.nodes:
            if not isinstance(node, (af.arch.Memory, af.arch.Compute)):
                continue
            comp = {}
            if node.area is not None:
                comp["area"] = float(node.area)
            if node.leak_power is not None:
                comp["leak_power"] = float(node.leak_power)
            actions = {}
            for a in node.actions:
                act = {}
                if a.energy is not None:
                    act["energy"] = float(a.energy)
                if a.latency is not None:
                    act["latency"] = float(a.latency)
                if act:
                    actions[a.name] = act
            if actions:
                comp["actions"] = actions
            if comp:
                components[node.name] = comp
        results[name] = components
    with open(HWCOMPONENTS_JSON_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {len(results)} arch results to {HWCOMPONENTS_JSON_PATH}")


if __name__ == "__main__":
    generate_hwcomponents()
    generate(fusion_choices=(True, False))
