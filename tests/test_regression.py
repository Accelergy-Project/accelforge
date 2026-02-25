"""
Regression tests for the FFM mapper.

    python tests/test_regression.py            # (re)generate reference json
    pytest tests/test_regression.py -v         # compare against reference
"""

import json
from numbers import Number
import os
import re
import unittest
from pathlib import Path

import accelforge as af
from accelforge.frontend.spec import Spec
from accelforge.mapper import Metrics

JSON_PATH = Path(__file__).parent / "regression_reference.json"

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
        self.assertEqual(cur["n_mappings"], ref["n_mappings"])
        self.assertAlmostEqual(cur["energy"], ref["energy"], delta=1e-12)
        self.assertAlmostEqual(cur["latency"], ref["latency"], delta=1e-12)
        print(f"Regression testing {arch=} {workload=} {fused=}")
        for s in ["energy_per_component", "latency_per_component", "actions"]:
            print(f"\tchecking {s}")
            for c in ref[s]:
                print(f"\t\tchecking {c}")
                s2 = f"for {arch=} {workload=} {fused=} {s}"
                self.assertIn(
                    c,
                    cur[s],
                    msg=f"{s2}, {c} not in {cur[s]}",
                )
                self.assertAlmostEqual(
                    cur[s][c],
                    ref[s][c],
                    delta=1e-12,
                    msg=f"{s2}, reference {ref[s][c]} -> current {cur[s][c]}",
                )


for _k, _a, _w, _f in _cases():
    _name = "test_" + re.sub(r"[^a-zA-Z0-9]", "_", _k)

    def _t(_k=_k, _a=_a, _w=_w, _f=_f):
        def t(self):
            self._check(_k, _a, _w, _f)

        return t

    setattr(TestFFMRegression, _name, _t())

if __name__ == "__main__":
    generate(fusion_choices=(True, False))
