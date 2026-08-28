"""
Compare the impact of binding policies on the mappings that FFM finds.

A *binding* decides how a physically-distributed memory's tensor is spread over its
physical instances: each spatial loop below the memory's Storage either partitions the
tensor along one of its ranks, or replicates it across that spatial dimension. This
script maps one workload under several binding policies and tabulates what each one
costs.

Run it as::

    python experiments/binding_constraints.py --arch tests/input_files/distributed_binding.arch.yaml

The interesting column is not only the cost. ``n_templates`` says how much of the
mapspace each policy left, and a policy that looks free may simply have removed the
mappings that would have shown its cost -- so read the two together.
"""

import argparse
import copy
import csv
import json
import logging
import sys
import time
from pathlib import Path

import accelforge as af
from accelforge.frontend.arch import BindingConstraints, Memory
from accelforge.frontend.mapping import binding_summary
from accelforge.frontend.spec import Spec
from accelforge.mapper import Metrics
from accelforge.mapper.FFM._make_pmappings.make_pmappings import get_jobs


# =================================================================================================
# The policies under comparison
# =================================================================================================
EXPENSIVE = {"explore-1-change", "explore-2-changes", "never-duplicate-explore-1"}
"""
Policies that let the mapper explore bindings rather than pinning one. Each binding of
each pmapping template becomes a template of its own, so these run orders of magnitude
longer than the pinned policies -- minutes to hours rather than seconds, depending on how
many spatial loops bind each shared tensor (only those are explored). Excluded from a
default run; name them with ``--policies`` to include them.
"""


def policies() -> dict[str, dict]:
    """
    Name to the settings that implement each policy. Add to this dict to compare a
    policy of your own; every entry is applied to a fresh copy of the same Spec. The
    ``binding`` entry goes on the memories of the arch (see `build_spec`); every other
    entry is a mapper setting.
    """
    return {
        # The mapper's default: one binding per template, chosen to minimize memory
        # usage and then network movement for that Einsum alone.
        "locally-optimal": {"binding": BindingConstraints()},
        # The same default, but the mapper may also try bindings that differ from it.
        # This is the upper bound the constrained policies are measured against.
        "explore-1-change": {
            "binding": BindingConstraints(max_binding_rank_changes=1)
        },
        "explore-2-changes": {
            "binding": BindingConstraints(max_binding_rank_changes=2)
        },
        # No tensor may be replicated across a spatial dimension: every binding loop
        # must partition its tensor. Trades memory footprint for network traffic.
        "never-duplicate": {
            "binding": BindingConstraints(may_duplicate="Nothing")
        },
        "never-duplicate-explore-1": {
            "binding": BindingConstraints(
                may_duplicate="Nothing", max_binding_rank_changes=1
            )
        },
        # The opposite extreme: every instance holds a whole copy of every tensor. The
        # cheapest possible network, the most expensive possible memory.
        "always-duplicate": {
            "binding": BindingConstraints(must_duplicate="All")
        },
        # Weights are replicated so that every instance can compute with them; the
        # activations flowing between Einsums are partitioned.
        "duplicate-inputs-only": {
            "binding": BindingConstraints(
                may_duplicate="Inputs", must_duplicate="Inputs"
            )
        },
    }


# =================================================================================================
# Running one policy
# =================================================================================================
def build_spec(arch: Path, workload, jinja_parse_data: dict, settings: dict) -> Spec:
    spec = Spec.from_yaml(arch, workload, jinja_parse_data=jinja_parse_data)
    settings = dict(settings)
    binding = settings.pop("binding", None)
    for key, value in settings.items():
        if not hasattr(spec.mapper, key):
            raise ValueError(f"spec.mapper has no setting {key!r}.")
        setattr(spec.mapper, key, copy.deepcopy(value))
    if binding is not None:
        # Binding constraints belong to the memory they constrain, so a policy is a
        # policy for the whole arch only if it is given to every memory. The memories
        # that are not physically distributed have no binding and ignore it, which keeps
        # this script usable with any --arch.
        for node, _ in spec.arch.iterate_hierarchically():
            if isinstance(node, Memory):
                node.binding = copy.deepcopy(binding)
    return spec


def count_templates(spec: Spec, einsum_names: list[str] | None) -> int:
    """
    How many pmapping templates the policy leaves. Templates dropped because no binding
    of them satisfies the constraints are logged by the mapper at WARNING level; this
    counts what survived.
    """
    spec = spec._spec_eval_expressions(eval_arch=False, eval_non_arch=True)
    jobs = get_jobs(
        spec,
        spec.mapper.metrics,
        einsum_names or spec.workload.einsum_names,
        fail_if_no_pmappings_for_einsum=False,
        print_progress=False,
    )
    return sum(
        len(job_list)
        for compatibility2jobs in jobs.values()
        for job_list in compatibility2jobs.values()
    )


def run_policy(
    name: str,
    settings: dict,
    arch: Path,
    workload,
    jinja_parse_data: dict,
    einsum_names: list[str] | None,
) -> dict:
    """Map the workload under one policy and return one row of the comparison."""
    row = {"policy": name, "settings": json.dumps(_describe(settings))}
    started = time.time()
    try:
        row["n_templates"] = count_templates(
            build_spec(arch, workload, jinja_parse_data, settings), einsum_names
        )

        spec = build_spec(arch, workload, jinja_parse_data, settings)
        result = spec.map_workload_to_arch(
            einsum_names=einsum_names, print_progress=False
        )
        if not len(result):
            row["error"] = "no mappings"
            return row

        energy = result.energy()
        latency = result.latency()
        # The mapper returns the Pareto front, so a metric is a list when more than one
        # mapping survives. Report the best by energy-delay product.
        energy, latency = _as_lists(energy), _as_lists(latency)
        best = min(range(len(energy)), key=lambda i: energy[i] * latency[i])
        row["n_mappings"] = len(energy)
        row["energy"] = energy[best]
        row["latency"] = latency[best]
        row["edp"] = energy[best] * latency[best]
        row["binding"] = _format_binding(result.mapping(best))
    except Exception as e:  # noqa: BLE001 -- one failing policy must not stop the sweep
        # A policy whose constraints admit no mapping at all is a result, not a crash:
        # record it and keep going, but keep the traceback recoverable.
        logging.exception("Policy %s failed.", name)
        row["error"] = f"{type(e).__name__}: {e}"
    finally:
        row["seconds"] = round(time.time() - started, 1)
    return row


def _as_lists(metric) -> list[float]:
    return list(metric) if isinstance(metric, (list, tuple)) else [metric]


def _describe(settings: dict) -> dict:
    """A JSON-friendly rendering of the mapper settings a policy applies."""
    return {
        key: json.loads(json.dumps(value, default=_jsonable))
        for key, value in settings.items()
    }


def _jsonable(value):
    if hasattr(value, "model_dump"):
        return json.loads(json.dumps(value.model_dump(), default=_jsonable))
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    return str(value)


def _format_binding(mapping) -> str:
    """``SharedBuffer/T1: X=M, Y=DUPLICATE; ...`` for the chosen mapping."""
    parts = []
    for (component, tensor), per_dimension in sorted(
        binding_summary(mapping.nodes).items()
    ):
        bound = ", ".join(f"{d}={r}" for d, r in per_dimension.items())
        parts.append(f"{component}/{tensor}: {bound}")
    return "; ".join(parts)


# =================================================================================================
# Top level
# =================================================================================================
def _resolve_workload(name: str):
    """A workload YAML path, or a dotted name under ``accelforge.examples.workloads``."""
    path = Path(name)
    if path.exists():
        return path
    value = af.examples.workloads
    for part in name.split("."):
        value = getattr(value, part, None)
        if value is None:
            raise SystemExit(
                f"{name!r} is neither a file nor a workload under "
                f"accelforge.examples.workloads."
            )
    return value



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arch",
        type=Path,
        default=Path(__file__).parent.parent
        / "tests"
        / "input_files"
        / "distributed_binding.arch.yaml",
        help="Architecture YAML. Must contain a physically-distributed memory, or "
        "every policy will give the same answer.",
    )
    parser.add_argument(
        "--einsums",
        nargs="*",
        default=None,
        help="Einsums to map. Defaults to all of them.",
    )
    parser.add_argument(
        "--policies",
        nargs="*",
        default=None,
        help=f"Policies to run, by name. Defaults to all but {sorted(EXPENSIVE)}, "
        f"which explore bindings rather than pinning one and take far longer.",
    )
    parser.add_argument(
        "--workload",
        default="basic.matmuls_any_einsums",
        help="Workload YAML, or a dotted name under accelforge.examples.workloads.",
    )
    parser.add_argument(
        "--jinja",
        default='{"N_EINSUMS": 2, "M": 8, "KN": 8}',
        help="JSON object of jinja_parse_data for the workload.",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("binding_constraints.csv"), help="Output CSV."
    )
    parser.add_argument(
        "--metrics",
        default="ENERGY_DELAY_PRODUCT",
        help="Metrics the mapper optimizes, by Metrics attribute name.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    workload = _resolve_workload(args.workload)
    jinja_parse_data = json.loads(args.jinja)
    metrics = getattr(Metrics, args.metrics)

    chosen = policies()
    if args.policies:
        missing = sorted(set(args.policies) - set(chosen))
        if missing:
            parser.error(f"Unknown policies {missing}. Known: {sorted(chosen)}.")
        chosen = {k: v for k, v in chosen.items() if k in args.policies}
    else:
        chosen = {k: v for k, v in chosen.items() if k not in EXPENSIVE}

    rows = []
    for name, settings in chosen.items():
        settings = {"metrics": metrics, **settings}
        print(f"Running {name}...", flush=True)
        row = run_policy(
            name, settings, args.arch, workload, jinja_parse_data, args.einsums
        )
        rows.append(row)
        print(f"  {row}", flush=True)

    fields = sorted({k for row in rows for k in row})
    fields = ["policy"] + [f for f in fields if f != "policy"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}.")

    baseline = rows[0]
    if "edp" in baseline:
        print(f"\nEDP relative to {baseline['policy']}:")
        for row in rows:
            if "edp" in row:
                print(
                    f"  {row['policy']:<32} {row['edp'] / baseline['edp']:>8.3f}× "
                    f"({row['n_templates']} templates)"
                )
            else:
                print(f"  {row['policy']:<32} {row.get('error', 'no result')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
