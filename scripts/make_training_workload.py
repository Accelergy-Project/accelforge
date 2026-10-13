#!/usr/bin/env python3

"""
VIBE CODE WARNING: Entirely LLM-generated. Humans have verified outputs on a small set
of samples. Inspect any generated workloads before using them.


Convert an inference workload into its training backward-pass workload.

For each Einsum Y = X * W, creates:

- dW = X * dY
- W = prev_W (copy)
- new_W = W * dW * MV_W
- dX = W * dY. 

Operands produced by other Einsums (e.g., K in QK = Q * K) get gradients but no
copy/update. Two-tensor Einsums become elementwise dX = dY. Tensors with several
consumers get one gradient contribution per consumer plus a sum.

Renames includes input, output, and weight for each Einsum (weight omitted for
two-tensor Einsums). Tensors with dependencies from the inference pass are renamed
forward_dependent, and should be kept in backing storage.

Bits per value references four variables: gradient_bits, weight_bits, activation_bits,
optimizer_bits. Optimizer bits are PER WEIGHT VALUE, not per optimizer value.

Usage:
  python scripts/make_training_workload.py workload.yaml -D BATCH_SIZE=1 -o
  training.yaml
"""

import argparse
import ast
import re
import sys

from ruamel.yaml.comments import CommentedSeq

from accelforge import Spec
from accelforge.frontend.workload import Einsum, TensorAccess, Workload, _ISL_REGEX
from accelforge.util import _yaml
from accelforge.util._setexpressions import InvertibleSet

BITS_DEFAULTS = {"gradient": 16, "weight": 16, "activation": 16, "optimizer": 64}

DEFAULT_RENAMES = {
    "einsums": [
        {
            "name": "default",
            "tensor_accesses": [
                {
                    "name": "input",
                    "source": "Inputs & Intermediates if len(All) == 3 else Inputs",
                    "expected_count": 1,
                },
                {"name": "output", "source": "Outputs", "expected_count": 1},
                {
                    "name": "weight",
                    "source": "~(input | output)",
                    "expected_count": "1 if len(All) == 3 else 0",
                },
                {"name": "forward_dependent", "source": "Nothing"},
            ],
        }
    ]
}


def idents(strings):
    return {v for s in strings for v in re.findall(_ISL_REGEX, s)}


def classify(einsum: Einsum) -> tuple[str, str | None, str]:
    """Return the (input, weight, output) tensors of an evaluated Einsum."""
    renames = {}
    for r in einsum.renames:
        if isinstance(r.source, InvertibleSet):
            renames.setdefault(r.name, r.source.instance)

    def single(slot):
        s = renames.get(slot, ())
        if len(s) != 1:
            raise ValueError(f"{einsum.name}: rename {slot!r} = {sorted(s)}, need 1")
        return next(iter(s))

    n = len(einsum.tensor_accesses)
    if n not in (2, 3):
        raise ValueError(f"Einsum {einsum.name} has {n} tensors; only 2 or 3 supported")
    return single("input"), single("weight") if n == 3 else None, single("output")


def grad(t):
    return "d" + t


def convert(workload: Workload) -> tuple[Workload, list[str]]:
    """Return the training workload and per-Einsum comments for an evaluated
    inference workload."""
    producers, consumers, ranks = {}, {}, {}
    for e in workload.einsums:
        for a in e.tensor_accesses:
            ranks.setdefault(a.name, tuple(a.projection))
            if a.output:
                producers[a.name] = e.name
            else:
                consumers.setdefault(a.name, []).append(e.name)
    pending = {t: len(c) for t, c in consumers.items()}
    einsums, comments, copied = [], [], set()

    def fresh(name):
        if name in ranks:
            raise ValueError(f"Generated tensor {name} collides with a forward tensor")
        return name

    def contribution(t, consumer):
        return grad(t) if len(consumers[t]) == 1 else f"{grad(t)}_{consumer}"

    def add(comment, out, ins, renames=None, source=None, copy=False):
        """out and ins are (tensor, projection, bits_per_value) triples; source
        is the forward Einsum whose per-Einsum bounds carry over."""
        comments.append(comment)
        accesses = [
            TensorAccess(name=t, projection=dict(p), bits_per_value=b, output=n == 0)
            for n, (t, p, b) in enumerate([out] + ins)
        ]
        kwargs = {}
        if source is not None:
            variables = idents(x for _, p, _ in [out] + ins for x in p.values())
            kwargs["iteration_space_shape"] = [
                s for s in source.iteration_space_shape if idents([s]) <= variables
            ]
        einsums.append(
            Einsum(
                name=out[0],
                tensor_accesses=accesses,
                renames=renames or {},
                is_copy_operation=copy,
                **kwargs,
            )
        )

    def add_sum(t):
        assert (
            pending[t] == 0
        ), f"{t} gradient incomplete; Einsums not in produce-before-consume order?"
        proj = {r: r.lower() for r in ranks[t]}
        add(
            f"Sum the gradient contributions to {t}.",
            (grad(t), proj, "gradient_bits"),
            [(contribution(t, c), proj, "gradient_bits") for c in consumers[t]],
            renames={"input": "Inputs", "weight": "Nothing"},
        )

    def ensure_copied(w, proj, source):
        if w in copied:
            return
        copied.add(w)
        prev = fresh(f"prev_{w}")
        add(
            f"Copy {w} from off-chip.",
            (w, proj, "weight_bits"),
            [(prev, proj, "weight_bits")],
            renames={
                "input": prev,
                "output": w,
                "weight": "Nothing",
                "forward_dependent": prev,
            },
            source=source,
            copy=True,
        )

    def add_update(w, proj, source):
        if len(consumers[w]) > 1:
            add_sum(w)
        ensure_copied(w, proj, source)
        new, mv = fresh(f"new_{w}"), fresh(f"MV_{w}")
        add(
            f"Update {w} with gradient {grad(w)}.",
            (new, proj, "weight_bits"),
            [
                (w, proj, "weight_bits"),
                (grad(w), proj, "gradient_bits"),
                (mv, proj, "optimizer_bits"),
            ],
            renames={
                "input": "Inputs",
                "output": new,
                "weight": "Nothing",
                "forward_dependent": new,
            },
            source=source,
        )

    for e in reversed(list(workload.einsums)):
        x, w, y = classify(e)
        dy = grad(y)
        if len(consumers.get(y, ())) > 1:
            add_sum(y)
        proj = {a.name: a.projection for a in e.tensor_accesses}

        if w is None:
            dx = fresh(contribution(x, e.name))
            add(
                f"Backward of {e.name}.",
                (dx, proj[x], "gradient_bits"),
                [(dy, proj[y], "gradient_bits")],
                source=e,
                copy=e.is_copy_operation,
            )
            pending[x] -= 1
            continue

        true_weight = w not in producers
        dw, dx = fresh(contribution(w, e.name)), fresh(contribution(x, e.name))

        # Weight gradient is input * output gradient.
        add(
            f"Backward of {e.name}: gradient of {w}.",
            (dw, proj[w], "gradient_bits"),
            [(x, proj[x], "activation_bits"), (dy, proj[y], "gradient_bits")],
            renames={"input": x, "output": dy, "weight": dw, "forward_dependent": x},
            source=e,
        )
        pending[w] -= 1
        if true_weight:
            if pending[w] == 0:
                add_update(w, proj[w], e)
            ensure_copied(w, proj[w], e)

        # Input gradient is weight * output gradient.
        renames = {"input": dy, "output": dx, "weight": w}
        if not true_weight:
            renames["forward_dependent"] = w
        add(
            f"Backward of {e.name}: gradient of {x}.",
            (dx, proj[x], "gradient_bits"),
            [
                (w, proj[w], "weight_bits" if true_weight else "activation_bits"),
                (dy, proj[y], "gradient_bits"),
            ],
            renames=renames,
            source=e,
        )
        pending[x] -= 1

    # Pristine inputs with several consumers never hit a producer; sum them here.
    for t in consumers:
        if t not in producers and t not in copied and len(consumers[t]) > 1:
            add_sum(t)
    assert all(v == 0 for v in pending.values()), f"Unconsumed contributions: {pending}"

    training = Workload(
        einsums=einsums,
        rank_sizes=dict(workload.rank_sizes),
        iteration_space_shape=dict(workload.iteration_space_shape),
        persistent_tensors="forward_dependent",
    )
    return training, comments


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("workload", help="path to the inference workload YAML")
    parser.add_argument("-o", "--output", help="output path (default: stdout)")
    parser.add_argument(
        "-D",
        "--define",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="jinja variable for parsing the workload, e.g. -D BATCH_SIZE=1",
    )
    for k, v in BITS_DEFAULTS.items():
        parser.add_argument(f"--{k}-bits", type=int, default=v, help=f"default {v}")
    args = parser.parse_args()

    defines = {}
    for d in args.define:
        k, _, v = d.partition("=")
        try:
            defines[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            defines[k] = v

    spec = Spec.from_yaml(args.workload, jinja_parse_data=defines)
    workload = spec._spec_eval_expressions(eval_arch=False).workload

    producers = {t for e in workload.einsums for t in e.output_tensor_names}
    print(f"{'Einsum':<24}{'input':<20}{'weight':<20}{'output':<20}", file=sys.stderr)
    for e in workload.einsums:
        x, w, y = classify(e)
        kind = (
            "copy"
            if e.is_copy_operation
            else (
                "elementwise"
                if w is None
                else "weight" if w not in producers else "forward intermediate"
            )
        )
        print(f"{e.name:<24}{x:<20}{w or '-':<20}{y:<20}{kind}", file=sys.stderr)

    training, comments = convert(workload)
    dump = training.model_dump(exclude_defaults=True)
    dump["einsums"] = CommentedSeq(dump["einsums"])
    for i, (einsum, comment) in enumerate(zip(dump["einsums"], comments)):
        if "renames" in einsum:
            einsum["renames"] = {r["name"]: r["source"] for r in einsum["renames"]}
        dump["einsums"].yaml_set_comment_before_after_key(i, f"\n{comment}", indent=2)

    externals = ", ".join(
        grad(t) for t in producers if not workload.einsums_with_tensor_as_input(t)
    )
    source_desc = args.workload + (f" (jinja: {defines})" if defines else "")
    text = (
        f"# Training backward pass generated from {source_desc}\n"
        f"# by scripts/make_training_workload.py.\n"
        f"#\n"
        f"# Assume as input {externals}. Precisions are gradient_bits, weight_bits, activation_bits,\n"
        f"# optimizer_bits. Note optimizer bits is optimizer bits PER WEIGHT VALUE, not per\n"
        f"# optimizer value. Tensors with dependencies from the inference pass are\n"
        f'# "forward_dependent", and should be kept in backing storage.\n'
    ) + _yaml.to_yaml_string(
        {
            "variables": {
                f"{k}_bits": getattr(args, f"{k}_bits") for k in BITS_DEFAULTS
            },
            "renames": DEFAULT_RENAMES,
            "workload": dump,
        }
    )
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
