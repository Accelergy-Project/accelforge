#!/usr/bin/env python3

"""
VIBE CODE WARNING: Entirely LLM-generated. Humans have verified outputs on a small set
of samples. Inspect any generated workloads before using them.

Convert an inference workload into its training backward-pass workload.

For each Einsum Y = X * W, creates:

- dY_copy = dY (copy) <IF AND ONLY IF dY IS SAVED & NOT COMPUTED>
- dW = X * dY_copy
- W_copy = W (copy)
- update_W, update_MV_W = W_copy * dW * MV_W (the optimizer state is written back too)
- dX = W_copy * dY_copy.

Operands produced by other Einsums (e.g., K in QK = Q * K) get gradients but no
copy/update. Two-tensor Einsums become elementwise dX = dY. Tensors with several
consumers get one gradient contribution per consumer plus a sum.

Forward tensors listed in --recompute are recomputed in the backward pass instead of
stored, recomputing only those that are necessary (in forward order). Stored forward
tensors read by several Einsums get an optional on-chip copy (copy_X reading prev_X)
right before their first reader.

Renames includes input, output, and weight for each Einsum (weight omitted for
two-tensor Einsums). Tensors with dependencies from the inference pass are renamed
forward_dependent, and should be kept in a storage that can hold them through both
passes.

Bits per value references four variables: gradient_bits, weight_bits, activation_bits,
optimizer_bits. Optimizer bits are PER WEIGHT VALUE, not per optimizer value.

Usage:
  python scripts/make_training_workload.py workload.yaml -D BATCH_SIZE=1 -o
  training.yaml --gradient-bits 16 --weight-bits 16 --activation-bits 16
  --optimizer-bits 64
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

BITS = ("gradient", "weight", "activation", "optimizer")

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
                {"name": "weight_gradient", "source": "Nothing"},
                {"name": "recomputed", "source": "Nothing"},
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


def convert(
    workload: Workload,
    recompute=(),
    include_weight_copies=True,
    delay_updates_to_end=False,
    dependencies=(),
) -> tuple[Workload, list[str]]:
    """Return the training workload and per-Einsum comments for an evaluated
    inference workload. dependencies are (forward tensor, backward Einsum) pairs: the
    Einsum also reads the tensor."""
    producers, consumers, ranks, out_proj = {}, {}, {}, {}
    for e in workload.einsums:
        for a in e.tensor_accesses:
            ranks.setdefault(a.name, tuple(a.projection))
            if a.output:
                producers[a.name] = e.name
                out_proj[a.name] = a.projection
            else:
                consumers.setdefault(a.name, []).append(e.name)
    pending = {t: len(c) for t, c in consumers.items()}
    einsums, comments, copies, summed, updates = [], [], {}, set(), []
    weight_gradient_tensors = set()
    fwd = {e.name: classify(e) for e in workload.einsums}
    order = [e.name for e in workload.einsums]

    # Count reads of forward tensors by the backward pass. Recomputing a tensor
    # replaces its reads with reads of its producer's inputs.
    reads = {}

    def read(t, is_weight=False):
        if not (is_weight and t not in producers):
            reads[t] = reads.get(t, 0) + 1

    for e in workload.einsums:
        x, w, _ = fwd[e.name]
        if w is not None:
            read(x)
            read(w, is_weight=True)
    extra_inputs = {}
    for t, einsum_name in dependencies:
        if t not in ranks:
            raise ValueError(f"Can not add dependency on {t}: not a forward tensor")
        extra_inputs.setdefault(einsum_name, []).append(t)
        read(t, is_weight=True)
    recomputed = set()
    for t in sorted(
        recompute,
        key=lambda t: order.index(producers[t]) if t in producers else -1,
        reverse=True,
    ):
        if t not in producers:
            raise ValueError(f"Can not recompute {t}: no forward Einsum produces it")
        if not reads.get(t):
            print(
                f"Not recomputing {t}: the backward pass never reads it",
                file=sys.stderr,
            )
            continue
        recomputed.add(t)
        del reads[t]
        px, pw, _ = fwd[producers[t]]
        read(px)
        if pw is not None:
            read(pw, is_weight=True)
    copy_acts = {t for t, n in reads.items() if n > 1}
    recompute_done, act_copied = set(), set()

    def stored(t):
        return t in reads and t not in copy_acts

    def fresh(name):
        if name in ranks:
            raise ValueError(f"Generated tensor {name} collides with a forward tensor")
        return name

    def contribution(t, consumer):
        return grad(t) if len(consumers[t]) == 1 else f"{grad(t)}_{consumer}"

    def add(
        comment,
        out,
        ins,
        renames=None,
        source=None,
        copy=False,
        name=None,
        tags=(),
        extra_outs=(),
    ):
        """out, extra_outs and ins are (tensor, projection, bits_per_value) triples;
        source is the forward Einsum whose per-Einsum bounds carry over."""
        renames = dict(renames or {})
        for t in extra_inputs.pop(name or out[0], ()):
            ensure_recomputed(t)
            proj = out_proj[t] if t in out_proj else {r: r.lower() for r in ranks[t]}
            ins = ins + [
                (t, proj, "activation_bits" if t in producers else "weight_bits")
            ]
            renames.setdefault("input", ins[0][0])
            renames.setdefault("output", out[0])
            renames["weight"] = " | ".join(
                [renames["weight"], t] if "weight" in renames else [t]
            )
            if stored(t):
                renames["forward_dependent"] = " | ".join(
                    [renames["forward_dependent"], t]
                    if "forward_dependent" in renames
                    else [t]
                )
        # A copy goes right before the first Einsum that reads the copied tensor
        for t, _, _ in ins:
            if t in copy_acts:
                ensure_activation_copied(t)
        comments.append(comment)
        outs = [out] + list(extra_outs)
        accesses = [
            TensorAccess(
                name=t, projection=dict(p), bits_per_value=b, output=n < len(outs)
            )
            for n, (t, p, b) in enumerate(outs + ins)
        ]
        for rename, tensors in (
            ("weight_gradient", weight_gradient_tensors),
            ("recomputed", recomputed),
        ):
            here = [t for t, _, _ in [out] + ins if t in tensors]
            if here:
                renames[rename] = " | ".join(here)
        kwargs = {}
        if source is not None:
            variables = idents(x for _, p, _ in [out] + ins for x in p.values())
            kwargs["iteration_space_shape"] = [
                s for s in source.iteration_space_shape if idents([s]) <= variables
            ]
        einsums.append(
            Einsum(
                name=name or out[0],
                tensor_accesses=accesses,
                renames=renames,
                is_copy_operation=copy,
                tags=list(tags),
                **kwargs,
            )
        )

    def add_sum(t):
        assert (
            pending[t] == 0
        ), f"{t} gradient incomplete; Einsums not in produce-before-consume order?"
        proj = {r: r.lower() for r in ranks[t]}
        summed.add(t)
        add(
            f"Sum the gradient contributions to {t}.",
            (grad(t), proj, "gradient_bits"),
            [(contribution(t, c), proj, "gradient_bits") for c in consumers[t]],
            renames={"input": "Inputs", "weight": "Nothing"},
            tags=["gradient_sum"],
        )

    def ensure_copied(t, proj, bits, source, forward_dependent=False):
        if t not in copies:
            copies[t] = fresh(f"{t}_copy")
            renames = {"input": t, "output": copies[t], "weight": "Nothing"}
            if forward_dependent:
                renames["forward_dependent"] = t
            add(
                f"Copy {t}.",
                (copies[t], proj, bits),
                [(t, proj, bits)],
                renames=renames,
                source=source,
                copy=True,
            )
        return copies[t]

    def ensure_activation_copied(t):
        if t in act_copied:
            return
        act_copied.add(t)
        proj = out_proj.get(t, {r: r.lower() for r in ranks[t]})
        prev = fresh(f"prev_{t}")
        add(
            f"Copy {t} from off-chip.",
            (t, proj, "activation_bits"),
            [(prev, proj, "activation_bits")],
            renames={
                "input": prev,
                "output": t,
                "weight": "Nothing",
                "forward_dependent": prev,
            },
            source=workload.einsums[producers[t]] if t in producers else None,
            copy=True,
            name=f"copy_{t}",
        )

    def ensure_recomputed(*tensors):
        """Recompute the tensors and what their recomputes read, in forward order."""
        needed = set()

        def collect(t):
            if t in recomputed and t not in recompute_done and t not in needed:
                needed.add(t)
                px, pw, _ = fwd[producers[t]]
                collect(px)
                if pw is not None:
                    collect(pw)

        for t in tensors:
            collect(t)
        for t in sorted(needed, key=lambda t: order.index(producers[t])):
            add_recompute(t)

    def add_recompute(t):
        recompute_done.add(t)
        p = workload.einsums[producers[t]]
        px, pw, _ = fwd[p.name]
        proj = {a.name: a.projection for a in p.tensor_accesses}
        ins, renames = [], {"input": px, "output": t}
        for i in (px, pw):
            if i is None:
                continue
            true_w = i == pw and pw not in producers
            # A true weight is read the same way the input gradient reads it
            if true_w and include_weight_copies:
                i = ensure_copied(i, proj[i], "weight_bits", p, forward_dependent=True)
            ins.append(
                (
                    i,
                    proj[pw if true_w else i],
                    "weight_bits" if true_w else "activation_bits",
                )
            )
        if pw is not None:
            renames["weight"] = ins[-1][0]
        fd = [i for i, _, _ in ins if stored(i) or (i == pw and pw not in producers)]
        if fd:
            renames["forward_dependent"] = " | ".join(fd)
        add(
            f"Recompute {t}.",
            (t, proj[t], "activation_bits"),
            ins,
            renames=renames,
            source=p,
            copy=p.is_copy_operation,
        )

    def add_update(w, proj, source):
        weight_gradient_tensors.add(grad(w))
        if len(consumers[w]) > 1:
            add_sum(w)
        w_read = w
        if include_weight_copies:
            w_read = ensure_copied(
                w, proj, "weight_bits", source, forward_dependent=True
            )
        update, mv = fresh(f"update_{w}"), fresh(f"MV_{w}")
        update_mv = fresh(f"update_MV_{w}")

        def add_update_einsum():
            add(
                f"Update {w} and its optimizer state with gradient {grad(w)}.",
                (update, proj, "weight_bits"),
                [
                    (w_read, proj, "weight_bits"),
                    (grad(w), proj, "gradient_bits"),
                    (mv, proj, "optimizer_bits"),
                ],
                renames={
                    "input": "Inputs",
                    "output": "Outputs",
                    "weight": "Nothing",
                    "forward_dependent": (
                        "Outputs" if w_read != w else f"{w} | Outputs"
                    ),
                },
                source=source,
                tags=["weight_update"],
                extra_outs=[(update_mv, proj, "optimizer_bits")],
            )

        if delay_updates_to_end:
            updates.append(add_update_einsum)
        else:
            add_update_einsum()

    for e in reversed(list(workload.einsums)):
        x, w, y = fwd[e.name]
        dy = grad(y)
        if len(consumers.get(y, ())) > 1:
            add_sum(y)
        proj = {a.name: a.projection for a in e.tensor_accesses}
        # Copy dY only if it is saved and not computed (no backward Einsum produces
        # it). The copy goes before any recompute so it does not sit between a
        # recomputed tensor and the Einsum that consumes it (FFM only fuses adjacent
        # Einsums).
        dy_copy = dy
        if w is not None and y not in consumers:
            dy_copy = ensure_copied(dy, proj[y], "gradient_bits", e)
        if w is not None:
            ensure_recomputed(x, w)

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
        if true_weight:
            weight_gradient_tensors.add(dw)

        # Weight gradient is input * output gradient.
        renames = {"input": dy_copy, "output": dw, "weight": x}
        if stored(x):
            renames["forward_dependent"] = x
        add(
            f"Backward of {e.name}: gradient of {w}.",
            (dw, proj[w], "gradient_bits"),
            [(x, proj[x], "activation_bits"), (dy_copy, proj[y], "gradient_bits")],
            renames=renames,
            source=e,
        )
        pending[w] -= 1
        w_read = w
        if true_weight:
            if pending[w] == 0:
                add_update(w, proj[w], e)
            if include_weight_copies:
                w_read = ensure_copied(
                    w, proj[w], "weight_bits", e, forward_dependent=True
                )

        # Input gradient is weight * output gradient.
        renames = {"input": dy_copy, "output": dx, "weight": w_read}
        # Reading a forward tensor (an uncopied weight or a stored operand) makes the
        # Einsum forward dependent.
        if w_read == w and (true_weight or stored(w)):
            renames["forward_dependent"] = w
        add(
            f"Backward of {e.name}: gradient of {x}.",
            (dx, proj[x], "gradient_bits"),
            [
                (w_read, proj[w], "weight_bits" if true_weight else "activation_bits"),
                (dy_copy, proj[y], "gradient_bits"),
            ],
            renames=renames,
            source=e,
        )
        pending[x] -= 1

    # Pristine inputs with several consumers never hit a producer; sum them here.
    for t in consumers:
        if t not in producers and t not in summed and len(consumers[t]) > 1:
            add_sum(t)
    for add_update_einsum in updates:
        add_update_einsum()
    assert all(v == 0 for v in pending.values()), f"Unconsumed contributions: {pending}"
    if extra_inputs:
        raise ValueError(
            f"No backward Einsum named {sorted(extra_inputs)} for --add-dependency"
        )

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
    parser.add_argument(
        "--recompute",
        nargs="*",
        default=[],
        metavar="TENSOR",
        help="recompute these forward tensors in the backward pass instead of storing "
        "them; they are renamed `recomputed` in every Einsum that produces or reads them",
    )
    parser.add_argument(
        "--add-dependency",
        nargs=2,
        action="append",
        default=[],
        metavar=("TENSOR", "EINSUM"),
        help="make backward Einsum EINSUM also read forward tensor TENSOR, e.g. "
        "QK_softmax dQK for the exact softmax backward",
    )
    parser.add_argument(
        "--include-weight-copies",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="copy each weight (W_copy = W) before the backward pass reads it",
    )
    parser.add_argument(
        "--delay-updates-to-end",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="put every weight update Einsum at the very end",
    )
    for k in BITS:
        parser.add_argument(f"--{k}-bits", type=int, required=True, help=f"{k}_bits")
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

    training, comments = convert(
        workload,
        args.recompute,
        args.include_weight_copies,
        args.delay_updates_to_end,
        args.add_dependency,
    )
    dump = training.model_dump(exclude_defaults=True)
    dump["einsums"] = CommentedSeq(dump["einsums"])
    for i, (einsum, comment) in enumerate(zip(dump["einsums"], comments)):
        for a in einsum["tensor_accesses"]:
            if all(k == v.upper() for k, v in a["projection"].items()):
                a["projection"] = list(a["projection"].values())
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
            "variables": {f"{k}_bits": getattr(args, f"{k}_bits") for k in BITS},
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
