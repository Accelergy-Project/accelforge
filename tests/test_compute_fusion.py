"""Unit tests for map_op/reduce_op op-profiles and arch-side op-kind fusion.

An einsum declares `map_op`/`reduce_op`; `effective_op_profile()` turns those
into per-iteration op_kind counts. A Compute component may declare a
ComputeAction with `fuses=[...]` that coalesces several op_kinds into a single
entry keyed under the action's primary `op_kind`. The canonical case is a
fused-MAC unit that pairs (mul, add) into one charge per iteration -- and an
`op_kind="mac"` action fuses [mul, add] by default, so legacy single-MAC arches
keep working unchanged.
"""

import pytest

from accelforge import Spec, examples
from accelforge.frontend.arch.components import ComputeAction
from accelforge.frontend.workload import Einsum, TensorAccess
from accelforge.model._looptree.reuse.symbolic._symbolic import _apply_compute_fusion


def _compute(actions):
    """Minimal object exposing an `.actions` list of real ComputeActions
    (so `effective_fuses`, including the op_kind-aware default, is exercised)."""

    class _C:
        pass

    c = _C()
    c.actions = [
        ComputeAction(name=f"a{i}", op_kind=ok, fuses=list(fz))
        for i, (ok, fz) in enumerate(actions)
    ]
    return c


# ---------- _apply_compute_fusion ----------


def test_apply_fusion_collapses_mul_add_to_mac():
    compute = _compute([("mac", ["mul", "add"])])
    assert _apply_compute_fusion({"mul": 5, "add": 5}, compute) == {"mac": 5}


def test_apply_fusion_requires_equal_counts():
    compute = _compute([("mac", ["mul", "add"])])
    assert _apply_compute_fusion({"mul": 3, "add": 5}, compute) == {"mul": 3, "add": 5}


def test_apply_fusion_missing_op_kind_no_op():
    compute = _compute([("mac", ["mul", "add"])])
    assert _apply_compute_fusion({"add": 1}, compute) == {"add": 1}


def test_apply_fusion_no_fuses_declared():
    # Separate mul/add units (no mac) -> nothing fuses.
    compute = _compute([("mul", []), ("add", [])])
    assert _apply_compute_fusion({"mul": 7, "add": 7}, compute) == {"mul": 7, "add": 7}


def test_apply_fusion_singleton_fuses_is_noop():
    compute = _compute([("mul", ["mul"])])
    assert _apply_compute_fusion({"mul": 4, "add": 4}, compute) == {"mul": 4, "add": 4}


def test_apply_fusion_preserves_unrelated_keys():
    compute = _compute([("mac", ["mul", "add"])])
    assert _apply_compute_fusion({"mul": 2, "add": 2, "max": 1}, compute) == {
        "mac": 2,
        "max": 1,
    }


def test_apply_fusion_multiple_actions():
    compute = _compute([("mac", ["mul", "add"]), ("logical", ["and", "or"])])
    assert _apply_compute_fusion(
        {"mul": 3, "add": 3, "and": 2, "or": 2}, compute
    ) == {"mac": 3, "logical": 2}


def test_apply_fusion_empty_profile_returns_empty():
    compute = _compute([("mac", ["mul", "add"])])
    assert _apply_compute_fusion({}, compute) == {}


def test_mac_action_fuses_mul_add_by_default():
    """An `op_kind="mac"` action with no explicit `fuses` still coalesces
    [mul, add] -- this is what keeps legacy single-MAC arches unchanged."""
    compute = _compute([("mac", [])])  # fuses unset
    assert _apply_compute_fusion({"mul": 6, "add": 6}, compute) == {"mac": 6}


def test_non_mac_action_does_not_fuse_by_default():
    compute = _compute([("mul", []), ("add", [])])  # fuses unset, not mac
    assert _apply_compute_fusion({"mul": 1, "add": 1}, compute) == {"mul": 1, "add": 1}


# ---------- Einsum.effective_op_profile ----------


def test_default_ops_are_mul_add():
    e = Einsum(
        name="GEMM",
        tensor_accesses=[
            TensorAccess(name="A", projection=["m", "k"]),
            TensorAccess(name="B", projection=["k", "n"]),
            TensorAccess(name="C", projection=["m", "n"], output=True),
        ],
    )
    assert e.effective_op_profile() == {"mul": 1, "add": 1}


def test_equal_map_reduce_collapses_to_count_two():
    e = Einsum(
        name="AddReduce",
        tensor_accesses=[
            TensorAccess(name="X", projection=["m", "k"]),
            TensorAccess(name="Y", projection=["m"], output=True),
        ],
        map_op="add",
        reduce_op="add",
    )
    assert e.effective_op_profile() == {"add": 2}


def test_square_map_op_rewritten_to_mul():
    e = Einsum(
        name="SumSq",
        tensor_accesses=[
            TensorAccess(name="X", projection=["m"]),
            TensorAccess(name="Y", projection=[], output=True),
        ],
        map_op="square",
    )
    assert e.effective_op_profile() == {"mul": 1, "add": 1}


def test_copy_operation_has_no_ops():
    e = Einsum(
        name="Copy",
        tensor_accesses=[
            TensorAccess(name="X", projection=["m"]),
            TensorAccess(name="Y", projection=["m"], output=True),
        ],
        is_copy_operation=True,
    )
    assert e.effective_op_profile() == {}


def test_square_plus_add_fuses_to_mac():
    """RMSNorm-style sum-of-squares: square->mul, then mul+add fuses to one MAC."""
    e = Einsum(
        name="SumSq",
        tensor_accesses=[
            TensorAccess(name="X", projection=["m"]),
            TensorAccess(name="Y", projection=[], output=True),
        ],
        map_op="square",
    )
    compute = _compute([("mac", ["mul", "add"])])
    assert _apply_compute_fusion(e.effective_op_profile(), compute) == {"mac": 1}


# ---------- end-to-end via Spec ----------


def test_simple_arch_mac_fuses_mul_add_by_default():
    """examples/arches/simple.yaml declares a single op_kind="mac" compute
    action with no explicit fuses; its effective_fuses defaults to [mul, add]."""
    spec = Spec.from_yaml(
        examples.arches.simple,
        examples.workloads.matmuls,
        jinja_parse_data={"N_EINSUMS": 1, "M": 8, "KN": 8},
    )
    mac = spec.arch.find("MAC")
    assert len(mac.actions) == 1
    assert mac.actions[0].op_kind == "mac"
    assert mac.actions[0].effective_fuses == ["mul", "add"]
