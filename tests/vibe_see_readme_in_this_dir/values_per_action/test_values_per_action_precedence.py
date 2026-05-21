"""
Tests for Memory._get_values_per_action precedence:
  1. action.values_per_action[t]
  2. component.values_per_action[t]
  3. action.bits_per_action / bpv
  4. component.bits_per_action / bpv (via action.bpa default expression)
"""

import unittest

from accelforge.frontend.spec import Spec
from accelforge.frontend.workload import Workload
from accelforge.frontend.arch import Arch, Memory, Compute


def _spec(
    *,
    workload_bpv=8,
    comp_bpa=None,
    comp_vpa=None,
    comp_bpv=None,
    read_bpa=None,
    read_vpa=None,
    write_bpa=None,
    write_vpa=None,
):
    comp_kwargs = {}
    if comp_bpa is not None:
        comp_kwargs["bits_per_action"] = comp_bpa
    if comp_vpa is not None:
        comp_kwargs["values_per_action"] = comp_vpa
    if comp_bpv is not None:
        comp_kwargs["bits_per_value"] = comp_bpv

    read_action = {"name": "read", "energy": 1, "latency": 0}
    if read_bpa is not None:
        read_action["bits_per_action"] = read_bpa
    if read_vpa is not None:
        read_action["values_per_action"] = read_vpa

    write_action = {"name": "write", "energy": 1, "latency": 0}
    if write_bpa is not None:
        write_action["bits_per_action"] = write_bpa
    if write_vpa is not None:
        write_action["values_per_action"] = write_vpa

    spec = Spec(
        workload=Workload(
            rank_sizes={"M": 16, "N": 16},
            bits_per_value={"All": workload_bpv},
            einsums=[
                {
                    "name": "E",
                    "tensor_accesses": [
                        {"name": "X", "projection": ["m"]},
                        {"name": "Y", "projection": ["m", "n"], "output": True},
                    ],
                }
            ],
        ),
        arch=Arch(
            nodes=[
                Memory(
                    name="Mem",
                    size=1024,
                    leak_power=0,
                    area=0,
                    actions=[read_action, write_action],
                    **comp_kwargs,
                ),
                Compute(
                    name="MAC",
                    actions=[{"name": "compute", "energy": 1, "latency": 1}],
                    leak_power=0,
                    area=0,
                ),
            ],
        ),
    )
    return spec._spec_eval_expressions(einsum_name="E").arch.find("Mem")


class TestValuesPerActionPrecedence(unittest.TestCase):
    def test_action_vpa_overrides_component_vpa(self):
        mem = _spec(comp_vpa={"X": 3}, read_vpa={"X": 7})
        self.assertEqual(mem._get_values_per_action("read", "X", 8), 7)

    def test_action_vpa_overrides_action_bpa(self):
        mem = _spec(read_bpa=64, read_vpa={"X": 5})
        self.assertEqual(mem._get_values_per_action("read", "X", 8), 5)

    def test_action_vpa_overrides_component_bpa(self):
        mem = _spec(comp_bpa=64, read_vpa={"X": 9})
        self.assertEqual(mem._get_values_per_action("read", "X", 8), 9)

    def test_component_vpa_used_when_action_vpa_empty(self):
        mem = _spec(comp_vpa={"X": 6})
        self.assertEqual(mem._get_values_per_action("read", "X", 8), 6)

    def test_component_vpa_overrides_action_bpa(self):
        mem = _spec(comp_vpa={"X": 11}, read_bpa=64)
        self.assertEqual(mem._get_values_per_action("read", "X", 8), 11)

    def test_component_vpa_overrides_component_bpa(self):
        mem = _spec(comp_bpa=64, comp_vpa={"X": 13})
        self.assertEqual(mem._get_values_per_action("read", "X", 8), 13)

    def test_action_bpa_used_when_vpa_empty(self):
        mem = _spec(read_bpa=64)
        self.assertEqual(mem._get_values_per_action("read", "X", 8), 8.0)

    def test_action_bpa_overrides_component_bpa(self):
        mem = _spec(comp_bpa=32, read_bpa=64)
        self.assertEqual(mem._get_values_per_action("read", "X", 8), 8.0)
        # write inherits component bpa via default expression
        self.assertEqual(mem._get_values_per_action("write", "X", 8), 4.0)

    def test_component_bpa_used_when_vpa_and_action_bpa_empty(self):
        mem = _spec(comp_bpa=32)
        self.assertEqual(mem._get_values_per_action("read", "X", 8), 4.0)

    def test_bpv_lookup_component_then_default(self):
        mem = _spec(comp_bpa=32, comp_bpv={"X": 16})
        # X uses component bpv (16): 32/16 = 2
        self.assertEqual(mem._get_values_per_action("read", "X", 8), 2.0)
        # Y falls back to passed-in default (8): 32/8 = 4
        self.assertEqual(mem._get_values_per_action("read", "Y", 8), 4.0)

    def test_bpv_default_arg_used_when_no_component_bpv(self):
        mem = _spec(comp_bpa=16)
        # Different defaults yield different results, proving the arg is used
        self.assertEqual(mem._get_values_per_action("read", "X", 8), 2.0)
        self.assertEqual(mem._get_values_per_action("read", "X", 4), 4.0)

    def test_default_bpa_is_one(self):
        mem = _spec()
        # No bpa or vpa set anywhere: action bpa defaults to 1, bpv = 8
        self.assertEqual(mem._get_values_per_action("read", "X", 8), 0.125)


class TestSetExpressionKeys(unittest.TestCase):
    """Set-expression keys (All, Inputs, Outputs, complements) must be expanded
    to per-tensor entries by _eval_tensor2number on BOTH action.values_per_action
    and component.values_per_action (and bits_per_value)."""

    def test_action_vpa_all_expands_to_each_tensor(self):
        mem = _spec(read_vpa={"All": 2})
        self.assertEqual(mem._get_values_per_action("read", "X", 8), 2)
        self.assertEqual(mem._get_values_per_action("read", "Y", 8), 2)

    def test_action_vpa_inputs_expands_only_to_inputs(self):
        mem = _spec(read_vpa={"Inputs": 3}, read_bpa=64)
        # Input X uses the action vpa
        self.assertEqual(mem._get_values_per_action("read", "X", 8), 3)
        # Output Y falls through to action bpa / bpv = 64/8 = 8
        self.assertEqual(mem._get_values_per_action("read", "Y", 8), 8.0)

    def test_action_vpa_inputs_and_outputs_non_overlapping_both_expand(self):
        mem = _spec(read_vpa={"Inputs": 4, "Outputs": 9})
        self.assertEqual(mem._get_values_per_action("read", "X", 8), 4)
        self.assertEqual(mem._get_values_per_action("read", "Y", 8), 9)

    def test_action_vpa_complement_key(self):
        mem = _spec(read_vpa={"Inputs": 5, "All - Inputs": 7})
        self.assertEqual(mem._get_values_per_action("read", "X", 8), 5)
        self.assertEqual(mem._get_values_per_action("read", "Y", 8), 7)

    def test_component_vpa_all_expands(self):
        # Regression: component vpa already gets eval'd via _eval_tensor2number
        mem = _spec(comp_vpa={"All": 6})
        self.assertEqual(mem._get_values_per_action("read", "X", 8), 6)
        self.assertEqual(mem._get_values_per_action("read", "Y", 8), 6)

    def test_component_vpa_inputs_outputs_expand(self):
        mem = _spec(comp_vpa={"Inputs": 2, "Outputs": 11})
        self.assertEqual(mem._get_values_per_action("read", "X", 8), 2)
        self.assertEqual(mem._get_values_per_action("read", "Y", 8), 11)

    def test_action_set_expr_beats_component_set_expr(self):
        # Precedence: action set-expr wins over component set-expr
        mem = _spec(comp_vpa={"All": 100}, read_vpa={"All": 7})
        self.assertEqual(mem._get_values_per_action("read", "X", 8), 7)
        self.assertEqual(mem._get_values_per_action("read", "Y", 8), 7)

    def test_action_inputs_beats_component_all(self):
        # Action Inputs:N wins for X; component All:M still applies for Y
        mem = _spec(comp_vpa={"All": 50}, read_vpa={"Inputs": 3})
        self.assertEqual(mem._get_values_per_action("read", "X", 8), 3)
        self.assertEqual(mem._get_values_per_action("read", "Y", 8), 50)

    def test_component_bpv_all_expands(self):
        # bits_per_value already evaluated via _eval_tensor2number on component
        mem = _spec(comp_bpa=32, comp_bpv={"All": 16})
        # bpa/bpv = 32/16 = 2 for any tensor
        self.assertEqual(mem._get_values_per_action("read", "X", 8), 2.0)
        self.assertEqual(mem._get_values_per_action("read", "Y", 8), 2.0)

    def test_component_bpv_inputs_outputs_expand(self):
        mem = _spec(comp_bpa=32, comp_bpv={"Inputs": 16, "Outputs": 4})
        self.assertEqual(mem._get_values_per_action("read", "X", 8), 2.0)
        self.assertEqual(mem._get_values_per_action("read", "Y", 8), 8.0)


if __name__ == "__main__":
    unittest.main()
