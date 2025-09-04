from collections.abc import Collection, Generator, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

import fastfusion.frontend.arch as arch
from fastfusion.frontend.mapping import MappingNode, TensorHolder
from fastfusion.frontend.specification import Specification
from fastfusion.frontend.workload.workload import TensorName, SymbolTable
from fastfusion.util.parse_expressions import MATH_FUNCS

from .bypass_keep_generator import make_tensor_choices_all_levels
from fastfusion.frontend.workload.workload import EinsumName

def eval_enabled(component: arch.Component, symbol_table: SymbolTable) -> bool:
    enabled = component.constraints.misc.enabled
    if isinstance(enabled, str):
        return eval(enabled, {"__builtins__": MATH_FUNCS}, symbol_table)
    if isinstance(enabled, bool):
        return enabled
    raise ValueError(
        f"enabled for {component.name} must be a bool or evaluate to a bool, "
        f"got {type(enabled)}: {enabled}"
    )

def get_tensor_choices(
    einsum_name: EinsumName,
    nodes: list[arch.Memory],
    symbol_table: SymbolTable,
    spec: Specification,
) -> Generator[tuple[list[TensorHolder], Any], None, None]:
    nodes, compute = nodes[:-1], nodes[-1]
    while True:
        if not nodes:
            return
        if not isinstance(nodes[0], arch.Memory):
            nodes = nodes[1:]
            continue
        if not eval_enabled(nodes[0], symbol_table):
            nodes = nodes[1:]
            continue
        break

    first_tensor_holder = nodes[0]

    tensors = spec.workload.einsums[einsum_name].tensor_names
    is_copy_op = spec.workload.einsums[einsum_name].is_copy_operation

    for choice, symbol_table in make_tensor_choices_all_levels(nodes, symbol_table, is_copy_op=is_copy_op):
        all_tensor_holders = []
        for v in choice.values():
            all_tensor_holders.extend(v)

        base_mapping = []
        for node in list(all_tensor_holders[::-1]):
            if node.component == first_tensor_holder.name:
                all_tensor_holders.remove(node)
                base_mapping.append(node)
                
        required_order = get_dataflow_constraint(
            nodes, symbol_table, tensors
        )

        if not eval_enabled(compute, symbol_table):
            continue

        for mapping in recursive_order_tensor_choices(
            einsum_name, tensors, base_mapping, nodes, all_tensor_holders, required_order, spec, is_copy_op
        ):
            yield mapping, symbol_table


def get_dataflow_constraint(nodes, symbol_table, tensors):
    required_order: dict[str, list[Order]] = {}
    for node in nodes:
        constraint = node.constraints.dataflow._parse(
            symbol_table, f"{node.name}.constraints.dataflow"
        )
        if constraint.tensor_order_options:
            for order_constraint in constraint.tensor_order_options:
                order = Order()
                for together_tensors in order_constraint:
                    in_mapping_together_tensors = [
                        tensor
                        for tensor in together_tensors
                        if tensor in tensors
                    ]
                    if len(in_mapping_together_tensors) == 1:
                        only_tensor = in_mapping_together_tensors[0]
                        order.add_tensor(only_tensor)
                    elif len(in_mapping_together_tensors) > 1:
                        order.add_together_tensors(in_mapping_together_tensors)
                if order.order:
                    required_order.setdefault(node.name, []).append(order)
    return required_order


def recursive_order_tensor_choices(
    einsum_name: EinsumName,
    tensors: set[TensorName],
    mapping: Sequence[MappingNode],
    nodes: list[arch.Memory],
    remaining_choices: list,
    required_order: list[list[TensorHolder]],
    spec: Specification,
    is_copy_op: bool = False,
) -> Generator[list[MappingNode], None, None]:
    
    def check_has_tensors(mapping: list[MappingNode]):
        tensor_holders = [node for node in mapping if isinstance(node, TensorHolder)]
        tensors_in_mapping = {tensor for tensor_holder in tensor_holders for tensor in tensor_holder.tensors}
        if tensors_in_mapping != tensors:
            raise ValueError(
                f"Einsum {einsum_name} has a mapping that is missing tensors. Ensure that "
                f"there is a node storing each tensor in the Einsum. Missing tensors: "
                f"{tensors - tensors_in_mapping}. Mapping:\n\t" + "\n\t".join(
                    m.compact_str() for m in mapping
                )
            )
    
    mapping = list(mapping)
    if not remaining_choices:
        check_has_tensors(mapping)
        yield mapping
        return
    
    # If it's a copy op and there's a node storing each tensor, then return immediately
    if is_copy_op:
        tensor_holders = [node for node in mapping if isinstance(node, TensorHolder)]
        seen_tensors = {tensor for tensor_holder in tensor_holders for tensor in tensor_holder.tensors}
        if seen_tensors == tensors:
            check_has_tensors(mapping)
            yield mapping
            return

    for choice in sorted(remaining_choices, key=lambda x: x.compact_str()):
        mapping.append(choice)
        new_remaining = [c for c in remaining_choices if c != choice]
        if valid_tensor_holder_order(mapping, [n.name for n in nodes], required_order, spec):
            yield from recursive_order_tensor_choices(
                einsum_name, tensors, mapping, nodes, new_remaining, required_order, spec, is_copy_op,
            )
        mapping.pop()


def valid_tensor_holder_order(
    mapping: Sequence[TensorHolder],
    node_names: list[str],
    required_orders: dict[str, list["Order"]],
    spec: Specification,
):
    memory_to_satisfied_constraints: dict[str, set] = {}
    for i in range(len(mapping)):
        for j in range(i, len(mapping)):

            s1, s2 = mapping[i].component, mapping[j].component
            s1_idx, s2_idx = node_names.index(s1), node_names.index(s2)

            assert len(mapping[i].tensors) == 1
            assert len(mapping[j].tensors) == 1

            if spec.mapper.ffm.force_memory_hierarchy_order:
                if i < j and s2_idx < s1_idx:
                    return False

            if s1 == s2 and s1 in required_orders and i != j:
                if s1 not in memory_to_satisfied_constraints:
                    memory_to_satisfied_constraints[s1] = {
                        i for i in range(len(required_orders[s1]))
                    }

                good = True
                for order_idx, order_choice in enumerate(required_orders[s1]):
                    if order_idx not in memory_to_satisfied_constraints[s1]:
                        continue

                    good = True
                    for t1, t2 in product(mapping[i].tensors, mapping[j].tensors):
                        idx_of_i_in_order = order_choice.index(t1)
                        idx_of_j_in_order = order_choice.index(t2)

                        if idx_of_i_in_order is None or idx_of_j_in_order is None:
                            continue

                        if idx_of_i_in_order > idx_of_j_in_order:
                            good = False
                            break
                    if not good:
                        memory_to_satisfied_constraints[s1].remove(order_idx)

                if len(memory_to_satisfied_constraints[s1]) == 0:
                    return False

            if not (set(mapping[i].tensors) & set(mapping[j].tensors)):
                continue

            if i < j and s2_idx < s1_idx:
                return False

            # If a tensor is stored in two levels back-to-back, then we should have
            # bypassed the outer TensorHolder if possible.
            either_backing = mapping[i]._backing & mapping[j]._backing
            if i == j or i == j - 1:
                if s1_idx < s2_idx and not (
                    (set(mapping[i]._must_keep_tensors) & set(mapping[j].tensors))
                    or either_backing
                ):
                    return False
                if s2_idx < s1_idx and not (
                    (set(mapping[j]._must_keep_tensors) & set(mapping[i].tensors))
                    or either_backing
                ):
                    return False

    for i in range(len(mapping)):
        for j in range(i, len(mapping)):
            s1, s2 = mapping[i].component, mapping[j].component
            if s1 != s2 or s1 not in memory_to_satisfied_constraints or i == j:
                continue

            satisfied_orders = memory_to_satisfied_constraints[s1]
            assert len(satisfied_orders) > 0

            for order_idx in satisfied_orders:
                order = required_orders[s1][order_idx]
                for tensor_i in mapping[i].tensors:
                    for tensor_j in mapping[j].tensors:
                        if order.index(tensor_i) != order.index(tensor_j):
                            continue
                break

    return True


@dataclass(frozen=True)
class Alone:
    tensor: Any


@dataclass(frozen=True)
class Together:
    tensors: Collection[Any]


class Order:
    """An ordering of tensors."""

    def __init__(self):
        self.order = []

    def __repr__(self):
        return f"Order({self.order})"

    def add_tensor(self, tensor):
        self.order.append(Alone(tensor))

    def add_together_tensors(self, together_tensors):
        self.order.append(Together(together_tensors))

    def index(self, tensor):
        for i, order_term in enumerate(self.order):
            if (isinstance(order_term, Alone) and order_term.tensor == tensor) or (
                isinstance(order_term, Together) and tensor in order_term.tensors
            ):
                return i
        return None
