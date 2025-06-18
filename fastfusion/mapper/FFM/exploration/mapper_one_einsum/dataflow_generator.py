from collections.abc import Collection, Generator, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

import fastfusion.frontend.architecture as architecture
from fastfusion.frontend.mapping import MappingNode, Storage
from fastfusion.frontend.specification import Specification
from fastfusion.frontend.workload.workload import TensorName, SymbolTable

from .bypass_keep_generator import make_storage_choices_all_levels


def get_storage_choices(
    nodes: list[architecture.Memory],
    symbol_table: SymbolTable,
    spec: Specification,
) -> Generator[tuple[list[Storage], Any], None, None]:
    while not isinstance(nodes[0], architecture.Memory):
        nodes = nodes[1:]
    first_storage = nodes[0]
    
    def collect_tensors(storage_nodes: list[Storage]):
        return {tensor for storage in storage_nodes for tensor in storage.tensors}

    for choice, symbol_table in make_storage_choices_all_levels(nodes, symbol_table):
        all_storage_nodes = []
        for v in choice.values():
            all_storage_nodes.extend(v)

        base_mapping = []
        for node in list(all_storage_nodes[::-1]):
            if node.memory == first_storage.name:
                all_storage_nodes.remove(node)
                base_mapping.append(node)
                
        tensors_in_mapping = collect_tensors(all_storage_nodes)
        required_order = get_dataflow_constraint(nodes,
                                                 symbol_table,
                                                 tensors_in_mapping)

        for mapping in recursive_order_storage_choices(base_mapping,
                                                       nodes,
                                                       all_storage_nodes,
                                                       required_order,
                                                       spec):
            yield mapping, symbol_table


def get_dataflow_constraint(nodes, symbol_table, tensors_in_mapping):
    required_order: dict[str, list[Order]] = {}
    for node in nodes:
        constraint = node.constraints.dataflow._parse(symbol_table, f"{node.name}.constraints.dataflow")
        if constraint.storage_orders:
            for order_constraint in constraint.storage_orders:
                order = Order()
                for together_tensors in order_constraint:
                    in_mapping_together_tensors = [
                        tensor for tensor in together_tensors
                        if tensor in tensors_in_mapping
                    ]
                    if len(in_mapping_together_tensors) == 1:
                        only_tensor = in_mapping_together_tensors[0]
                        order.add_tensor(only_tensor)
                    else:
                        order.add_together_tensors(in_mapping_together_tensors)
                required_order.setdefault(node.name, []).append(order)
    return required_order


def recursive_order_storage_choices(
    mapping: Sequence[MappingNode],
    nodes: list[architecture.Memory],
    remaining_choices: list,
    required_order: list[list[Storage]],
    spec: Specification,
) -> Generator[list[MappingNode], None, None]:
    mapping = list(mapping)
    if not remaining_choices:
        yield mapping
        return

    for choice in sorted(remaining_choices, key=lambda x: x.compact_string()):
        mapping.append(choice)
        new_remaining = [c for c in remaining_choices if c != choice]
        if valid_storage_order(mapping, [n.name for n in nodes], required_order, spec):
            yield from recursive_order_storage_choices(mapping,
                                                       nodes,
                                                       new_remaining,
                                                       required_order,
                                                       spec)
        mapping.pop()


def valid_storage_order(
    mapping: Sequence[Storage],
    node_names: list[str],
    required_orders: dict[str, list["Order"]],
    spec: Specification,
):
    for node in mapping:
        node._even_with_below = False

    memory_to_satisfied_constraints: dict[str, set] = {}
    for i in range(len(mapping)):
        for j in range(i, len(mapping)):

            s1, s2 = mapping[i].memory, mapping[j].memory
            s1_idx, s2_idx = node_names.index(s1), node_names.index(s2)

            assert len(mapping[i].tensors) == 1
            assert len(mapping[j].tensors) == 1

            if spec.mapper_ffm.force_memory_hierarchy_order:
                if i < j and s2_idx < s1_idx:
                    return False

            if s1 == s2 and s1 in required_orders and i != j:
                if s1 not in memory_to_satisfied_constraints:
                    memory_to_satisfied_constraints[s1] = \
                        {i for i in range(len(required_orders[s1]))}

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
            # bypassed the outer storage if possible.
            either_backing = mapping[i]._backing & mapping[j]._backing
            if i == j or i == j - 1:
                if s1_idx < s2_idx and not ((set(mapping[i]._must_keep_tensors) & set(mapping[j].tensors)) or either_backing):
                    return False
                if s2_idx < s1_idx and not ((set(mapping[j]._must_keep_tensors) & set(mapping[i].tensors)) or either_backing):
                    return False

    for i in range(len(mapping)):
        for j in range(i, len(mapping)):
            s1, s2 = mapping[i].memory, mapping[j].memory
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
                        mapping[i]._even_with_below = True
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
        return f'Order({self.order})'

    def add_tensor(self, tensor):
        self.order.append(Alone(tensor))

    def add_together_tensors(self, together_tensors):
        self.order.append(Together(together_tensors))

    def index(self, tensor):
        for i, order_term in enumerate(self.order):
            if (
                (isinstance(order_term, Alone) and order_term.tensor == tensor)
                or
                (isinstance(order_term, Together) and tensor in order_term.tensors)
            ):
                return i
        return None
