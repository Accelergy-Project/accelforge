from collections.abc import Collection, Generator, Sequence
from dataclasses import dataclass
from itertools import product
import logging
from typing import Any

import accelforge.frontend.arch as arch
from accelforge.frontend.mapping import MappingNode, Storage, Toll, TensorHolder
from accelforge.frontend.spec import Spec
from accelforge.frontend.workload import TensorName, SymbolTable
from accelforge.util._eval_expressions import MATH_FUNCS

from accelforge.util._frozenset import oset
from accelforge.mapper.FFM._make_pmappings.make_pmapping_templates.make_storages import (
    make_storage_choices_all_levels,
)
from accelforge.frontend.workload import EinsumName


def get_tensor_choices(
    einsum_name: EinsumName,
    nodes: list[arch.Memory],
    symbol_table: SymbolTable,
    spec: Spec,
    first_memory: arch.Memory,
    fusable_tensors: set[TensorName],
    fanouts: dict[str, int],
    prioritize_reuse_of_unfused_tensors: bool,
) -> Generator[tuple[list[TensorHolder], SymbolTable, arch.Compute], None, None]:
    nodes, compute = nodes[:-1], nodes[-1]
    # while True:
    #     if not nodes:
    #         return
    #     if not isinstance(nodes[0], arch.Memory):
    #         nodes = nodes[1:]
    #         continue
    #     assert isinstance(nodes[0].enabled, bool)
    #     if not nodes[0].enabled:
    #         nodes = nodes[1:]
    #         continue
    #     break
    nodes = list(filter(lambda n: isinstance(n, arch.TensorHolder), nodes))

    einsum = spec.workload.einsums[einsum_name]
    persistent_tensors = oset(
        t.name
        for t in spec.workload.einsums[einsum_name].tensor_accesses
        if t.persistent
    )

    for choice, symbol_table in make_storage_choices_all_levels(
        nodes=nodes,
        symbol_table=symbol_table,
        is_copy_op=einsum.is_copy_operation,
        persistent_tensors=persistent_tensors,
        seen_tensors=oset(),
        einsum_name=einsum_name,
        prioritize_reuse_of_unfused_tensors=prioritize_reuse_of_unfused_tensors,
    ):
        x = [y for z in choice.values() for y in z]
        logging.info(
            f"\t\tUnordered storage choice: {", ".join(n.compact_str() for n in x)}"
        )
        all_tensor_holders = [v2 for v in choice.values() for v2 in v]
        storage_holders = [th for th in all_tensor_holders if isinstance(th, Storage)]
        toll_holders = [th for th in all_tensor_holders if isinstance(th, Toll)]

        # Start out the mapping with the outermost memory name
        base_mapping = []

        # make_storage_chocies_all_levels will parse the "tensors" field for each node,
        # and we'll want those parsed values for later steps. This code grabs it from
        # the nodes in the mapping. We only check the first storage node for each memory
        # level since they're all the same.
        parsed_nodes_in_mapping = []
        for k, v in choice.items():
            for s in v[:1]:
                parsed_nodes_in_mapping.append(s.component_object)

        # Get the dataflow constraints for the mapping
        required_order = get_tensor_order_constraint(
            parsed_nodes_in_mapping, symbol_table, einsum.tensor_names
        )

        symbol_table["arch_attributes"] = {}
        cur_compute = compute._eval_expressions(
            symbol_table,
            location=f"arch.{compute.name}",
            musteval_tryeval_to=True,
            must_copy=False,
        )[0]
        assert isinstance(cur_compute.enabled, bool)
        if not cur_compute.enabled:
            continue
        # TODO: check if compute supports the Einsum operations

        for mapping in recursive_order_tensor_choices(
            einsum_name,
            base_mapping,
            nodes,
            storage_holders,
            required_order,
            spec,
            first_memory,
            fusable_tensors,
            fanouts,
        ):
            mapping = insert_tolls(mapping, toll_holders, nodes, fanouts)
            yield mapping, symbol_table, cur_compute


def get_tensor_order_constraint(nodes, symbol_table, tensors):
    required_order: dict[str, list[Order]] = {}
    seen_tensors = oset()
    for node in nodes:
        if isinstance(node, arch.Container):
            continue
        node_tensors: arch.Tensors = node.tensors._eval_expressions(
            symbol_table=symbol_table,
            musteval_tryeval_to=True,
            must_copy=False,
            location=f"arch.{node.name}.tensors",
        )[0]
        tensor_order_options = list(node_tensors.tensor_order_options)
        for order_constraint in tensor_order_options:
            order = Order()
            for together_tensors in order_constraint:
                in_mapping_together_tensors = [
                    tensor for tensor in together_tensors if tensor in tensors
                ]
                if len(in_mapping_together_tensors) == 1:
                    only_tensor = in_mapping_together_tensors[0]
                    order.add_tensor(only_tensor)
                elif len(in_mapping_together_tensors) > 1:
                    order.add_together_tensors(in_mapping_together_tensors)
            if order.order:
                required_order.setdefault(node.name, []).append(order)
        seen_tensors.update(node_tensors.keep)
    return required_order


def insert_tolls(
    mapping: list[TensorHolder],
    toll_holders: list[Toll],
    arch_nodes: list[arch.TensorHolder],
    fanouts: dict[str, int],
) -> list[MappingNode]:
    if not toll_holders:
        return mapping

    mapping = list(mapping)

    arch_order = {n.name: i for i, n in enumerate(arch_nodes)}

    # Sort tolls by arch order, then by tensor name
    toll_holders = sorted(
        toll_holders,
        key=lambda t: (arch_order[t.component], sorted(t.tensors)),
    )

    for toll in toll_holders:
        toll_arch_idx = arch_order[toll.component]
        toll_fanout = fanouts.get(toll.component, 0)

        # Must go below the storage node above them
        toll_tensors = oset(toll.tensors)
        min_pos = 0
        for i, m in enumerate(mapping):
            if (
                oset(m.tensors) & toll_tensors
                and arch_order[m.component] < toll_arch_idx
            ):
                min_pos = max(min_pos, i + 1)

        # Rule 2: Must go above the storage node below them
        max_pos = len(mapping)
        for i, m in enumerate(mapping):
            if (
                oset(m.tensors) & toll_tensors
                and arch_order[m.component] > toll_arch_idx
            ):
                max_pos = min(max_pos, i)
                break

        assert min_pos <= max_pos

        # If possible, go below fanout above them
        for i in range(min_pos, min(max_pos, len(mapping))):
            if fanouts.get(mapping[i].component, 0) < toll_fanout:
                min_pos = i + 1

        # Go below any already-inserted entry from an arch level above this toll, and
        # maintain alphabetical order for tolls at the same arch level
        for i in range(min_pos, min(max_pos, len(mapping))):
            m = mapping[i]
            m_arch_idx = arch_order.get(m.component, -1)
            if m_arch_idx < toll_arch_idx:
                min_pos = i + 1
            elif m_arch_idx == toll_arch_idx and sorted(m.tensors) < sorted(
                toll.tensors
            ):
                min_pos = i + 1

        # Rule 4: Go as high as possible
        mapping.insert(min_pos, toll)

    return mapping


def recursive_order_tensor_choices(
    einsum_name: EinsumName,
    mapping: Sequence[MappingNode],
    nodes: list[arch.Memory],
    remaining_choices: list,
    required_order: list[list[TensorHolder]],
    spec: Spec,
    first_memory: arch.Memory,
    fusable_tensors: set[TensorName],
    fanouts: dict[str, int],
) -> Generator[list[MappingNode], None, None]:
    einsum = spec.workload.einsums[einsum_name]

    def check_has_tensors(mapping: list[MappingNode]):
        tensor_holders = [node for node in mapping if isinstance(node, TensorHolder)]
        tensors_in_mapping = oset(
            tensor
            for tensor_holder in tensor_holders
            for tensor in tensor_holder.tensors
        )
        if tensors_in_mapping != einsum.tensor_names:
            raise ValueError(
                f"Einsum {einsum_name} has a pmapping template that is missing tensors. Ensure "
                f"that there is a storage node storing each tensor in the Einsum. Missing "
                f"tensors: {einsum.tensor_names - tensors_in_mapping}. Pmapping template:\n\t"
                + "\n\t".join(m.compact_str() for m in mapping)
            )

    mapping = list(mapping)
    if not remaining_choices:
        check_has_tensors(mapping)
        yield mapping
        return

    # If it's a copy op and we have the backing storage for every tensor, return
    # immediately
    if einsum.is_copy_operation:
        tensor_holders = [node for node in mapping if isinstance(node, TensorHolder)]
        if oset().union(*[t._backing for t in tensor_holders]) == einsum.tensor_names:
            check_has_tensors(mapping)
            yield mapping
            return

    for choice in sorted(remaining_choices, key=lambda x: x.compact_str()):
        mapping.append(choice)
        new_remaining = [c for c in remaining_choices if c != choice]
        valid, reason = valid_tensor_holder_order(
            mapping,
            [n.name for n in nodes],
            required_order,
            spec,
            first_memory,
            fusable_tensors,
            fanouts,
            new_remaining,
        )
        if valid:
            yield from recursive_order_tensor_choices(
                einsum_name,
                mapping,
                nodes,
                new_remaining,
                required_order,
                spec,
                first_memory,
                fusable_tensors,
                fanouts,
            )
        else:
            logging.info(
                "\t\t"
                + " " * len(mapping)
                + f"Invalid tensor holder order: {", ".join(n.compact_str() for n in mapping)}: {reason}"
            )
        mapping.pop()


def valid_tensor_holder_order(
    mapping: Sequence[TensorHolder],
    node_names: list[str],
    required_orders: dict[str, list["Order"]],
    spec: Spec,
    first_memory: arch.Memory,
    fusable_tensors: set[TensorName],
    fanouts: dict[str, int],
    remaining_choices: list[TensorHolder],
):
    memory_to_satisfied_constraints: dict[str, set] = {}

    for i, m0 in enumerate(mapping):
        # There are some checks for back-to-back nodes, but ones in remaining_choices
        # aren't yet ordered so we can't decide if anything is directly back-to-back.
        # However we can check longer-distance relations between nodes in our mapping
        # and remaining choices because we know that they will be placed somewhere
        # eventually. So we'll add a None between to act as a separator, then we'll
        # never get straight back-to-back nodes between the mapping and remaining
        # choices.
        following = mapping[i + 1 :] + [None] + remaining_choices
        for j, m1 in enumerate(following):
            if following[j] is None:
                continue

            j += i + 1

            s1, s2 = m0.component, m1.component
            s1_idx, s2_idx = node_names.index(s1), node_names.index(s2)
            s1_persistent, s2_persistent = m0.persistent, m1.persistent
            either_persistent = s1_persistent or s2_persistent

            assert len(m0.tensors) == 1
            assert len(m1.tensors) == 1

            # If they're persistent they're forced to be at the top.
            force_order = (
                spec.mapper.force_memory_hierarchy_order and not either_persistent
            )
            force_order &= m0.component_object.tensors.force_memory_hierarchy_order
            force_order &= m1.component_object.tensors.force_memory_hierarchy_order

            # Ctrl-F for CONTIGUOUS_ITERATION_SPACE_DISCUSSION: The following line does
            # not let backing storage be above in the mapping anything that is below it
            # in the memory hierarchy. THIS IS NOT FUNDAMENTAL. If we remove this
            # constraint, then the fused loops may be different across different backing
            # storages, so we would need to update make_pmappings_from_templates.py to
            # make compatibility from the mapping for each tensor.
            force_order |= bool(m0._backing & fusable_tensors)

            if force_order and i < j and s2_idx < s1_idx:
                return (
                    False,
                    f"Memory {s1} is below memory {s2}, violating memory hierarchy order.",
                )

            s1_outermost = s1_persistent
            s2_outermost = s2_persistent
            if not spec.mapper._can_lower_outermost_memory:
                s1_outermost |= s1 == first_memory.name
                s2_outermost |= s2 == first_memory.name

            # Persistent tensors must be at the top of the hierarchy
            if s2_outermost and not s1_outermost and i < j:
                return (
                    False,
                    f"Outermost {m0.compact_str()}, persistent {s1_persistent} is below non-outermost {m1.compact_str()}, persistent {s2_persistent}.",
                )

            # If they're both the first memory and we can't lower the outermost memory,
            # then relative order doesn't matter because we can't have any loops between
            # them.
            if (
                s1 == first_memory.name
                and s2 == first_memory.name
                and not spec.mapper._can_lower_outermost_memory
            ):
                if sorted(m0.tensors) < sorted(m1.tensors):
                    return (
                        False,
                        f"Force alphabetical order for storage in outermost memory. {m0.compact_str()} is before {m1.compact_str()}.",
                    )
                # Ignore the following constraints
                continue

            if s1 == s2 and s1 in required_orders and i != j:
                if s1 not in memory_to_satisfied_constraints:
                    memory_to_satisfied_constraints[s1] = oset(
                        i for i in range(len(required_orders[s1]))
                    )

                good = True
                for order_idx, order_choice in enumerate(required_orders[s1]):
                    if order_idx not in memory_to_satisfied_constraints[s1]:
                        continue

                    good = True
                    for t1, t2 in product(m0.tensors, m1.tensors):
                        idx_of_i_in_order = order_choice.index(t1)
                        idx_of_j_in_order = order_choice.index(t2)

                        if idx_of_i_in_order is None or idx_of_j_in_order is None:
                            continue

                        if idx_of_i_in_order > idx_of_j_in_order:
                            good = False
                            reason = f"Tensor {t1} is before tensor {t2} in the order {order_choice}"
                            break
                    if not good:
                        memory_to_satisfied_constraints[s1].remove(order_idx)

                if len(memory_to_satisfied_constraints[s1]) == 0:
                    return False, reason

            if not (oset(m0.tensors) & oset(m1.tensors)):
                continue

            if i < j and s2_idx < s1_idx:
                return False, f"{m0.compact_str()} is below {m1.compact_str()}"

            # If a tensor is stored in two levels back-to-back, then we should have
            # bypassed the outer TensorHolder if possible.
            either_backing = m0._backing & m1._backing
            same_fanout = fanouts[s1] == fanouts[s2]
            if (
                "redundant_dataplacements"
                not in spec.mapper._count_option_for_mapsapce_size_evaluation
            ):
                if same_fanout and (i == j or i == j - 1):
                    if s1_idx < s2_idx and not (
                        (oset(m0._must_keep_tensors) & oset(m1.tensors))
                        or either_backing
                    ):
                        shared = oset(m0._must_keep_tensors) & oset(m1.tensors)
                        return (
                            False,
                            f"{shared} stored in back-to-back storage nodes, and could have bypassed the outer one.",
                        )
                    if s2_idx < s1_idx and not (
                        (oset(m1._must_keep_tensors) & oset(m0.tensors))
                        or either_backing
                    ):
                        shared = oset(m1._must_keep_tensors) & oset(m0.tensors)
                        return (
                            False,
                            f"{shared} is stored in back-to-back storage nodes, and could have bypassed the outer one.",
                        )

    for i, m0 in enumerate(mapping):
        for j, m1 in enumerate(mapping[i:]):
            s1, s2 = m0.component, m1.component
            if s1 != s2 or s1 not in memory_to_satisfied_constraints or i == j:
                continue

            satisfied_orders = memory_to_satisfied_constraints[s1]
            assert len(satisfied_orders) > 0

            for order_idx in satisfied_orders:
                order = required_orders[s1][order_idx]
                for tensor_i in m0.tensors:
                    for tensor_j in m1.tensors:
                        if order.index(tensor_i) != order.index(tensor_j):
                            continue
                break

    return True, ""


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
