import copy
from collections.abc import Generator
from itertools import chain, combinations

import fastfusion.frontend.architecture as architecture
from fastfusion.frontend.mapping import Storage
from fastfusion.frontend.workload.workload import TensorName, SymbolTable

from fastfusion.util.setexpressions import InvertibleSet


def make_storage_choices_one_level(
    node: architecture.Leaf,
    symbol_table: dict[str, InvertibleSet],
) -> Generator[tuple[list[Storage], SymbolTable], None, None]:
    """
    Generate combinations of storage nodes based on keep and bypass
    constraints.
    
    Each generated list contains storage nodes for single tensors.
    """
    assert "All" in symbol_table
    tensors = symbol_table["All"]

    if not isinstance(node, architecture.Memory):
        yield [], symbol_table
        return

    new_symbol_table = copy.copy(symbol_table)
    storage_constraints = node.constraints.storage._parse_keep_bypass(symbol_table)
    must_keep = tensors.to_my_space(storage_constraints.keep)
    must_bypass = tensors.to_my_space(storage_constraints.bypass)

    if must_keep - tensors:
        raise KeyError(f"Keep constraint for {node.name} includes tensors that are "
                       f"not in the einsum: {must_keep - new_symbol_table['All']}")
    if must_bypass - tensors:
        raise KeyError(f"Bypass constraint for {node.name} includes tensors that are "
                       f"not in the einsum: {must_bypass - tensors.full_space}")
    if must_keep & must_bypass:
        raise KeyError(f"Keep and bypass constraints for {node.name} intersect: "
                       f"{must_keep & must_bypass}")
    
    may_keep = tensors - must_bypass - must_keep

    for subset in powerset(sorted(may_keep, key=str)):
        # Make keep choice & update symbol table
        subset = tensors.to_my_space(set(subset))
        keep_choice = tensors.to_my_space(subset | must_keep)
        keep_choice.tensors = lambda: keep_choice # So users can do MainMemory().tensors(). Optional.
        new_symbol_table[node.name] = keep_choice

        # Make sure they're all tensors
        assert all(isinstance(k, TensorName) for k in keep_choice)
        keep_choice = keep_choice.to_my_space({copy.copy(t) for t in keep_choice})
        storage_nodes = []

        for t in sorted(keep_choice, key=str):
            storage_nodes.append(Storage(tensors=[t], memory=node.name, memory_object=node))
            if t in must_keep:
                storage_nodes[-1]._must_keep_tensors = [t]
        yield storage_nodes, new_symbol_table


def make_storage_choices_all_levels(
    nodes: list[Storage], 
    symbol_table: dict[str, InvertibleSet],
) -> Generator[tuple[dict[str, list[Storage]], SymbolTable], None, None]:
    """
    Generate combinations of storage nodes based on keep and bypass
    constraints.
    
    Each generated dict maps memory name to a list of storage nodes for
    single tensors.
    """
    if len(nodes) == 0:
        yield dict(), symbol_table
        return
    for choice, symbol_table in make_storage_choices_one_level(nodes[0], symbol_table):
        for subchoices, symbol_table in make_storage_choices_all_levels(nodes[1:], symbol_table):
            yield {**subchoices, nodes[0].name: choice}, symbol_table


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
