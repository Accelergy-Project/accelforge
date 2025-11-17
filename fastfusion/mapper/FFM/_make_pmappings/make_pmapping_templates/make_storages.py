import copy
from collections.abc import Generator
from itertools import chain, combinations

import fastfusion.frontend.arch as arch
from fastfusion.frontend.mapping import Storage, TensorHolder, ProcessingStage
from fastfusion.frontend.workload.workload import TensorName, SymbolTable

from fastfusion.util.setexpressions import InvertibleSet


def make_tensor_choices_one_level(
    node: arch.Leaf,
    symbol_table: dict[str, InvertibleSet],
    persistent_tensors: set[TensorName],
    seen_tensors: set[TensorName] = (),
    is_copy_op: bool = False,
) -> Generator[tuple[list[TensorHolder], SymbolTable, set[TensorName]], None, None]:
    """
    Generate combinations of TensorHolder nodes based on keep and bypass
    constraints.

    Each generated list contains TensorHolder nodes for single tensors.
    """
    assert "All" in symbol_table
    tensors = symbol_table["All"]

    if not isinstance(node, arch.TensorHolder):
        yield [], symbol_table, set(seen_tensors)
        return

    if isinstance(node, arch.Memory):
        target_type = Storage
    elif isinstance(node, arch.ProcessingStage):
        target_type = ProcessingStage
    else:
        raise ValueError(f"Unexpected tensor holder type: {type(node)}")

    new_symbol_table = copy.copy(symbol_table)
    tensor_constraints = node.constraints.tensors._parse_keep(
        symbol_table, f"{node.name}.constraints.tensors"
    )
    must_keep = tensors.to_my_space(tensor_constraints.keep)
    may_keep = tensors.to_my_space(tensor_constraints.may_keep)
    may_keep -= must_keep

    if must_keep - tensors:
        raise KeyError(
            f"Keep constraint for {node.name} includes tensors that are "
            f"not in the workload: {must_keep - new_symbol_table['All']}"
        )
    if may_keep - tensors:
        raise KeyError(
            f"Bypass constraint for {node.name} includes tensors that are "
            f"not in the workload: {may_keep - tensors.full_space}"
        )

    # No reuse in copy operations, so no need to keep tensors in more places
    if is_copy_op:
        may_keep -= tensors.to_my_space(seen_tensors)

    for subset in powerset(sorted(may_keep, key=str)):
        # Make keep choice & update symbol table
        subset = tensors.to_my_space(set(subset))
        keep_choice = tensors.to_my_space(subset | must_keep)
        # Below line is so users can do MainMemory().tensors() or MainMemory.tensors
        keep_choice.tensors = keep_choice
        new_symbol_table[node.name] = keep_choice
        new_symbol_table["Above"] |= keep_choice
        new_seen_tensors = seen_tensors | set(keep_choice)

        # Make sure they're all tensors
        assert all(isinstance(k, TensorName) for k in keep_choice)
        keep_choice = keep_choice.to_my_space({copy.copy(t) for t in keep_choice})
        nodes = []

        for t in sorted(keep_choice, key=str):
            nodes.append(
                target_type(tensors=[t], component=node.name, component_object=node)
            )
            if t not in seen_tensors:
                nodes[-1]._backing.add(t)
                nodes[-1]._must_keep_tensors = [t]
                nodes[-1].persistent = t in persistent_tensors
            elif t in must_keep:
                nodes[-1]._must_keep_tensors = [t]

        yield nodes, new_symbol_table, new_seen_tensors


def make_storage_choices_all_levels(
    nodes: list[TensorHolder],
    symbol_table: dict[str, InvertibleSet],
    persistent_tensors: set[TensorName],
    seen_tensors: set[TensorName] = None,
    is_copy_op: bool = False,
) -> Generator[tuple[dict[str, list[TensorHolder]], SymbolTable], None, None]:
    """
    Generate combinations of TensorHolder nodes based on keep and bypass
    constraints.

    Each generated dict maps memory name to a list of TensorHolder nodes for
    single tensors.
    """
    seen_tensors = set() if seen_tensors is None else seen_tensors
    if len(nodes) == 0:
        yield dict(), symbol_table
        return
    for choice, symbol_table, new_seen_tensors in make_tensor_choices_one_level(
        node=nodes[0],
        symbol_table=symbol_table,
        persistent_tensors=persistent_tensors,
        seen_tensors=seen_tensors,
        is_copy_op=is_copy_op,
    ):
        for subchoices, symbol_table in make_storage_choices_all_levels(
            nodes=nodes[1:],
            symbol_table=symbol_table,
            persistent_tensors=persistent_tensors,
            seen_tensors=new_seen_tensors,
            is_copy_op=is_copy_op,
        ):
            yield {**subchoices, nodes[0].name: choice}, symbol_table


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
