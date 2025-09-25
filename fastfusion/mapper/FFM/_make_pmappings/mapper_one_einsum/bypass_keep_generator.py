import copy
from collections.abc import Generator
from itertools import chain, combinations

import fastfusion.frontend.arch as arch
from fastfusion.frontend.arch import ProcessingStage
from fastfusion.frontend.mapping import Storage, TensorHolder
from fastfusion.frontend.workload.workload import TensorName, SymbolTable

from fastfusion.util.setexpressions import InvertibleSet


def make_tensor_choices_one_level(
    node: arch.Leaf,
    symbol_table: dict[str, InvertibleSet],
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
    tensor_constraints = node.constraints.tensors._parse_keep_bypass(
        symbol_table, f"{node.name}.constraints.tensors"
    )
    must_keep = tensors.to_my_space(tensor_constraints.keep)
    must_bypass = tensors.to_my_space(tensor_constraints.bypass)

    if must_keep - tensors:
        raise KeyError(
            f"Keep constraint for {node.name} includes tensors that are "
            f"not in the workload: {must_keep - new_symbol_table['All']}"
        )
    if must_bypass - tensors:
        raise KeyError(
            f"Bypass constraint for {node.name} includes tensors that are "
            f"not in the workload: {must_bypass - tensors.full_space}"
        )
    if must_keep & must_bypass:
        raise KeyError(
            f"Keep and bypass constraints for {node.name} intersect: "
            f"{must_keep & must_bypass}"
        )

    may_keep = tensors - must_bypass - must_keep

    # No reuse in copy operations, so no need to keep tensors in more places
    if is_copy_op:
        may_keep -= tensors.to_my_space(seen_tensors)

    for subset in powerset(sorted(may_keep, key=str)):
        # Make keep choice & update symbol table
        subset = tensors.to_my_space(set(subset))
        keep_choice = tensors.to_my_space(subset | must_keep)
        keep_choice.tensors = (
            lambda: keep_choice
        )  # So users can do MainMemory().tensors(). Optional.
        new_symbol_table[node.name] = keep_choice
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
            elif t in must_keep:
                nodes[-1]._must_keep_tensors = [t]

        yield nodes, new_symbol_table, new_seen_tensors


def make_tensor_choices_all_levels(
    nodes: list[TensorHolder],
    symbol_table: dict[str, InvertibleSet],
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
        nodes[0], symbol_table, seen_tensors, is_copy_op
    ):
        for subchoices, symbol_table in make_tensor_choices_all_levels(
            nodes[1:], symbol_table, new_seen_tensors, is_copy_op
        ):
            yield {**subchoices, nodes[0].name: choice}, symbol_table


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
