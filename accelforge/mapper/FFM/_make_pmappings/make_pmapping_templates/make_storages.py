import copy
from collections.abc import Generator
from itertools import chain, combinations
import logging

import accelforge.frontend.arch as arch
from accelforge.frontend.mapping import Storage, TensorHolder, Toll
from accelforge.frontend.workload import TensorName, SymbolTable

from accelforge.util.exceptions import EvaluationError
from accelforge.util._setexpressions import InvertibleSet


def make_tensor_choices_one_level(
    node: arch.Leaf,
    symbol_table: dict[str, InvertibleSet],
    persistent_tensors: set[TensorName],
    seen_tensors: set[TensorName] = (),
    is_copy_op: bool = False,
    einsum_name: str = None,
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
    elif isinstance(node, arch.Toll):
        target_type = Toll
    elif isinstance(node, arch.Dummy):
        yield [], symbol_table, set(seen_tensors)
        return
    else:
        raise ValueError(f"Unexpected tensor holder type: {type(node)}")

    new_symbol_table = copy.copy(symbol_table)

    node = copy.copy(node)
    try:
        node.tensors: arch.Tensors = node.tensors._eval_expressions(
            symbol_table=symbol_table,
            musteval_tryeval_to=True,
            must_copy=False,
            location=f"arch.{node.name}.tensors",
        )[0]
    except EvaluationError as e:
        e.add_field(f"Einsum {einsum_name} arch.{node.name}.tensors")
        raise e

    must_keep = tensors.to_my_space(node.tensors.keep | node.tensors.back)
    may_keep = tensors.to_my_space(node.tensors.may_keep)
    may_keep -= must_keep

    if seen_tensors & set(node.tensors.back):
        return

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

    logging.info(
        f"\t\t{node.name} must keep {sorted(must_keep)}, may keep {sorted(may_keep)}"
    )

    # No reuse in copy operations, so no need to keep tensors in more places
    if is_copy_op:
        may_keep -= tensors.to_my_space(seen_tensors)

    for subset in powerset(sorted(may_keep, key=str)):
        # Make keep choice & update symbol table
        subset = tensors.to_my_space(set(subset))
        keep_choice = tensors.to_my_space(subset | must_keep)
        # Below line is so users can do MainMemory().tensors() or MainMemory.tensors
        new_symbol_table[node.name] = keep_choice
        new_symbol_table["Above"] |= keep_choice
        new_seen_tensors = seen_tensors | set(keep_choice)

        # Make sure they're all tensors
        assert all(isinstance(k, TensorName) for k in keep_choice)
        keep_choice = keep_choice.to_my_space({copy.copy(t) for t in keep_choice})
        nodes = []

        # Create storage nodes. Sort them to keep this deterministic. Ordering is done
        # later.
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
    einsum_name: str = None,
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

    if "Above" not in symbol_table:
        symbol_table["Above"] = symbol_table["Nothing"]

    for choice, symbol_table, new_seen_tensors in make_tensor_choices_one_level(
        node=nodes[0],
        symbol_table=symbol_table,
        persistent_tensors=persistent_tensors,
        seen_tensors=seen_tensors,
        is_copy_op=is_copy_op,
        einsum_name=einsum_name,
    ):
        for subchoices, symbol_table in make_storage_choices_all_levels(
            nodes=nodes[1:],
            symbol_table=symbol_table,
            persistent_tensors=persistent_tensors,
            seen_tensors=new_seen_tensors,
            is_copy_op=is_copy_op,
            einsum_name=einsum_name,
        ):
            yield {**subchoices, nodes[0].name: choice}, symbol_table


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
