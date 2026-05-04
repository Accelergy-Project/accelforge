from collections import defaultdict
import logging
from typing import List
from accelforge._accelerated_imports import np
from accelforge.frontend._workload_isl._symbolic import PartiallyRelevant, Relevant
import accelforge.frontend.arch as arch
from accelforge.frontend.arch import (
    Comparison,
    _MinUsageConstraintLambda,
    _TileShapeConstraintLambda,
    _LoopBoundsConstraintLambda,
    _ConstraintLambda,
)
from accelforge.frontend.mapping import (
    Loop,
    MappingNode,
    TensorHolder,
    Temporal,
    Spatial,
)
from accelforge.frontend.renames import TensorName
from accelforge.frontend.workload import EinsumName, RankVariable
from accelforge.util._setexpressions import InvertibleSet
from accelforge.util._frozenset import fzs, oset


# =================================================================================================
# Attach constraints to mapping
# =================================================================================================
class MappingConstraints:
    def __init__(self):
        self.tile_shape_constraints: list[_TileShapeConstraintLambda] = []
        self.loop_bounds_constraints: list[_LoopBoundsConstraintLambda] = []
        self.min_usage_constraints: dict[tuple[str, str], _MinUsageConstraintLambda] = (
            {}
        )

    def get_all_constraints(self) -> list[_ConstraintLambda]:
        return (
            self.tile_shape_constraints
            + self.loop_bounds_constraints
            + list(self.min_usage_constraints.values())
        )

    def set_loop_indices(self, nodes: list[MappingNode]):
        loops = [n for n in nodes if isinstance(n, Loop)]
        for c in self.get_all_constraints():
            c._target_node_indices = [nodes.index(t) for t in c.target_mapping_nodes]
            c._target_loop_indices = [loops.index(t) for t in c.target_mapping_nodes]

        # Min usage constraints also depend on the loop ABOVE the target loop
        # because the loop above determines the number of tiles
        for c in self.min_usage_constraints.values():
            # Rank variables must be unique between mapping nodes
            rank_variables = oset(t.rank_variable for t in c.target_mapping_nodes)
            assert len(rank_variables) == len(
                c.target_mapping_nodes
            ), "Rank variables must be unique between mapping nodes"

            for target_mapping_node in c.target_mapping_nodes:
                assert isinstance(target_mapping_node, Spatial)
                loop_index = loops.index(target_mapping_node) - 1
                while loop_index >= 0:
                    loop = loops[loop_index]
                    if loop.rank_variable in rank_variables:
                        c._target_loop_indices.append(loop_index)
                        c._target_node_indices.append(nodes.index(loop))
                        break
                    loop_index -= 1

    def clear_constrained_to_one(
        self, mapping: list["MappingNode"], einsum_name: EinsumName
    ) -> list["MappingNode"]:
        # Not constrained to one --> Can't remove
        node2constraints = defaultdict(list)
        do_not_remove = oset()
        for c in self.tile_shape_constraints:
            for t in c.target_mapping_nodes:
                node2constraints[id(t)].append(c)
                do_not_remove.add(id(t))
        for c in self.loop_bounds_constraints:
            if not c.constraint._constrained_to_one():
                for t in c.target_mapping_nodes:
                    node2constraints[id(t)].append(c)
                    do_not_remove.add(id(t))

        # Constrained to one --> remove iff not in do_not_remove
        to_remove = oset()
        for c in self.loop_bounds_constraints:
            if c.constraint._constrained_to_one():
                my_remove = oset(id(t) for t in c.target_mapping_nodes)
                if my_remove & do_not_remove:
                    loops = [n for n in mapping if id(n) in my_remove]
                    p = len(loops) == 1
                    loops = (", ".join(n.compact_str() for n in loops)).strip()
                    isare = "is" if p else "are"
                    all_others = ", ".join(
                        str(c2) for c2 in node2constraints[id(t)] if c2 != c
                    )
                    logging.warning(
                        f"For Einsum {einsum_name}, loop{'s' * (not p)} {loops} "
                        f"{isare} set to be removed by {c} and also appear{'s' * p} in "
                        f"{all_others}. The loop{'s' * (not p)} will not be removed "
                        f"from the mapping, but it may be subject to conflicting "
                        f"constraints."
                    )

                c.target_mapping_nodes = [
                    t for t in c.target_mapping_nodes if id(t) not in my_remove
                ]
                to_remove.update(my_remove)
        self.loop_bounds_constraints = [
            c
            for c in self.loop_bounds_constraints
            if not c.constraint._constrained_to_one()
        ]

        for c in self.get_all_constraints():
            c.target_mapping_nodes = [
                n for n in c.target_mapping_nodes if id(n) not in to_remove
            ]

        return [m for m in mapping if id(m) not in to_remove]

    def pretty_str(self) -> str:
        s = ""
        all_constraints = self.get_all_constraints()
        s += "Tile shape constraints:\n"
        for c in self.tile_shape_constraints:
            s += f"\t{all_constraints.index(c)} {c.pretty_str()}\n"
        s += "Loop bounds constraints:\n"
        for c in self.loop_bounds_constraints:
            s += f"\t{all_constraints.index(c)} {c.pretty_str()}\n"
        s += "Min usage constraints:\n"
        for c in self.min_usage_constraints.values():
            s += f"\t{all_constraints.index(c)} {c.pretty_str()}\n"
        return s

    def remove_missing_targets(self, mapping: list[MappingNode]):
        for c in self.get_all_constraints():
            c.target_mapping_nodes = [n for n in c.target_mapping_nodes if n in mapping]

        self.tile_shape_constraints = [c for c in self.tile_shape_constraints if c]
        self.loop_bounds_constraints = [c for c in self.loop_bounds_constraints if c]
        self.min_usage_constraints = {
            k: c for k, c in self.min_usage_constraints.items() if c
        }


def first_tensor_holder_index(mapping: list["MappingNode"], memory_name: str) -> int:
    for i, m in enumerate(mapping):
        if isinstance(m, TensorHolder) and m.component == memory_name:
            return i
    return None


def constrained_loops(
    mapping: list["MappingNode"],
    rank_variables: set[RankVariable],
    start_index: int = None,
    look_behind: bool = False,
    component: str = None,
    one_loop_per_rank_variable: bool = True,
) -> list[Loop]:
    """The first loop nodes containing given rank variables."""
    nodes = []
    remaining_rank_variables = oset(rank_variables)

    if look_behind:
        to_check = list(enumerate(mapping))
        to_check.reverse()
        if start_index is not None:
            to_check = [m for i, m in to_check if i <= start_index]
    else:
        to_check = list(enumerate(mapping))
        to_check = [m for i, m in to_check if start_index is None or i >= start_index]

    for m in to_check:
        if not isinstance(m, Loop):
            continue
        if component is not None and (
            not isinstance(m, Spatial) or m.component != component
        ):
            continue
        assert isinstance(m.rank_variable, RankVariable)
        if m.rank_variable in remaining_rank_variables:
            nodes.append(m)
            if one_loop_per_rank_variable:
                remaining_rank_variables.discard(m.rank_variable)
    # TODO: what is this supposed to do?
    # for r in remaining_rank_variables:
    #     assert (
    #         component is None
    #     ), "There should be a spatial loop for every rank variable"
    return nodes


def get_constraints(
    flattened_arch: list[arch.Leaf],
    mapping: List[MappingNode],
    symbol_table: dict[str, InvertibleSet],
    einsum_name: EinsumName,
    tensor_to_relevancy: dict[
        TensorName, dict[RankVariable, Relevant | PartiallyRelevant]
    ],
    is_copy_operation: bool,
) -> tuple[List[MappingNode], MappingConstraints]:

    constraints = MappingConstraints()

    # Tensor constraints
    for m in flattened_arch:
        # Ignore if not a memory
        if not isinstance(m, arch.Memory):
            continue

        # Ignore if it doesn't hold any tensors
        if (index := first_tensor_holder_index(mapping, m.name)) is None:
            continue

        # Tile shape constraints
        for c in m.tensors.tile_shape:
            nodes = constrained_loops(
                mapping, c.expression, index - 1, look_behind=True
            )
            for exp in c._split_expression():
                new_nodes = [n for n in nodes if n.rank_variable in exp]
                constraint = _TileShapeConstraintLambda(c, new_nodes, exp)
                constraints.tile_shape_constraints.append(constraint)

        no_resend_refetch_constraint = _loop_bound_constraint_from_no_refetch_and_resend(
            mapping,
            index,
            m,
            symbol_table,
            tensor_to_relevancy
        )
        if no_resend_refetch_constraint is not None:
            constraints.loop_bounds_constraints.append(no_resend_refetch_constraint)

    # Spatial constraints
    for m in flattened_arch:
        if not isinstance(m, arch.Leaf):
            continue

        for dim in m.spatial:
            loops = [
                n
                for n in mapping
                if isinstance(n, Spatial)
                and (n.component, n.name) == (m.name, dim.name)
            ]
            loop_bounds = list(dim.loop_bounds)
            if dim.reuse:
                loop_bounds.append(
                    Comparison(
                        expression=dim.reuse.rank_variables,
                        operator="==",
                        value=1,
                    )
                )
                loop_bounds[-1]._str_repr = f"reuse {oset(dim.reuse)}"

            # Loop bounds constraints
            if loop_bounds:
                for c in loop_bounds:
                    nodes = constrained_loops(loops, c.expression, component=m.name)
                    for exp in c._split_expression():
                        new_nodes = [l for l in loops if l.rank_variable in exp]
                        constraint = _LoopBoundsConstraintLambda(c, new_nodes, exp)
                        constraints.loop_bounds_constraints.append(constraint)

            # Min usage constraints
            target_mapping_nodes = [
                n
                for n in mapping
                if isinstance(n, Spatial)
                and n.component == m.name
                and n.name == dim.name
            ]
            # Min usage constraints don't apply to copy operations
            if dim.min_usage > 0 and not is_copy_operation:
                if not target_mapping_nodes:
                    continue
                rank_variables = oset(t.rank_variable for t in target_mapping_nodes)
                constraint = _MinUsageConstraintLambda(
                    target_mapping_nodes,
                    rank_variables,
                    dim.min_usage,
                )
                key = (m.name, dim.name)
                constraints.min_usage_constraints[key] = constraint

            for t in target_mapping_nodes:
                t._may_reuse = dim.may_reuse

    # Additional spatial constraints
    for m in mapping:
        if isinstance(m, Spatial) and m._constrained_to_one:
            constraints.loop_bounds_constraints.append(
                _LoopBoundsConstraintLambda(
                    Comparison(expression=m.rank_variable, operator="==", value=1),
                    [m],
                    m.rank_variable,
                )
            )

    mapping = constraints.clear_constrained_to_one(mapping, einsum_name)

    return mapping, constraints


def _loop_bound_constraint_from_no_refetch_and_resend(mapping, index, arch_node, symbol_table, tensor_to_relevancy):
    """Create a loop bound constraint representing no refetch and no resend."""
    no_refetch = mapping[index].component_object.tensors.no_refetch_from_above
    exp = symbol_table[arch_node.name] & no_refetch

    nodes = []
    seen = oset()
    for no_refetch in exp.iter_one_element_sets():
        # Start from the first index of the tensor holder, stop at index - 1
        start_index = 0
        n = next(iter(no_refetch))
        while start_index < len(mapping):
            if (
                isinstance(mapping[start_index], TensorHolder)
                and n in mapping[start_index].tensors
            ):
                break
            start_index += 1

        end_index = start_index + 1
        while end_index < len(mapping):
            if (
                isinstance(mapping[end_index], TensorHolder)
                and n in mapping[end_index].tensors
                and mapping[end_index].component == m.name
            ):
                break
            end_index += 1
        else:
            # This tensor isn't stored in this component at all. Don't look at any
            # loops. We can also end up here if the tensor is backed in this
            # component, in which case we'll start looking below the component &
            # never find it.
            end_index = start_index

        for i in range(start_index, min(end_index, len(mapping))):
            if isinstance(mapping[i], Temporal) and not isinstance(
                tensor_to_relevancy[n][mapping[i].rank_variable], Relevant
            ):
                if id(mapping[i]) not in seen:
                    seen.add(id(mapping[i]))
                    nodes.append(mapping[i])

    no_resend = mapping[index].component_object.tensors.no_resend_to_below
    for no_resend in no_resend.iter_one_element_sets():
        # Start from the first index of this one, stop when we find someone else
        # below
        start_index = 0
        n = next(iter(no_resend))
        while start_index < len(mapping):
            if (
                isinstance(mapping[start_index], TensorHolder)
                and n in mapping[start_index].tensors
                and mapping[start_index].component == m.name
            ):
                break
            start_index += 1

        end_index = start_index + 1
        while end_index < len(mapping):
            if (
                isinstance(mapping[end_index], TensorHolder)
                and n in mapping[end_index].tensors
            ):
                # Can't have two tensor holders for the same tensor + component
                assert mapping[end_index].component != m.name
                break
            end_index += 1

        for i in range(start_index, min(end_index, len(mapping))):
            if isinstance(mapping[i], Temporal) and not isinstance(
                tensor_to_relevancy[n][mapping[i].rank_variable], Relevant
            ):
                if id(mapping[i]) not in seen:
                    seen.add(id(mapping[i]))
                    nodes.append(mapping[i])

    if nodes:
        return _LoopBoundsConstraintLambda(
            Comparison(expression=exp, operator="==", value=1), nodes, exp
        )
    else:
        return None