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
from accelforge.util._frozenset import fzs


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

    def check_tile_shape_constraints(
        self, tile_shapes: np.ndarray, complete_indices: list[int]
    ):
        mask = np.ones(tile_shapes.shape[0], dtype=np.bool)
        for c in self.tile_shape_constraints:
            mask = mask & c(complete_indices, tile_shapes[:, c._target_loop_indices])
        return mask

    def check_min_usage_constraints(
        self,
        component_name: str,
        name: str,
        usage: np.ndarray,
        complete_indices: list[int],
    ):
        if (component_name, name) not in self.min_usage_constraints:
            return np.ones(usage.shape[0], dtype=np.bool)

        return self.min_usage_constraints[(component_name, name)](
            complete_indices, usage
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
            rank_variables = set(t.rank_variable for t in c.target_mapping_nodes)
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
        do_not_remove = set()
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
        to_remove = set()
        for c in self.loop_bounds_constraints:
            if c.constraint._constrained_to_one():
                my_remove = set(id(t) for t in c.target_mapping_nodes)
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
    nodes = []
    remaining_rank_variables = set(rank_variables)

    if look_behind:
        to_check = list(enumerate(mapping))
        to_check.reverse()
        if start_index is not None:
            to_check = [
                m for i, m in to_check if start_index is None or i <= start_index
            ]
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

        exp = symbol_table[m.name] & m.tensors.no_refetch_from_above

        nodes = []
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

            end_index = start_index
            while end_index < len(mapping):
                if (
                    isinstance(mapping[end_index], TensorHolder)
                    and n in mapping[end_index].tensors
                    and mapping[end_index].component == m.name
                ):
                    break
                end_index += 1

            for i in range(start_index, end_index):
                if isinstance(mapping[i], Temporal) and not isinstance(
                    tensor_to_relevancy[n][mapping[i].rank_variable], Relevant
                ):
                    if mapping[i] not in nodes:
                        nodes.append(mapping[i])

        if nodes:
            constraints.loop_bounds_constraints.append(
                _LoopBoundsConstraintLambda(
                    Comparison(expression=exp, operator="==", value=1), nodes, exp
                )
            )

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
                loop_bounds[-1]._str_repr = f"reuse {set(dim.reuse)}"

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
            if dim.min_usage > 0:
                if not target_mapping_nodes:
                    continue
                rank_variables = {t.rank_variable for t in target_mapping_nodes}
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
