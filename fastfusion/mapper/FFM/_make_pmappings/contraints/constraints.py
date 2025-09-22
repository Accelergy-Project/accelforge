from collections import defaultdict
from typing import List
from fastfusion.accelerated_imports import np
import fastfusion.frontend.arch as arch
from fastfusion.frontend.constraints import Comparison, ConstraintGroup, MinUtilizationConstraintLambda, TileShapeConstraintLambda, LoopBoundsConstraintLambda, ConstraintLambda
from fastfusion.frontend.constraints import Spatial as SpatialConstraint
from fastfusion.frontend.mapping import Iteration, MappingNode, TensorHolder, Temporal, Spatial
from fastfusion.frontend.workload.workload import EinsumName, RankVariableName
from fastfusion.util.setexpressions import InvertibleSet
from fastfusion.util.util import fzs

    
# =================================================================================================
# Attach constraints to mapping
# =================================================================================================
class MappingConstraints:
    def __init__(self):
        self.tile_shape_constraints: list[TileShapeConstraintLambda] = []
        self.loop_bounds_constraints: list[LoopBoundsConstraintLambda] = []
        self.min_utilization_constraints: dict[tuple[str, str], MinUtilizationConstraintLambda] = {}

    def get_all_constraints(self) -> list[ConstraintLambda]:
        return self.tile_shape_constraints + self.loop_bounds_constraints + list(self.min_utilization_constraints.values())

    def check_tile_shape_constraints(
            self,
            tile_shapes: np.ndarray, 
            complete_indices: list[int]
        ):
        mask = np.ones(tile_shapes.shape[0], dtype=np.bool)
        for c in self.tile_shape_constraints:
            mask = mask & c(complete_indices, tile_shapes[:, c._target_loop_indices])
        return mask
    
    def check_min_utilization_constraints(
            self,
            component_name: str,
            name: str,
            utilization: np.ndarray,
            complete_indices: list[int]
        ):
        if (component_name, name) not in self.min_utilization_constraints:
            return np.ones(utilization.shape[0], dtype=np.bool)

        return self.min_utilization_constraints[(component_name, name)](complete_indices, utilization)
    
    def set_loop_indices(self, nodes: list[MappingNode]):
        loops = [n for n in nodes if isinstance(n, Iteration)]
        for c in self.get_all_constraints():
            c._target_node_indices = [nodes.index(t) for t in c.target_mapping_nodes]
            c._target_loop_indices = [loops.index(t) for t in c.target_mapping_nodes]
        
        # Min utilization constraints also depend on the loop ABOVE the target loop
        # because the loop above determines the number of tiles
        for c in self.min_utilization_constraints.values():
            # Rank variables must be unique between mapping nodes
            rank_variables = set(t.rank_variable for t in c.target_mapping_nodes)
            assert len(rank_variables) == len(c.target_mapping_nodes), "Rank variables must be unique between mapping nodes"
            
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

    def clear_constrained_to_one(self, mapping: list["MappingNode"]) -> list["MappingNode"]:
        # Not constrained to one --> Can't remove
        do_not_remove = set()
        for c in self.tile_shape_constraints:
            for t in c.target_mapping_nodes:
                do_not_remove.add(id(t))
        for c in self.loop_bounds_constraints:
            if not c.constraint.constrained_to_one():
                for t in c.target_mapping_nodes:
                    do_not_remove.add(id(t))
        
        # Constrained to one --> remove iff not in do_not_remove
        to_remove = set()
        for c in self.loop_bounds_constraints:
            if c.constraint.constrained_to_one():
                my_remove = set(id(t) for t in c.target_mapping_nodes) - do_not_remove
                c.target_mapping_nodes = [t for t in c.target_mapping_nodes if id(t) not in my_remove]
                to_remove.update(my_remove)
        self.loop_bounds_constraints = [c for c in self.loop_bounds_constraints if not c.constraint.constrained_to_one()]
        
        for c in self.get_all_constraints():
            c.target_mapping_nodes = [n for n in c.target_mapping_nodes if id(n) not in to_remove]

        return [m for m in mapping if id(m) not in to_remove]
    
    def pretty_str(self) -> str:
        s = ''
        all_constraints = self.get_all_constraints()
        s += 'Tile shape constraints:\n'
        for c in self.tile_shape_constraints:
            s += f'\t{all_constraints.index(c)} {c.pretty_str()}\n'
        s += 'Loop bounds constraints:\n'
        for c in self.loop_bounds_constraints:
            s += f'\t{all_constraints.index(c)} {c.pretty_str()}\n'
        s += 'Min utilization constraints:\n'
        for c in self.min_utilization_constraints.values():
            s += f'\t{all_constraints.index(c)} {c.pretty_str()}\n'
        return s

def first_tensor_holder_index(mapping: list["MappingNode"], memory_name: str) -> int:
    for i, m in enumerate(mapping):
        if isinstance(m, TensorHolder) and m.component == memory_name:
            return i
    return None

def constrained_loops(
        mapping: list["MappingNode"], 
        rank_variables: set[RankVariableName], 
        start_index: int=None, 
        look_behind: bool=False, 
        component: str=None,
        one_loop_per_rank_variable: bool=True,
    ) -> list[Iteration]:
    nodes = []
    remaining_rank_variables = set(rank_variables)
    
    if look_behind:
        to_check = list(enumerate(mapping))
        to_check.reverse()
        if start_index is not None:
            to_check = [m for i, m in to_check if start_index is None or i <= start_index]
    else:
        to_check = list(enumerate(mapping))
        to_check = [m for i, m in to_check if start_index is None or i >= start_index]
    
    for m in to_check:
        if not isinstance(m, Iteration):
            continue
        if component is not None and (not isinstance(m, Spatial) or m.component != component):
            continue
        assert isinstance(m.rank_variable, RankVariableName)
        if m.rank_variable in remaining_rank_variables:
            nodes.append(m)
            if one_loop_per_rank_variable:
                remaining_rank_variables.discard(m.rank_variable)
    for r in remaining_rank_variables:
        assert component is None, "There should be a spatial loop for every rank variable"
    return nodes

def get_constraints(
    arch_flattened: list[arch.Leaf],
    mapping: List[MappingNode],
    symbol_table: dict[str, InvertibleSet],
    einsum_name: EinsumName,
) -> tuple[List[MappingNode], MappingConstraints]:
    
    constraints = MappingConstraints()
    
    
    # Tensor constraints
    for m in arch_flattened:
        # Ignore if not a memory
        if not isinstance(m, arch.Memory):
            continue
        
        # Ignore if it doesn't hold any tensors
        if (index := first_tensor_holder_index(mapping, m.name)) is None:
            continue

        tensor_constraints = m.constraints.tensors._parse_non_keep_bypass(symbol_table, f"{m.name}.constraints.tensors")

        # Tile shape constraints
        for c in tensor_constraints.tile_shape:
            nodes = constrained_loops(mapping, c.expression, index - 1, look_behind=True)
            for exp in c.split_expression():
                new_nodes = [n for n in nodes if n.rank_variable in exp]
                constraint = TileShapeConstraintLambda(c, new_nodes, exp)
                constraints.tile_shape_constraints.append(constraint)
                
        # No refetch from above constraints
        exp = tensor_constraints.no_refetch_from_above & symbol_table[m.name]
        result = set()
        for no_refetch in exp.iter_one_element_sets():
            result.update(~no_refetch.rank_variables())
        nodes = constrained_loops(mapping, result, index - 1, look_behind=True, one_loop_per_rank_variable=False)
        constraints.loop_bounds_constraints.append(LoopBoundsConstraintLambda(
            Comparison(
                expression=exp,
                operator='==',
                value=1
            ),
            nodes,
            exp
        ))
                
    # Temporal loop bounds constraints
    # TODO: Implement
            
    # Spatial constraints
    for m in arch_flattened:
        if not isinstance(m, arch.Memory):
            continue

        for dim in m.spatial:
            dim = dim.name
            if dim not in m.constraints.spatial:
                continue
            loops = [n for n in mapping if isinstance(n, Spatial) and (n.component, n.name) == (m.name, dim)]
            spatial_constraint = m.constraints.spatial[dim]._parse(symbol_table, f"{m.name}.constraints.spatial")

            # Loop bounds constraints
            if spatial_constraint.loop_bounds:
                for c in spatial_constraint.loop_bounds:
                    nodes = constrained_loops(loops, c.expression, component=m.name)
                    for exp in c.split_expression():
                        new_nodes = [l for l in loops if l.rank_variable in exp]
                        constraint = LoopBoundsConstraintLambda(c, new_nodes, exp)
                        constraints.loop_bounds_constraints.append(constraint)

            # Min utilization constraints
            if spatial_constraint.min_utilization > 0:
                target_mapping_nodes = [
                    n for n in mapping if isinstance(n, Spatial) and n.component == m.name and n.name == dim
                ]
                if not target_mapping_nodes:
                    continue
                rank_variables = {t.rank_variable for t in target_mapping_nodes}
                constraint = MinUtilizationConstraintLambda(target_mapping_nodes, rank_variables, spatial_constraint.min_utilization)
                key = (m.name, dim)
                constraints.min_utilization_constraints[key] = constraint

    mapping = constraints.clear_constrained_to_one(mapping)
    constraints.set_loop_indices(mapping)
    
    return mapping, constraints
