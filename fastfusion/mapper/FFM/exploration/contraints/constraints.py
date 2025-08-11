from collections import defaultdict
from typing import List
from fastfusion.accelerated_imports import np
import fastfusion.frontend.architecture as architecture
from fastfusion.frontend.constraints import Comparison, ConstraintGroup, MinUtilizationConstraintLambda, TileShapeConstraintLambda, LoopBoundsConstraintLambda
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

    def check_tile_shape_constraints(
            self,
            tile_shapes: np.ndarray, 
            rank_vars: set[RankVariableName]
        ):
        mask = np.ones(tile_shapes.shape[0], dtype=np.bool)
        for c in self.tile_shape_constraints:
            mask = mask & c(rank_vars, tile_shapes[:, c._target_indices])
        return mask
    
    def check_min_utilization_constraints(
            self,
            component_name: str,
            dimension: str,
            utilization: np.ndarray,
            rank_vars: set[RankVariableName]
        ):
        if (component_name, dimension) not in self.min_utilization_constraints:
            return np.ones(utilization.shape[0], dtype=np.bool)
        return self.min_utilization_constraints[(component_name, dimension)](rank_vars, utilization)
    
    def set_loop_indices(self, loops: list[Iteration]):
        for c in self.tile_shape_constraints:
            c._target_indices = [loops.index(t) for t in c.target_mapping_nodes]

        for c in self.loop_bounds_constraints:
            c._target_indices = [loops.index(t) for t in c.target_mapping_nodes]
            
    def clear_constrained_to_one(self, mapping: list[MappingNode]) -> list[MappingNode]:
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

        return [m for m in mapping if id(m) not in to_remove]

def first_tensor_holder_index(mapping: list[MappingNode], memory_name: str) -> int:
    for i, m in enumerate(mapping):
        if isinstance(m, TensorHolder) and m.component == memory_name:
            return i
    return None

def constrained_loops(mapping: list[MappingNode], rank_variables: set[RankVariableName], start_index: int=None, look_behind: bool=False, across: str=None) -> list[Iteration]:
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
        if across is not None and (not isinstance(m, Spatial) or m.across != across):
            continue
        assert isinstance(m.rank_variable, RankVariableName)
        if m.rank_variable in remaining_rank_variables:
            nodes.append(m)
            remaining_rank_variables.discard(m.rank_variable)
    for r in remaining_rank_variables:
        assert across is None, "There should be a spatial loop for every rank variable"
        node = Temporal(rank_variable=r, tile_shape='symbol')
        mapping.insert(start_index, node)
        nodes.append(node)
    return nodes

def get_constraints(
    arch_flattened: list[architecture.Leaf],
    mapping: List[MappingNode],
    symbol_table: dict[str, InvertibleSet],
    einsum_name: EinsumName,
) -> tuple[List[MappingNode], MappingConstraints]:
    
    constraints = MappingConstraints()
    
    # Tensor constraints
    for m in arch_flattened:
        if not isinstance(m, architecture.Memory):
            continue
        
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
                
    # Temporal loop bounds constraints
    # TODO: Implement
            
    # Spatial constraints
    for m in arch_flattened:
        if not isinstance(m, architecture.Memory):
            continue

        for dim in m.spatial.fanout:
            if dim not in m.constraints.spatial:
                continue
            loops = [n for n in mapping if isinstance(n, Spatial) and (n.across, n.dimension) == (m.name, dim)]
            spatial_constraint = m.constraints.spatial[dim]._parse(symbol_table, f"{m.name}.constraints.spatial")

            # Loop bounds constraints
            if spatial_constraint.loop_bounds:
                for c in spatial_constraint.loop_bounds:
                    nodes = constrained_loops(loops, c.expression, across=m.name)
                    for exp in c.split_expression():
                        new_nodes = [l for l in loops if l.rank_variable in exp]
                        constraint = LoopBoundsConstraintLambda(c, new_nodes, exp)
                        constraints.loop_bounds_constraints.append(constraint)

            # Min utilization constraints
            if spatial_constraint.min_utilization > 0:
                target_mapping_nodes = [
                    n for n in mapping if isinstance(n, Spatial) and n.across == m.name and n.dimension == dim
                ]
                if not target_mapping_nodes:
                    continue
                rank_variables = {t.rank_variable for t in target_mapping_nodes}
                constraint = MinUtilizationConstraintLambda(target_mapping_nodes, rank_variables, spatial_constraint.min_utilization)
                key = (m.name, dim)
                constraints.min_utilization_constraints[key] = constraint

    mapping = constraints.clear_constrained_to_one(mapping)
    constraints.set_loop_indices([m for m in mapping if isinstance(m, Iteration)])
    
    return mapping, constraints
