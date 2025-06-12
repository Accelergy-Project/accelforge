from collections import defaultdict
from typing import List
import numpy as np
import fastfusion.frontend.architecture as architecture
from fastfusion.frontend.constraints import Comparison, ConstraintGroup, MaximizeUtilizationConstraintLambda, TileShapeConstraintLambda, LoopBoundsConstraintLambda
from fastfusion.frontend.constraints import Spatial as SpatialConstraint
from fastfusion.frontend.mapping import Iteration, MappingNode, Storage, Temporal, Spatial
from fastfusion.frontend.workload.workload import RankVariableName
from fastfusion.util.setexpressions import InvertibleSet
from fastfusion.util.util import fzs

    
# =================================================================================================
# Attach constraints to mapping
# =================================================================================================
class MappingConstraints:
    def __init__(
        self,
        tile_shape_constraints: list[TileShapeConstraintLambda] = (),
        loop_bounds_constraints: list[LoopBoundsConstraintLambda] = (),
        maximize_utilization_constraints: dict[tuple[str, str], MaximizeUtilizationConstraintLambda] = (),
    ):
        self.tile_shape_constraints = list(tile_shape_constraints)
        self.loop_bounds_constraints = list(loop_bounds_constraints)
        self.maximize_utilization_constraints = dict(maximize_utilization_constraints)

    def check_tile_shape_constraints(
            self,
            tile_shapes: np.ndarray, 
            rank_vars: set[RankVariableName]
        ):
        mask = np.ones(tile_shapes.shape[0], dtype=np.bool)
        for c in self.tile_shape_constraints:
            mask = mask & c(rank_vars, tile_shapes[:, c._target_indices])
        return mask
    
    
    def check_maximize_utilization_constraints(
            self,
            component_name: str,
            dimension: str,
            utilization: np.ndarray,
            rank_vars: set[RankVariableName]
        ):
        if (component_name, dimension) not in self.maximize_utilization_constraints:
            return np.ones(utilization.shape[0], dtype=np.bool)
        return self.maximize_utilization_constraints[(component_name, dimension)](rank_vars, utilization)



def get_constraints(
    arch_flattened: list[architecture.Leaf],
    mapping: List[MappingNode],
    symbol_table: dict[str, InvertibleSet],
) -> MappingConstraints:
    parsed = {}
    def get_parsed_storage_constraint(constraint: ConstraintGroup) -> ConstraintGroup:
        key = id(constraint.storage)
        if key not in parsed:
            parsed[key] = constraint.storage._parse_non_keep_bypass(symbol_table)
        return parsed[key]
    
    def get_parsed_spatial_constraint(constraint: ConstraintGroup, dimension: str) -> SpatialConstraint:
        key = (id(constraint), dimension)
        if dimension not in constraint.spatial:
            return None
        if key not in parsed and dimension in constraint.spatial:
            parsed[key] = constraint.spatial[dimension]._parse(symbol_table)
        return parsed[key]
            
    tile_shape_constraint_id_to_mapping_nodes = defaultdict(list)
    loop_bounds_constraint_id_to_mapping_nodes = defaultdict(list)
    tile_shape_constraints: list[Comparison] = []
    loop_bounds_constraints: list[Comparison] = []

    def add_storage_constraint(m: Storage, constraint: Comparison):
        if id(constraint) not in tile_shape_constraint_id_to_mapping_nodes:
            tile_shape_constraints.append(constraint)
        tile_shape_constraint_id_to_mapping_nodes[id(constraint)].append(m)
        
    def add_loop_bounds_constraint(m: Iteration, constraint: Comparison):
        if id(constraint) not in loop_bounds_constraint_id_to_mapping_nodes:
            loop_bounds_constraints.append(constraint)
        loop_bounds_constraint_id_to_mapping_nodes[id(constraint)].append(m)
        
    def add_parsed_dataflow_constraint_to_list(constraint: ConstraintGroup, dataflow_list: list[Comparison]):
        key = id(constraint)
        if key not in parsed:
            parsed[key] = constraint._parse(symbol_table)
        return parsed[key]
    
    for m in mapping:
        if isinstance(m, Storage):
            constraint = get_parsed_storage_constraint(m.memory_object.constraints)
            for c in constraint.tile_shape:
                add_storage_constraint(m, c)

        if isinstance(m, Spatial):
            constraint = get_parsed_spatial_constraint(m.across_object.constraints, m.dimension)
            if constraint is not None:
                for c in constraint.loop_bounds:
                    if m.rank_variable in c.expression:
                        add_loop_bounds_constraint(m, c)

    constraints = MappingConstraints()

    constraint_lambdas = []
    do_not_remove = set()
    for constraint in tile_shape_constraints:
        mapping_nodes = tile_shape_constraint_id_to_mapping_nodes[id(constraint)]
        targets = []
        for expression in constraint.split_expression():
            for m in mapping_nodes:
                target_loops = []
                remaining_rank_vars = set(expression)
                idx = mapping.index(m)
                for i in range(idx, -1, -1):
                    if isinstance(mapping[i], Iteration):
                        if mapping[i].rank_variable in remaining_rank_vars:
                            target_loops.append(mapping[i])
                            remaining_rank_vars.discard(mapping[i].rank_variable)
                for r in remaining_rank_vars:
                    mapping.insert(
                        idx,
                        Temporal(rank_variable=r, tile_shape='symbol')
                    )
                    target_loops.append(mapping[idx])
                targets.append(target_loops)
                for t in target_loops:
                    do_not_remove.add(id(t))
                
        seen = set()
        for t in targets:
            seen_key = fzs(id(x) for x in t)
            if seen_key in seen:
                continue
            seen.add(seen_key)
            constraint = TileShapeConstraintLambda(constraint, t, expression)
            constraints.tile_shape_constraints.append(constraint)

    for constraint in loop_bounds_constraints:
        mapping_nodes = loop_bounds_constraint_id_to_mapping_nodes[id(constraint)]
        if constraint.constrained_to_one():
            for m in mapping_nodes:
                if id(m) not in do_not_remove:
                    mapping.remove(m)
            continue
        raise NotImplementedError("Loop bounds constraints not implemented")
        constraint_lambdas.append(LoopBoundsConstraintLambda(constraint, mapping_nodes))

    loops = [n for n in mapping if isinstance(n, Iteration)]
                        
    for node in arch_flattened:
        if not isinstance(node, architecture.Memory):
            continue
        for dim in node.spatial.fanout:
            spatial_constraint = get_parsed_spatial_constraint(node.constraints, dim)
            if spatial_constraint is None:
                    continue
            if not spatial_constraint.maximize_utilization:
                continue
            
            
            target_loops = []
            for loop in mapping:
                if isinstance(loop, Spatial) and loop.across == node.name and loop.dimension == dim:
                    target_loops.append(loop)
            if not loops:
                continue        
            
            rank_variables = {t.rank_variable for t in target_loops}
            constraint = MaximizeUtilizationConstraintLambda(target_loops, rank_variables)
            key = (node.name, dim)
            constraints.maximize_utilization_constraints[key] = constraint

    for c in constraint_lambdas:
        c._target_indices = [loops.index(t) for t in c.target_mapping_nodes]
        assert c._target_indices
    
    return constraints
