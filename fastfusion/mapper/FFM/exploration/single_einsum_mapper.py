from collections import defaultdict
from itertools import chain, combinations
import copy
from typing import List

import numpy as np
from pandas import DataFrame

from fastfusion.frontend.constraints import Comparison, ConstraintGroup
from fastfusion.frontend.mapping import Iteration, Mapping, MappingNode, Storage, Temporal, Spatial, Compute
import fastfusion.frontend.architecture as architecture
from fastfusion.frontend.architecture import Leaf
from fastfusion.mapper.FFM.exploration.tile_shape_exploration import dummy_tile_shape_exploration
from fastfusion.mapper.FFM.joining.mappinginfo import Compatibility, Loop, Reservation
from fastfusion.mapper.FFM.joining.sim import SIM
from fastfusion.mapper.FFM.pareto import MAPPING_COLUMN, Pareto
from fastfusion.util.setexpressions import InvertibleSet
from fastfusion.frontend.specification import Specification
from fastfusion.frontend.workload.workload import Einsum, RankVariableName, TensorName
from fastfusion.util.util import fzs

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

# =================================================================================================
# Choose what data to store in each memory
# =================================================================================================

def make_storage_choices_one_level(
        node: Leaf,
        symbol_table: dict[str, InvertibleSet],
    ):
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

    for subset in powerset(may_keep):
        # Make keep choice & update symbol table
        subset = tensors.to_my_space(set(subset))
        keep_choice = tensors.to_my_space(subset | must_keep)
        keep_choice.tensors = lambda: keep_choice # So users can do MainMemory().tensors(). Optional.
        new_symbol_table[node.name] = keep_choice
        
        # Make sure they're all tensors
        assert all(isinstance(k, TensorName) for k in keep_choice)
        keep_choice = keep_choice.to_my_space({copy.copy(t) for t in keep_choice})
        storage_nodes = []
        
        # Create storage nodes
        for t in keep_choice:
            storage_nodes.append(Storage(tensors=[t], memory=node.name, memory_object=node))
            storage_nodes[-1]._must_exist = t in must_keep
        yield storage_nodes, new_symbol_table

def make_storage_choices_all_levels(
    nodes: list[Storage], 
    symbol_table: dict[str, InvertibleSet],
):
    if len(nodes) == 0:
        yield dict(), symbol_table
        return
    for choice, symbol_table in make_storage_choices_one_level(nodes[0], symbol_table):
        for subchoices, symbol_table in make_storage_choices_all_levels(nodes[1:], symbol_table):
            yield {**subchoices, nodes[0].name: choice}, symbol_table
            

# =================================================================================================
# Order storage nodes (dataflow).
# =================================================================================================

def label_backing_storages(mapping: List[MappingNode]):
    seen_tensors = set()
    for i, s in enumerate(mapping):
        if isinstance(s, Storage):
            s._backing = s.tensor not in seen_tensors
            seen_tensors.add(s.tensor)

def valid_storage_order(mapping: List[MappingNode], node_names: list[str]):
    for i in range(len(mapping)):
        for j in range(i, len(mapping)):
            t1, t2 = mapping[i].tensor, mapping[j].tensor
            if t1 != t2:
                continue
            s1, s2 = mapping[i].memory, mapping[j].memory
            s1_idx, s2_idx = node_names.index(s1), node_names.index(s2)
            
            # If a tensor is stored in two levels back-to-back, then we
            # should have bypassed the outer storage if possible.
            if i == j or i == j - 1:
                if s1_idx < s2_idx and not (mapping[i]._must_exist or mapping[i]._backing):
                    return False
                if s2_idx < s1_idx and not (mapping[j]._must_exist or mapping[j]._backing):
                    return False
                
            # Ensure order
            if i < j and s2_idx < s1_idx:
                return False
    return True

def recursive_order_storage_choices(
    mapping: List[MappingNode],
    nodes: list[architecture.Memory],
    remaining_choices: list,
):
    if not remaining_choices:
        yield mapping
        return

    for choice in list(remaining_choices):
        mapping.append(choice)
        remaining_choices.remove(choice)
        label_backing_storages(mapping)
        if valid_storage_order(mapping, [n.name for n in nodes]):
            yield from recursive_order_storage_choices(mapping, nodes, remaining_choices)
        mapping.pop()
        remaining_choices.append(choice)


def get_storage_choices(
    nodes: list[architecture.Memory],
    symbol_table: dict[str, InvertibleSet],
):
    while not isinstance(nodes[0], architecture.Memory):
        nodes = nodes[1:]
    first_storage = nodes[0]
    
    for choice, symbol_table in make_storage_choices_all_levels(nodes, symbol_table):
        all_storage_nodes = []
        for v in choice.values():
            all_storage_nodes.extend(v)
            
        base_mapping = []
        for node in list(all_storage_nodes[::-1]):
            if node.memory == first_storage.name:
                all_storage_nodes.remove(node)
                base_mapping.append(node)
            
        for mapping in recursive_order_storage_choices(base_mapping, nodes, all_storage_nodes):
            yield mapping, symbol_table

# =================================================================================================
# Insert loops
# =================================================================================================

def insert_temporal_loops(
    mapping: List[MappingNode],
    einsum: Einsum,
    first_memory: architecture.Memory,
):
    # First establish insertion points. Insertion points are:
    # - Below the last instance of the first memory
    # - Between any two storage nodes
    # - After the last storage node
    
    split_mapping = []
    for m in mapping:
        split_mapping.append([])
        split_mapping[-1].append(m)
        if m.memory == first_memory.name:
            while len(split_mapping) > 1:
                split_mapping[0].extend(split_mapping.pop(1))
    
    
    full_mapping = []
    seen_tensors = set()
    for i, prev in enumerate(split_mapping):
        full_mapping.extend(prev)
        cur = split_mapping[i+1] if i < len(split_mapping) - 1 else []

        rank_variables = set(einsum.rank_variables)
        seen_tensors |= set(t.tensor for t in prev)
        
        # If we haven't seen a tensor yet, must only iterate over relevant rank
        # variables.
        for t in einsum.tensors - seen_tensors:
            rank_variables &= einsum.tensor2rank_variables[t]

        
        # If there is a backing storage in the next block, include everything.
        if any(s._backing for s in cur):
            rank_variables = set(einsum.rank_variables)
            
        # Otherwise include loops that will get reuse for the previous block of
        # storage nodes
        else:
            rank_variables = set()
            for s in prev:
                rank_variables |= einsum.tensor2rank_variables[s.tensor]
                
        # Loops must also index into the next block of storage nodes.
        for s in cur:
            rank_variables &= einsum.tensor2rank_variables[s.tensor]
                
        # If there are any tensors we haven't seen yet, we may only iterate over
        # their relevant rank variables.
        for t in einsum.tensors - seen_tensors:
            rank_variables &= einsum.tensor2rank_variables[t]
                    
        if rank_variables:
            full_mapping.append(Temporal(rank_variable=rank_variables))

    full_mapping = list(full_mapping)
    return full_mapping

def insert_spatial_loops(
    mapping: List[MappingNode],
    einsum: Einsum,
    arch_flattened: list[architecture.Memory],
):
    nodes_with_fanout = [n for n in arch_flattened if n.spatial.get_fanout() > 1]
    arch_node_names = [n.name for n in arch_flattened]
    
    # Place spatials above the last instance of the first memory ABOVE each fanout
    for fanout in nodes_with_fanout:
        insertion_point = 0
        for i in range(len(mapping)):
            if not isinstance(mapping[i], Storage):
                continue
            memory_name = mapping[i].memory
            if arch_node_names.index(memory_name) < arch_node_names.index(fanout.name):
                insertion_point = i + 1

        rv = einsum.rank_variables
        if fanout.spatial.fanout_Y > 1:
            mapping.insert(
                insertion_point, 
                Spatial(rank_variable=rv, dimension="Y", across_object=fanout, across=fanout.name))
        if fanout.spatial.fanout_X > 1:
            mapping.insert(
                insertion_point, 
                Spatial(rank_variable=rv, dimension="X", across_object=fanout, across=fanout.name))

def unpack_loops_to_rank_variables(mapping: List[MappingNode]):
    mapping_new = []
    for node in mapping:
        if not isinstance(node, Iteration) or not isinstance(node.rank_variable, set):
            mapping_new.append(node)
            continue

        for r in sorted(node.rank_variable):
            mapping_new.append(
                type(node)(
                    rank_variable=r,
                    **node.model_dump(exclude={"rank_variable"}),
                )
            )
    return mapping_new

# =================================================================================================
# Iterate over mappings
# =================================================================================================
def iterate_mappings_no_constraints(
    spec: Specification,
    einsum_name: str,
    arch_flattened: list[architecture.Leaf],
):
    first_memory = None
    for node in arch_flattened:
        if isinstance(node, architecture.Memory):
            first_memory = node
            break
    if first_memory is None:
        raise ValueError("No memory found in architecture")

    symbol_table = spec.workload.get_constraint_symbol_table(einsum_name, spec.renames)
    einsum = spec.workload.einsums[einsum_name]
    for mapping, symbol_table in get_storage_choices(arch_flattened, symbol_table):
        mapping = insert_temporal_loops(mapping, einsum, first_memory)
        insert_spatial_loops(mapping, einsum, arch_flattened)
        mapping = unpack_loops_to_rank_variables(mapping)
        yield mapping, symbol_table

# =================================================================================================
# Attach constraints to mapping
# =================================================================================================
class TileSizeConstraintLambda:
    def __init__(
        self,
        constraint: Comparison,
        target_mapping_nodes: list[Storage],
        rank_variables: set[str],
    ):
        self.constraint = constraint
        self.constraint_lambda = constraint.to_constraint_lambda(True)
        self.target_mapping_nodes = target_mapping_nodes
        self.rank_variables = rank_variables

    def __call__(self, final: bool, sizes: np.ndarray) -> bool:
        return self.constraint(final, sizes)
    
class LoopBoundsConstraintLambda:
    def __init__(
        self,
        constraint: Comparison,
        target_mapping_nodes: list[Iteration],
    ):
        self.constraint = constraint
        self.constraint_lambda = constraint.to_constraint_lambda(True)
        self.target_mapping_nodes = target_mapping_nodes

    def __call__(self, final: bool, sizes: np.ndarray) -> bool:
        return self.constraint(final, sizes)

def get_constraints(
    mapping: List[MappingNode],
    symbol_table: dict[str, InvertibleSet],
):
    parsed = {}
    def get_parsed_storage_constraint(constraint: ConstraintGroup) -> ConstraintGroup:
        key = id(constraint.storage)
        if key not in parsed:
            parsed[key] = constraint.storage._parse_non_keep_bypass(symbol_table)
        return parsed[key]
    
    def get_parsed_spatial_constraint(constraint: ConstraintGroup, for_X: bool) -> ConstraintGroup:
        key = id((id(constraint), for_X))
        if key not in parsed:
            constraint = constraint.get_spatial_constraint(for_X=for_X, for_Y=not for_X)
            parsed[key] = constraint._parse(symbol_table)
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
    
    for m in mapping:
        if isinstance(m, Storage):
            constraint = get_parsed_storage_constraint(m.memory_object.constraints)
            for c in constraint.tile_shape:
                add_storage_constraint(m, c)

        for dim, for_x in [("X", True), ("Y", False)]:
            if isinstance(m, Spatial) and m.dimension == dim:
                constraint = get_parsed_spatial_constraint(m.across_object.constraints, for_x)
                for c in constraint.loop_bounds:
                    if m.rank_variable in c.expression:
                        add_loop_bounds_constraint(m, c)

    constraint_lambdas = []
    for constraint in tile_shape_constraints:
        mapping_nodes = tile_shape_constraint_id_to_mapping_nodes[id(constraint)]
        constraint_lambdas.append(TileSizeConstraintLambda(constraint, mapping_nodes, constraint.expression))

    for constraint in loop_bounds_constraints:
        mapping_nodes = loop_bounds_constraint_id_to_mapping_nodes[id(constraint)]
        if constraint.constrained_to_one():
            for m in mapping_nodes:
                mapping.remove(m)
            continue
        constraint_lambdas.append(LoopBoundsConstraintLambda(constraint, mapping_nodes))
    
    return constraint_lambdas

def iterate_mappings_constraints(
    spec: Specification,
    einsum_names: list[str] | str | None = None,
):
    arch_flattened = spec.get_flattened_architecture()
    compute_name = arch_flattened[-1].name

    if isinstance(einsum_names, str):
        einsum_names = [einsum_names]
    if einsum_names is None:
        einsum_names = [e.name for e in spec.workload.einsums]

    for einsum_name in einsum_names:
        for mapping, symbol_table in iterate_mappings_no_constraints(spec, einsum_name, arch_flattened):
            constraints = get_constraints(mapping, symbol_table)
            mapping.append(Compute(einsum=einsum_name, compute=compute_name))
            mapping = Mapping(nodes=mapping)
            yield mapping, constraints

# =================================================================================================
# Make sims
# =================================================================================================
def make_compatibility(mapping: Mapping, tile_shape: list[int], rank_variable_to_size: dict[RankVariableName, int]):
        loops = mapping.loops
        null_loops = []
        for i, t in enumerate(tile_shape):
            this_loop = loops[i]
            prev_size = rank_variable_to_size[this_loop.rank_variable]
            prev_loop = next(iter(l for l in loops[i-1::-1] if l.rank_variable == this_loop.rank_variable), None)
            if prev_loop is not None:
                prev_size = tile_shape[loops.index(prev_loop)]
            if prev_size == t:
                null_loops.append(this_loop)
        tile_shape = [t if i not in null_loops else None for i, t in enumerate(tile_shape)]
        
        compatibility = mapping.make_pmapping_compatibility(tile_shape)
        fused_loops = []
        reservations = {}
        for node in compatibility.nodes:
            if isinstance(node, Iteration):
                fused_loops.append(node)
            elif isinstance(node, Storage):
                reservations.setdefault(len(fused_loops), []).append(node)
            else:
                raise ValueError(f"Unexpected node type: {type(node)}")
            
        compatibility_loops = []
        for loop in fused_loops:
            loop = Loop(
                rank_variable_names=fzs(loop.rank_variable),
                bound=tile_shape[fused_loops.index(loop)],
                is_spatial=isinstance(loop, Spatial)
            )
            compatibility_loops.append(loop)
        compatibility_reservations = []
        for above_loop_index, storages in reservations.items():
            for storage in storages:
                for t in storage.tensors:
                    compatibility_reservations.append(
                        Reservation(
                            name=t,
                            above_loop_index=above_loop_index,
                            resource_name=storage.memory,
                            size=0 # TODO: Get size
                        )
                    )
        compatibility = Compatibility(
            loops=tuple(compatibility_loops),
            storage=fzs(compatibility_reservations)
        )
        return compatibility

def add_to_compatibility2sim(compatibility2sim: dict[Compatibility, SIM], sim: SIM):
    if sim.compatibility not in compatibility2sim:
        compatibility2sim[sim.compatibility] = sim
    prev = compatibility2sim[sim.compatibility]
    prev.mappings = Pareto.concat([prev.mappings, sim.mappings])

def make_sims(mapping: Mapping, explored_results: DataFrame, rank_variable_to_size: dict[RankVariableName, int], compatibility2sim: dict[Compatibility, SIM]):
    n_fused_loops = mapping.n_fused_loops
    
    # Replace columns with "tile_shape" with "__tile_shape"
    rename = lambda col: col.replace("tile_shape", "__tile_shape")
    explored_results.rename(columns=rename, inplace=True)
    
    fused_loop_columns = [f"__tile_shape{i}" for i in range(n_fused_loops)]

    for tile_shape, mappings in explored_results.groupby(fused_loop_columns):
        # Check for null loops
        compatibility = make_compatibility(mapping, tile_shape, rank_variable_to_size)
        mappings.drop(columns=fused_loop_columns, inplace=True)
        mappings[MAPPING_COLUMN] = [mapping] * len(mappings)
        sim = SIM(compatibility, Pareto(mappings))
        add_to_compatibility2sim(compatibility2sim, sim)

# =================================================================================================
# Top level
# =================================================================================================
def get_single_einsum_sims(
    spec: Specification,
    einsum_name: str,
    rank_variable_to_size: dict[RankVariableName, int],
) -> dict[Compatibility, SIM]:
    compatibility2sim = {}
    workload = spec.workload
    for i, (mapping, constraints) in enumerate(iterate_mappings_constraints(spec, einsum_name)):
        result = dummy_tile_shape_exploration(mapping, workload, constraints)
        make_sims(mapping, result, rank_variable_to_size, compatibility2sim)
    return compatibility2sim