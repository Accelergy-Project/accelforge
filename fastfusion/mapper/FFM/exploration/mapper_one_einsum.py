from collections import defaultdict
from collections.abc import Generator
import hashlib
from itertools import chain, combinations
import copy
import itertools
from typing import Any, Callable, List

from joblib import delayed
import numpy as np
from pandas import DataFrame
from tqdm import tqdm

from fastfusion.frontend.constraints import Comparison, ConstraintGroup, TileShapeConstraintLambda, LoopBoundsConstraintLambda
from fastfusion.frontend.mapping import Iteration, Mapping, MappingNode, Storage, Temporal, Spatial, Compute, ModelOnlyNode
from fastfusion.frontend.mapping import Reservation as ReservationNode
import fastfusion.frontend.architecture as architecture
from fastfusion.frontend.architecture import Leaf
from fastfusion.frontend.workload.isl import get_rank_variable_bounds
from fastfusion.mapper.FFM.exploration.tile_shape_exploration import explore_tile_shapes
from fastfusion.mapper.FFM.joining.mappinginfo import Compatibility, Loop, Reservation
from fastfusion.mapper.FFM.joining.sim import SIM
from fastfusion.mapper.FFM.pareto import TAGS_COLUMN, MAPPING_COLUMN, PartialMappings, col2nameloop, is_reservation_col, nameloop2col, tensor2col, DecompressData
from fastfusion.util.setexpressions import InvertibleSet
from fastfusion.frontend.specification import Specification
from fastfusion.frontend.workload.workload import Einsum, EinsumName, RankVariableName, TensorName
from fastfusion.util.util import fzs, parallel
from fastfusion.mapper.FFM.tags import Tags

MAX_N_LOOPS = 250

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
            tensors = set(s.tensors)
            s._backing = tensors - seen_tensors
            s._must_keep_tensors.extend(tensors - seen_tensors) # Backed tensors must be kept.
            seen_tensors.update(tensors)

def valid_storage_order(mapping: List[MappingNode], node_names: list[str], required_order: list[list[Storage]]):
    for i in range(len(mapping)):
        for j in range(i, len(mapping)):

            s1, s2 = mapping[i].memory, mapping[j].memory
            s1_idx, s2_idx = node_names.index(s1), node_names.index(s2)
            
            # Ensure order # TODO: FIXME. Moved this above the continue to
            # shrink the mapspace. This prevents local buffer from being above
            # global buffer.
            if i < j and s2_idx < s1_idx:
                return False
            
            if not (set(mapping[i].tensors) & set(mapping[j].tensors)):
                continue
            
            # If a tensor is stored in two levels back-to-back, then we
            # should have bypassed the outer storage if possible.
            if i == j or i == j - 1:
                if s1_idx < s2_idx and not ((set(mapping[i]._must_keep_tensors) & set(mapping[j].tensors)) or mapping[i]._backing):
                    return False
                if s2_idx < s1_idx and not ((set(mapping[j]._must_keep_tensors) & set(mapping[i].tensors)) or mapping[j]._backing):
                    return False
                
            
            for r in required_order:
                if mapping[i] in r and mapping[j] in r:
                    a, b = r.index(mapping[i]), r.index(mapping[j])
                    if a > b:
                        return False
    return True

def recursive_order_storage_choices(
    mapping: List[MappingNode],
    nodes: list[architecture.Memory],
    remaining_choices: list,
    required_order: list[list[Storage]],
) -> Generator[list[MappingNode], None, None]:
    mapping = list(mapping)
    if not remaining_choices:
        yield mapping
        return

    for choice in sorted(remaining_choices, key=lambda x: x.compact_string()):
        mapping.append(choice)
        new_remaining = [c for c in remaining_choices if c != choice]
        if valid_storage_order(mapping, [n.name for n in nodes], required_order):
            yield from recursive_order_storage_choices(mapping, nodes, new_remaining, required_order)
        mapping.pop()


def get_storage_choices(
    nodes: list[architecture.Memory],
    symbol_table: dict[str, InvertibleSet],
) -> Generator[tuple[list[Storage], Any], None, None]:
    while not isinstance(nodes[0], architecture.Memory):
        nodes = nodes[1:]
    first_storage = nodes[0]
    
    def pop_choice(storage_nodes: list[Storage], memory_name: str, tensor_name: TensorName):
        for i, node in enumerate(storage_nodes):
            if node.memory == memory_name and tensor_name in node.tensors:
                return storage_nodes.pop(i)
        return None
        
    
    for choice, symbol_table in make_storage_choices_all_levels(nodes, symbol_table):
        all_storage_nodes = []
        for v in choice.values():
            all_storage_nodes.extend(v)                            
            
        base_mapping = []
        for node in list(all_storage_nodes[::-1]):
            if node.memory == first_storage.name:
                all_storage_nodes.remove(node)
                base_mapping.append(node)
        
        required_order = []
        for node in nodes:
            # dataflow_constraints.append((node, node.constraints.dataflow._parse(symbol_table)))
            constraint = node.constraints.dataflow._parse(symbol_table)
            if constraint.storage_order:
                order = []
                for s in constraint.storage_order:
                    parent_storage = None
                    for tensor in s:
                        new_storage = pop_choice(all_storage_nodes, node.name, tensor)
                        if new_storage is None:
                            continue
                        if parent_storage is None:
                            parent_storage = new_storage
                        else:
                            parent_storage.tensors += new_storage.tensors
                            parent_storage._must_keep_tensors += new_storage._must_keep_tensors
                    if parent_storage is not None:
                        all_storage_nodes.append(parent_storage)
                        order.append(parent_storage)
                required_order.append(order)

        for mapping in recursive_order_storage_choices(base_mapping, nodes, all_storage_nodes, required_order):
            yield mapping, symbol_table

# =================================================================================================
# Insert loops
# =================================================================================================

def insert_temporal_loops(
    mapping: List[MappingNode],
    einsum: Einsum,
    first_memory: architecture.Memory,
    rank_variable_bounds: dict[RankVariableName, int],
):
    # First establish insertion points. Insertion points are:
    # - Below the last instance of the first memory
    # - Between any two storage nodes
    # - After the last storage node
    
    split_mapping = [[]]
    for m in mapping:
        split_mapping.append([])
        split_mapping[-1].append(m)
        if m.memory == first_memory.name:
            while len(split_mapping) > 1:
                split_mapping[0].extend(split_mapping.pop(1))
    
    
    full_mapping = []
    seen_tensors = set()
    for i, prev_storages in enumerate(split_mapping):
        full_mapping.extend(prev_storages)
        next_storages = split_mapping[i+1] if i < len(split_mapping) - 1 else []
        assert len(next_storages) <= 1

        rank_variables = einsum.rank_variables
        rank_variables = {r for r in rank_variables if rank_variable_bounds[r] > 1}
        seen_tensors |= set.union(*(set(t.tensors) for t in prev_storages), set())
        
        # If we haven't seen a tensor yet, must only iterate over relevant rank
        # variables.
        for t in einsum.tensors - seen_tensors:
            rank_variables &= einsum.tensor2rank_variables[t]

        # If there is no backing storage in the next block, only include loops
        # that reuse tensors in the previous block.
        if not any(s._backing for s in next_storages):
            prev_relevant = [einsum.tensor2rank_variables[t] for s in prev_storages for t in s.tensors]
            if prev_relevant:
                rank_variables -= set.intersection(*prev_relevant)

        # Only include loops that will index into the next block of storage nodes.
        if next_storages:
            next_relevant = [einsum.tensor2rank_variables[t] for s in next_storages for t in s.tensors]
            rank_variables &= set.union(*next_relevant, set())

        # if i == len(split_mapping) - 1:
        #     rank_variables = set(einsum.rank_variables)

        if rank_variables:
            full_mapping.append(Temporal(rank_variable=rank_variables, tile_shape='symbol'))

    full_mapping = list(full_mapping)

    return full_mapping

def insert_spatial_loops(
    mapping: List[MappingNode],
    einsum: Einsum,
    arch_flattened: list[architecture.Memory],
    rank_variable_bounds: dict[RankVariableName, int],
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
        rv = {r for r in rv if rank_variable_bounds[r] > 1}
        for fanout_dim, fanout_size in fanout.spatial.fanout.items():
            mapping.insert(
                insertion_point, 
                Spatial(rank_variable=rv, dimension=fanout_dim, across_object=fanout, across=fanout.name, tile_shape='symbol'))

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

def label_fused_loops(mapping: List[MappingNode]):
    last_backing_storage = None
    for i, node in enumerate(mapping):
        if isinstance(node, Storage) and node._backing:
            last_backing_storage = i
    if last_backing_storage is None:
        raise ValueError(f"No backing storage found in mapping {", ".join(m.compact_string() for m in mapping)}")

    for i, node in enumerate(mapping):
        if isinstance(node, Iteration):
            node._fused = i < last_backing_storage
    return mapping

# =================================================================================================
# Iterate over mappings
# =================================================================================================
def temporal_fused_constraint_thing_fix_me(mapping: List[MappingNode], rank_variables: list[RankVariableName]):
    # Only one fused loop is allowed per rank variable
    rank_variables = list(rank_variables)
    if not rank_variables:
        yield mapping
        return

    my_rank_variable = RankVariableName(rank_variables.pop())
    # indent = " " * (10 - len(rank_variables))
    fused_loops = [i for i, node in enumerate(mapping) if isinstance(node, Iteration) and node._fused and my_rank_variable == node.rank_variable]
    
    if not fused_loops or len(fused_loops) == 1:
        # print(indent + f"Yielding for rank variable {my_rank_variable}. Length: {len(mapping)}")
        # print(indent + ", ".join(m.compact_string() for m in mapping))
        yield from temporal_fused_constraint_thing_fix_me(mapping, rank_variables)
        return
    
    for choice in fused_loops:
        mapping_new = list(mapping)
        for f in fused_loops[::-1]:
            if f != choice:
                mapping_new.pop(f)
        # print(indent + f"Yielding for rank variable {my_rank_variable}. Length: {len(mapping_new)}")
        # print(indent + ", ".join(m.compact_string() for m in mapping_new))
        yield from temporal_fused_constraint_thing_fix_me(mapping_new, rank_variables)

def temporal_constraint_2_fix_me(mapping: List[MappingNode], einsum: Einsum):
    return mapping
    # Between two storage nodes for the same memory, pop all loops that index
    # into the tensors for both storage nodes.
    # return mapping
    to_pop = set()
    for i, node in enumerate(mapping):
        for j, node2 in enumerate(mapping):
            if i >= j:
                continue
            if not isinstance(node, Storage) or not isinstance(node2, Storage):
                continue
            if node.memory != node2.memory:
                continue
            rv1 = set.union(*(einsum.tensor2rank_variables[t] for t in node.tensors))
            rv2 = set.union(*(einsum.tensor2rank_variables[t] for t in node2.tensors))
            to_drop = rv1 & rv2
            if not to_drop:
                continue
            for k in range(i+1, j):
                if isinstance(mapping[k], Temporal) and mapping[k].rank_variable in to_drop:
                    to_pop.add(k)
    return [node for i, node in enumerate(mapping) if i not in to_pop]

def place_missing_temporal_loops(mapping: List[MappingNode], einsum: Einsum, rank_variable_bounds: dict[RankVariableName, int]):
    # If any rank variables are missing, add them as high as possible.
    rank_variables = einsum.rank_variables
    rank_variables = {r for r in rank_variables if rank_variable_bounds[r] > 1}
    for m in mapping:
        if isinstance(m, Temporal) and not m._fused:
            rank_variables.discard(m.rank_variable)
            
    # insert_point = 0
    # while insert_point < len(mapping) and not isinstance(mapping[insert_point], Temporal):
    #     insert_point += 1
    # Insert point: Right under the last backing storage
    for i in range(len(mapping)-1, -1, -1):
        if isinstance(mapping[i], Storage) and mapping[i]._backing:
            insert_point = i + 1
            break

    temporals = [
        Temporal(rank_variable=r, tile_shape='symbol')
        for r in sorted(rank_variables)
    ]

    if insert_point == len(mapping):
        mapping.extend(temporals)
    else:
        for t in temporals:
            mapping.insert(insert_point, t)
    # mapping.extend(copy.deepcopy(temporals))
    
    seen = set()
    # for i in range(len(mapping) - 1, -1, -1):
    #     if isinstance(mapping[i], Storage):
    #         if mapping[i]._backing:
    #             break
    #         seen = set()
    #     if isinstance(mapping[i], Temporal):
    #         if mapping[i].rank_variable in seen:
    #             mapping.pop(i)
    #         else:
    #             seen.add(mapping[i].rank_variable)
    
def iterate_mappings_n_loops_constraint(mapping: Mapping, einsum: Einsum):
    n_loops = sum(isinstance(m, Iteration) for m in mapping.nodes)
    n_to_drop = n_loops - MAX_N_LOOPS
    if n_to_drop <= 0:
        yield mapping
        return
    
    rank_variables = einsum.rank_variables

    index2iteration = [i for i, node in enumerate(mapping.nodes) if isinstance(node, Iteration)]

    # Don't drop the innermost loop of any rank variable
    need_rank_variables = set(rank_variables)
    for i, node in list(enumerate(mapping.nodes))[::-1]:
        if isinstance(node, Iteration) and node.rank_variable in need_rank_variables:
            need_rank_variables.discard(node.rank_variable)
            index2iteration.remove(i)
        
    assert not need_rank_variables
    
    for choices in itertools.combinations(index2iteration, n_to_drop):
        mapping_new = [m for i, m in enumerate(mapping.nodes) if i not in choices]
        yield Mapping(nodes=mapping_new)

def iterate_mappings_no_constraints(
    spec: Specification,
    einsum_name: str,
    arch_flattened: list[architecture.Leaf],
    rank_variable_bounds: dict[RankVariableName, int],
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
        mapping = copy.deepcopy(mapping)
        label_backing_storages(mapping)
        # print(", ".join(m.compact_string() for m in mapping))
        mapping = insert_temporal_loops(mapping, einsum, first_memory, rank_variable_bounds)
        # print(", ".join(m.compact_string() for m in mapping))
        insert_spatial_loops(mapping, einsum, arch_flattened, rank_variable_bounds)
        # print(", ".join(m.compact_string() for m in mapping))
        mapping = unpack_loops_to_rank_variables(mapping)
        label_fused_loops(mapping)
        # print(", ".join(m.compact_string() for m in mapping))
        for mapping2 in temporal_fused_constraint_thing_fix_me(mapping, list(spec.workload.einsums[einsum_name].rank_variables)): # TODO
            mapping2 = temporal_constraint_2_fix_me(mapping2, einsum)
            place_missing_temporal_loops(mapping2, einsum, rank_variable_bounds)
            yield mapping2, symbol_table

# =================================================================================================
# Attach constraints to mapping
# =================================================================================================
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
    
    def get_parsed_spatial_constraint(constraint: ConstraintGroup, dimension: str) -> ConstraintGroup:
        key = (id(constraint), dimension)
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

    constraint_lambdas = []
    for constraint in tile_shape_constraints:
        mapping_nodes = tile_shape_constraint_id_to_mapping_nodes[id(constraint)]
        constraint_lambdas.append(TileShapeConstraintLambda(constraint, mapping_nodes, constraint.expression))

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
    arch_flattened: list[architecture.Leaf] | None = None,
    rank_variable_bounds: dict[RankVariableName, int] | None = None,
):
    if arch_flattened is None:
        arch_flattened = spec.get_flattened_architecture()
    compute_name = arch_flattened[-1].name

    if isinstance(einsum_names, str):
        einsum_names = [einsum_names]
    if einsum_names is None:
        einsum_names = [e.name for e in spec.workload.einsums]
        
    if rank_variable_bounds is None:
        rank_variable_bounds = get_rank_variable_bounds(spec, einsum_names)

    for einsum_name in einsum_names:
        for mapping, symbol_table in iterate_mappings_no_constraints(spec, einsum_name, arch_flattened, rank_variable_bounds):
            constraints = get_constraints(mapping, symbol_table)
            mapping.append(Compute(einsum=einsum_name, compute=compute_name))
            # mapping = copy.copy(mapping)
            mapping = Mapping(nodes=mapping)
            # yield mapping, constraints
            for mapping2 in iterate_mappings_n_loops_constraint(mapping, spec.workload.einsums[einsum_name]):
                yield Mapping(nodes=[copy.copy(n) for n in mapping2.nodes]), symbol_table

# =================================================================================================
# Make sims
# =================================================================================================
def make_compatibility(
    mapping: Mapping,
    intermediate_tensors: set[TensorName],
) -> Compatibility:
    fused_slice = mapping.get_fused_slice(intermediate_tensors)
    fused_loops = []
    reservations: dict[int, list[ReservationNode]] = {}
    for node in fused_slice.nodes:
        if isinstance(node, Iteration):
            fused_loops.append(node)
        elif isinstance(node, ReservationNode):
            reservations.setdefault(len(fused_loops), []).append(node)
        elif isinstance(node, ModelOnlyNode):
            continue
        elif isinstance(node, Storage):
            continue
        else:
            raise ValueError(f"Unexpected node type: {type(node)}")

    compatibility_loops = []
    for loop in fused_loops:
        loop = Loop(
            rank_variable_names=fzs((loop.rank_variable,)),
            bound=0, # Populated later
            is_spatial=isinstance(loop, Spatial)
        )
        compatibility_loops.append(loop)
    compatibility_reservations = []
    for above_loop_index, reservation_nodes in reservations.items():
        for reservation in reservation_nodes:
            compatibility_reservations.append(
                Reservation(
                    name=reservation.tensor,
                    above_loop_index=above_loop_index,
                    resource_name=reservation.memory,
                    size=0 # TODO: Get size
                )
            )

    compatibility = Compatibility(
        loops=tuple(compatibility_loops),
        storage=fzs(compatibility_reservations),
    )
    return compatibility


def add_to_compatibility2sim(compatibility2sim: dict[Compatibility, SIM], sim: SIM):
    if sim.compatibility not in compatibility2sim:
        compatibility2sim[sim.compatibility] = sim
    prev = compatibility2sim[sim.compatibility]

    # Make columns in previous and current SIM match (by adding empty column)
    # if not a reservation column.
    for col in prev.mappings.data.columns:
        if col not in sim.mappings.data.columns:
            if not is_reservation_col(col):
                sim.mappings.data[col] = 0
    for col in sim.mappings.data.columns:
        if col not in prev.mappings.data.columns:
            if not is_reservation_col(col):
                prev.mappings.data[col] = 0
    prev.mappings = PartialMappings.concat([prev.mappings, sim.mappings])
    
def get_equivalent_sims(sim: SIM, tagger: Callable[[Mapping], Tags]) -> list[SIM]:
    equivalent_permutations = sim.compatibility.make_equivalent_permutations()
    result = []
    for c in equivalent_permutations:
        tags = Tags() if tagger is None else tagger(c)
        if not tags.matches(Tags(("INVALID",))):
            result.append(SIM(c.update(tags=tags), sim.mappings))
    return result

def get_compatibility_loops(mapping: Mapping, tile_shapes: list[int]) -> "Mapping":
    compatibility = Mapping(nodes=[])
    i = 0
    for node in mapping.nodes:
        while i < len(tile_shapes) and tile_shapes[i] is None:
            i += 1
        if i >= len(tile_shapes):
            break
        new_node = copy.deepcopy(node)
        if isinstance(node, Iteration):
            new_node.tile_shape = tile_shapes[i]
            i += 1
        compatibility.nodes.append(new_node)
    return compatibility

def drop_cols(mappings: DataFrame):
    from fastfusion.mapper.FFM.pareto import col2nameloop
    to_drop = []
    for col in mappings.columns:
        if col2nameloop(col) is None:
            continue
        name, loop_index = col2nameloop(col)
        if name == "LocalBuffer" or name == "Register" or name == "MainMemory":
            to_drop.append(col)
    return mappings.drop(columns=to_drop)

def shift_reservations_by_null_loop_indices(mappings: DataFrame, null_loop_indices: set[int]):
    prev = copy.deepcopy(mappings) # TODO: Is this needed?
    target2newabovename = {}
    dropcols = []
    for c in mappings.columns:
        if not is_reservation_col(c):
            continue
        name, above = col2nameloop(c)
        new_above = above - sum(above > i for i in null_loop_indices)
        target = nameloop2col(name, new_above)
        if target in target2newabovename:
            if above > target2newabovename[target][1]:
                dropcols.append(nameloop2col(*target2newabovename[target]))
                target2newabovename[target] = (name, above)
            else:
                dropcols.append(c)
        else:
            target2newabovename[target] = (name, above)

    mappings.drop(columns=dropcols, inplace=True)
    renames = {}
    for target, (name, above) in target2newabovename.items():
        renames[nameloop2col(name, above)] = target
    mappings.rename(columns=renames, inplace=True)
    if len(mappings.columns) != len(mappings.columns.unique()):
        shift_reservations_by_null_loop_indices(prev, null_loop_indices)
        raise ValueError(f"Duplicate columns: {mappings.columns}")
    assert len(mappings.columns) == len(mappings.columns.unique())
    return mappings

# def matches_storage_order(mapping: Mapping, storage_order: list[str]):
#     found = [s.tensor for s in mapping.nodes if isinstance(s, Storage)]
#     return len(found) >= len(storage_order) and all(s1 == s2 for s1, s2 in zip(found, storage_order))

# def has_tensors(mapping: Mapping, tensors: list[TensorName]):
#     found = set(s.tensor for s in mapping.nodes if isinstance(s, Storage))
#     return found >= set(tensors)

def make_sims(
        mapping: Mapping,
        explored_results: DataFrame,
        rank_variable_bounds: dict[RankVariableName, int],
        intermediate_tensors: set[TensorName],
        tagger: Callable[[Mapping], Tags] =  None,
        total_pmappings: int = None
    ):    
    if explored_results.empty:
        return {}
    compatibility = make_compatibility(mapping, intermediate_tensors)
    # print(compatibility)
    # if len(compatibility.loops) == 0:
    #     print(compatibility)
    # if compatibility.tags.matches(Tags(("INVALID",))):
    #     return {}

    fused_loop_columns = [f"__tile_shape{i}" for i in range(len(compatibility.loops))]
        
    explored_results = drop_cols(explored_results)
        
    if fused_loop_columns:
        groups = list(explored_results.groupby(fused_loop_columns))
    else:
        groups = [((), explored_results)]
        
    pmappings_per_group = None if total_pmappings is None else total_pmappings / len(groups)

    sims = []

    for tile_shape, mappings in groups:
        tensor2size = {}
    
        dropcols = []
        for tensor in intermediate_tensors: # Sizes are all the same
            tensor2size[tensor] = mappings[tensor2col(tensor)].iloc[0]
            dropcols.append(tensor2col(tensor))
        mappings.drop(columns=dropcols, inplace=True)
        
        new_compatibility, null_loop_indices = compatibility.populate_tile_shape(tile_shape, rank_variable_bounds, tensor2size)

        tags = Tags() if tagger is None else tagger(new_compatibility)
        if tags.matches(Tags(("INVALID",))):
            continue
        new_compatibility = new_compatibility.update(tags=tags)

        shift_reservations_by_null_loop_indices(mappings, null_loop_indices)
        partial_mappings = PartialMappings(mappings, free_to_loop_index=len(new_compatibility.loops), n_pmappings=pmappings_per_group, skip_pareto=len(mappings) < 1000)
        sim = SIM(new_compatibility, partial_mappings)
        sim.mappings.data[TAGS_COLUMN] = [compatibility.tags] * len(sim.mappings.data)
        sims.append(sim)

    new_sims = []
    for sim in sims:
        for equivalent_sim in get_equivalent_sims(sim, tagger):
            new_sims.append(equivalent_sim)
    return new_sims

# =================================================================================================
# Top level
# =================================================================================================
def _per_proc_compatibility2sim(
    mapping: Mapping,
    constraints: list[Comparison],
    spec: Specification,
    rank_variable_bounds: dict[RankVariableName, int],
    intermediate_tensors: set[TensorName],
    flattened_arch: list[architecture.Leaf],
    einsum_name: EinsumName,
    job_id: int,
    tagger=None,
) -> tuple[str, dict[Compatibility, SIM], str, Mapping]:
    # print(f", ".join(m.compact_string() for m in mapping.nodes))
    result, total_pmappings = explore_tile_shapes(mapping, constraints, spec, flattened_arch)
    sims = make_sims(mapping, result, rank_variable_bounds, intermediate_tensors, tagger=tagger, total_pmappings=total_pmappings)
    decompress_data = PartialMappings.compress_paretos(
        einsum_name, 
        [s.mappings for s in sims],
        job_id=job_id,
        extra_data={MAPPING_COLUMN: mapping}
    )
    return einsum_name, sims, decompress_data, job_id

def get_single_einsum_jobs(
    spec: Specification,
    einsum_name: EinsumName,
    rank_variable_bounds: dict[RankVariableName, int] | None = None,
    flattened_arch: list[architecture.Leaf] | None = None,
    tagger: Callable[[Mapping], Tags] | None = None,
    start_index: int = 0,
) -> list[SIM] | tuple[dict[EinsumName, dict[Compatibility, list[SIM]]], DecompressData]:
    einsum_name = EinsumName(einsum_name)

    if rank_variable_bounds is None:
        rank_variable_bounds = get_rank_variable_bounds(spec.workload, einsum_name)
    
    workload = spec.workload
    intermediate_tensors = workload.intermediate_tensors() & workload.einsums[einsum_name].tensor_names

    if flattened_arch is None:
        flattened_arch = spec.get_flattened_architecture()

    mappings_constraints = tqdm(iterate_mappings_constraints(spec,
                                einsum_name,
                                flattened_arch,
                                rank_variable_bounds,
                                ),
                                desc=f"Generating storage and loop choices for Einsum {einsum_name}")

    return  [
        delayed(_per_proc_compatibility2sim)(
            mapping=mapping,
            constraints=constraints,
            spec=spec,
            rank_variable_bounds=rank_variable_bounds,
            intermediate_tensors=intermediate_tensors,
            flattened_arch=flattened_arch,
            einsum_name=einsum_name,
            tagger=tagger,
            job_id=start_index + i,
       )
        for i, (mapping, constraints) in enumerate(mappings_constraints)
    ]
