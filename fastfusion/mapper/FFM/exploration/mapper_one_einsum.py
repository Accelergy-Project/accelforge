from collections import defaultdict
from itertools import chain, combinations
import copy
import itertools
from typing import Callable, List

from joblib import delayed
import numpy as np
from pandas import DataFrame
from tqdm import tqdm

from fastfusion.frontend import constraints
from fastfusion.frontend.constraints import Comparison, ConstraintGroup, TileShapeConstraintLambda, LoopBoundsConstraintLambda
from fastfusion.frontend.mapping import Iteration, Mapping, MappingNode, Storage, Temporal, Spatial, Compute, ModelOnlyNode
import fastfusion.frontend.architecture as architecture
from fastfusion.frontend.architecture import Leaf
from fastfusion.frontend.workload.isl import get_rank_variable_bounds
from fastfusion.mapper.FFM.exploration.tile_shape_exploration import explore_tile_shapes
from fastfusion.mapper.FFM.joining.mappinginfo import Compatibility, Loop, Reservation
from fastfusion.mapper.FFM.joining.sim import SIM
from fastfusion.mapper.FFM.joining.simexplore import compress_sims
from fastfusion.mapper.FFM.pareto import MAPPING_COLUMN, PartialMappings, is_reservation_col
from fastfusion.util.setexpressions import InvertibleSet
from fastfusion.frontend.specification import Specification
from fastfusion.frontend.workload.workload import Einsum, EinsumName, RankVariableName, TensorName, Workload
from fastfusion.util.util import fzs, parallel
from fastfusion.mapper.FFM.tags import Tags

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

def _rename_columns_we_should_fix_this_later(df):
    # tile_shape --> __tile_shape
    def rename(col: str):
        if col.startswith("tile_shape"):
            col = col.replace("tile_shape", "__tile_shape")
        if col == "energy" or col == "latency":
            col = f"metric_{col}"
        return col
    df.rename(columns=rename, inplace=True)
    return df

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
):
    if not remaining_choices:
        yield mapping
        return

    for choice in list(remaining_choices):
        mapping.append(choice)
        remaining_choices.remove(choice)
        if valid_storage_order(mapping, [n.name for n in nodes], required_order):
            yield from recursive_order_storage_choices(mapping, nodes, remaining_choices, required_order)
        mapping.pop()
        remaining_choices.append(choice)


def get_storage_choices(
    nodes: list[architecture.Memory],
    symbol_table: dict[str, InvertibleSet],
):
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
        seen_tensors |= set.union(*(set(t.tensors) for t in prev), set())
        
        # If we haven't seen a tensor yet, must only iterate over relevant rank
        # variables.
        for t in einsum.tensors - seen_tensors:
            rank_variables &= einsum.tensor2rank_variables[t]

        
        rank_variables = set(einsum.rank_variables)
        
        # If there is no backing storage in the next block, only include loops
        # that reuse tensors in the previous block.
        if not any(s._backing for s in cur):
            prev_relevant = [einsum.tensor2rank_variables[t] for s in prev for t in s.tensors]
            if prev_relevant:
                rank_variables -= set.intersection(*prev_relevant)

        # Only include loops that will index into the next block of storage nodes.
        if cur:
            next_relevant = [einsum.tensor2rank_variables[t] for s in cur for t in s.tensors]
            rank_variables &= set.union(*next_relevant, set())
            
        # If there are any tensors we haven't seen yet, we may only iterate over
        # their relevant rank variables.
        for t in einsum.tensors - seen_tensors:
            rank_variables &= einsum.tensor2rank_variables[t]

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

def place_missing_temporal_loops(mapping: List[MappingNode], einsum: Einsum):
    # If any rank variables are missing, add them as high as possible.
    rank_variables = set(einsum.rank_variables)
    for m in mapping:
        if isinstance(m, Temporal):
            rank_variables.discard(m.rank_variable)
            
    insert_point = 0
    while insert_point < len(mapping) and not isinstance(mapping[insert_point], Temporal):
        insert_point += 1

    temporals = [
        Temporal(rank_variable=r, tile_shape='symbol')
        for r in rank_variables
    ]

    if insert_point == len(mapping):
        mapping.extend(temporals)
    else:
        for t in temporals:
            mapping.insert(insert_point, t)

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
        label_backing_storages(mapping)
        # print(", ".join(m.compact_string() for m in mapping))
        mapping = insert_temporal_loops(mapping, einsum, first_memory)
        # print(", ".join(m.compact_string() for m in mapping))
        insert_spatial_loops(mapping, einsum, arch_flattened)
        # print(", ".join(m.compact_string() for m in mapping))
        mapping = unpack_loops_to_rank_variables(mapping)
        label_fused_loops(mapping)
        # print(", ".join(m.compact_string() for m in mapping))
        for mapping2 in temporal_fused_constraint_thing_fix_me(mapping, list(spec.workload.einsums[einsum_name].rank_variables)): # TODO
            mapping2 = temporal_constraint_2_fix_me(mapping2, einsum)
            place_missing_temporal_loops(mapping2, einsum)
            yield copy.deepcopy(mapping2), symbol_table

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
):
    if arch_flattened is None:
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
            mapping = copy.copy(mapping)
            mapping = Mapping(nodes=mapping)
            number_of_loops = sum(isinstance(m, Iteration) for m in mapping.nodes)
            # print(f"Number of loops: {number_of_loops}")
            # print(", ".join(m.compact_string() for m in mapping.nodes))
            yield mapping, constraints

# =================================================================================================
# Make sims
# =================================================================================================
def make_compatibility(
    mapping: Mapping,
    intermediate_tensors: set[TensorName],
    tagger: Callable[[Mapping, set[TensorName]], Tags] | None = None
) -> Compatibility:
    compatibility = mapping.get_fused_slice(intermediate_tensors)
    fused_loops = []
    reservations = {}
    for node in compatibility.nodes:
        if isinstance(node, Iteration):
            fused_loops.append(node)
        elif isinstance(node, Storage):
            reservations.setdefault(len(fused_loops), []).append(node)
        elif isinstance(node, ModelOnlyNode):
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

    if tagger is None:
        tags = Tags()
    else:
        tags = tagger(mapping, intermediate_tensors)

    compatibility = Compatibility(
        loops=tuple(compatibility_loops),
        storage=fzs(compatibility_reservations),
        tags=tags,
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
    
def get_equivalent_sims(sim: SIM) -> list[SIM]:
    equivalent_permutations = sim.compatibility.make_equivalent_permutations()
    return [SIM(c, sim.mappings) for c in equivalent_permutations]

    
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

def make_sims(mapping: Mapping,
              explored_results: DataFrame,
              rank_variable_bounds: dict[RankVariableName, int],
              intermediate_tensors: set[TensorName], tagger=None):
    compatibility = make_compatibility(mapping, intermediate_tensors, tagger=tagger)
    fused_loop_columns = [f"__tile_shape{i}" for i in range(len(compatibility.loops))]
        
    explored_results = drop_cols(explored_results)
        
    if fused_loop_columns:
        groups = explored_results.groupby(fused_loop_columns)
    else:
        groups = [((), explored_results)]
    compatibility2sim = {}

    for tile_shape, mappings in groups: #tqdm(groups, desc="Generating SIMs"):
        # Check for null loops
        new_compatibility = compatibility.populate_tile_shape(tile_shape, rank_variable_bounds)
        # mappings.drop(columns=fused_loop_columns, inplace=True)
        sim = SIM(new_compatibility, PartialMappings(mappings, free_to_loop_index=len(new_compatibility.loops) - 1))
        sim.mappings.data[MAPPING_COLUMN] = [mapping] * len(sim.mappings.data)
        for equivalent_sim in get_equivalent_sims(sim):
            compatibility2sim.setdefault(equivalent_sim.compatibility, []).append(equivalent_sim)
    
    return compatibility2sim

# =================================================================================================
# Top level
# =================================================================================================
def _concat_sims(einsum_name: EinsumName, sims: list[SIM]):
    return einsum_name, SIM.concat(sims)

def concat_sims(sims: dict[EinsumName, dict[Compatibility, list[SIM]]]):
    to_compress = {
        einsum_name: list(itertools.chain.from_iterable(compatibility2sim.values()))
        for einsum_name, compatibility2sim in sims.items()
    }
    decompress_data = compress_sims(to_compress)
    to_pack = [
        delayed(_concat_sims)(einsum_name, compatibility2sim)
        for einsum_name, compatibility2sim in sims.items()
        for compatibility2sim in compatibility2sim.values()
    ]
    sims = {}
    for einsum_name, sim in parallel(to_pack, pbar="Concatenating SIMs"):
        sims.setdefault(einsum_name, []).append(sim)
    return sims, decompress_data

def _per_proc_compatibility2sim(
    mapping: Mapping,
    constraints: list[Comparison],
    specification: Specification,
    rank_variable_bounds: dict[RankVariableName, int],
    intermediate_tensors: set[TensorName],
    flattend_arch: list[architecture.Leaf],
    einsum_name: EinsumName,
    tagger=None,
) -> tuple[str, dict[Compatibility, SIM]]:
    print(", ".join(m.compact_string() for m in mapping.nodes))
    result = explore_tile_shapes(mapping, constraints, specification, flattend_arch)
    return einsum_name, make_sims(mapping, result, rank_variable_bounds, intermediate_tensors, tagger=tagger)

def get_single_einsum_sims(
    spec: Specification,
    einsum_name: EinsumName,
    rank_variable_bounds: dict[RankVariableName, int] | None = None,
    flattened_arch: list[architecture.Leaf] | None = None,
    return_jobs: bool = False,
    tagger: Callable[[Mapping, set[TensorName]], Tags] | None = None,
) -> list[SIM] | list[Callable[[], tuple[str, list[SIM]]]]:
    einsum_name = EinsumName(einsum_name)
    
    if rank_variable_bounds is None:
        rank_variable_bounds = get_rank_variable_bounds(spec.workload, einsum_name)
    
    compatibility2sim = {}
    workload = spec.workload
    intermediate_tensors = workload.intermediate_tensors()

    if flattened_arch is None:
        flattened_arch = spec.get_flattened_architecture()

    mappings_constraints = list(iterate_mappings_constraints(spec,
                                                             einsum_name,
                                                             flattened_arch))
    
    jobs = [
        delayed(_per_proc_compatibility2sim)(
            mapping,
            constraints,
            spec,
            rank_variable_bounds,
            intermediate_tensors,
            flattened_arch,
            einsum_name,
            tagger
       )
        for mapping, constraints in mappings_constraints
    ]
    
    if return_jobs:
        return jobs
    
    sims = {}
    for einsum_name, new_sims in parallel(
        jobs,
        pbar="Generating SIMs",
        return_as="generator_unordered"
    ):
        for compatibility, ns in new_sims.items():
            sims.setdefault(compatibility, []).extend(ns)
                        
    return concat_sims({einsum_name: sims})


    # compatibility2sim = {}
    # for _, sims in parallel(
    #     jobs,
    #     pbar=f"Generating pmappings for Einsum {einsum_name}",
    #     return_as="generator"
    # ):
    #     pass
    #     # for sim in sims:
    #     #     compatibility2sim.setdefault(sim.compatibility, []).append(sim)
            
    # return list(compatibility2sim.values())
