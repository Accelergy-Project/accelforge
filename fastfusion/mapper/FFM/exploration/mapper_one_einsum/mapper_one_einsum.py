import copy
import itertools
from collections import defaultdict
from collections.abc import Sequence
from numbers import Number
from typing import Callable, Iterator, List

from joblib import delayed
from pandas import DataFrame
from tqdm import tqdm

import fastfusion.frontend.architecture as architecture
from fastfusion.frontend.mapping import Compute, Iteration, Mapping, MappingNode, ModelOnlyNode, Storage, Temporal, Spatial
from fastfusion.frontend.specification import Specification
from fastfusion.frontend.workload.isl import get_rank_variable_bounds
from fastfusion.frontend.workload.workload import Einsum, EinsumName, RankVariableName, TensorName, Workload

from fastfusion.mapper.FFM.exploration import metrics
from fastfusion.mapper.FFM.exploration.mapper_one_einsum.dataflow_generator import get_storage_choices
from fastfusion.mapper.FFM.exploration.tile_shape_exploration import explore_tile_shapes
from fastfusion.mapper.FFM.joining.mappinginfo import Compatibility, Loop, Reservation, TilePattern
from fastfusion.mapper.FFM.joining.sim import SIM
from fastfusion.mapper.FFM.pareto import TAGS_COLUMN, MAPPING_COLUMN, PartialMappings, col2nameloop, is_reservation_col, nameloop2col, tensor2col, DecompressData
from fastfusion.mapper.FFM.exploration.contraints.constraints import MappingConstraints, get_constraints
from fastfusion.mapper.FFM.tags import Tags
from fastfusion.util.util import fzs
from fastfusion.frontend.mapping import Reservation as ReservationNode



MAX_N_LOOPS = 250

# =================================================================================================
# Insert loops
# =================================================================================================

def insert_temporal_loops(
    mapping: List[MappingNode],
    einsum: Einsum,
    first_memory: architecture.Memory,
    rank_variable_bounds: dict[RankVariableName, int],
    ranks_with_tile_pattern: set,
    workload: Workload,
):
    # First establish insertion points. Insertion points are:
    # - Below the last instance of the first memory
    # - Between any two storage nodes
    # - After the last storage node

    split_mapping: list[list[Storage]] = [[]]
    for m in mapping:
        split_mapping.append([])
        split_mapping[-1].append(m)
        if m.memory == first_memory.name:
            while len(split_mapping) > 1:
                split_mapping[0].extend(split_mapping.pop(1))

    # These Einsum properties are recalculated since Einsum is mutable
    # We're pre-computing and reusing for efficiency
    tensor2fully_relevant_rank_vars = einsum.tensor2fully_relevant_rank_variables
    tensor2partially_relevant_rank_vars = einsum.tensor2partially_relevant_rank_variables
    tensor2rank_vars = einsum.tensor2rank_variables

    intermediate_tensors = einsum.tensors & workload.intermediate_tensors
    is_fused_loops = True
    accumulative_prev_relevant = set()
    seen_tensors = set()
    choices = []
    for i, prev_storages in enumerate(split_mapping):
        # full_mapping.extend(prev_storages)
        next_storages = split_mapping[i+1] if i < len(split_mapping) - 1 else []
        assert len(next_storages) <= 1

        rank_variables = einsum.rank_variables
        rank_variables = {r for r in rank_variables if rank_variable_bounds[r] > 1}
        seen_tensors |= set.union(*(set(t.tensors) for t in prev_storages), set())
        is_fused_loops = is_fused_loops and len(intermediate_tensors - seen_tensors) > 0

        # No recomputation:
        # If we haven't seen a tensor yet, must only iterate over fully-relevant
        # rank variables.
        for t in intermediate_tensors - seen_tensors:
            rank_variables &= tensor2fully_relevant_rank_vars[t]

        # If there is no backing storage in the next block, only include loops
        # that reuse at least one tensor in the previous block.
        # If must be even, then it has to reuse accumulatively all the storages
        # above that must be even.
        if not any(s._backing for s in next_storages):
            prev_relevant_uneven = [
                tensor2fully_relevant_rank_vars[t]
                for s in prev_storages for t in s.tensors
                if not s._even_with_below
            ]
            prev_relevant_even = [
                tensor2fully_relevant_rank_vars[t]
                for s in prev_storages for t in s.tensors
                if s._even_with_below
            ]
            if not any(s._even_with_below for s in prev_storages):
                accumulative_prev_relevant = set()
                assert len(prev_relevant_even) == 0
            if prev_relevant_uneven:
                prev_relevant_uneven = set.intersection(*prev_relevant_uneven)
            else:
                prev_relevant_uneven = set()
            
            if prev_relevant_even:
                prev_relevant_even = set.union(*prev_relevant_even)
            else:
                prev_relevant_even = set()
            accumulative_prev_relevant |= prev_relevant_even

            rank_variables -= prev_relevant_uneven | accumulative_prev_relevant

        # Only include loops that will index into the next block of storage nodes.
        if next_storages:
            next_relevant = [tensor2rank_vars[t] for s in next_storages for t in s.tensors]
            rank_variables &= set.union(*next_relevant, set())

        if not rank_variables:
            choices.append([[]])
        elif is_fused_loops:
            choices.append(list(itertools.permutations(rank_variables)))
        else:
            choices.append([set(rank_variables)])

    for loop_orders in itertools.product(*choices):
        full_mapping = []
        lowering_choices = []
        seen_tensors = set()
        for prev_storages, loop_order in zip(split_mapping, loop_orders):
            full_mapping.extend(prev_storages)
            seen_tensors |= set.union(*(set(t.tensors) for t in prev_storages), set())

            if (einsum.output_tensors() & seen_tensors) != einsum.output_tensors():
                maybe_tile_pattern_rank_vars = ranks_with_tile_pattern & set(loop_order)
            else:
                maybe_tile_pattern_rank_vars = set()

            if isinstance(loop_order, Sequence):
                prev_has_backing = any(s._backing for s in prev_storages)
                first_loop_partially_relevant = None
                for loop in loop_order:
                    if first_loop_partially_relevant is None:
                        first_loop_partially_relevant = all(
                            loop in tensor2partially_relevant_rank_vars[tensor]
                            for storage in prev_storages
                            for tensor in storage.tensors
                        )
                    if loop in maybe_tile_pattern_rank_vars:
                        full_mapping.append(Temporal(rank_variable=loop,
                                                     tile_pattern='symbol'))
                    else:
                        full_mapping.append(Temporal(rank_variable=loop,
                                                     tile_shape='symbol'))
                if prev_has_backing and first_loop_partially_relevant:
                    assert len(prev_storages) == 1
                    lowering_choices.append([True, False])
                elif all(storage.memory == first_memory.name for storage in prev_storages):
                    lowering_choices.extend([[False]]*len(prev_storages))
                elif prev_has_backing:
                    lowering_choices.extend([[False]]*len(prev_storages))
                else:
                    lowering_choices.extend([[True]]*len(prev_storages))
            elif isinstance(loop_order, set) and len(prev_storages) > 1:
                assert all(storage.memory == first_memory.name for storage in prev_storages)
                full_mapping.append(Temporal(rank_variable=loop_order, tile_shape='symbol'))
                lowering_choices.extend([[True]]*len(prev_storages))
            elif isinstance(loop_order, set):  # if set, cannot be fused
                assert len(prev_storages) == 1 and len(prev_storages[0].tensors) == 1
                tensor = next(iter(prev_storages[0].tensors))
                fully_relevant_rank_vars = \
                    loop_order & einsum.tensor2fully_relevant_rank_variables[tensor]
                irrelevant_rank_vars = \
                    loop_order - einsum.tensor2fully_relevant_rank_variables[tensor]
                partially_relevant_rank_vars = \
                    loop_order - irrelevant_rank_vars - fully_relevant_rank_vars

                # Place loops in order of best lowering
                if fully_relevant_rank_vars:
                    full_mapping.append(Temporal(rank_variable=fully_relevant_rank_vars,
                                                tile_shape='symbol'))

                if partially_relevant_rank_vars:
                    full_mapping.append(Temporal(rank_variable=partially_relevant_rank_vars, tile_shape='symbol'))

                if irrelevant_rank_vars:
                    full_mapping.append(Temporal(rank_variable=irrelevant_rank_vars, tile_shape='symbol'))

                lowering_choices.extend([[True]]*len(prev_storages))
            else:
                raise RuntimeError('BUG')

        for lowering_choice in itertools.product(*lowering_choices):
            assert len(lowering_choice) == sum(isinstance(node, Storage) for node in full_mapping)
            for lower, node in zip(lowering_choice,
                                   (node for node in full_mapping
                                    if isinstance(node, Storage))):
                node._lower = lower
            yield list(full_mapping)


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

def timeloop_style_even(mapping: list[MappingNode]):
    # Iterate through the mapping. If there are >2 storage nodes for the same
    # memory, combine all but the innermost one
    memory2indices = defaultdict(list)
    i = 0
    for i, node in enumerate(mapping):
        if not isinstance(mapping[i], Storage):
            i += 1
            continue
        seen = memory2indices[node.memory]
        if len(seen) <= 1:
            seen.append(i)
        else:
            mapping[i] = None
            mapping[seen[-1]].tensors.extend(node.tensors)
    return [m for m in mapping if m is not None]

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

    ranks_with_tile_pattern = get_ranks_with_tile_pattern(einsum_name, spec.workload)
    # print('RANKS WITH TILE PATTERN', ranks_with_tile_pattern)

    symbol_table = spec.workload.get_constraint_symbol_table(einsum_name, spec.renames)
    einsum = spec.workload.einsums[einsum_name]
    for mapping, symbol_table in get_storage_choices(arch_flattened, symbol_table, spec):
        mapping = copy.deepcopy(mapping)
        if spec.mapper_ffm.timeloop_style_even:
            mapping = timeloop_style_even(mapping)
        label_backing_storages(mapping)
        # print(", ".join(m.compact_string() for m in mapping))
        for mapping in insert_temporal_loops(mapping, einsum, first_memory, rank_variable_bounds, ranks_with_tile_pattern, spec.workload):
            mapping = copy.deepcopy(mapping)
            # print(", ".join(m.compact_string() for m in mapping))
            insert_spatial_loops(mapping, einsum, arch_flattened, rank_variable_bounds)
            # print(", ".join(m.compact_string() for m in mapping))
            mapping = unpack_loops_to_rank_variables(mapping)
            label_fused_loops(mapping)
            print(", ".join(m.compact_string() for m in mapping))
            for mapping2 in temporal_fused_constraint_thing_fix_me(mapping, list(spec.workload.einsums[einsum_name].rank_variables)): # TODO
                mapping2 = temporal_constraint_2_fix_me(mapping2, einsum)
                place_missing_temporal_loops(mapping2, einsum, rank_variable_bounds)
                yield mapping2, symbol_table

def iterate_mappings_constraints(
    spec: Specification,
    einsum_names: list[str] | str | None = None,
    arch_flattened: list[architecture.Leaf] | None = None,
    rank_variable_bounds: dict[RankVariableName, int] | None = None,
) -> Iterator[tuple[Mapping, MappingConstraints]]:
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
            # MAPPING MUST NOT BE MODIFIED AFTER THIS POINT
            constraints = get_constraints(arch_flattened, mapping, symbol_table) 
            mapping.append(Compute(einsum=einsum_name, compute=compute_name))
            # mapping = copy.copy(mapping)
            mapping = Mapping(nodes=[copy.copy(n) for n in mapping])
            yield mapping, constraints
            # yield mapping, constraints
            # for mapping2 in iterate_mappings_n_loops_constraint(mapping, spec.workload.einsums[einsum_name]):
            #     yield Mapping(nodes=[copy.copy(n) for n in mapping2.nodes]), constraints

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
        if loop.tile_shape is not None:
            bound = 0  # populated later, but type is important
        elif loop.tile_pattern is not None:
            bound = TilePattern(0, 0)
        else:
            raise RuntimeError('BUG')

        loop = Loop(
            rank_variable_names=fzs((loop.rank_variable,)),
            bound=bound,
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


def get_equivalent_sims(sim: SIM, tagger: Callable[[Mapping], Tags]) -> list[SIM]:
    equivalent_permutations = sim.compatibility.make_equivalent_permutations()
    result = []
    for c in equivalent_permutations:
        try:
            tags = Tags() if tagger is None else tagger(c)
        except ValueError:
            continue
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

    n_tile_shapes = sum(1 if isinstance(l.bound, Number) else 2 for l in compatibility.loops)
    fused_loop_columns = [f"__tile_shape{i}" for i in range(n_tile_shapes)]

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
        try:
            tags = Tags() if tagger is None else tagger(new_compatibility)
        except ValueError as e:
            continue

        new_compatibility = new_compatibility.update(tags=tags)
        print(new_compatibility)

        shift_reservations_by_null_loop_indices(mappings, null_loop_indices)
        partial_mappings = PartialMappings(mappings, free_to_loop_index=len(new_compatibility.loops) - 1, n_pmappings=pmappings_per_group, skip_pareto=len(mappings) < 1000)
        sim = SIM(new_compatibility, partial_mappings)
        if tagger is not None:
            sim.mappings.data[TAGS_COLUMN] = [compatibility.tags] * len(sim.mappings.data)
        sims.append(sim)

    new_sims = []
    for sim in sims:
        # for equivalent_sim in get_equivalent_sims(sim, tagger):
        #     new_sims.append(equivalent_sim)
        new_sims.append(sim)
    return new_sims

# =================================================================================================
# Top level
# =================================================================================================
def _per_proc_compatibility2sim(
    mapping: Mapping,
    constraints: MappingConstraints,
    spec: Specification,
    rank_variable_bounds: dict[RankVariableName, int],
    intermediate_tensors: set[TensorName],
    flattened_arch: list[architecture.Leaf],
    einsum_name: EinsumName,
    metrics: metrics.Metrics,
    job_id: int,
    tagger=None,
) -> tuple[str, dict[Compatibility, SIM], str, Mapping]:
    # print(f", ".join(m.compact_string() for m in mapping.nodes))
    result, total_pmappings = explore_tile_shapes(mapping, constraints, spec, flattened_arch, metrics)
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
    metrics: metrics.Metrics,
    rank_variable_bounds: dict[RankVariableName, int] | None = None,
    flattened_arch: list[architecture.Leaf] | None = None,
    tagger: Callable[[Mapping], Tags] | None = None,
    start_index: int = 0,
) -> list[SIM] | tuple[dict[EinsumName, dict[Compatibility, list[SIM]]], DecompressData]:
    einsum_name = EinsumName(einsum_name)

    if rank_variable_bounds is None:
        rank_variable_bounds = get_rank_variable_bounds(spec.workload, einsum_name)
    
    workload = spec.workload
    intermediate_tensors = workload.intermediate_tensors & workload.einsums[einsum_name].tensor_names

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
            metrics=metrics,
            job_id=start_index + i,
       )
        for i, (mapping, constraints) in enumerate(mappings_constraints)
    ]


def label_backing_storages(mapping: Sequence[MappingNode]):
    seen_tensors = set()
    for i, s in enumerate(mapping):
        if isinstance(s, Storage):
            tensors = set(s.tensors)
            s._backing = tensors - seen_tensors
            s._must_keep_tensors.extend(tensors - seen_tensors) # Backed tensors must be kept.
            seen_tensors.update(tensors)


def get_ranks_with_tile_pattern(producer_name: EinsumName, workload: Workload):
    producer = workload.einsums[producer_name]
    output_tensors = producer.output_tensors()
    if len(output_tensors) > 1:
        return set()

    ranks_with_tile_pattern = set()
    for tensor in output_tensors:
        prod_rank2rank_vars = producer.tensor_accesses[tensor].rank2rank_variables
        for consumer in workload.einsums_that_read_tensor(tensor):
            cons_rank2rank_vars = consumer.tensor_accesses[tensor].rank2rank_variables
            for cons_rank, cons_rank_vars in cons_rank2rank_vars.items():
                if cons_rank not in prod_rank2rank_vars:
                    continue
                if len(cons_rank_vars) == 1:
                    continue  # Not an affine expr at the consumer
                prod_rank_vars = prod_rank2rank_vars[cons_rank]
                if len(prod_rank_vars) != 1:
                    continue  # Unclear what to do in this case
                prod_rank_var = next(iter(prod_rank_vars))
                ranks_with_tile_pattern.add(prod_rank_var)
    return ranks_with_tile_pattern
