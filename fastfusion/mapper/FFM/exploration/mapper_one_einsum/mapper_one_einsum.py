import copy
import gc
import itertools
from collections import defaultdict
from collections.abc import Sequence
from numbers import Number
from typing import Callable, Iterator, List

from joblib import delayed
from pandas import DataFrame
from tqdm import tqdm

import fastfusion.frontend.architecture as architecture
from fastfusion.frontend.mapping import Compute, Iteration, Mapping, MappingNode, ModelOnlyNode, Storage, Temporal, Spatial, Fill
from fastfusion.frontend.specification import Specification
from fastfusion.frontend.workload.isl import get_rank_variable_bounds
from fastfusion.frontend.workload.workload import Einsum, EinsumName, RankVariableName, TensorName, Workload

from fastfusion.mapper.FFM.exploration import metrics
from fastfusion.mapper.FFM.exploration.mapper_one_einsum.dataflow_generator import get_storage_choices
from fastfusion.mapper.FFM.exploration.tile_shape_exploration import explore_tile_shapes, get_initial_delta_choices
from fastfusion.mapper.FFM.joining.mappinginfo import Compatibility, Loop, Reservation, TilePattern
from fastfusion.mapper.FFM.joining.sim import SIM
from fastfusion.mapper.FFM.pareto import TAGS_COLUMN, MAPPING_COLUMN, PartialMappings, col2nameloop, is_reservation_col, nameloop2col, tensor2col, DecompressData
from fastfusion.mapper.FFM.exploration.contraints.constraints import MappingConstraints, get_constraints
from fastfusion.mapper.FFM.tags import Tags
from fastfusion.util.util import defaultintersection, fzs
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
    except_from_imperfect: set,
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
    tensor2irrelevant_rank_vars = einsum.tensor2irrelevant_rank_variables
    tensor2rank_vars = einsum.tensor2rank_variables

    intermediate_tensors = einsum.tensor_names & workload.intermediate_tensor_names
    is_fused_loops = True
    seen_tensors = set()
    choices = []
    lowering_choices = []

    for i, prev_storages in enumerate(split_mapping):
        # =============================================================================
        # Choose what temporal loops to insert between prev_storages and the next
        # storage node(s).
        # =============================================================================

        next_storages = split_mapping[i+1] if i < len(split_mapping) - 1 else []
        assert sum(len(set(s.tensors) - s._backing) for s in prev_storages) <= 1
        assert sum(len(s.tensors) for s in next_storages) <= 1

        rank_variables = einsum.rank_variables
        # rank_variables = {r for r in rank_variables if rank_variable_bounds[r] > 1}
        seen_tensors |= set.union(*(set(t.tensors) for t in prev_storages), set())
        is_fused_loops = is_fused_loops and len(intermediate_tensors - seen_tensors) > 0
        prev_tensors = set.union(set(), *(set(t.tensors) for t in prev_storages))

        # Generally we want to only use rank variables that are irrelevant to the
        # previous tensors, else we'd just lower those tensors. However, we can't lower
        # backing storage nodes because this will add loops to compatibility.

        # No recomputation: If we haven't seen a tensor yet, must only iterate over
        # fully-relevant rank variables.
        for t in intermediate_tensors - seen_tensors:
            rank_variables &= tensor2fully_relevant_rank_vars[t]

        # Optimality-preserving optimizations: We can trivially lower non-backing
        # storage nodes through fully-relevant loops. Can't do this if the loops are
        # fused because that'd add loops to the compatibility.
        for s in prev_storages:
            for t in s.tensors:
                if t not in s._backing and not s._must_be_here:
                    rank_variables -= tensor2fully_relevant_rank_vars[t]
                    
        # Optimality-preserving optimization: We can trivially raise storage nodes
        # through irrelevant unfused loops. Can't do this if the loops are fused because
        # that'd increase the lifetime of the storage node.
        if not is_fused_loops:
            for s in next_storages:
                if not s._must_be_here:
                    for t in s.tensors:
                        rank_variables -= tensor2irrelevant_rank_vars[t]

        
        # Test permutations of partially-relevant rank variables because we'll be
        # lowering through them. Don't permute fully-relevant rank variables because
        # we won't lower through them.
        partially_relevant_to_previous = set.union(set(), *(tensor2partially_relevant_rank_vars[t] for t in prev_tensors))
        partially_relevant_choices = list(itertools.permutations(rank_variables & partially_relevant_to_previous))
        irrelevant_choices = tuple(sorted(rank_variables - partially_relevant_to_previous))
        choices.append([x + irrelevant_choices for x in partially_relevant_choices])

        # =============================================================================
        # Choose whether to lower storage nodes through partially-relevant loops.
        # =============================================================================

        # Option 1: Previous storage is backing and the loop(s) are partially-relevant.
        # We want to explore both lowering and non-lowering. Partially-relevant loop
        # becomes fused if we lower.
        prev_has_backing = any(s._backing for s in prev_storages)
        if prev_has_backing and partially_relevant_to_previous:
            assert len(prev_storages) == 1
            lowering_choices.append([True, False]*len(prev_storages))

        # Option 2: No backing in previous. Lower all. No cost to lowering. Conditioned
        # on option 1 being false.
        elif not prev_has_backing:
            lowering_choices.extend([[True]]*len(prev_storages))

        # Option 3: Fused, but all previous storages are for the first memory. Don't
        # lower. We don't need to reduce memory usage for DRAM.
        elif all(storage.memory == first_memory.name for storage in prev_storages):
            lowering_choices.extend([[False]]*len(prev_storages))

        # Option 4: Previous storage is backing. Don't lower this; needs to be alive for
        # the other Einsum(s).
        elif prev_has_backing:
            lowering_choices.extend([[False]]*len(prev_storages))
            
        else:
            raise RuntimeError('BUG')

    # =======================================================================================
    # Iterate over all possible mappings
    # =======================================================================================
    for loop_orders in itertools.product(*choices):
        full_mapping = []
        for prev_storages, loop_order in zip(split_mapping, loop_orders):
            full_mapping.extend(prev_storages)
            full_mapping.extend(Temporal(rank_variable=r, tile_shape='symbol') for r in loop_order)
        storage_nodes = [node for node in full_mapping if isinstance(node, Storage)]
        assert len(lowering_choices) == len(storage_nodes)
        for lowering_choice in itertools.product(*lowering_choices):
            for lower, node in zip(lowering_choice, storage_nodes):
                node._lower = lower
                
            yield list(full_mapping)

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
        # rv = {r for r in rv if rank_variable_bounds[r] > 1}
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
def temporal_fused_constraint_thing_fix_me(mapping: List[MappingNode], rank_variables: list[RankVariableName], rank_variable_bounds: dict[RankVariableName, int]):
    # Only one fused loop is allowed per rank variable
    rank_variables = list(rank_variables)
    if not rank_variables:
        yield mapping
        return

    my_rank_variable = RankVariableName(rank_variables.pop())
    # indent = " " * (10 - len(rank_variables))
    fused_loops = [
        i for i, node in enumerate(mapping) if isinstance(node, Iteration) 
        and node._fused and my_rank_variable == node.rank_variable
        and rank_variable_bounds[my_rank_variable] > 1 # Don't worry about loops with size 1
    ]
    
    if not fused_loops or len(fused_loops) == 1:
        # print(indent + f"Yielding for rank variable {my_rank_variable}. Length: {len(mapping)}")
        # print(indent + ", ".join(m.compact_string() for m in mapping))
        yield from temporal_fused_constraint_thing_fix_me(mapping, rank_variables, rank_variable_bounds)
        return

    for choice in fused_loops:
        mapping_new = list(mapping)
        for f in fused_loops[::-1]:
            if f != choice:
                mapping_new.pop(f)
        # print(indent + f"Yielding for rank variable {my_rank_variable}. Length: {len(mapping_new)}")
        # print(indent + ", ".join(m.compact_string() for m in mapping_new))
        yield from temporal_fused_constraint_thing_fix_me(mapping_new, rank_variables, rank_variable_bounds)


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
    rank_variables = einsum.rank_variables
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


def pad_with_bottom_loops(mapping: list[MappingNode], einsum: Einsum):
    rank_variables = einsum.rank_variables
    rank_var_to_count = defaultdict(lambda: 0)
    for node in mapping:
        if isinstance(node, Temporal):
            rank_var_to_count[node.rank_variable] += 1

    for rank_var in rank_variables:
        if rank_var_to_count[rank_var] < 2:
            mapping.append(Temporal(rank_variable=rank_var, tile_shape='symbol'))
    

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
    except_from_imperfect: set,
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
        # print(", ".join(m.compact_string() for m in mapping))
        for mapping in insert_temporal_loops(mapping, einsum, first_memory, rank_variable_bounds, ranks_with_tile_pattern, spec.workload, except_from_imperfect):
            mapping = copy.deepcopy(mapping)
            # print(", ".join(m.compact_string() for m in mapping))
            insert_spatial_loops(mapping, einsum, arch_flattened)
            # print(", ".join(m.compact_string() for m in mapping))
            mapping = unpack_loops_to_rank_variables(mapping)
            label_fused_loops(mapping)
            # print('POST-LABEL')
            # print(", ".join(m.compact_string() for m in mapping))
            # print(f'{einsum_name}: {", ".join(m.compact_string() for m in mapping)}')
            for mapping2 in temporal_fused_constraint_thing_fix_me(mapping, list(spec.workload.einsums[einsum_name].rank_variables), rank_variable_bounds): # TODO
                # mapping2 = temporal_constraint_2_fix_me(mapping2, einsum)
                # print('PRE-PADDING')
                # print(", ".join(m.compact_string() for m in mapping2))
                place_missing_temporal_loops(mapping, einsum)
                # pad_with_bottom_loops(mapping2, einsum)
                # print('POST-PADDING')
                # print(", ".join(m.compact_string() for m in mapping2))
                # print('FINAL')
                # print(", ".join(m.compact_string() for m in mapping))

                yield mapping2, symbol_table

def iterate_mappings_constraints(
    spec: Specification,
    einsum_names: list[str] | str | None = None,
    arch_flattened: list[architecture.Leaf] | None = None,
    rank_variable_bounds: dict[RankVariableName, int] | None = None,
    except_from_imperfect: set = set(),
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
        for mapping, symbol_table in iterate_mappings_no_constraints(spec, einsum_name, arch_flattened, rank_variable_bounds, except_from_imperfect):
            # MAPPING MUST NOT BE MODIFIED AFTER THIS POINT
            mapping, constraints = get_constraints(arch_flattened, mapping, symbol_table, einsum_name) 
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


def get_equivalent_sims(sim: SIM, tagger: Callable[[Mapping], Tags], reservation_levels: set[int]) -> list[SIM]:
    equivalent_permutations = sim.compatibility.make_equivalent_permutations(reservation_levels)
    result = []
    for c in equivalent_permutations:
        try:
            tags = Tags() if tagger is None else tagger(c)
        except ValueError:
            continue
        result.append(SIM(c.update(tags=tags), sim.mappings.copy()))
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


def make_sims(
        compatibility: Compatibility,
        explored_results: DataFrame,
        rank_variable_bounds: dict[RankVariableName, int],
        intermediate_tensors: set[TensorName],
        tagger: Callable[[Mapping], Tags] =  None,
        total_pmappings: int = None,
        tensor2compatibilties: dict[TensorName, set[Compatibility]] | None = None,
    ):
    
    if explored_results.empty:
        # print(f"INVALID: {compatibility}")
        return {}
    # else:
    #     print(f"VALID: {compatibility}")


    n_tile_shapes = sum(1 if isinstance(l.bound, Number) else 2 for l in compatibility.loops)
    fused_loop_columns = [f"__tile_shape{i}" for i in range(n_tile_shapes)]

    if fused_loop_columns:
        groups = list(explored_results.groupby(fused_loop_columns))
    else:
        groups = [((), explored_results)]
        
    pmappings_per_group = None if total_pmappings is None else total_pmappings / len(groups)

    sims = []
    
    n_skipped = 0
    total = 0
    
    for tile_shape, mappings in groups:
        tensor2size = {}

        dropcols = []
        for tensor in intermediate_tensors: # Sizes are all the same
            tensor2size[tensor] = mappings[tensor2col(tensor)].iloc[0]
            dropcols.append(tensor2col(tensor))
        mappings.drop(columns=dropcols, inplace=True)

        # print(tile_shape)
        # print(', '.join(m.compact_string() for m in mapping.nodes if not isinstance(m, Fill)))
        new_compatibility, null_loop_indices = compatibility.populate_tile_shape(tile_shape, rank_variable_bounds, tensor2size)

        try:
            tags = Tags() if tagger is None else tagger(new_compatibility)
        except ValueError as e:
            continue

        new_compatibility = new_compatibility.update(tags=tags)

        total += len(mappings)
        fail = False
        for t, c in new_compatibility.per_tensor_compatibility().items():
            if t in tensor2compatibilties and c not in tensor2compatibilties[t]:
                fail = True
                break
        if fail:
            n_skipped += len(mappings)
            continue

        shift_reservations_by_null_loop_indices(mappings, null_loop_indices)
        # TODO: Redundant capacity checks because limit_capacity is called. We want it
        # so we can drop dead reservations though.
        partial_mappings = PartialMappings(mappings, next_shared_loop_index=len(new_compatibility.loops) - 1, n_pmappings=pmappings_per_group, skip_pareto=len(mappings) < 1000)
        # partial_mappings = PartialMappings(mappings, free_to_loop_index=len(new_compatibility.loops) - 1, n_pmappings=pmappings_per_group, skip_pareto=False)
        reservation_levels = partial_mappings.all_reservation_levels()
        sim = SIM(new_compatibility, partial_mappings)
        if tagger is not None:
            sim.mappings.data[TAGS_COLUMN] = [compatibility.tags] * len(sim.mappings.data)
            
        for equivalent_sim in get_equivalent_sims(sim, tagger, reservation_levels):
            sims.append(equivalent_sim)
            
    # print(f'{n_skipped} / {total} skipped ({n_skipped / total * 100:.2f}%)')

    return sims

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
    tensor2compatibilties: dict[TensorName, set[Compatibility]] | None = None,
    tensor2boundless_compatibilities: dict[TensorName, set[Compatibility]] | None = None,
    df_lambda: Callable[[DataFrame], DataFrame] = lambda df: df
) -> tuple[str, dict[Compatibility, SIM], str, Mapping]:
       
    # print(f', '.join(m.compact_string() for m in mapping.nodes))
    mapping_copy = copy.deepcopy(mapping)
    explore_tile_shapes(mapping_copy, constraints, spec, flattened_arch, metrics, _fix_me=True)
    compatibility = make_compatibility(mapping_copy, intermediate_tensors)

    if tensor2boundless_compatibilities is not None:
        for t, c in compatibility.per_tensor_compatibility().items():
            if t not in tensor2boundless_compatibilities:
                continue
            for c2 in c.subsets_of_loops(clear_bounds=True):
                if c2 in tensor2boundless_compatibilities[t]:
                    break
            else:
                # print(f'Skipping {c} due to incompatibility.')
                return einsum_name, [], None, job_id
    
    mstr = ', '.join(m.compact_string() for m in mapping.nodes)
    result, total_pmappings = explore_tile_shapes(mapping, constraints, spec, flattened_arch, metrics, _fix_me=False)
    result = df_lambda(result)
    compatibility = make_compatibility(mapping, intermediate_tensors)
    # print(f'{mstr}')
    sims = make_sims(compatibility, result, rank_variable_bounds, intermediate_tensors, tagger=tagger, total_pmappings=total_pmappings, tensor2compatibilties=tensor2compatibilties)
    decompress_data = PartialMappings.compress_paretos(
        einsum_name, 
        [s.mappings for s in sims],
        job_id=job_id,
        extra_data={MAPPING_COLUMN: mapping}
    )
    gc.collect()
    # print(f"FINISHED: " + ', '.join(m.compact_string() for m in mapping.nodes))
    return einsum_name, sims, decompress_data, job_id


def get_single_einsum_jobs(
    spec: Specification,
    einsum_name: EinsumName,
    metrics: metrics.Metrics,
    rank_variable_bounds: dict[RankVariableName, int] | None = None,
    flattened_arch: list[architecture.Leaf] | None = None,
    tensor2compatibilties: dict[TensorName, set[Compatibility]] | None = None,
    tensor2boundless_compatibilities: dict[TensorName, set[Compatibility]] | None = None,
    tagger: Callable[[Mapping], Tags] | None = None,
    start_index: int = 0,
    except_from_imperfect: set = set(),
    df_lambda: Callable[[DataFrame], DataFrame] = lambda df: df
) -> list[SIM] | tuple[dict[EinsumName, dict[Compatibility, list[SIM]]], DecompressData]:
    einsum_name = EinsumName(einsum_name)

    if rank_variable_bounds is None:
        rank_variable_bounds = get_rank_variable_bounds(spec.workload, einsum_name)
    
    workload = spec.workload
    intermediate_tensors = workload.intermediate_tensor_names & workload.einsums[einsum_name].tensor_names

    if flattened_arch is None:
        flattened_arch = spec.get_flattened_architecture()

    mappings_constraints = tqdm(iterate_mappings_constraints(spec,
                                einsum_name,
                                flattened_arch,
                                rank_variable_bounds,
                                except_from_imperfect,
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
            tensor2compatibilties=tensor2compatibilties,
            tensor2boundless_compatibilities=tensor2boundless_compatibilities,
            df_lambda=df_lambda,
        )
        for i, (mapping, constraints) in enumerate(mappings_constraints)
    ]


def get_ranks_with_tile_pattern(producer_name: EinsumName, workload: Workload):
    initial_choices = get_initial_delta_choices(producer_name, workload)
    return {
        rank_var
        for rank_var in workload.einsums[producer_name].rank_variables
        if len(initial_choices[rank_var]) > 1
    }
