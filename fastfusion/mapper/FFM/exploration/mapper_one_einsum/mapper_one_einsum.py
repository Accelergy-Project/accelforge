import copy
from collections import defaultdict
import math
from numbers import Number
import time
from typing import Any, Callable, Iterator, List
import uuid

from fastfusion.accelerated_imports import pd
from tqdm import tqdm

import fastfusion.frontend.architecture as architecture
from fastfusion.frontend.mapping import (
    Compute,
    Iteration,
    Mapping,
    MappingNode,
    TensorHolder,
    Temporal,
)
from fastfusion.frontend.specification import Specification
from fastfusion.frontend.workload.isl import get_rank_variable_bounds
from fastfusion.frontend.workload.symbolic import get_stride_and_halo_of_einsum
from fastfusion.frontend.workload.workload import (
    Einsum,
    EinsumName,
    RankVariableName,
    Workload,
)
from fastfusion.mapper.FFM.exploration.mapper_one_einsum.dataflow_generator import (
    get_tensor_choices,
)
from fastfusion.mapper.metrics import Metrics
from fastfusion.mapper.FFM.exploration.tile_shape_exploration import (
    explore_tile_shapes,
    get_initial_delta_choices,
)
from fastfusion.mapper.FFM.joining.sim import SIM
from fastfusion.mapper.FFM.pareto import (
    MAPPING_COLUMN,
    PartialMappings,
    TILE_SHAPE_PREFIX,
    col2nameloop,
    col_used_in_pareto,
    is_reservation_col,
    makepareto,
    nameloop2col,
    tensor2col,
)
from fastfusion.mapper.FFM.exploration.contraints.constraints import (
    MappingConstraints,
    get_constraints,
)
from fastfusion.mapper.FFM.tags import Tags
from fastfusion.frontend.mapping import Reservation as ReservationNode
from fastfusion.mapper.FFM.exploration.mapper_one_einsum.loop_generator import (
    insert_temporal_loops,
    insert_spatial_loops,
)
from fastfusion.mapper.FFM.exploration.mapper_one_einsum.mapper_job import (
    Job,
    SameCompatibilityJobs,
    SameEinsumJobs
)
from fastfusion.util.setexpressions import eval_set_expression


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
    last_backer = None
    for i, node in enumerate(mapping):
        if isinstance(node, TensorHolder) and node._backing:
            last_backer = i
    if last_backer is None:
        raise ValueError(
            f"No backing TensorHolder found in mapping {", ".join(m.compact_string() for m in mapping)}"
        )

    for i, node in enumerate(mapping):
        if isinstance(node, Iteration):
            node._fused = i < last_backer
    return mapping


# =================================================================================================
# Iterate over mappings
# =================================================================================================
def max_fused_loops_per_rank(
    spec: Specification,
    mapping: List[MappingNode],
    rank_variables: list[RankVariableName],
    rank_variable_bounds: dict[RankVariableName, int],
):
    rank_variables = list(rank_variables)
    if not rank_variables:
        yield mapping
        return

    my_rank_variable = RankVariableName(rank_variables.pop())
    fused_loops = [
        i
        for i, node in enumerate(mapping)
        if isinstance(node, Iteration)
        and node._fused
        and my_rank_variable == node.rank_variable
        and rank_variable_bounds[my_rank_variable]
        > 1  # Don't worry about loops with size 1
    ]

    if len(fused_loops) <= spec.mapper_ffm.max_fused_loops_per_rank:
        yield from max_fused_loops_per_rank(
            spec,
            mapping, rank_variables, rank_variable_bounds
        )
        return

    for choice in fused_loops:
        mapping_new = list(mapping)
        for f in fused_loops[::-1]:
            if f != choice:
                mapping_new.pop(f)
        yield from max_fused_loops_per_rank(
            spec,
            mapping_new, rank_variables, rank_variable_bounds
        )


def place_missing_temporal_loops(mapping: List[MappingNode], einsum: Einsum):
    # If any rank variables are missing, add them as high as possible.
    rank_variables = einsum.rank_variables
    for m in mapping:
        if isinstance(m, Temporal) and not m._fused:
            rank_variables.discard(m.rank_variable)

    # insert_point = 0
    # while insert_point < len(mapping) and not isinstance(mapping[insert_point], Temporal):
    #     insert_point += 1
    # Insert point: Right under the last backing
    for i in range(len(mapping) - 1, -1, -1):
        if isinstance(mapping[i], TensorHolder) and mapping[i]._backing:
            insert_point = i + 1
            break

    temporals = [
        Temporal(rank_variable=r, tile_shape="symbol") for r in sorted(rank_variables)
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
            mapping.append(Temporal(rank_variable=rank_var, tile_shape="symbol"))


def timeloop_style_even(mapping: list[MappingNode]):
    # Iterate through the mapping. If there are >2 TensorHolder nodes for the same
    # memory, move all below the 2nd to the same level as the 2nd.
    mapping = copy.deepcopy(mapping)
    memory2indices = defaultdict(list)
    i = 0
    while i < len(mapping):
        node = mapping[i]
        if not isinstance(mapping[i], TensorHolder):
            i += 1
            continue
        node: TensorHolder
        seen = memory2indices[node.component]
        mapping[i]._lower = len(seen) > 0
        if len(seen) <= 1:
            seen.append(i)
        else:
            mapping[i]._lower = False
            mapping.insert(seen[-1] + 1, mapping.pop(i))
        i += 1
    return mapping


def get_ranks_with_tile_pattern(producer_name: EinsumName, workload: Workload):
    initial_choices = get_initial_delta_choices(producer_name, workload)
    return {
        rank_var
        for rank_var in workload.einsums[producer_name].rank_variables
        if len(initial_choices[rank_var]) > 1
    }


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
    for mapping, symbol_table in get_tensor_choices(
        einsum_name, arch_flattened, symbol_table, spec
    ):
        mapping = copy.deepcopy(mapping)
        # print(", ".join(m.compact_string() for m in mapping))
        for mapping in insert_temporal_loops(
            mapping,
            einsum,
            first_memory,
            rank_variable_bounds,
            ranks_with_tile_pattern,
            spec.workload,
            except_from_imperfect,
        ):
            mapping = copy.deepcopy(mapping)
            # print(", ".join(m.compact_string() for m in mapping))
            insert_spatial_loops(mapping, einsum, arch_flattened)
            # print(", ".join(m.compact_string() for m in mapping))
            mapping = unpack_loops_to_rank_variables(mapping)
            label_fused_loops(mapping)
            # print('POST-LABEL')
            # print(", ".join(m.compact_string() for m in mapping))
            # print(f'{einsum_name}: {", ".join(m.compact_string() for m in mapping)}')
            if spec.mapper_ffm.timeloop_style_even:
                mapping = timeloop_style_even(mapping)

            for mapping2 in max_fused_loops_per_rank(
                spec,
                mapping,
                list(spec.workload.einsums[einsum_name].rank_variables),
                rank_variable_bounds,
            ):  # TODO
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
) -> Iterator[tuple[Mapping, MappingConstraints, dict[str, str]]]:
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
        for mapping, symbol_table in iterate_mappings_no_constraints(
            spec,
            einsum_name,
            arch_flattened,
            rank_variable_bounds,
            except_from_imperfect,
        ):
            # MAPPING MUST NOT BE MODIFIED AFTER THIS POINT
            mapping, constraints = get_constraints(
                arch_flattened, mapping, symbol_table, einsum_name
            )
            mapping.append(Compute(einsum=einsum_name, compute=compute_name))
            mapping = Mapping(nodes=[copy.copy(n) for n in mapping])
            yield mapping, constraints, symbol_table


# =================================================================================================
# Make sims
# =================================================================================================


def get_equivalent_sims(
    sim: SIM, tagger: Callable[[Mapping], Tags], reservation_levels: set[int]
) -> list[SIM]:
    equivalent_permutations = sim.compatibility.make_equivalent_permutations(
        reservation_levels
    )
    result = []
    for c in equivalent_permutations:
        try:
            tags = Tags() if tagger is None else tagger(c)
            result.append(SIM(c.update(tags=tags), None))
        except ValueError:
            pass
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


def shift_reservations_by_null_loop_indices(
    mappings: pd.DataFrame, null_loop_indices: set[int]
):
    prev = copy.deepcopy(mappings)  # TODO: Is this needed?
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

def parse_flattened_arch(
    job: Job,
    symbol_table: dict[str, str],
) -> list[architecture.Leaf]:
    flattened_arch = copy.deepcopy(job.flattened_arch)
    
    tensor_names = job.spec.workload.einsums[job.einsum_name].tensor_names

    def parse_tensor2bits(to_parse: dict[str, Any], location: str, symbol_table: dict[str, str]) -> dict[str, Any]:
        result = {}
        if not isinstance(to_parse, dict):
            raise ValueError(
                f"Expected a dict, got {type(to_parse)}: {to_parse}"
            )
        for key, value in to_parse.items():
            key_parsed = eval_set_expression(
                expression=key,
                symbol_table=symbol_table,
                expected_space_name="tensors",
                location=f"{location} {key}"
            )
            for k2 in key_parsed:
                if k2 in result and result[k2] != value:
                    raise ValueError(
                        f"Multiple entries for {k2} in {location}: "
                        f"{result[k2]} and {value}"
                    )
                result[k2] = value
                
        for tensor_name in tensor_names:
            if tensor_name not in result:
                raise ValueError(
                    f"Tensor {tensor_name} not found in {location}. "
                    f"Available tensors: {', '.join(result.keys())}. Original "
                    f"expressions: {', '.join(to_parse.keys())}. Symbol table:\n\t"
                    + "\n\t".join(f"{k}: {v}" for k, v in symbol_table.items())
                )

        return result

    for node in flattened_arch:
        if not isinstance(node, architecture.TensorHolder):
            continue

        node.attributes.datawidth = parse_tensor2bits(
            node.attributes.datawidth,
            location=f"datawidth of {node.name} for Einsum {job.einsum_name}",
            symbol_table=symbol_table
        )

        for action in node.actions:
            action.arguments.bits_per_action = parse_tensor2bits(
                action.arguments.bits_per_action,
                location=f"bits_per_action of {node.name} action {action.name} for Einsum {job.einsum_name}",
                symbol_table=symbol_table
            )
            
    return flattened_arch

# =================================================================================================
# Top level
# =================================================================================================
def get_single_einsum_jobs(job: Job) -> SameEinsumJobs:
    mappings_constraints = tqdm(
        iterate_mappings_constraints(
            job.spec,
            job.einsum_name,
            job.flattened_arch,
            job.rank_variable_bounds,
            job.except_from_imperfect,
        ),
        desc=f"Generating tensor order and loop choices for Einsum {job.einsum_name}",
    )
    rank_variable_bounds = get_rank_variable_bounds(
                job.spec.workload, job.einsum_name
            )

    jobs = SameEinsumJobs()
    for i, (mapping, constraints, symbol_table) in enumerate(mappings_constraints):
        new_job = copy.copy(job)
        new_job.mapping = mapping
        new_job.constraints = constraints
        new_job.job_id = uuid.uuid4()
        new_job.flattened_arch = parse_flattened_arch(new_job, symbol_table)
        new_job.rank_variable_bounds = rank_variable_bounds
        new_job.stride_and_halo = get_stride_and_halo_of_einsum(job.einsum_name,
                                                                job.spec.workload,
                                                                rank_variable_bounds)
        jobs.append(new_job)

    return jobs


def generate_pmappings(
    jobs_with_similar_compatibilities: SameCompatibilityJobs
):
    total_pmappings = 0
    results = []
    
    job_ids = [job.job_id for job in jobs_with_similar_compatibilities]
    einsum_name = jobs_with_similar_compatibilities.einsum_name
    
    for job in jobs_with_similar_compatibilities:
        result, n_pmappings = explore_tile_shapes(job)
        # This changes the pmapping count to include permutations
        # n_loops = []
        # cur_n_loops = 0
        # for node in job.mapping.nodes:
        #     if isinstance(node, Iteration):
        #         cur_n_loops += 1
        #     elif isinstance(node, ReservationNode):
        #         if cur_n_loops >= 1:
        #             n_loops.append(cur_n_loops)
        #         cur_n_loops = 0
        # if cur_n_loops >= 1:
        #     n_loops.append(cur_n_loops)
        # Uncomment below to include permutations
        # n_pmappings *= math.prod(math.factorial(len(job.rank_variable_bounds)) for n in n_loops)

        # Uncomment below to not include permutations AND assume that the permutation
        # engine has no knowledge of relevant/irrelevant rank variables. Note that the
        # space will still be much bigger than reported here since the extra loops would
        # also increase the index factorization space size.
        # n_pmappings *= math.prod(math.factorial(n) for n in n_loops)
        
        result[MAPPING_COLUMN] = job.job_id
        cols_to_drop = []
        for col in result.columns:
            if is_reservation_col(col) and col2nameloop(col)[0] in job.memories_track_pmappings_only:
                cols_to_drop.append(col)
        result.drop(columns=cols_to_drop, inplace=True)
        
        total_pmappings += n_pmappings
        results.append(result)

    compatibility = jobs_with_similar_compatibilities.compatibility
    intermediate_tensors = jobs_with_similar_compatibilities.intermediate_tensors
    einsum_name = jobs_with_similar_compatibilities.einsum_name
    tagger = jobs_with_similar_compatibilities.tagger
    compatibility_updater = jobs_with_similar_compatibilities.update_compatibility_with_tile_shapes
    metrics = jobs_with_similar_compatibilities.metrics
    limit_capacity_drop_valid_reservations = not (Metrics.RESOURCE_USAGE & metrics)

    # Creating a PartialMappings fills in reservation columns since different partial
    # mappings have different ones.
    next_shared_loop_index = compatibility.n_loops - 1
    results = PartialMappings.concat(
        [
            PartialMappings(
                r,
                skip_pareto=True,
                next_shared_loop_index=next_shared_loop_index,
                limit_capacity_drop_valid_reservations=limit_capacity_drop_valid_reservations,
            )
            for r in results
        ],
        skip_pareto=True,
    ).data

    if results.empty:
        return einsum_name, [], None, job_ids

    # fused_loop_cols = [col for col in results if col.startswith(TILE_SHAPE_PREFIX)]
    fused_loop_cols = [f"{einsum_name}___tile_shape{i}" for i in range(compatibility.n_loops)] # TODO: Make this work for extended Einsums
    
    tensor_cols = [tensor2col(tensor) for tensor in intermediate_tensors]
    
    results.columns = [
        c if col_used_in_pareto(c) or c in tensor_cols
        else f"{einsum_name}_{c}"
        for c in results.columns
    ]
    
    # Pareto prune
    prev_size = len(results)
    results = makepareto(results, split_by_cols=fused_loop_cols)
    new_size = len(results)
    
    jobs_passed_pareto = sorted(results[f"{einsum_name}_{MAPPING_COLUMN}"].unique())
    pmapping_objects = {job.job_id: job.mapping for job in jobs_with_similar_compatibilities if job.job_id in jobs_passed_pareto}
    # print(f'Pareto pruned from {prev_size} to {new_size} pmappings ({new_size / prev_size * 100:.2f}%)')

    if fused_loop_cols:
        groups = list(results.groupby(fused_loop_cols))
    else:
        groups = [((), results)]

    pmappings_per_group = (
        None if total_pmappings is None else total_pmappings / len(groups)
    )

    sims = []

    seen_compatibilities = set()

    for tile_shape, mappings in groups:
        tensor2size = {}

        dropcols = []#list(fused_loop_cols)
        for tensor in intermediate_tensors:  # Sizes are all the same
            tensor2size[tensor] = mappings[tensor2col(tensor)].iloc[0]
            dropcols.append(tensor2col(tensor))
        mappings = mappings.drop(columns=dropcols)

        new_compatibility, null_loop_indices = compatibility_updater(tile_shape,
                                                                     tensor2size)

        shift_reservations_by_null_loop_indices(mappings, null_loop_indices)

        # TODO: Redundant capacity checks because limit_capacity is called. We want it
        # so we can drop dead reservations though.
        # Skip pareto because we already did it above
        # prev_len = len(mappings)
        next_shared_loop_index_this_group = new_compatibility.n_loops - 1
        partial_mappings = PartialMappings(
            mappings,
            next_shared_loop_index=next_shared_loop_index_this_group,
            n_pmappings=pmappings_per_group,
            skip_pareto=next_shared_loop_index_this_group == next_shared_loop_index,
            limit_capacity_drop_valid_reservations=limit_capacity_drop_valid_reservations,
        )
        reservation_levels = partial_mappings.all_reservation_levels()
        sim = SIM(new_compatibility, partial_mappings)

        sim._equivalent_sims = get_equivalent_sims(sim, tagger, reservation_levels)
        sim._equivalent_sims = [
            e
            for e in sim._equivalent_sims
            if e.compatibility not in seen_compatibilities
        ]
        seen_compatibilities.update(e.compatibility for e in sim._equivalent_sims)
        sims.append(sim)

    return einsum_name, sims, pmapping_objects