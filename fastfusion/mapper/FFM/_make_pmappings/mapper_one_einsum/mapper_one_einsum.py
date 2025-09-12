import copy
from typing import Callable
import pandas as pd
from uuid import UUID
from fastfusion.frontend.mapping import Mapping
from fastfusion.frontend.workload.workload import EinsumName
from fastfusion.mapper.FFM._pmapping_group import (
    col2nameloop,
    is_reservation_col,
    nameloop2col,
)

from fastfusion.frontend.mapper.metrics import Metrics
from fastfusion.mapper.FFM._make_pmappings.tile_shape_exploration import (
    explore_tile_shapes,
)
from fastfusion.mapper.FFM._join_pmappings.sim import SIM
from fastfusion.mapper.FFM._pmapping_group import (
    MAPPING_COLUMN,
    PmappingGroup,
    col2nameloop,
    col_used_in_pareto,
    is_reservation_col,
    makepareto,
    tensor2col,
)
from fastfusion.mapper.FFM._make_pmappings.mapper_one_einsum.mapper_job import (
    SameCompatibilityJobs,
)
from fastfusion.mapper.FFM.deprecate_maybe.tags import Tags

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


def generate_pmappings(
    jobs_with_similar_compatibilities: SameCompatibilityJobs
) -> tuple[EinsumName, list[SIM], dict[UUID, Mapping], SameCompatibilityJobs]:
    total_pmappings = 0
    results = []
    
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

        # print(job.pretty_str())
        
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

    # Creating a PmappingGroup fills in reservation columns since different partial
    # mappings have different ones.
    next_shared_loop_index = compatibility.n_loops - 1
    results = PmappingGroup.concat(
        [
            PmappingGroup(
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
        return einsum_name, [], {}, jobs_with_similar_compatibilities

    fused_loop_cols = [
        f"{einsum_name}\0tile_shape{i}"
        for i in range(compatibility.n_loops)
    ]  # TODO: Make this work for extended Einsums

    tensor_cols = [tensor2col(tensor) for tensor in intermediate_tensors]

    results.columns = [
        c if col_used_in_pareto(c) or c in tensor_cols else f"{einsum_name}\0{c}"
        for c in results.columns
    ]

    # Pareto prune
    prev_size = len(results)
    results = makepareto(results, split_by_cols=fused_loop_cols)
    new_size = len(results)
    
    jobs_passed_pareto = sorted(results[f"{einsum_name}\0{MAPPING_COLUMN}"].unique())
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
        partial_mappings = PmappingGroup(
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

    return einsum_name, sims, pmapping_objects, jobs_with_similar_compatibilities