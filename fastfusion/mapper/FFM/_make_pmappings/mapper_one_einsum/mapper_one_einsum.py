import copy
import math
from typing import Callable
import pandas as pd
from uuid import UUID

import sympy
from fastfusion.frontend.arch import TensorHolder
from fastfusion.frontend.mapping import Iteration, Mapping, TilePattern
from fastfusion.frontend.renames import RankVariableName, TensorName
from fastfusion.frontend.workload.workload import EinsumName
from fastfusion.mapper.FFM._join_pmappings.compatibility import (
    Compatibility,
    TensorReservation,
)
from fastfusion.mapper.FFM._make_pmappings.mapper_one_einsum.tile_shape_exploration import (
    EXPERIMENTAL_TILE_SHAPE_EXPLORATION,
)
from fastfusion.mapper.FFM._pmapping_group import (
    col2nameloop,
    is_reservation_col,
    nameloop2col,
)

from fastfusion.frontend.mapper.metrics import Metrics
from fastfusion.mapper.FFM._make_pmappings.mapper_one_einsum.tile_shape_exploration import (
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
    Job,
    SameCompatibilityJobs,
)
from fastfusion.mapper.FFM._pmapping_group.df_convention import is_fused_loop_col
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


def generate_pmappings_old(
    jobs_with_similar_compatibilities: SameCompatibilityJobs,
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
            if (
                is_reservation_col(col)
                and col2nameloop(col)[0] in job.memories_track_pmappings_only
            ):
                cols_to_drop.append(col)
        result.drop(columns=cols_to_drop, inplace=True)

        total_pmappings += n_pmappings
        results.append(result)

    compatibility = jobs_with_similar_compatibilities.compatibility
    intermediate_tensors = jobs_with_similar_compatibilities.intermediate_tensors
    einsum_name = jobs_with_similar_compatibilities.einsum_name
    tagger = jobs_with_similar_compatibilities.tagger
    compatibility_updater = (
        jobs_with_similar_compatibilities.update_compatibility_with_tile_shapes
    )
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
        f"{einsum_name}<SEP>tile_shape{i}" for i in range(compatibility.n_loops)
    ]  # TODO: Make this work for extended Einsums

    tensor_cols = [tensor2col(tensor) for tensor in intermediate_tensors]

    results.columns = [
        c if col_used_in_pareto(c) or c in tensor_cols else f"{einsum_name}<SEP>{c}"
        for c in results.columns
    ]

    # Pareto prune
    prev_size = len(results)
    results = makepareto(results, split_by_cols=fused_loop_cols)
    new_size = len(results)

    jobs_passed_pareto = sorted(results[f"{einsum_name}<SEP>{MAPPING_COLUMN}"].unique())
    pmapping_objects = {
        job.job_id: job.mapping
        for job in jobs_with_similar_compatibilities
        if job.job_id in jobs_passed_pareto
    }
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
        compatibility = jwsc.compatibility
        tensor2size = {}

        dropcols = []  # list(fused_loop_cols)
        for tensor in intermediate_tensors:  # Sizes are all the same
            tensor2size[tensor] = mappings[tensor2col(tensor)].iloc[0]
            dropcols.append(tensor2col(tensor))
        mappings = mappings.drop(columns=dropcols)

        compatibility, null_loop_indices = compatibility_updater(
            tile_shape, tensor2size
        )

        shift_reservations_by_null_loop_indices(mappings, null_loop_indices)

        # TODO: Redundant capacity checks because limit_capacity is called. We want it
        # so we can drop dead reservations though.
        # Skip pareto because we already did it above
        # prev_len = len(mappings)
        next_shared_loop_index_this_group = compatibility.n_loops - 1
        partial_mappings = PmappingGroup(
            mappings,
            next_shared_loop_index=next_shared_loop_index_this_group,
            n_pmappings=pmappings_per_group,
            skip_pareto=next_shared_loop_index_this_group == next_shared_loop_index,
            limit_capacity_drop_valid_reservations=limit_capacity_drop_valid_reservations,
        )
        reservation_levels = partial_mappings.all_reservation_levels()
        sim = SIM(compatibility, partial_mappings)

        sim._equivalent_sims = get_equivalent_sims(sim, tagger, reservation_levels)
        sim._equivalent_sims = [
            e
            for e in sim._equivalent_sims
            if e.compatibility not in seen_compatibilities
        ]
        seen_compatibilities.update(e.compatibility for e in sim._equivalent_sims)
        sims.append(sim)

    return einsum_name, sims, pmapping_objects, jobs_with_similar_compatibilities


def get_n_permutations(job: Job) -> int:
    n_loops: list[int] = []
    cur_n_loops = 0
    for node in job.mapping.nodes:
        if isinstance(node, Iteration):
            cur_n_loops += 1
        elif isinstance(node, TensorHolder):
            if cur_n_loops > 1:
                n_loops.append(cur_n_loops)
            cur_n_loops = 0
    if cur_n_loops > 1:
        n_loops.append(cur_n_loops)
    return math.prod(math.factorial(n) for n in n_loops)


def mapping2fused_loop_cols(mapping: Mapping, einsum_name: EinsumName):
    cols = []
    for loop in [l for l in mapping.nodes if isinstance(l, Iteration) and l._fused]:
        if loop.tile_shape is not None:
            cols.append(loop.tile_shape)
        elif loop.tile_pattern is not None:
            cols.append(loop.tile_pattern.stride)
            cols.append(loop.tile_pattern.initial_tile_shape)
        else:
            raise ValueError(f"Can't find tile shape or tile pattern for loop {loop}")
        return [f"{einsum_name}<SEP>{c}" if isinstance(c, str) else c for c in cols]


def get_fused_loop_indices(
    df: pd.DataFrame,
    compatibility: Compatibility,
    einsum_name: EinsumName,
    return_as_int: bool = False,
) -> pd.Series | int:
    result = []

    loops = compatibility.loops
    for i, loop in enumerate(loops):
        col = loop.tile_pattern.calculated_n_iterations
        assert col is not None, f"Loop {loop} has no calculated n_iterations"
        if isinstance(col, str):
            col = df[f"{einsum_name}<SEP>{col}"]
        elif isinstance(col, sympy.Symbol):
            col = df[f"{einsum_name}<SEP>{col.name}"]
        result.append(col != 1)

    if return_as_int:
        n = 0
        for b in result:
            n = n * 2 + b
        return n
    else:
        r2 = []
        for b in result:
            if len(b.unique()) > 1:
                raise ValueError(f"This won't work if there's more than one")
            r2.append(b.iloc[0])
        return r2


def generate_pmappings_new(
    jobs_with_similar_compatibilities: SameCompatibilityJobs,
) -> tuple[EinsumName, list[SIM], dict[UUID, Mapping], SameCompatibilityJobs]:
    jwsc = jobs_with_similar_compatibilities
    prev_jobs = copy.deepcopy(jwsc)

    # Ensure that all the symbols are the same
    symbols: set[tuple[tuple[RankVariableName, tuple[str, ...]]]] = set()
    for job in jwsc:
        cur_symbols = []
        for node in job.mapping.nodes:
            if not isinstance(node, Iteration):
                continue
            if not node._fused:
                break
            cur_symbols.append(
                (
                    node.rank_variable,
                    tuple(sorted(node.tile_pattern.symbols_as_strings())),
                )
            )
        symbols.add(tuple(cur_symbols))

    if len(symbols) > 1:
        raise ValueError(f"Symbols are not the same: {symbols}")

    results = []

    for job in jobs_with_similar_compatibilities:
        result = explore_tile_shapes(job)
        job.compatibility = job.compatibility.populate_loops(job.mapping)
        # This changes the pmapping count to include superfluous permutations
        # TODO: Add a multiplier for the permutations that we include in the fusion
        # piece, which are NOT known to be superfluous
        # job.total_pmappings *= get_n_permutations(job)

        result[MAPPING_COLUMN] = job.job_id
        cols_to_drop = []
        for col in result.columns:
            if (
                is_reservation_col(col)
                and col2nameloop(col)[0] in job.memories_track_pmappings_only
            ):
                cols_to_drop.append(col)
        result.drop(columns=cols_to_drop, inplace=True)
        results.append(result)

    intermediate_tensors = jwsc.intermediate_tensors
    einsum_name = jwsc.einsum_name
    metrics = jwsc.metrics
    limit_capacity_drop_valid_reservations = not (Metrics.RESOURCE_USAGE & metrics)
    compatibility = jwsc.compatibility

    # Creating a PmappingGroup fills in reservation columns since different pmappings
    # have different ones.
    next_shared_loop_index = compatibility.n_loops - 1
    df = PmappingGroup.concat(
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
    if df.empty:
        return einsum_name, [], {}, jobs_with_similar_compatibilities

    tensor_cols = [tensor2col(tensor) for tensor in intermediate_tensors]
    df.columns = [
        c if col_used_in_pareto(c) or c in tensor_cols else f"{einsum_name}<SEP>{c}"
        for c in df.columns
    ]

    fused_loop_cols = [f"{einsum_name}<SEP>{c}" for c in compatibility.symbols()]

    job0 = next(iter(jobs_with_similar_compatibilities))

    # Pareto prune
    try:
        df = makepareto(df, split_by_cols=fused_loop_cols)
    except:
        for job in prev_jobs:
            result = explore_tile_shapes(job)
            job.compatibility = job.compatibility.populate_loops(job.mapping)
            # This changes the pmapping count to include superfluous permutations
            # TODO: Add a multiplier for the permutations that we include in the fusion
            # piece, which are NOT known to be superfluous
            # job.total_pmappings *= get_n_permutations(job)

            result[MAPPING_COLUMN] = job.job_id
            cols_to_drop = []
            for col in result.columns:
                if (
                    is_reservation_col(col)
                    and col2nameloop(col)[0] in job.memories_track_pmappings_only
                ):
                    cols_to_drop.append(col)
            result.drop(columns=cols_to_drop, inplace=True)
            results.append(result)

    jobs_passed_pareto = sorted(df[f"{einsum_name}<SEP>{MAPPING_COLUMN}"].unique())
    pmapping_objects = {
        job.job_id: job.mapping
        for job in jobs_with_similar_compatibilities
        if job.job_id in jobs_passed_pareto
    }

    # Assert all jobs have the same symbols for compatibility n_iterations. If they
    # don't, this logic will break.
    iteration2symbols = []
    for j in jobs_with_similar_compatibilities:
        for t in j.compatibility.tensors:
            for i, l in enumerate(t.loops):
                if len(iteration2symbols) <= i:
                    iteration2symbols.append(set())
                iteration2symbols[i].add(l.tile_pattern.calculated_n_iterations)
    assert all(
        len(s) == 1 for s in iteration2symbols
    ), "All jobs must have the same symbols for compatibility n_iterations"

    df["fused_loop_indices"] = get_fused_loop_indices(
        df, job0.compatibility, einsum_name, return_as_int=True
    )
    groups = list(df.groupby(["fused_loop_indices"]))
    pmappings_per_group = sum(
        j.total_pmappings for j in jobs_with_similar_compatibilities
    ) / len(groups)

    sims = []
    for _, mappings in groups:
        compatibility = jwsc.compatibility
        fused_loop_indices = []

        for i, f in enumerate(
            get_fused_loop_indices(
                mappings, compatibility, einsum_name, return_as_int=False
            )
        ):
            if f:
                fused_loop_indices.append(i)

        null_loop_indices = tuple(
            i for i in range(compatibility.n_loops) if i not in fused_loop_indices
        )

        dropcols = ["fused_loop_indices"]
        mappings = mappings.drop(columns=dropcols)

        compatibility = compatibility.drop_loop_indices(null_loop_indices)

        symbol_renames, compatibility = compatibility.make_fused_loop_symbols()
        for k, v in symbol_renames.items():
            mappings[v] = mappings[f"{einsum_name}<SEP>{k}"]
        shift_reservations_by_null_loop_indices(mappings, null_loop_indices)

        symbols = compatibility.symbols()
        dropcols = [
            c for c in mappings.columns if is_fused_loop_col(c) and c not in symbols
        ]
        mappings = mappings.drop(columns=dropcols)

        # TODO: Redundant capacity checks because limit_capacity is called. We want it
        # so we can drop dead reservations though.
        # Skip pareto because we already did it above
        # prev_len = len(mappings)
        next_shared_loop_index_this_group = compatibility.n_loops - 1
        partial_mappings = PmappingGroup(
            mappings,
            next_shared_loop_index=next_shared_loop_index_this_group,
            n_pmappings=pmappings_per_group,
            skip_pareto=next_shared_loop_index_this_group == next_shared_loop_index,
            limit_capacity_drop_valid_reservations=limit_capacity_drop_valid_reservations,
        )
        sim = SIM(compatibility, partial_mappings)
        sims.append(sim)

    return einsum_name, sims, pmapping_objects, jobs_with_similar_compatibilities


def generate_pmappings(
    jobs_with_similar_compatibilities: SameCompatibilityJobs,
) -> tuple[EinsumName, list[SIM], dict[UUID, Mapping], SameCompatibilityJobs]:
    if EXPERIMENTAL_TILE_SHAPE_EXPLORATION:
        return generate_pmappings_new(jobs_with_similar_compatibilities)
    return generate_pmappings_old(jobs_with_similar_compatibilities)
