import copy
import math
import pandas as pd
from uuid import UUID
from collections import defaultdict

import sympy
from accelforge.frontend.mapping import Loop, Mapping, Spatial, Temporal
from accelforge.frontend.workload import EinsumName
from accelforge.mapper.FFM._join_pmappings.compatibility import (
    Compatibility,
)
from accelforge.mapper.FFM._join_pmappings.pmapping_dataframe import (
    MAPPING_COLUMN,
    PmappingDataframe,
    col2nameloop,
    col_used_in_pareto,
    is_reservation_col,
    makepareto,
    tensor2col,
    col2nameloop,
    is_reservation_col,
    nameloop2col,
)

from accelforge.frontend.mapper.metrics import Metrics
from accelforge.mapper.FFM._make_pmappings.make_pmappings_from_templates.make_tile_shapes import (
    make_tile_shapes,
    IMPERFECT,
)
from accelforge.mapper.FFM._join_pmappings.pmapping_group import PmappingGroup
from accelforge.mapper.FFM._make_pmappings.pmapper_job import (
    Job,
    SameCompatibilityJobs,
)
from accelforge.mapper.FFM._pareto_df.df_convention import (
    is_fused_loop_col,
    is_n_iterations_col,
)
from accelforge.util._mathfuncs import _count_factorizations


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


def get_equivalent_pmappings(
    pmapping_group: PmappingGroup, reservation_levels: set[int]
) -> list[PmappingGroup]:
    equivalent_permutations = pmapping_group.compatibility.make_equivalent_permutations(
        reservation_levels
    )
    result = [PmappingGroup(c, None) for c in equivalent_permutations]
    return result


def mapping2fused_loop_cols(mapping: Mapping, einsum_name: EinsumName):
    cols = []
    for loop in [l for l in mapping.nodes if isinstance(l, Loop) and l._fused]:
        if loop.tile_shape is not None:
            cols.append(loop.tile_shape)
        elif loop.tile_pattern is not None:
            cols.append(loop.tile_pattern.tile_shape)
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


def _count_loops(job: Job) -> tuple[list[int], list[int], dict[str, int]]:
    nodes = job.mapping.nodes
    temporal_n_loops = []
    spatial_n_loops = []
    rv_spatial_count = defaultdict(int)
    rv_temporal_count = defaultdict(int)
    cur_n_loops = 0
    spatial_dim = None

    def pop_loop():
        nonlocal cur_n_loops
        if cur_n_loops >= 1:
            if spatial_dim is not None:
                spatial_n_loops.append(cur_n_loops)
            else:
                temporal_n_loops.append(cur_n_loops)
        cur_n_loops = 0

    for node in nodes:
        cur_spatial_dim = None
        if isinstance(node, Spatial):
            cur_spatial_dim = node.name
            rv_spatial_count[node.rank_variable] += 1
        if cur_spatial_dim != spatial_dim:
            pop_loop()
            spatial_dim = cur_spatial_dim
        if isinstance(node, Loop):
            cur_n_loops += 1
            if isinstance(node, Temporal):
                rv_temporal_count[node.rank_variable] += 1
        else:
            pop_loop()
    pop_loop()
    return temporal_n_loops, spatial_n_loops, rv_spatial_count, rv_temporal_count


def multiply_n_pmappings_by_permutations(n_pmappings: int, job: Job) -> int:
    option = job.spec.mapper._count_option_for_mapsapce_size_evaluation
    # if option == "normal":
    #     return n_pmappings

    temporal_n_loops, spatial_n_loops, rv_spatial_count, rv_temporal_count = (
        _count_loops(job)
    )

    rv = {k: v for k, v in job.rank_variable_bounds.items()}

    if "non_helpful_tile_shapes" in option:
        rv_temporal_count = {r: len(temporal_n_loops) for r in rv.keys()}

    if "non_helpful_loops_for_loop_orders" in option:
        for i in range(len(temporal_n_loops)):
            temporal_n_loops[i] = len(rv)

    # Count number of tile shapes
    rv2loops = {r: rv_spatial_count[r] + rv_temporal_count[r] for r in rv}
    n_factorizations = math.prod(
        _count_factorizations(b, rv2loops[r], imperfect=IMPERFECT)
        for r, b in rv.items()
    )
    n_temporal_loop_orders = math.prod(math.factorial(n) for n in temporal_n_loops)

    n = n_factorizations

    # assert n >= n_pmappings, f"n_pmappings: {n_pmappings} > n: {n}"

    if "redundant_loop_orders" in option:
        # job.mapping._n_loop_orders is the number of permutations that we actually
        # evaluate. Don't want to double count them.
        n *= n_temporal_loop_orders / job.mapping._n_loop_orders

    # assert n >= n_pmappings, f"n_pmappings: {n_pmappings} > n: {n}"

    return n


def assert_all_jobs_have_same_symbols(
    jobs_with_similar_compatibilities: SameCompatibilityJobs,
):
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


def make_pmappings_from_templates(
    jobs_with_similar_compatibilities: SameCompatibilityJobs,
) -> tuple[EinsumName, list[PmappingGroup], dict[UUID, Mapping], SameCompatibilityJobs]:
    jwsc = jobs_with_similar_compatibilities

    results = []

    for job in jobs_with_similar_compatibilities:
        try:
            result, tensor2mapping = make_tile_shapes(job)
        except Exception as e:
            e.add_note(f"Einsum {jwsc.einsum_name} compatibility {job.compatibility}")
            raise
        job.compatibility = job.compatibility.populate_loops()

        # Ctrl-F for CONTIGUOUS_ITERATION_SPACE_DISCUSSION TODO: Turn tensor2pmapping
        # into per-tensor compatibility

        # This changes the pmapping count to include superfluous permutations
        # TODO: Add a multiplier for the permutations that we include in the fusion
        # piece, which are NOT known to be superfluous

        # prev = job.spec.mapper._count_option_for_mapsapce_size_evaluation
        # job.spec.mapper._count_option_for_mapsapce_size_evaluation = "redundant_loop_orders_and_irrelevant_loops"
        # a = multiply_n_pmappings_by_permutations(job.n_total_pmappings, job)
        # job.spec.mapper._count_option_for_mapsapce_size_evaluation = "redundant_loop_orders"
        # b = multiply_n_pmappings_by_permutations(job.n_total_pmappings, job)

        # if a < b:
        #     job.spec.mapper._count_option_for_mapsapce_size_evaluation = "redundant_loop_orders_and_irrelevant_loops"
        #     a = multiply_n_pmappings_by_permutations(job.n_total_pmappings, job)
        #     job.spec.mapper._count_option_for_mapsapce_size_evaluation = "redundant_loop_orders"
        #     b = multiply_n_pmappings_by_permutations(job.n_total_pmappings, job)
        #     assert False

        # job.spec.mapper._count_option_for_mapsapce_size_evaluation = prev
        job.n_total_pmappings = multiply_n_pmappings_by_permutations(
            job.n_total_pmappings, job
        )

        result[MAPPING_COLUMN] = job.job_id
        cols_to_drop = []
        for col in result.columns:
            if is_reservation_col(col):
                resource = col2nameloop(col)[0]
                if resource in job.memories_track_pmappings_only:
                    cols_to_drop.append(col)
        result.drop(columns=cols_to_drop, inplace=True)
        results.append(result)

    fusable_tensors = jwsc.fusable_tensors
    einsum_name = jwsc.einsum_name
    metrics = jwsc.metrics
    limit_capacity_drop_valid_reservations = not (Metrics.RESOURCE_USAGE & metrics)
    compatibility = jwsc.compatibility

    # Creating a PmappingDataframe fills in reservation columns since different pmappings
    # have different ones.
    next_shared_loop_index = compatibility.n_loops - 1
    df = PmappingDataframe.concat(
        [
            PmappingDataframe(
                r,
                skip_pareto=True,
                next_shared_loop_index=next_shared_loop_index,
                n_total_pmappings=1,  # Unused for now, just making an initial Pareto
                n_valid_pmappings=1,  # Unused for now, just making an initial Pareto
                ignored_resources=job.ignored_resources,
                # False because we may have lifetimes that stretch through this Einsum
                # due to data dependencies, not loops
                limit_capacity_drop_valid_reservations=False,
            )
            for r in results
        ],
        skip_pareto=True,
    ).data
    if df.empty:
        return einsum_name, [], {}, jobs_with_similar_compatibilities

    tensor_cols = [tensor2col(tensor) for tensor in fusable_tensors]
    df.columns = [
        c if col_used_in_pareto(c) or c in tensor_cols else f"{einsum_name}<SEP>{c}"
        for c in df.columns
    ]

    fused_loop_cols = [
        f"{einsum_name}<SEP>{c}"
        for c in compatibility.symbols()
        if not is_n_iterations_col(c)
    ]

    job0 = next(iter(jobs_with_similar_compatibilities))

    # Pareto prune
    df = makepareto(df, split_by_cols=fused_loop_cols).copy()

    jobs_passed_pareto = sorted(df[f"{einsum_name}<SEP>{MAPPING_COLUMN}"].unique())
    pmapping_objects = {
        job.job_id: job.mapping
        for job in jobs_with_similar_compatibilities
        if job.job_id in jobs_passed_pareto
    }

    assert_all_jobs_have_same_symbols(jobs_with_similar_compatibilities)
    # Otherwise, following logic fails

    df["fused_loop_indices"] = get_fused_loop_indices(
        df, job0.compatibility, einsum_name, return_as_int=True
    )
    groups = list(df.groupby(["fused_loop_indices"]))
    total_pmappings_per_group = sum(
        j.n_total_pmappings for j in jobs_with_similar_compatibilities
    ) / len(groups)
    valid_pmappings_per_group = sum(
        j.n_valid_pmappings for j in jobs_with_similar_compatibilities
    ) / len(groups)

    pmapping_groups = []
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

        symbol_renames, compatibility = compatibility.make_fused_loop_symbols(
            einsum_name
        )
        for k, v in symbol_renames.items():
            mappings[v] = mappings[f"{einsum_name}<SEP>{k}"]
        shift_reservations_by_null_loop_indices(mappings, null_loop_indices)

        energy_cols = [c for c in mappings.columns if "Total<SEP>energy" in c]
        if (mappings[energy_cols] < 0).any(axis=None):
            mapping_with_negative_energy = mappings[
                (mappings[energy_cols] < 0).any(axis=1)
            ]
            msg = ""
            for _, row in mapping_with_negative_energy.iterrows():
                for k, v in row.items():
                    msg += f"{k}: {v}\n"
                msg += "\n"
            raise RuntimeError(f"negative energy:\n{msg}")

        # Skip pareto because we already did it above
        next_shared_loop_index_this_group = compatibility.n_loops - 1
        mappings = compatibility.clear_unrelated_columns(mappings)
        partial_mappings = PmappingDataframe(
            mappings,
            next_shared_loop_index=next_shared_loop_index_this_group,
            n_total_pmappings=total_pmappings_per_group,
            n_valid_pmappings=valid_pmappings_per_group,
            skip_pareto=next_shared_loop_index_this_group == next_shared_loop_index,
            ignored_resources=job.ignored_resources,
            # False because we may have lifetimes that stretch through this Einsum
            # due to data dependencies, not loops
            limit_capacity_drop_valid_reservations=False,
        )
        pmapping_groups.append(PmappingGroup(compatibility, partial_mappings))

    return (
        einsum_name,
        pmapping_groups,
        pmapping_objects,
        jobs_with_similar_compatibilities,
    )
