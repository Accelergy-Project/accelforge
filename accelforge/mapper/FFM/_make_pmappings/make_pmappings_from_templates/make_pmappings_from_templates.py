import copy
import itertools
import math
from accelforge._accelerated_imports import pandas as pd
from uuid import UUID, uuid4
from collections import defaultdict

import sympy
from accelforge.frontend.mapping import (
    Loop,
    Mapping,
    Spatial,
    Temporal,
    Binding,
    iter_binding_blocks,
)
from accelforge.util._basetypes import EvalableList
from accelforge.frontend.workload import EinsumName
from accelforge.util._frozenset import oset
from accelforge.mapper.FFM._join_pmappings.compatibility import (
    Compatibility,
)
from accelforge.mapper.FFM._join_pmappings.pmapping_dataframe import (
    MAPPING_COLUMN,
    PmappingDataframe,
    col2complatency,
    col2reservation,
    col_used_in_pareto,
    complatency2col,
    is_reservation_col,
    makepareto,
    tensor2col,
    reservation2col,
)

from accelforge.frontend.mapper.metrics import Metrics
from accelforge.mapper.FFM._make_pmappings.make_pmappings_from_templates.make_tile_shapes import (
    make_tile_shapes,
)
from accelforge.frontend.mapping import Loop as _Loop


def _imperfect_per_loop(rank_var, mapping) -> tuple[bool, ...]:
    return tuple(
        bool(n._may_cause_imperfect)
        for n in mapping.nodes
        if isinstance(n, _Loop) and n.rank_variable == rank_var
    )


from accelforge.mapper.FFM._join_pmappings.pmapping_group import PmappingGroup
from accelforge.mapper.FFM._make_pmappings.pmapper_job import (
    Job,
    SameCompatibilityJobs,
    SameTemplateJobs,
)
from accelforge.mapper.FFM._pareto_df.df_convention import (
    col2commlatency,
    commlatency2col,
    is_energy_col,
    is_fused_loop_col,
    is_n_iterations_col,
)
from accelforge.util._mathfuncs import _count_factorizations


def shift_reservations_by_null_loop_indices(
    mappings: pd.DataFrame, null_loop_indices: set[int], n_loops: int = 0
):
    def shift(level):
        return level - sum(level > i for i in null_loop_indices)

    non_null_loops = [i for i in range(n_loops) if i not in null_loop_indices]

    # Group columns by the column they land on once the null loops are removed.
    target2sources = {}
    for c in mappings.columns:
        if is_reservation_col(c):
            key = col2reservation(c)
            target = reservation2col(key.name, shift(key.nloops))
        elif (key := col2complatency(c)) is not None:
            target = complatency2col(key.name, shift(key.nloops))
        elif (key := col2commlatency(c)) is not None:
            # A wind-up pours into the outermost remaining loop at or below its
            # own or, with none remaining, is private and pours into the total.
            remaining = [i for i in non_null_loops if i >= key.nloops]
            if remaining:
                target = commlatency2col(key.direction, shift(min(remaining)))
            else:
                target = "Total<SEP>latency"
        else:
            continue
        target2sources.setdefault(target, []).append((key.nloops, c))

    renames = {}
    for target, sources in target2sources.items():
        # Wind-ups are serialized --> sum
        if col2commlatency(target) is not None or target == "Total<SEP>latency":
            for _, c in sources:
                if c != target:
                    mappings[target] = mappings.get(target, 0) + mappings.pop(c)
        else:
            # Reservation/latency columns include one another, so keep the one that
            # includes the rest: the deepest reservation (cumulative downward) or the
            # shallowest latency (cumulative upward).
            _, keep = (max if is_reservation_col(target) else min)(sources)
            renames[keep] = target
            mappings = mappings.drop(columns=[c for _, c in sources if c != keep])
    mappings = mappings.rename(columns=renames)

    if len(mappings.columns) != len(mappings.columns.unique()):
        raise ValueError(f"Duplicate columns: {mappings.columns}")
    return mappings


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


def _get_non_degenerate_loops(
    df: pd.DataFrame,
    n_iterations: list,
    einsum_name: EinsumName,
    return_as_int: bool,
) -> pd.Series | int | list[bool]:
    """
    For each loop bound in `n_iterations`, whether the loop is non-degenerate (i.e.,
    its bound is not one).

    If `return_as_int`, the flags are packed into one integer (or an integer Series
    if any of them is data-dependent) that can be grouped on. Otherwise, a list of
    per-loop booleans is returned; this requires that each flag is the same for
    every row of `df`.
    """
    result = []
    for col in n_iterations:
        if isinstance(col, sympy.Symbol):
            col = col.name
        if isinstance(col, str):
            col = df[f"{einsum_name}<SEP>{col}"]
        result.append(col != 1)

    if return_as_int:
        n = 0
        for b in result:
            n = n * 2 + b
        return n
    else:
        r2 = []
        for b in result:
            if isinstance(b, pd.Series):
                if len(b.unique()) > 1:
                    raise ValueError(f"This won't work if there's more than one")
                b = b.iloc[0]
            r2.append(b)
        return r2


def get_fused_loop_indices(
    df: pd.DataFrame,
    compatibility: Compatibility,
    einsum_name: EinsumName,
    return_as_int: bool = False,
) -> pd.Series | int:
    n_iterations = []
    for loop in compatibility.loops:
        col = loop.tile_pattern.calculated_n_iterations
        assert col is not None, f"Loop {loop} has no calculated n_iterations"
        n_iterations.append(col)
    return _get_non_degenerate_loops(df, n_iterations, einsum_name, return_as_int)


def get_non_degenerate_physical_spatial_loops(
    df: pd.DataFrame,
    compatibility: Compatibility,
    einsum_name: EinsumName,
    return_as_int: bool = False,
) -> pd.Series | int | list[bool]:
    """
    Same as `get_fused_loop_indices`, but for the physical spatial (binding) loops
    of physically distributed storages.
    """
    return _get_non_degenerate_loops(
        df,
        compatibility.physical_spatial_loop_iterations(),
        einsum_name,
        return_as_int,
    )


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
    option = job.spec_one_einsum.mapper._count_option_for_mapsapce_size_evaluation
    # if option == "normal":
    #     return n_pmappings

    temporal_n_loops, spatial_n_loops, rv_spatial_count, rv_temporal_count = (
        _count_loops(job)
    )

    rv = {r: job.rank_variable_bounds[r] for r in job.einsum.rank_variables}

    n = 1
    for r, b in rv.items():
        imperfects = _imperfect_per_loop(r, job.mapping)
        if "non_helpful_tile_shapes" in option:
            assert not any(imperfects)
            nloops = len(temporal_n_loops) + rv_spatial_count[r]
            n *= _count_factorizations(b, (False,) * nloops)
        else:
            n *= _count_factorizations(b, imperfects)

    if "non_helpful_loops_for_loop_orders" in option:
        for i in range(len(temporal_n_loops)):
            temporal_n_loops[i] = len(rv)

    n_temporal_loop_orders = math.prod(math.factorial(n) for n in temporal_n_loops)

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
                    iteration2symbols.append(oset())
                iteration2symbols[i].add(l.tile_pattern.calculated_n_iterations)
    assert all(
        len(s) == 1 for s in iteration2symbols
    ), "All jobs must have the same symbols for compatibility n_iterations"


def _binding_expansions(
    aliases: dict[tuple[str, str, int], tuple[str, ...]]
) -> list[dict[tuple[str, str, int], str]]:
    """
    Every concrete binding that a template's representative stands for, the one the
    template was actually modeled with first. One entry, and an empty one, when the
    template's binding stands only for itself.
    """
    if not aliases:
        return [{}]
    keys = sorted(aliases)
    return [
        dict(zip(keys, combination))
        for combination in itertools.product(*[aliases[k] for k in keys])
    ]


def _rebind_mapping_column(
    mappings: "pd.DataFrame",
    einsum_name: EinsumName,
    assignment: dict[tuple[str, str, int], str],
    pmapping_objects: dict[UUID, Mapping],
) -> "pd.DataFrame":
    """
    A copy of `mappings` whose rows point at rebound copies of the mappings they point
    at now, registering each copy in `pmapping_objects` under a fresh id. The rows carry
    an id rather than a mapping, and the join resolves it back through
    `pmapping_objects`, so an expanded row needs a mapping of its own to resolve to --
    otherwise it would report the binding the template was modeled with rather than the
    one its compatibility was expanded to.
    """
    column = f"{einsum_name}<SEP>{MAPPING_COLUMN}"
    rebound_ids = {}
    for job_id in mappings[column].unique():
        mapping = pmapping_objects.get(job_id)
        if mapping is None:
            continue

        # The Binding nodes are copied too, rather than shared with the template: a
        # rebound copy says something different about the same (memory, tensor) pair.
        new_mapping = Mapping(
            nodes=[
                copy.deepcopy(n) if isinstance(n, Binding) else copy.copy(n)
                for n in mapping.nodes
            ]
        )
        new_mapping._n_loop_orders = mapping._n_loop_orders
        new_mapping._template_index = mapping._template_index
        for block in iter_binding_blocks(new_mapping.nodes):
            rebound = EvalableList()
            changed = False
            for i, loop in enumerate(block.binding):
                rank = assignment.get((block.tensor, block.component, i))
                if rank is not None and rank != loop.rank_variable:
                    loop = loop.model_copy()
                    loop.rank_variable = rank
                    changed = True
                rebound.append(loop)
            if changed:
                block.set_binding(rebound)

        rebound_id = uuid4()
        pmapping_objects[rebound_id] = new_mapping
        rebound_ids[job_id] = rebound_id

    mappings = mappings.copy()
    mappings[column] = mappings[column].map(lambda i: rebound_ids.get(i, i))
    return mappings


def make_pmappings_from_templates(
    jobs_with_similar_compatibilities: SameTemplateJobs,
) -> tuple[EinsumName, list[PmappingGroup], dict[UUID, Mapping]]:
    jwsc = jobs_with_similar_compatibilities

    results = []
    pmapping_keep_rates = []

    for job in jobs_with_similar_compatibilities:
        try:
            result, tensor2mapping = make_tile_shapes(job)
        except Exception as e:
            e.add_note(f"Einsum {jwsc.einsum_name} compatibility {job.compatibility}")
            raise

        job.compatibility = job.compatibility.populate_loops(
            job.ranks_with_tile_pattern, job.mapping
        )

        # Ctrl-F for CONTIGUOUS_ITERATION_SPACE_DISCUSSION TODO: Turn tensor2pmapping
        # into per-tensor compatibility

        # This changes the pmapping count to include superfluous permutations
        # TODO: Add a multiplier for the permutations that we include in the fusion
        # piece, which are NOT known to be superfluous

        # prev = job.spec_one_einsum.mapper._count_option_for_mapsapce_size_evaluation
        # job.spec_one_einsum.mapper._count_option_for_mapsapce_size_evaluation = "redundant_loop_orders_and_irrelevant_loops"
        # a = multiply_n_pmappings_by_permutations(job.n_total_pmappings, job)
        # job.spec_one_einsum.mapper._count_option_for_mapsapce_size_evaluation = "redundant_loop_orders"
        # b = multiply_n_pmappings_by_permutations(job.n_total_pmappings, job)

        # if a < b:
        #     job.spec_one_einsum.mapper._count_option_for_mapsapce_size_evaluation = "redundant_loop_orders_and_irrelevant_loops"
        #     a = multiply_n_pmappings_by_permutations(job.n_total_pmappings, job)
        #     job.spec_one_einsum.mapper._count_option_for_mapsapce_size_evaluation = "redundant_loop_orders"
        #     b = multiply_n_pmappings_by_permutations(job.n_total_pmappings, job)
        #     assert False

        # job.spec_one_einsum.mapper._count_option_for_mapsapce_size_evaluation = prev
        job.n_total_pmappings = multiply_n_pmappings_by_permutations(
            job.n_total_pmappings, job
        )

        result[MAPPING_COLUMN] = job.job_id
        drop_set = set()
        for col in result.columns:
            if is_reservation_col(col):
                resource = col2reservation(col)[0]
                if resource in job.memories_track_pmappings_only:
                    drop_set.add(col)
        if drop_set:
            result = result[[c for c in result.columns if c not in drop_set]]
        results.append(result)
        pmapping_keep_rates.append(
            (
                job.job_id,
                dict(job.pmapping_keep_rates),
                job.n_total_pmappings,
                job.n_evaluated_pmappings,
            )
        )

    fusable_tensors = jwsc.fusable_tensors
    einsum_name = jwsc.einsum_name
    metrics = jwsc.metrics
    drop_valid_reservations = not (Metrics.RESOURCE_USAGE & metrics)
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
                # due to data dependencies, not loops. For example, if we have no fused
                # loops, we can't free our reservations because another Einsum may
                # reserve a resource that lasts through us because it needs to propagate
                # to another Einsum. drop_valid_reservations is turned on later at the
                # joining stage because we have full information of live tensors then.
                drop_valid_reservations=False,
            )
            for r in results
        ],
        skip_pareto=True,
    ).data
    if df.empty:
        return einsum_name, [], {}, pmapping_keep_rates

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

    resource_usage_tolerance = job0.resource_usage_tolerance
    if not drop_valid_reservations:
        resource_usage_tolerance = job0.objective_tolerance

    # Absolute error can sum across Einsums, so we need to divide by it here. Relative
    # error doesn't (x * (1 += 0.1) + y * (1 += 0.1) = (x + y) * (1 +- <= 1.1)), so no
    # divide needed. NOTE: Pruning using tolerance happens in exactly two places that
    # affect results (a third is when there's dirty initial joining, but that doesn't
    # affect final results). Here and in the tile shape creation when a formula is fully
    # evaluated. There's no transformation of the values between these steps, so error
    # doesn't stack (it's just doing the same pruning, perhaps with a different number
    # of objectives).
    absolute_resource_usage_tolerance = (
        resource_usage_tolerance / job0.workload_n_einsums
    )

    # Pareto prune
    df = makepareto(
        df,
        split_by_cols=fused_loop_cols,
        resource_usage_tolerance=resource_usage_tolerance,
        absolute_resource_usage_tolerance=absolute_resource_usage_tolerance,
        objective_tolerance=job0.objective_tolerance,
    ).copy()

    jobs_passed_pareto = sorted(df[f"{einsum_name}<SEP>{MAPPING_COLUMN}"].unique())
    pmapping_objects = {
        job.job_id: job.mapping
        for job in jobs_with_similar_compatibilities
        if job.job_id in jobs_passed_pareto
    }

    assert_all_jobs_have_same_symbols(jobs_with_similar_compatibilities)
    # Otherwise, following logic fails

    # Every job here shares a compatibility, and `_allocate_jobs` hands this function
    # one job at a time, so one set of aliases covers the call. Asserted rather than
    # handled per-job: if that ever stops holding, expanding one job's aliases over
    # another job's rows would quietly give those rows a binding they were not modeled
    # for.
    binding_rank_aliases = job0.binding_rank_aliases or {}
    assert all(
        (j.binding_rank_aliases or {}) == binding_rank_aliases
        for j in jobs_with_similar_compatibilities
    ), "Jobs of one compatibility disagree on which ranks their binding stands for."

    df["fused_loop_indices"] = get_fused_loop_indices(
        df, job0.compatibility, einsum_name, return_as_int=True
    )
    df["physical_spatial_loop_indices"] = get_non_degenerate_physical_spatial_loops(
        df, job0.compatibility, einsum_name, return_as_int=True
    )
    groups = list(df.groupby(["fused_loop_indices", "physical_spatial_loop_indices"]))
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

        null_physical_spatial_loops = {
            n
            for n, non_degenerate in zip(
                compatibility.physical_spatial_loop_iterations(),
                get_non_degenerate_physical_spatial_loops(
                    mappings, compatibility, einsum_name
                ),
            )
            if not non_degenerate
        }

        dropcols = ["fused_loop_indices", "physical_spatial_loop_indices"]
        mappings = mappings.drop(columns=dropcols)

        def _for_this_group(c: Compatibility):
            """
            `c` as this group sees it: the loops this group's rows do not use dropped,
            and the symbols named for this Einsum. None of it depends on which rank a
            binding loop binds, so an expanded compatibility takes exactly these steps
            too -- and takes them from the binding the mapping wrote, since dropping
            degenerate binding loops is what makes a loop's position stop matching.
            """
            c = c.drop_loop_indices(null_loop_indices)
            c = c.drop_physical_spatial_loops(null_physical_spatial_loops)
            return c.make_fused_loop_symbols(einsum_name)

        base_compatibility = compatibility
        symbol_renames, compatibility = _for_this_group(compatibility)
        for new_name, old_name in symbol_renames.items():
            mappings[new_name] = mappings[f"{einsum_name}<SEP>{old_name}"]
        mappings = shift_reservations_by_null_loop_indices(
            mappings,
            null_loop_indices,
            n_loops=compatibility.n_loops + len(null_loop_indices),
        )

        energy_cols = [c for c in mappings.columns if is_energy_col(c)]
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
            ignored_resources=job.ignored_resources
            | oset(job.memories_track_pmappings_only),
            skip_pareto=True,
            # False because we may have lifetimes that stretch through this Einsum due
            # to data dependencies, not loops. For example, if we have no fused loops,
            # we can't free our reservations because another Einsum may reserve a
            # resource that lasts through us because it needs to propagate to another
            # Einsum. drop_valid_reservations is turned on later at the joining stage
            # because we have full information of live tensors then.
            drop_valid_reservations=False,
        )
        # If we have fewer fused loops, reservations likely got freed. We can free!
        if next_shared_loop_index_this_group != next_shared_loop_index:
            partial_mappings.make_pareto(
                resource_usage_tolerance=resource_usage_tolerance,
                objective_tolerance=job0.objective_tolerance,
                absolute_resource_usage_tolerance=absolute_resource_usage_tolerance,
            )
        pmapping_groups.append(PmappingGroup(compatibility, partial_mappings))

        # The template was modeled once for a rank that stands for several. Each of the
        # others gets the same numbers under its own compatibility -- a different thing
        # for a neighboring Einsum to agree with -- without being modeled again. An
        # expansion whose compatibility comes out unchanged (its binding loop was
        # degenerate and dropped above) really would be a duplicate, so it is skipped.
        seen_compatibilities = {compatibility}
        for assignment in _binding_expansions(binding_rank_aliases)[1:]:
            _, expanded = _for_this_group(
                base_compatibility.rebind_physical_spatial_loops(assignment)
            )
            if expanded in seen_compatibilities:
                continue
            seen_compatibilities.add(expanded)
            expanded_mappings = PmappingDataframe(
                # From the rows the representative ended up with, not the ones it
                # started with, so that an expansion carries exactly what was modeled
                # and survived the Pareto pass just above.
                _rebind_mapping_column(
                    partial_mappings.data, einsum_name, assignment, pmapping_objects
                ),
                next_shared_loop_index=next_shared_loop_index_this_group,
                n_total_pmappings=total_pmappings_per_group,
                n_valid_pmappings=valid_pmappings_per_group,
                ignored_resources=job.ignored_resources
                | oset(job.memories_track_pmappings_only),
                skip_pareto=True,
                drop_valid_reservations=False,
            )
            pmapping_groups.append(PmappingGroup(expanded, expanded_mappings))

    # Defragment to speed up pickling
    for pg in pmapping_groups:
        pg.mappings._data = pg.mappings._data.copy()

    return (
        einsum_name,
        pmapping_groups,
        pmapping_objects,
        pmapping_keep_rates,
    )
