from copy import deepcopy
from accelforge.mapper.FFM._join_pmappings.compatibility import Compatibility
from collections import defaultdict
import itertools
import logging
import time
from typing import Any, Callable

from accelforge._accelerated_imports import pd, np
from accelforge.frontend.spec import Spec
from accelforge.frontend.mapping import Mapping
from accelforge.frontend.mapper.metrics import Metrics
from accelforge.frontend.workload import EinsumName
from accelforge.mapper.FFM.mappings import Mappings
from accelforge.mapper.FFM.pmappings import MultiEinsumPmappings
from accelforge.mapper.FFM._join_pmappings.compress_pmappings import (
    compress_einsum2pmappings,
    decompress_pmappings,
)
from accelforge.mapper.FFM._make_pmappings.make_pmappings import (
    get_rank_variable_bounds_for_all_einsums,
)
from accelforge.mapper.FFM._join_pmappings.pmapping_dataframe import (
    row2pmappings,
)
from accelforge.mapper.FFM._pareto_df.df_convention import (
    MAPPING_COLUMN,
    col_used_in_pareto,
    is_objective_col,
    is_reservation_col,
)
from accelforge.mapper.FFM._join_pmappings.pmapping_group import (
    PmappingGroup,
    Compatibility,
)
from accelforge.mapper.FFM._pareto_df.df_convention import col2reservation
from accelforge.util import _fillna_and__numeric_cast, parallel, delayed, oset

logger = logging.getLogger(__name__)


class JoiningTimer:
    def __init__(self):
        self.prev_time = time.time()
        self.total_time = defaultdict(int)

    def print_time(self, what: str):
        t = time.time() - self.prev_time
        logger.info(f"{what}: {t:.2f} seconds")
        self.total_time[what] += t
        self.prev_time = time.time()

    def log_total_time(self):
        logger.info(f"\n======== Total time ========")
        for k, v in self.total_time.items():
            logger.info(f"{k}: {v:.2f} seconds")
        total = sum(self.total_time.values())
        if total > 60:
            logger.info(f"\nTotal: {total:.2f} seconds ({total/60:.2f} minutes)")
        else:
            logger.info(f"\nTotal: {total:.2f} seconds")
        logger.info(f"============================\n")


class OptimalityThresholder:
    def __init__(
        self,
        prev_solutions: Mappings,
        _pmapping_row_filter_function: Callable[[pd.DataFrame], np.ndarray],
        aggregator: str,
        print_progress: bool,
    ):
        compare_to = prev_solutions.data
        compare_cols = [c for c in compare_to.columns if col_used_in_pareto(c)]
        self._pmapping_row_filter_function = _pmapping_row_filter_function
        self.aggregator = aggregator

        if self.aggregator in ("prod", "sum"):
            objective_cols = [c for c in compare_cols if is_objective_col(c)]
            self._agg_cols = objective_cols
            if objective_cols:
                values = np.column_stack([compare_to[c].values for c in objective_cols])
                if self.aggregator == "prod":
                    agg = np.prod(values, axis=1)
                else:
                    agg = np.sum(values, axis=1)
                self._agg_threshold = agg.min()
            else:
                self._agg_threshold = float("inf")
            if print_progress:
                label = "product" if self.aggregator == "prod" else "sum"
                print(
                    f"Filtering out pmappings with {label} > "
                    f"{self._agg_threshold:.2e}"
                )
        else:  # "any"
            compare_to = compare_to.sort_values(by=compare_cols, ascending=False)

            if len(compare_to) > 10:
                chosen_indices = np.round(np.linspace(0, len(compare_to) - 1, 10))
            else:
                chosen_indices = np.round(np.arange(len(compare_to)))

            self.compare_to: list[dict[str, float]] = []
            if print_progress:
                print(f"Filtering out pmappings worse than the following:")

            for i in chosen_indices.astype(int):
                self.compare_to.append({c: compare_to[c].iloc[i] for c in compare_cols})
                if print_progress:
                    print(
                        "\t"
                        + "    ".join(
                            f"{k}={float(v):.2e}"
                            for k, v in self.compare_to[-1].items()
                        )
                    )

    def __call__(self, mapping: pd.DataFrame) -> bool:
        nondominated_by_all = np.ones(len(mapping), dtype=bool)

        if self.aggregator in ("prod", "sum"):
            cols_present = [c for c in self._agg_cols if c in mapping.columns]
            if cols_present:
                values = np.column_stack([mapping[c].values for c in cols_present])
                if self.aggregator == "prod":
                    agg = np.prod(values, axis=1)
                else:
                    agg = np.sum(values, axis=1)
                nondominated_by_all = agg <= self._agg_threshold
        else:  # "any"
            for c in self.compare_to:
                nondominated = np.zeros(len(mapping), dtype=bool)
                for k, v in c.items():
                    if k not in mapping.columns:
                        nondominated |= True
                    else:
                        nondominated |= mapping[k] <= v
                nondominated_by_all &= nondominated

        if self._pmapping_row_filter_function is not None:
            nondominated_by_all &= self._pmapping_row_filter_function(mapping)

        return nondominated_by_all


def prune_with_tolerance(
    pmappings: dict[EinsumName, list[PmappingGroup]],
    objective_tolerance: float,
    resource_usage_tolerance: float,
    print_progress: bool = True,
    is_last: bool = False,
):
    if objective_tolerance == 0 and resource_usage_tolerance == 0:
        return pmappings

    prev_n = sum(len(pg.mappings) for p in pmappings.values() for pg in p)

    def prune(einsum_name: EinsumName, pg: PmappingGroup):
        pg = PmappingGroup(
            pg.compatibility,
            pg.mappings.make_pareto(
                objective_tolerance=objective_tolerance,
                resource_usage_tolerance=resource_usage_tolerance,
                inplace=False,
            ),
        )
        return einsum_name, pg

    jobs = [delayed(prune)(e, pg) for e, p in pmappings.items() for pg in p]

    result = {einsum_name: [] for einsum_name in pmappings.keys()}
    for einsum_name, pg in parallel(
        jobs, pbar="Dirty pruning pmappings" if print_progress else None
    ):
        result[einsum_name].append(pg)

    new_n = sum(len(pg.mappings) for p in result.values() for pg in p)
    if new_n == prev_n and not is_last:
        return None

    if print_progress:
        print(f"Dirty joining uses {new_n / prev_n * 100:.2f}% of the pmappings")

    return result


def join_strategy_2(
    spec: Spec,
    compressed: dict[EinsumName, list[PmappingGroup]],
    print_progress: bool,
    metrics: Metrics,
    for_model: bool,
    _pmapping_row_filter_function: Callable[[pd.DataFrame], np.ndarray] | None = None,
    resource_usage_tolerance: float = 0,
):
    thresholds = [1, 0]
    thresholds = [t for t in thresholds if t > spec.mapper.objective_tolerance]
    thresholds.append(spec.mapper.objective_tolerance)

    filter_func = _pmapping_row_filter_function
    _runtime_log_file = spec.mapper._runtime_log_file
    for i, threshold in enumerate(thresholds):
        is_dirty = i < len(thresholds) - 1
        if not for_model and print_progress:
            if is_dirty:
                print(f"Dirty joining with objectives <= {1 + threshold}× optimal")
            else:
                print("Final clean join.")
        # Write round marker so the notebook can distinguish dirty vs clean
        if _runtime_log_file and is_dirty:
            import json

            with open(_runtime_log_file, "a") as f:
                f.write(json.dumps({"round": i, "threshold": threshold}) + "\n")
        try:
            cur_compressed = prune_with_tolerance(
                compressed,
                objective_tolerance=threshold,
                resource_usage_tolerance=resource_usage_tolerance,
                print_progress=print_progress,
                is_last=i == len(thresholds) - 1,
            )
            if cur_compressed is None:
                continue

            joined = join_pmappings(
                cur_compressed,
                spec,
                _pmapping_row_filter_function=filter_func,
                print_progress=print_progress,
                metrics=metrics,
            )
            if i < len(thresholds) - 1:
                filter_func = OptimalityThresholder(
                    joined,
                    _pmapping_row_filter_function,
                    spec.mapper._metric_aggregator,
                    print_progress,
                )
        except Exception as e:
            if i == len(thresholds) - 1:
                raise
            if print_progress:
                print(f"Error with optimality threshold {threshold}: {e}")

    return joined


def multi_strategy_join(
    spec: Spec,
    compressed: dict[EinsumName, list[PmappingGroup]],
    print_progress: bool,
    metrics: Metrics,
    for_model: bool,
    _pmapping_row_filter_function: Callable[[pd.DataFrame], np.ndarray] | None = None,
):
    for _, p in compressed.items():
        for pg in p:
            pg.mappings.drop_valid_reservations = not (Metrics.RESOURCE_USAGE & metrics)

    # If it's for the model, just join things directly
    if for_model:
        return join_pmappings(
            deepcopy(compressed),
            spec,
            print_progress=print_progress,
            metrics=metrics,
            _pmapping_row_filter_function=_pmapping_row_filter_function,
        )

    if metrics & Metrics.RESOURCE_USAGE:
        return join_strategy_2(
            spec,
            compressed,
            print_progress,
            metrics,
            for_model,
            _pmapping_row_filter_function,
        )

    resource_usage_thresholds = [
        0.2,
        0.1,
        0.05,
        0.02,
        0.01,
        0.005,
        0.002,
        0.001,
        0.0001,
        0.00001,
        0,  # Give up, do full precision join
    ]
    for i, threshold in enumerate(resource_usage_thresholds):
        for p in compressed.values():
            for pg in p:
                pg.mappings.excess_resource_tolerance = threshold
        if i < len(resource_usage_thresholds) - 1 and print_progress:
            print(f"Dirty joining with resource usage <= {1 + threshold}× optimal")
        joined = join_strategy_2(
            spec,
            compressed,
            print_progress,
            metrics,
            for_model,
            _pmapping_row_filter_function,
            resource_usage_tolerance=threshold,
        )
        for c in joined.data.columns:
            if is_reservation_col(c):
                maxvalue = joined.data[c].max()
                if maxvalue > 1:
                    if print_progress:
                        oversubscribed = (
                            f"{col2reservation(c).name} ({maxvalue * 100:.2f}%)"
                        )
                        print(f"Oversubscribed {oversubscribed}. Reducing threshold...")
                    break
        else:
            if print_progress:
                print("Dirty joining mapping(s) valid & optimal! Returning...")
            return joined
    return joined


def clean_compress_and_join_pmappings(
    pmappings: MultiEinsumPmappings,
    metrics: Metrics,
    for_model: bool,
    require_all_einsums: bool = True,
    _pmapping_row_filter_function: Callable[[pd.Series], bool] | None = None,
    print_progress: bool = True,
) -> Mappings:
    einsum2pmappings = pmappings.einsum2pmappings
    if not require_all_einsums:
        einsum2pmappings = {
            k: v
            for k, v in pmappings.einsum2pmappings.items()
            if k in pmappings.einsums_with_pmappings_generated
        }
    _check_einsum2pmappings_not_empty(einsum2pmappings, pmappings)

    compressed, decompress_data = compress_einsum2pmappings(
        einsum2pmappings, print_progress
    )

    joined = multi_strategy_join(
        pmappings.spec,
        compressed,
        print_progress,
        metrics,
        for_model,
        _pmapping_row_filter_function,
    )

    joined = decompress_pmappings(joined, decompress_data)

    for einsum_name in einsum2pmappings:
        col = f"{einsum_name}<SEP>{MAPPING_COLUMN}"
        joined.data[col] = joined.data[col].apply(
            lambda x: pmappings.pmapping_objects[einsum_name][x]
        )
    joined._data = _fillna_and__numeric_cast(joined.data, 0).reset_index(drop=True)
    joined._data = joined._data.copy()  # Defrag

    rank_variable_bounds = get_rank_variable_bounds_for_all_einsums(pmappings.spec)
    einsum_names = list(einsum2pmappings.keys())
    joined.data[f"Total<SEP>{MAPPING_COLUMN}"] = [
        MappingFromRow(r, rank_variable_bounds, einsum_names)
        for _, r in joined.data.iterrows()
    ]
    # Fill nans with 0. We might get missing columns for some mapping entries if there
    # are energy entries for some pmappings but not others (e.g., one pmapping accesses
    # DRAM while another doesn't.)
    return Mappings(
        pmappings.spec,
        list(
            x
            for x in list(einsum2pmappings.keys())
            if x in pmappings.einsums_with_pmappings_generated
        ),
        joined.data,
        total_mappings=joined.n_total_pmappings,
        valid_mappings=joined.n_valid_pmappings,
        flattened_arches=pmappings.flattened_arches,
        evaluated_specs=pmappings.evaluated_specs,
    )


class PmappingsOneEinsum:
    def __init__(self, einsum_name: str, pm_group_list: list[PmappingGroup]):
        self.einsum_name: str = einsum_name
        self.pmapping_groups: list[PmappingGroup] = pm_group_list
        self.tensor_names: set[str] = oset(pm_group_list[0].tensor_names)

    def __getitem__(self, i):
        return self.pmapping_groups[i]


def make_full_equivalent_rank_variables(pairwise_equivalent_rank_variables):
    full_equivalent_rank_variables = {
        k: oset(v) for k, v in pairwise_equivalent_rank_variables.items()
    }
    changed = True
    while changed:
        changed = False
        for r in full_equivalent_rank_variables:
            for r2 in list(full_equivalent_rank_variables[r]):
                for r3 in list(full_equivalent_rank_variables[r2]):
                    if r3 in full_equivalent_rank_variables[r]:
                        continue
                    changed = True
                    full_equivalent_rank_variables[r].add(r3)
    return full_equivalent_rank_variables


def get_memories_to_track(
    pmapping_groups: dict[str, list[PmappingGroup]],
    print_progress: bool = True,
) -> tuple[dict[str, list[PmappingGroup]], set[str], set[str]]:

    always_below = oset()
    for _, einsum_pmapping_groups in pmapping_groups.items():
        for s in einsum_pmapping_groups:
            for col in s.mappings.data.columns:
                reservation_key = col2reservation(col)
                if reservation_key is not None:
                    always_below.add(reservation_key.name)

    total_sizes = {}
    ignored_resources = oset()

    for _, einsum_pmapping_groups in pmapping_groups.items():
        max_sizes = {}
        for s in einsum_pmapping_groups:
            n_fused_loops = s.compatibility.n_loops
            for col in s.mappings.data.columns:
                reservation_key = col2reservation(col)
                if reservation_key is None:
                    continue

                name = reservation_key.name
                nloops = reservation_key.nloops
                if name in always_below and nloops < n_fused_loops:
                    always_below.remove(name)
                # Check each of the compatibility's tensors
                for tensor in s.compatibility.tensors:
                    if tensor.resource_name in always_below:
                        always_below.remove(tensor.resource_name)
                size = s.mappings.data[col].max()
                max_sizes[name] = max(max_sizes.get(name, 0), size)

                # nloops < 0 means that the reservation will live through all Einsums
                if nloops < 0:
                    ignored_resources.add(name)

        for name, size in max_sizes.items():
            total_sizes[name] = total_sizes.get(name, 0) + size

    ignore = oset(t for t, s in total_sizes.items() if s <= 1) | always_below

    if not ignore:
        return pmapping_groups, ignore

    def remove_unneeded_columns(s: PmappingGroup):
        data = s.mappings.data
        keep_cols = []
        for col in data.columns:
            name_nloops = col2reservation(col)
            if name_nloops is None or name_nloops[0] not in ignore:
                keep_cols.append(col)
        run_pareto = len(keep_cols) < len(data.columns)
        data = data[keep_cols].copy() if len(keep_cols) < len(data.columns) else data
        return PmappingGroup(
            s.compatibility,
            s.mappings.update(data=data, skip_pareto=not run_pareto),
        )

    for a in sorted(always_below):
        if print_progress:
            print(
                f"Not tracking {a} because it is never reserved for multiple pmappings."
            )
    for t, s in sorted(total_sizes.items(), key=lambda x: x[1], reverse=True):
        if s <= 1:
            if print_progress:
                print(
                    f"Not tracking {t} because its size is enough for the sum of all "
                    f"reservations ({s * 100:.2f}% of the total)"
                )
            break

    new_pmapping_groups = {}
    for einsum_name, einsum_pmapping_groups in pmapping_groups.items():
        new_pmapping_groups[einsum_name] = parallel(
            [delayed(remove_unneeded_columns)(s) for s in einsum_pmapping_groups],
            pbar=(
                f"Removing unneeded reservations for {einsum_name}"
                if print_progress
                else None
            ),
        )
    return new_pmapping_groups, ignore


def join_pmappings(
    pmapping_groups: dict[str, list[PmappingGroup]],
    spec: Spec,
    lookahead_filter: bool = True,
    metrics: Metrics = None,
    _pmapping_row_filter_function: Callable[[pd.Series], bool] | None = None,
    print_progress: bool = True,
):
    """
    CONTRACT FOR MAPPINGS GETTING TO THIS POINT:

    - Reservations at a level include reservations at all levels above it.
    - If one Einsum uses an aliased tensor more than once, then only one
      reservation is made for it. If overlapping lifetimes cause the aliases to
      be alive at the same time, then it is handled here.
    - Memory names should be sorted with higher memory names representing
      memories lower in the hierarchy. e.g., memory 0 is the largest,
      memory 1 the next largest, and memory N is the smallest.
    """
    skip_invalid = spec.mapper._skip_invalid
    combine_reservations = spec.mapper._combine_reservations
    _runtime_log_file = spec.mapper._runtime_log_file

    assert (
        skip_invalid
    ), "Joining only joins valid compatibilities in the for loops in this function."

    drop_valid_reservations = not (Metrics.RESOURCE_USAGE & metrics)
    ignored_resources = oset()

    if _pmapping_row_filter_function is not None:
        n = sum(len(s.mappings.data) for sg in pmapping_groups.values() for s in sg)
        pmapping_groups = {
            e: [
                PmappingGroup(
                    s.compatibility,
                    s.mappings.filter_rows(_pmapping_row_filter_function),
                )
                for s in pmapping_groups[e]
            ]
            for e in pmapping_groups
        }
        new_n = sum(len(s.mappings.data) for sg in pmapping_groups.values() for s in sg)
        if print_progress:
            print(f"Filtered {n} -> {new_n} ({new_n / n:.2%} kept) pmappings")

    if drop_valid_reservations:
        pmapping_groups, ignored_resources = get_memories_to_track(
            pmapping_groups, print_progress
        )

    for einsum_name, einsum_pmapping_groups in pmapping_groups.items():
        for s in einsum_pmapping_groups:
            s.mappings.drop_valid_reservations = drop_valid_reservations

    aliased_tensors = spec.workload.get_tensor_copies()

    runtime = {}

    pmapping_groups = list(pmapping_groups.items())

    if not skip_invalid:
        lookahead_filter = False

    for einsum_name, s in pmapping_groups:
        if not s:
            raise ValueError(f"No pmappings for {einsum_name}")

    timer = JoiningTimer()

    pmgroups = [PmappingsOneEinsum(*s) for s in pmapping_groups]

    if not pmgroups:
        raise ValueError("No pmappings to join")

    # ======================================================================
    # Initial consolidate and group all PmappingGroups
    # ======================================================================
    for i, einsum_pmappings in enumerate(pmgroups):
        cur_tensors = einsum_pmappings.tensor_names
        right_tensors = oset.union(oset(), *[s.tensor_names for s in pmgroups[i + 1 :]])
        # First Einsum: Remove dead tensors and left consolidate. This is because the
        # first Einsum will have the first pmappigns that are joined from the left
        if i == 0:
            if cur_tensors - right_tensors:
                PmappingGroup.remove_dead_tensors(
                    einsum_pmappings.pmapping_groups, right_tensors
                )
                for s in einsum_pmappings.pmapping_groups:
                    s.compatibility = s.compatibility.clear_dead_tensors(right_tensors)
            einsum_pmappings.pmapping_groups = PmappingGroup.left_consolidate(
                einsum_pmappings.pmapping_groups,
                right_tensors,
                parallelize=False,  # We're not pareto pruning, so parallelization doesn't help.
                pbar=(
                    f"Inital consolidate {einsum_pmappings.einsum_name} ({i+1}/{len(pmgroups)})"
                    if print_progress
                    else None
                ),
            )
            continue

        # All other Einsums: Will be joined from the right. Remove dead tensors, right
        # consolidate, combine, group.
        t0 = time.time()
        left_tensors = oset.union(oset(), *[s.tensor_names for s in pmgroups[:i]])
        live_tensors = right_tensors
        shared_tensors = left_tensors & einsum_pmappings.tensor_names

        if cur_tensors - (right_tensors | left_tensors):
            PmappingGroup.remove_dead_tensors(
                einsum_pmappings.pmapping_groups, right_tensors | left_tensors
            )
            for s in einsum_pmappings.pmapping_groups:
                s.compatibility = s.compatibility.clear_dead_tensors(
                    right_tensors | left_tensors
                )

        einsum_pmappings.pmapping_groups = sorted(
            einsum_pmappings.pmapping_groups,
            key=lambda x: len(x.mappings.data),
            reverse=True,
        )
        einsum_pmappings.pmapping_groups = PmappingGroup.right_consolidate(
            einsum_pmappings.pmapping_groups,
            live_tensors,
            shared_tensors,
            parallelize=False,  # We're not pareto pruning, so parallelization doesn't help.
            pbar=(
                f"Inital consolidate {einsum_pmappings.einsum_name} ({i+1}/{len(pmgroups)})"
                if print_progress
                else None
            ),
        )
        einsum_pmappings.pmapping_groups = PmappingGroup.combine_combineable(
            einsum_pmappings.pmapping_groups,
            left_tensors | right_tensors,
            _combine_reservations=combine_reservations,
            pbar_postfix=f" for {einsum_pmappings.einsum_name} ({i+1}/{len(pmgroups)})",
            print_progress=print_progress,
        )
        einsum_pmappings.pmapping_groups = PmappingGroup.group(
            einsum_pmappings.pmapping_groups, left_tensors
        )
        einsum, prev_einsum = einsum_pmappings.einsum_name, pmgroups[i - 1].einsum_name
        step_time = time.time() - t0
        runtime[f"{prev_einsum} → {einsum}"] = step_time
        if _runtime_log_file:
            import json as _json

            with open(_runtime_log_file, "a") as _f:
                _f.write(
                    _json.dumps(
                        {
                            "step": f"{prev_einsum} → {einsum}",
                            "phase": "consolidate",
                            "time": step_time,
                        }
                    )
                    + "\n"
                )
        t0 = time.time()
    timer.print_time(f"Initial consolidate and group")

    n_iterations = 0
    total_iterations = len(pmgroups)

    def grab_einsum_pmappings() -> (
        tuple[dict[Compatibility, list[PmappingGroup]], str, set[str]]
    ):
        nonlocal n_iterations
        n_iterations += 1
        holder = pmgroups.pop(0)
        return holder.pmapping_groups, holder.einsum_name, holder.tensor_names

    if pmgroups:
        left, left_einsum, left_tensors = grab_einsum_pmappings()

    partial_mapping_size = 1
    while pmgroups:
        t0 = time.time()
        # ======================================================================
        # Check that data dependencies are satisfied.
        # ======================================================================
        for s in pmgroups:
            output_tensors = spec.workload.einsums[s.einsum_name].output_tensor_names
            shared_fail = left_tensors & output_tensors
            if shared_fail:
                raise ValueError(
                    f"Einsum {left_einsum} uses tensors {sorted(shared_fail)} that "
                    f"are outputs of Einsum {s.einsum_name}, which is later in the "
                    f"joining order."
                )

        # ======================================================================
        # Grab new Einsum from the right. Record logging data and find still
        # tensors that will be live after this Einsum.
        # ======================================================================
        right, right_einsum, right_tensors = grab_einsum_pmappings()
        logger.info(f"Einsum {right_einsum} ({n_iterations}/{total_iterations})")

        partial_mapping_size += 1

        live_tensors = oset.union(oset(), *[s.tensor_names for s in pmgroups])
        shared_tensors = oset(left_tensors) & oset(right_tensors)
        live_tensors_with_right = live_tensors | right_tensors

        # ======================================================================
        # Clean up the previously-combined PmappingGroups. Consolidate, combine, group
        # them into buckets.
        # ======================================================================
        # print_time(f"Consolidating")

        left = PmappingGroup.combine_combineable(
            left,
            live_tensors | right_tensors,
            _combine_reservations=combine_reservations,
            print_progress=print_progress,
        )

        # print_time(f"Combining")
        # Group left and right into buckets
        left = PmappingGroup.group(left, right_tensors)
        # print_time("Grouping")

        # ======================================================================
        # Remove dead tensors from left and right. This happens after grouping because
        # we only reserve space for shared tensors after they're dead (alive is handled
        # by the normal reservation system). This is in case the tensor lifetime extends
        # beyond the Einsums for which it is used.
        # ======================================================================
        PmappingGroup.remove_dead_tensors(
            [s for lr in [left, right] for v in lr.values() for s, _ in v], live_tensors
        )

        DO_PRINT = False
        DELAY = True
        # ======================================================================
        # Merge the left and right buckets.
        # ======================================================================
        combined: list[PmappingGroup] = []
        combined_ids: set[tuple[int, int, tuple[tuple[int, int], ...]]] = oset()

        for k in left:
            found = False
            if DO_PRINT:
                print(f"Left key {k}")
            for (a, perm_a), (b, perm_b) in itertools.product(
                left[k], right.get(k, [])
            ):
                a: PmappingGroup
                b: PmappingGroup
                perm_a: list[int]
                perm_b: list[int]
                key_check = (id(a), id(b))
                if key_check in combined_ids:
                    continue
                combined_ids.add(key_check)
                found = True

                compatibility_a = a.compatibility.permute(perm_a)
                compatibility_b = b.compatibility.permute(perm_b)
                try:
                    compatibility_joined = compatibility_a.merge_next(
                        compatibility_b,
                        live_tensors,
                    )
                    if DO_PRINT:
                        print(
                            f"\t{a.compatibility}        <-->        {b.compatibility}"
                        )
                except ValueError as e:  # Incompatible!
                    # if DO_PRINT:
                    #     print(f"\tIncompatible: {e}")
                    continue

                t0 = time.time()

                combined.append(
                    a.merge_next(
                        b,
                        live_tensors,
                        live_tensors_with_right,
                        aliased_tensors,
                        compatibility_joined=compatibility_joined,
                        permuted_compatibility_left=compatibility_a,
                        permuted_compatibility_right=compatibility_b,
                        delay=DELAY,
                        _pmapping_row_filter_function=_pmapping_row_filter_function,
                        ignored_resources=ignored_resources,
                    )
                )

                if DO_PRINT:
                    # s = f"\t-->\n\t{combined[-1].compatibility}"
                    # s += f"({len(a.mappings.data)})x({len(b.mappings.data)})"
                    # print(s)
                    pass
            if DO_PRINT and not found:
                for a, _ in left[k]:
                    print(f"\tNo match for {a.compatibility}")

        if DO_PRINT:
            for k in right:
                if k not in left:
                    for b, _ in right[k]:
                        print(f"\tREVERSE: No match for {b.compatibility} using {k}")

        for l in left.values():
            for s, _ in l:
                s.mappings = None
        for r in right.values():
            for s, _ in r:
                s.mappings = None

        # print_time("Bucket merging")
        def raise_no_match_error():
            estr = f"No match found for any group.\n"
            estr += f"Left compatibility:\n\t" + "\n\t".join(
                str(c) for c in left.keys()
            )
            estr += f"\nRight compatibility:\n\t" + "\n\t".join(
                str(c) for c in right.keys()
            )
            raise ValueError(estr)

        def no_match_lookahead_error(
            combined: list[PmappingGroup],
            next_keys: set[tuple[int, int, tuple[tuple[int, int], ...]]],
        ):
            estr = f"No match found for any group. Left and right joined successfully, "
            estr += f"but will not be compatible with following Einsums.\n"
            estr += f"Left compatibility:\n\t" + "\n\t".join(
                str(s.compatibility) for g in left.values() for s, _ in g
            )
            estr += f"\nRight compatibility:\n\t" + "\n\t".join(
                str(s.compatibility) for g in right.values() for s, _ in g
            )
            estr += f"\nCombined compatibility:\n\t" + "\n\t".join(
                str(s.compatibility) for s in combined
            )
            estr += f"\nFollowing Einsum compatibility:\n\t" + "\n\t".join(
                str(c) for c in next_keys
            )
            raise ValueError(estr)

        # ======================================================================
        # Look ahead to the next Einsum and see if any of our groups will not
        # be able to merge with it. If so, we can drop them immediately.
        # ======================================================================
        lookahead_filter = True
        if lookahead_filter:
            cur_tensors = left_tensors | right_tensors
            for next_pmapping_groups in pmgroups:
                next_right_tensors = next_pmapping_groups.tensor_names
                if not next_right_tensors & cur_tensors:
                    continue
                prev_combined = combined
                combined = PmappingGroup.group(combined, next_right_tensors)
                next_keys = oset(
                    c.clear_dead_tensors(
                        cur_tensors
                    ).clear_tile_patterns_and_reservation_indices()
                    for c in next_pmapping_groups.pmapping_groups
                )
                for k in list[PmappingGroup](combined):
                    perms = k.make_equivalent_permutations()
                    perms = [
                        p[0]
                        .clear_dead_tensors(next_right_tensors)
                        .clear_tile_patterns_and_reservation_indices()
                        for p in perms
                    ]
                    if not any(p in next_keys for p in perms):
                        if DO_PRINT:
                            for b, _ in combined[k]:
                                print(
                                    f"\tLOOKAHEAD to {next_pmapping_groups.einsum_name}: No match for {b.compatibility}"
                                )
                        del combined[k]
                if not combined:
                    PmappingGroup.group(prev_combined, next_right_tensors)
                    no_match_lookahead_error(prev_combined, next_keys)

                combined = list(itertools.chain.from_iterable(combined.values()))
                combined = [c[0] for c in combined]
                # Remove duplicates
                id2combined = {id(c): c for c in combined}
                combined = list(id2combined.values())
                # print(
                #     f"Removed {prev_len - len(combined)}/{prev_len} ({len(combined)/prev_len*100:.2f}% remaining)"
                # )
                # print_time("Removing mappings that can't be combined later")

        if not combined:
            raise_no_match_error()

        # ======================================================================
        # If we delayed the mapping merging, do it now.
        # ======================================================================
        import copy

        if DELAY:
            mappings = parallel(
                [c.mappings for c in combined],
                pbar=(
                    f"Joining pmappings for {left_einsum} <--> {right_einsum} ({n_iterations}/{total_iterations})"
                    if print_progress
                    else None
                ),
            )
            for c, mapping in zip(combined, mappings):
                c.mappings = mapping
        timer.print_time("Pmapping merging")

        if not any(len(s.mappings.data) for s in combined):
            # for c in prev_combined:  # For debugging the joining
            #     x = c.mappings
            #     x[0](*x[1], **x[2])
            raise ValueError(f"No mappings found for {left_einsum} <--> {right_einsum}")

        step_time = time.time() - t0
        runtime[f"{left_einsum} → {right_einsum}"] += step_time
        if _runtime_log_file:
            import json as _json

            with open(_runtime_log_file, "a") as _f:
                _f.write(
                    _json.dumps(
                        {
                            "step": f"{left_einsum} → {right_einsum}",
                            "phase": "join",
                            "time": step_time,
                        }
                    )
                    + "\n"
                )
        # # ======================================================================
        # # Print statements
        # # ======================================================================
        # logger.info(
        #     f"\tCombining {sum(len(s) for s in left.values())}({len(left)}) x {sum(len(s) for s in right.values())}({len(right)}) -> {len(combined)}"
        # )

        nmappings = sum(len(s.mappings.data) for s in combined)
        for_einsum_text = f"for Einsum {right_einsum}"
        # print(f"\tNumber of groups {for_einsum_text}: {len(combined)}")
        # for c in combined:
        #     print(f"\t\t{c.compatibility}")
        # print(f"\tNumber of mappings {for_einsum_text}: {nmappings}")
        # print(
        #     f"\tMappings per group {for_einsum_text}: {nmappings / len(combined)}"
        # )
        # logger.info(
        #     f"\tLargest left: {max(len(s2.mappings.data) for s in left.values() for s2, _ in s)}"
        # )
        # logger.info(
        #     f"\tLargest right: {max(len(s2.mappings.data) for s in right.values() for s2, _ in s)}"
        # )

        # ======================================================================
        # Update left for the next iteration.
        # =================================================================
        left = combined
        left_einsum = right_einsum
        left_tensors |= right_tensors

    # ======================================================================
    # Final consolidate and group
    # ======================================================================
    t0 = time.time()
    left = PmappingGroup.left_consolidate(
        left, None, pbar="Final consolidate" if print_progress else None
    )
    s_final = PmappingGroup.combine_combineable(
        left, oset(), print_progress=print_progress
    )
    assert len(s_final) == 1
    mappings = s_final[0].mappings
    mappings.limit_capacity(next_shared_loop_index=-1, finished=True)
    mappings.free_to_loop_index(-2)
    mappings.make_pareto()

    timer.log_total_time()
    # if evaluations_tracker is not None and "Total_latency" in data.columns and "Total_energy" in data.columns:
    #     edp = data["Total_latency"] * data["Total_energy"]
    #     edp_min = edp.min()
    #     evaluations_tracker.add_evaluation(n_evaluations, edp_min)
    #     evaluations_tracker.n_mappings.update(n_mappings)
    #     evaluations_tracker.runtime.update(runtime)

    return mappings


def _check_einsum2pmappings_not_empty(
    einsum2pmappings: dict[EinsumName, list[PmappingGroup]],
    pmappings: MultiEinsumPmappings,
):
    for einsum_name, einsum_pmappings in einsum2pmappings.items():
        total = sum(len(p.mappings.data) for p in einsum_pmappings)
        n_compatibilities = len(einsum_pmappings)
        logger.info(
            f"Einsum {einsum_name} has {total} pmappings with {n_compatibilities} compatibilities"
        )
        if total == 0:
            if einsum_name in pmappings.einsums_with_pmappings_generated:
                keep_rates = pmappings.pmapping_keep_rates(per_einsum=True)[einsum_name]
                keep_rates_text = "\n\t".join(
                    f"{k}: {v:.2e}" for k, v in keep_rates.items()
                )
                raise ValueError(
                    f"Einsum {einsum_name} has no pmappings. This likely means that "
                    f"no pmappings satisfied constraints for the Einsum. Please check "
                    f"the stats outputs from the MultiEinsumPmappings object returned "
                    f"by `af.mapper.FFM.make_pmappings(spec)`. The following are the "
                    f"keep rates (porportion of pmappings that are NOT pruned) for "
                    f"various causes of pmapping removal:\n\t{keep_rates_text}"
                )

            raise ValueError(
                f"Einsum {einsum_name} has no pmappings generated. It looks like you "
                "may have used `make_pmappings` with `einsum_names` set. You may set "
                "`require_all_einsums=False` to ignore this error and map only the "
                "Einsums that have pmappings."
            )


class MappingFromRow:
    def __init__(
        self,
        row: pd.Series,
        rank_variable_bounds: dict[str, dict[str, int]],
        einsum_names: list[EinsumName] | None = None,
    ):
        self.row = row
        self.rank_variable_bounds = rank_variable_bounds
        self.einsum_names = einsum_names

    def __call__(self, _for_model: bool = False) -> Mapping:
        return Mapping._from_pmappings(
            row2pmappings(self.row, self.einsum_names, self.rank_variable_bounds),
            rank_variable_bounds=self.rank_variable_bounds,
            _for_model=_for_model,
        )

    def _repr_svg_(self) -> str:
        return self.render()

    def render(self, **kwargs) -> str:
        return self().render(**kwargs)
