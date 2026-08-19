from copy import deepcopy
from accelforge.frontend.renames import TensorName
from accelforge.mapper.FFM._join_pmappings.compatibility import (
    Compatibility,
    TensorReservation,
    _reservation_structure,
)
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
    TensorPairConstraint,
    row2pmappings,
)
from accelforge.mapper.FFM._pareto_df.df_convention import (
    MAPPING_COLUMN,
    col_used_in_pareto,
    is_objective_col,
    is_reservation_col,
    is_tensor_col,
)
from accelforge.mapper.FFM._join_pmappings.pmapping_group import (
    PmappingGroup,
    Compatibility,
)
from accelforge.mapper.FFM._pareto_df.df_convention import (
    col2reservationiters,
    col2reservationsize,
    reservationkey2iterscol,
)
from accelforge.util import (
    _fillna_and_numeric_cast,
    delayed,
    fzs,
    get_n_parallel_jobs,
    oset,
    parallel,
)

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


def _apply_edp_columns(df: pd.DataFrame, metrics: Metrics) -> pd.DataFrame:
    if not metrics & Metrics.ENERGY_DELAY_PRODUCT:
        return df

    energy = df["Total<SEP>energy"]
    latency = df["Total<SEP>latency"]
    df["Total<SEP>energy_delay_product"] = energy * latency
    if not (metrics & Metrics.ENERGY):
        del df["Total<SEP>energy"]
    if not (metrics & Metrics.LATENCY):
        del df["Total<SEP>latency"]
    return df


class OptimalityThresholder:
    def __init__(
        self,
        prev_solutions: Mappings,
        _pmapping_row_filter_function: Callable[[pd.DataFrame], np.ndarray],
        print_progress: bool,
        metrics: Metrics,
    ):
        self.metrics = metrics
        compare_to = _apply_edp_columns(prev_solutions.data.copy(), metrics)
        compare_cols = [c for c in compare_to.columns if col_used_in_pareto(c)]
        self._pmapping_row_filter_function = _pmapping_row_filter_function

        compare_to = compare_to.sort_values(by=compare_cols, ascending=False)

        if len(compare_to) > 100:
            chosen_indices = np.round(np.linspace(0, len(compare_to) - 1, 100))
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
                        f"{k}={float(v):.2e}" for k, v in self.compare_to[-1].items()
                    )
                )

    def __call__(self, mapping: pd.DataFrame) -> bool:
        nondominated_by_all = np.ones(len(mapping), dtype=bool)

        edp_mapping = _apply_edp_columns(mapping.copy(), self.metrics)

        for c in self.compare_to:
            nondominated = np.zeros(len(edp_mapping), dtype=bool)
            for k, v in c.items():
                if k not in edp_mapping.columns:
                    nondominated |= True
                else:
                    nondominated |= edp_mapping[k] <= v
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
                    print_progress,
                    metrics,
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
            key = col2reservationsize(c)
            if key is not None:
                maxvalue = joined.data[c].max()
                if maxvalue > 1:
                    if print_progress:
                        oversubscribed = f"{key.name} ({maxvalue * 100:.2f}%)"
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

    # These are normally dropped during each join, but that never happens if there's
    # just one Einsum, so we do it once more here.
    joined.data.drop(
        columns=[c for c in joined.data.columns if is_tensor_col(c)], inplace=True
    )

    _apply_edp_columns(joined.data, metrics)
    # Pareto prune again in case the EDP column application reduced number of
    # objectives.
    joined.make_pareto()

    for einsum_name in einsum2pmappings:
        col = f"{einsum_name}<SEP>{MAPPING_COLUMN}"
        joined.data[col] = joined.data[col].apply(
            lambda x: pmappings.pmapping_objects[einsum_name][x]
        )
    joined._data = _fillna_and_numeric_cast(joined.data, 0).reset_index(drop=True)
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


def _prefix_concat(*dfs: pd.DataFrame) -> pd.DataFrame:
    prefixed = [df.add_prefix(f"{i}_") for i, df in enumerate(dfs)]
    return pd.concat(prefixed, axis=1, copy=False)


def make_valid_tensor_combinations(
    pmgroups: list[PmappingsOneEinsum],
) -> dict[oset[TensorName], dict[tuple, pd.DataFrame]]:
    """
    Returns a dict of {tensor names: {reservation structure: DataFrame of all possible
    shapes, one row per shape combination}}

    Tensor names, reservation structure, and shapes are all tuples, even if there's just
    one tensor. All are sorted in increasing tensor name order. Entires include both
    single tensors and every pair of tensors.

    Omits entries that only appear in one Einsum, since those will trivially be valid.
    """

    result: dict[oset[TensorName], dict[tuple, pd.DataFrame]] = {}
    in_gt_one_einsum = set()
    for einsum_pmappings in pmgroups:
        einsum_result = {}
        for pg in einsum_pmappings.pmapping_groups:
            tensors = sorted(pg.compatibility.tensors, key=lambda t: t.name)
            tensor2df = {
                t.name: t._compatibility_values_df(pg.mappings.data) for t in tensors
            }
            tensor_choices = list(itertools.combinations(tensors, 2))
            tensor_choices += [(t,) for t in pg.compatibility.tensors]

            for choice in tensor_choices:
                choice = sorted(choice, key=lambda t: t.name)
                tensor_name_key = tuple(t.name for t in choice)
                structure_key = tuple(_reservation_structure(x) for x in choice)
                dfs = [tensor2df[t.name] for t in choice]

                catted = _prefix_concat(*dfs).drop_duplicates()

                cur = einsum_result.setdefault(tensor_name_key, {})
                if structure_key in cur:
                    cur[structure_key] = pd.concat([cur[structure_key], catted])
                else:
                    cur[structure_key] = catted

        for tensor_key, s2valid in einsum_result.items():
            s2valid = {k: v.drop_duplicates() for k, v in s2valid.items()}
            # In other Einsums -> only keep matching structures and, for those
            # structures, only keep matching shapes
            if tensor_key in result:
                new = {}
                for k, v in s2valid.items():
                    if k in result[tensor_key]:
                        new[k] = result[tensor_key][k].merge(v)
                result[tensor_key] = new
                in_gt_one_einsum.add(tensor_key)
            # First time we're seeing this/these tensor(s) -> keep all combos
            else:
                result[tensor_key] = s2valid

    result = {k: v for k, v in result.items() if k in in_gt_one_einsum}

    return result


def filter_compatible_tensor_combos(
    pmgroups: list[PmappingsOneEinsum],
    valid_combinations: dict[tuple, dict[tuple, pd.DataFrame]],
):
    for einsum_pmappings in pmgroups:
        cur_groups = einsum_pmappings.pmapping_groups
        for pg in cur_groups:
            data = pg.mappings.data
            reservations = {t.name: t for t in pg.compatibility.tensors}
            keep = np.ones(len(data), dtype=bool)

            for names, structure2valid in valid_combinations.items():
                if not all(n in reservations for n in names):
                    continue
                if not keep.any():
                    break
                cur_reservations = [reservations[n] for n in names]

                # Match on reservation loop structure
                structure_key = tuple(
                    _reservation_structure(m) for m in cur_reservations
                )
                if structure_key not in structure2valid:
                    keep[:] = False
                    break

                # Match on the shapes
                valid_shapes = structure2valid[structure_key]
                dfs = [r._compatibility_values_df(data) for r in cur_reservations]
                catted = _prefix_concat(*dfs)

                keep &= pd.MultiIndex.from_frame(catted).isin(
                    pd.MultiIndex.from_frame(valid_shapes)
                )

            if not keep.all():
                pg.mappings = pg.mappings.update(
                    data=data[keep].reset_index(drop=True), skip_pareto=True
                )

        non_null = [g for g in cur_groups if len(g.mappings.data) > 0]
        einsum_pmappings.pmapping_groups = non_null


def _tensor_pair_constraints(
    left: Compatibility,
    right: Compatibility,
    valid_combinations: dict[tuple[str, str], dict[tuple, pd.DataFrame]],
) -> list[TensorPairConstraint] | None:
    """
    For two Compatibilities, return a list of TensorPairConstraints that describe valid
    combinations of fused tensor shapes. If no valid combinations exist, returns None.
    """
    # Shared tensor structure must match or we can't join
    for t in left.tensor_names & right.tensor_names:
        left_structure = left.get_reservation_of_tensor(t).loop_structure()
        right_structure = right.get_reservation_of_tensor(t).loop_structure()
        if left_structure != right_structure:
            return None

    # If a reservation is in both, overwriting is OK because joining will ensure that
    # the shape of all shared tensors is the same.
    reservations: dict[str, TensorReservation] = {}
    reservations.update({t.name: t for t in right.tensors})
    reservations.update({t.name: t for t in left.tensors})

    left_tensors = oset(t.name for t in left.tensors)

    # Pre-filtering should guarantee this
    for t, reservation in reservations.items():
        if (t,) in valid_combinations:
            assert (reservation.loop_structure(),) in valid_combinations[(t,)]

    constraints = []
    for ta, tb in itertools.combinations(sorted(reservations), 2):
        ta, tb = sorted((ta, tb))
        ra, rb = reservations[ta], reservations[tb]

        # Missing entries happen when all combinations are valid
        if (ta, tb) not in valid_combinations:
            continue

        # Both came from the same side -> we've checked this in an earlier step or
        # pre-filtering.
        is_left = tuple(n in left_tensors for n in (ta, tb))
        if len(set(is_left)) == 1:
            continue

        structure2valid = valid_combinations[(ta, tb)]
        structure_a = reservations[ta].loop_structure()
        structure_b = reservations[tb].loop_structure()

        # Missing structure -> no valid combinations exist for this choice
        if (structure_a, structure_b) not in structure2valid:
            return None

        valid_shapes = structure2valid[(structure_a, structure_b)]

        shape_symbols_a = ra.compatibility_shape_symbols()
        shape_symbols_b = rb.compatibility_shape_symbols()
        if not shape_symbols_a and not shape_symbols_b:
            continue

        constraints.append(
            TensorPairConstraint(
                valid_combinations=valid_shapes,
                shape_symbols_a=shape_symbols_a,
                shape_symbols_b=shape_symbols_b,
                is_left_a=is_left[0],
                is_left_b=is_left[1],
            )
        )

    return constraints


def get_memories_to_track(
    pmapping_groups: dict[str, list[PmappingGroup]],
    print_progress: bool = True,
) -> tuple[dict[str, list[PmappingGroup]], set[str], set[str]]:

    always_below = oset()
    for _, einsum_pmapping_groups in pmapping_groups.items():
        for s in einsum_pmapping_groups:
            for col in s.mappings.data.columns:
                reservation_key = col2reservationsize(col)
                if reservation_key is not None:
                    always_below.add(reservation_key.name)

    total_sizes = {}
    ignored_resources = oset()

    for _, einsum_pmapping_groups in pmapping_groups.items():
        max_sizes = {}
        for s in einsum_pmapping_groups:
            data = s.mappings.data
            n_fused_iters = 1
            for b in s.compatibility.loops:
                for l in b.loops:
                    n_fused_iters *= data[l.tile_pattern.calculated_n_iterations]
            for col in data.columns:
                reservation_key = col2reservationsize(col)
                if reservation_key is None:
                    continue

                name = reservation_key.name
                iters_above = data[reservationkey2iterscol(reservation_key)]
                if name in always_below and not np.all(n_fused_iters <= iters_above):
                    always_below.remove(name)
                # Check each of the compatibility's tensors
                for tensor in s.compatibility.tensors:
                    if tensor.resource_name in always_below:
                        always_below.remove(tensor.resource_name)
                size = data[col].max()
                max_sizes[name] = max(max_sizes.get(name, 0), size)

                # 0 iters_above above = persistent, lives through all Einsums
                if np.all(iters_above.values == 0):
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
            name_nloops = col2reservationsize(col) or col2reservationiters(col)
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

    valid_combinations = make_valid_tensor_combinations(pmgroups)
    # prev_len = {g.einsum_name: sum(len(pg.mappings.data) for pg in g.pmapping_groups) for g in pmgroups}
    # print(f"Length before filtering: {prev_len}")
    filter_compatible_tensor_combos(pmgroups, valid_combinations)
    # new_len = {g.einsum_name: sum(len(pg.mappings.data) for pg in g.pmapping_groups) for g in pmgroups}
    # print(f"Length after filtering: {new_len}")
    timer.print_time("Valid tensor combinations")

    _constraints_cache = {}

    def tensor_pair_constraints(
        a: Compatibility, b: Compatibility
    ) -> list[TensorPairConstraint] | None:
        key = (a, b)
        if key not in _constraints_cache:
            _constraints_cache[key] = _tensor_pair_constraints(a, b, valid_combinations)
        return _constraints_cache[key]

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

    def grab_einsum_pmappings() -> tuple[list[PmappingGroup], str, set[str]]:
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

        # =============================================================================
        # If we're multiprocessing and the left side has fewer groups than the number of
        # processes, repeatedly split the largest group in half so that the merge work
        # below can fan out across all workers.
        # =============================================================================
        n_procs = get_n_parallel_jobs()
        for _ in range(n_procs - len(left)):
            best_i, best_len = None, -1
            for i, pg in enumerate(left):
                length = len(pg.mappings.data) if pg.mappings is not None else 0
                if length > best_len:
                    best_len = length
                    best_i = i
            if best_i is None or best_len < 2:
                break  # nothing left worth splitting
            first, second = left[best_i].split_in_half()
            left[best_i] = first
            left.append(second)

        # ======================================================================
        # Remove dead tensors from left and right. This happens after grouping because
        # we only reserve space for shared tensors after they're dead (alive is handled
        # by the normal reservation system). This is in case the tensor lifetime extends
        # beyond the Einsums for which it is used.
        # ======================================================================
        PmappingGroup.remove_dead_tensors(left + right, live_tensors)

        DO_PRINT = False
        DELAY = True
        # =============================================================================
        # Merge each compatible (left, right) pair
        # =============================================================================
        combined: list[PmappingGroup] = []

        for a, b in itertools.product(left, right):
            a: PmappingGroup
            b: PmappingGroup
            constraints = tensor_pair_constraints(a.compatibility, b.compatibility)
            if constraints is None:
                continue
            try:
                join_options = a.compatibility.merge_next(
                    b.compatibility,
                    live_tensors,
                )
                if DO_PRINT:
                    print(f"\t{a.compatibility}        <-->        {b.compatibility}")
            except ValueError as e:  # Incompatible!
                continue

            t0 = time.time()

            for compatibility_joined, left_loop_to_right_loop in join_options:
                combined.append(
                    a.merge_next(
                        b,
                        live_tensors,
                        live_tensors_with_right,
                        aliased_tensors,
                        compatibility_joined=compatibility_joined,
                        left_loop_to_right_loop=left_loop_to_right_loop,
                        tensor_pair_constraints=constraints,
                        delay=DELAY,
                        _pmapping_row_filter_function=_pmapping_row_filter_function,
                        ignored_resources=ignored_resources,
                    )
                )

        for s in left:
            s.mappings = None
        for s in right:
            s.mappings = None

        # print_time("Bucket merging")
        def raise_no_match_error():
            estr = f"No match found for any group.\n"
            estr += f"Left compatibility:\n\t" + "\n\t".join(
                str(s.compatibility) for s in left
            )
            estr += f"\nRight compatibility:\n\t" + "\n\t".join(
                str(s.compatibility) for s in right
            )
            raise ValueError(estr)

        def no_match_lookahead_error(
            combined: list[PmappingGroup],
            next_keys: set[tuple[int, int, tuple[tuple[int, int], ...]]],
        ):
            estr = f"No match found for any group. Left and right joined successfully, "
            estr += f"but will not be compatible with following Einsums.\n"
            estr += f"Left compatibility:\n\t" + "\n\t".join(
                str(s.compatibility) for s in left
            )
            estr += f"\nRight compatibility:\n\t" + "\n\t".join(
                str(s.compatibility) for s in right
            )
            estr += f"\nCombined compatibility:\n\t" + "\n\t".join(
                str(s.compatibility) for s in combined
            )
            estr += f"\nFollowing Einsum compatibility:\n\t" + "\n\t".join(
                str(c) for c in next_keys
            )
            raise ValueError(estr)

        # # ======================================================================
        # # Look ahead to the next Einsum and see if any of our groups will not
        # # be able to merge with it. If so, we can drop them immediately.
        # # ======================================================================
        # lookahead_filter = True
        # if lookahead_filter:
        #     cur_tensors = left_tensors | right_tensors
        #     for next_pmapping_groups in pmgroups:
        #         next_right_tensors = next_pmapping_groups.tensor_names
        #         if not next_right_tensors & cur_tensors:
        #             continue
        #         prev_combined = combined
        #         combined = PmappingGroup.group(combined, next_right_tensors)
        #         next_keys = oset(
        #             c.clear_dead_tensors(
        #                 cur_tensors
        #             ).clear_tile_patterns_and_reservation_indices()
        #             for c in next_pmapping_groups.pmapping_groups
        #         )
        #         for k in list[Compatibility](combined):
        #             perms = k.make_equivalent_compatibilities()
        #             perms = [
        #                 p[0]
        #                 .clear_dead_tensors(next_right_tensors)
        #                 .clear_tile_patterns_and_reservation_indices()
        #                 for p in perms
        #             ]
        #             if not any(p in next_keys for p in perms):
        #                 if DO_PRINT:
        #                     for b, _ in combined[k]:
        #                         print(
        #                             f"\tLOOKAHEAD to {next_pmapping_groups.einsum_name}: No match for {b.compatibility}"
        #                         )
        #                 del combined[k]
        #         if not combined:
        #             PmappingGroup.group(prev_combined, next_right_tensors)
        #             no_match_lookahead_error(prev_combined, next_keys)

        #         combined = list(itertools.chain.from_iterable(combined.values()))
        #         combined = [c[0] for c in combined]
        #         # Remove duplicates
        #         id2combined = {id(c): c for c in combined}
        #         combined = list(id2combined.values())
        #         # print(
        #         #     f"Removed {prev_len - len(combined)}/{prev_len} ({len(combined)/prev_len*100:.2f}% remaining)"
        #         # )
        #         # print_time("Removing mappings that can't be combined later")

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
    mappings.free_all_reservations()
    mappings.drop_redundant_reservations()
    mappings.limit_capacity(finished=True)
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
