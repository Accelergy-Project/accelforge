from collections import defaultdict
import itertools
import logging
import time
from typing import Callable

from fastfusion._accelerated_imports import pd
from fastfusion.frontend.spec import Spec
from fastfusion.frontend.mapping import Mapping
from fastfusion.frontend.mapper.metrics import Metrics
from fastfusion.frontend.workload import EinsumName
from fastfusion.mapper.FFM.mappings import Mappings
from fastfusion.mapper.FFM.pmappings import MultiEinsumPmappings
from fastfusion.mapper.FFM._join_pmappings.compress_pmappings import (
    compress_einsum2pmappings,
    decompress_pmappings,
)
from fastfusion.mapper.FFM._make_pmappings.make_pmappings import (
    get_rank_variable_bounds_for_all_einsums,
)
from fastfusion.mapper.FFM._join_pmappings.pmapping_dataframe import (
    row2pmappings,
)
from fastfusion.mapper.FFM._pareto_df.df_convention import MAPPING_COLUMN
from fastfusion.mapper.FFM._join_pmappings.pmapping_group import (
    PmappingGroup,
    Compatibility,
)
from fastfusion.mapper.FFM._pareto_df.df_convention import col2nameloop
from fastfusion.util import parallel, delayed


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


def clean_compress_and_join_pmappings(
    spec: Spec,
    pmappings: MultiEinsumPmappings,
    require_all_einsums: bool = True,
    _pmapping_row_filter_function: Callable[[pd.Series], bool] | None = None,
) -> Mappings:
    einsum2pmappings = pmappings.einsum2pmappings
    if not require_all_einsums:
        einsum2pmappings = {
            k: v
            for k, v in pmappings.einsum2pmappings.items()
            if k in pmappings.einsums_with_pmappings_generated
        }
    _check_einsum2pmappings_not_empty(einsum2pmappings, pmappings)

    compressed, decompress_data = compress_einsum2pmappings(einsum2pmappings)
    joined = join_pmappings(
        compressed,
        spec,
        pmappings.resource2capacity,
        _pmapping_row_filter_function=_pmapping_row_filter_function,
    )
    joined = decompress_pmappings(joined, decompress_data)

    for einsum_name in einsum2pmappings:
        col = f"{einsum_name}<SEP>{MAPPING_COLUMN}"
        joined.data[col] = joined.data[col].apply(
            lambda x: pmappings.pmapping_objects[einsum_name][x]
        )
    joined._data = joined.data.fillna(0).reset_index(drop=True)

    rank_variable_bounds = get_rank_variable_bounds_for_all_einsums(spec)
    einsum_names = list(einsum2pmappings.keys())
    joined.data[f"Total<SEP>{MAPPING_COLUMN}"] = [
        MappingFromRow(r, rank_variable_bounds, einsum_names)
        for _, r in joined.data.iterrows()
    ]
    # Fill nans with 0. We might get missing columns for some mapping entries if there
    # are energy entries for some pmappings but not others (e.g., one pmapping accesses
    # DRAM while another doesn't.)
    return Mappings(
        spec,
        list(
            x
            for x in list(einsum2pmappings.keys())
            if x in pmappings.einsums_with_pmappings_generated
        ),
        joined.data,
        total_mappings=joined.n_total_pmappings,
        valid_mappings=joined.n_valid_pmappings,
    )


class PmappingsOneEinsum:
    def __init__(self, einsum_name: str, pm_group_list: list[PmappingGroup]):
        self.einsum_name: str = einsum_name
        self.pmapping_groups: list[PmappingGroup] = pm_group_list
        self.tensor_names: set[str] = set(pm_group_list[0].tensor_names)

    def __getitem__(self, i):
        return self.pmapping_groups[i]


def make_full_equivalent_rank_variables(pairwise_equivalent_rank_variables):
    full_equivalent_rank_variables = {
        k: set(v) for k, v in pairwise_equivalent_rank_variables.items()
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
    resource2capacity: dict[str, int],
) -> tuple[dict[str, list[PmappingGroup]], set[str], set[str]]:

    always_below = set(resource2capacity.keys())
    total_sizes = {}
    ignored_resources = set()

    for _, einsum_pmapping_groups in pmapping_groups.items():
        max_sizes = {}
        for s in einsum_pmapping_groups:
            n_fused_loops = s.compatibility.n_loops
            for col in s.mappings.data.columns:
                name_nloops = col2nameloop(col)
                if name_nloops is None:
                    continue

                name, nloops = name_nloops
                if name in always_below and nloops < n_fused_loops:
                    always_below.remove(name)
                size = s.mappings.data[col].max()
                max_sizes[name] = max(max_sizes.get(name, 0), size)

                # nloops < 0 means that the reservation will live through all Einsums
                if nloops < 0:
                    ignored_resources.add(name)

        for name, size in max_sizes.items():
            total_sizes[name] = total_sizes.get(name, 0) + size

    ignore = (
        set(t for t, s in total_sizes.items() if resource2capacity[t] >= s)
        | always_below
    )

    if not ignore:
        return pmapping_groups, ignore

    def remove_unneeded_columns(s: PmappingGroup):
        data = s.mappings.data
        keep_cols = []
        for col in data.columns:
            name_nloops = col2nameloop(col)
            if name_nloops is None or name_nloops[0] not in ignore:
                keep_cols.append(col)
        run_pareto = len(keep_cols) < len(data.columns)
        return PmappingGroup(
            s.compatibility,
            s.mappings.update(data=data[keep_cols], skip_pareto=not run_pareto),
        )

    for a in sorted(always_below):
        print(f"Not tracking {a} because it is never reserved for multiple pmappings.")
    for t, s in sorted(total_sizes.items(), key=lambda x: x[1], reverse=True):
        if resource2capacity[t] >= s:
            print(
                f"Not tracking {t} because its size {resource2capacity[t]} is enough "
                f"for the sum of all reservations ({s})"
            )
            break

    new_pmapping_groups = {}
    for einsum_name, einsum_pmapping_groups in pmapping_groups.items():
        new_pmapping_groups[einsum_name] = list(
            parallel(
                [delayed(remove_unneeded_columns)(s) for s in einsum_pmapping_groups],
                pbar=f"Removing unneeded reservations for {einsum_name}",
                return_as="generator",
            )
        )
    return new_pmapping_groups, ignore


def join_pmappings(
    pmapping_groups: dict[str, list[PmappingGroup]],
    spec: Spec,
    resource2capacity: dict[str, int],
    # Optimality-maintaining optimizations.
    skip_invalid: bool = True,
    combine_reservations: bool = True,
    lookahead_filter: bool = True,
    metrics: Metrics = None,
    _pmapping_row_filter_function: Callable[[pd.Series], bool] | None = None,
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

    metrics = spec.mapper.ffm.metrics

    drop_valid_reservations = not (Metrics.RESOURCE_USAGE & metrics)
    ignored_resources = set()

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
        print(f"Filtered {n} -> {new_n} ({new_n / n:.2%} kept) pmappings")

    if drop_valid_reservations:
        pmapping_groups, ignored_resources = get_memories_to_track(
            pmapping_groups, resource2capacity
        )

    mixable_ranks = spec.workload._get_ranks_that_share_indexing_rank_variables()

    aliased_tensors = spec.workload.get_tensor_copies()

    n_mappings = {}
    runtime = {}
    nbuckets = []

    n_evaluations = 0

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
    n_mappings["Post Intra-Layer"] = 0
    for i, einsum_pmappings in enumerate(pmgroups):
        cur_tensors = einsum_pmappings.tensor_names
        right_tensors = set.union(set(), *[s.tensor_names for s in pmgroups[i + 1 :]])
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
                pbar=f"Inital consolidate {einsum_pmappings.einsum_name} ({i+1}/{len(pmgroups)})",
            )
            continue

        # All other Einsums: Will be joined from the right. Remove dead tensors, right
        # consolidate, combine, group.
        t0 = time.time()
        left_tensors = set.union(set(), *[s.tensor_names for s in pmgroups[:i]])
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
            pbar=f"Inital consolidate {einsum_pmappings.einsum_name} ({i+1}/{len(pmgroups)})",
        )
        einsum_pmappings.pmapping_groups = PmappingGroup.combine_combineable(
            einsum_pmappings.pmapping_groups,
            left_tensors | right_tensors,
            combine_reservations=combine_reservations,
            pbar_postfix=f" for {einsum_pmappings.einsum_name} ({i+1}/{len(pmgroups)})",
        )
        n_mappings["Post Intra-Layer"] += sum(
            len(s.mappings.data) for s in einsum_pmappings.pmapping_groups
        )
        einsum_pmappings.pmapping_groups = PmappingGroup.group(
            einsum_pmappings.pmapping_groups, left_tensors
        )
        einsum, prev_einsum = einsum_pmappings.einsum_name, pmgroups[i - 1].einsum_name
        runtime[f"{prev_einsum} → {einsum}"] = time.time() - t0
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
        # Grab new Einsum from the right. Record logging data and find still
        # tensors that will be live after this Einsum.
        # ======================================================================
        nbuckets.append(len(left))
        # nmappings.append(sum(len(s.mappings.data) for s in left))
        right, right_einsum, right_tensors = grab_einsum_pmappings()
        logger.info(f"Einsum {right_einsum} ({n_iterations}/{total_iterations})")

        partial_mapping_size += 1

        live_tensors = set.union(set(), *[s.tensor_names for s in pmgroups])
        shared_tensors = set(left_tensors) & set(right_tensors)
        live_tensors_with_right = live_tensors | right_tensors

        # ======================================================================
        # Clean up the previously-combined PmappingGroups. Consolidate, combine, group
        # them into buckets.
        # ======================================================================
        # print_time(f"Consolidating")

        left = PmappingGroup.combine_combineable(
            left,
            live_tensors | right_tensors,
            combine_reservations=combine_reservations,
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
        cur_nmappings = 0
        combined_ids: set[tuple[int, int, tuple[tuple[int, int], ...]]] = set()

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
                key_check = (
                    id(a),
                    id(b),
                    tuple((pa, pb) for pa, pb in zip(perm_a, perm_b)),
                )
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
                        mixable_ranks,
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
                        resource2capacity=resource2capacity,
                        drop_valid_reservations=drop_valid_reservations,
                        delay=DELAY,
                        _pmapping_row_filter_function=_pmapping_row_filter_function,
                        ignored_resources=ignored_resources,
                    )
                )
                t1 = time.time()
                # print(f'Took {t1 - t0:.2f} seconds to generate {len(combined[-1].mappings.data)} mappings')

                if not DELAY:
                    cur_nmappings += len(a.mappings.data) * len(b.mappings.data)
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
        if lookahead_filter:
            cur_tensors = left_tensors | right_tensors
            for next_pmapping_groups in pmgroups:
                next_right_tensors = next_pmapping_groups.tensor_names
                if not next_right_tensors & cur_tensors:
                    continue
                prev_combined = combined
                combined = PmappingGroup.group(combined, next_right_tensors)
                next_keys = {
                    c.clear_dead_tensors(
                        cur_tensors
                    ).clear_tile_patterns_and_reservation_indices()
                    for c in next_pmapping_groups.pmapping_groups
                }
                for k in list(combined):
                    k_cleared = k.clear_dead_tensors(
                        next_right_tensors
                    ).clear_tile_patterns_and_reservation_indices()
                    if k_cleared not in next_keys:
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
        if DELAY:
            mappings = parallel(
                [c.mappings for c in combined],
                pbar=f"Joining pmappings for {left_einsum} <--> {right_einsum} ({n_iterations}/{total_iterations})",
                return_as="generator",
            )
            for c, mapping in zip(combined, mappings):
                c.mappings = mapping
                cur_nmappings += c.n_pre_prune_mappings
        timer.print_time("Pmapping merging")

        prev_nmappings = cur_nmappings
        if not skip_invalid:
            left_nmappings = sum(len(s.mappings.data) for k in left.values() for s in k)
            right_nmappings = sum(
                len(s.mappings.data) for k in right.values() for s in k
            )
            cur_nmappings = left_nmappings * right_nmappings
        n_mappings[f"{left_einsum} → {right_einsum}"] = cur_nmappings
        n_evaluations += cur_nmappings
        runtime[f"{left_einsum} → {right_einsum}"] += (time.time() - t0) * (
            cur_nmappings / prev_nmappings
        )
        # print(
        #     f'Scaled runtime by {cur_nmappings / prev_nmappings}. Runtime: {runtime[f"{prev_einsum} → {einsum}"]:.2f}'
        # )

        # ======================================================================
        # Print statements
        # ======================================================================
        logger.info(
            f"\tCombining {sum(len(s) for s in left)}({len(left)}) x {sum(len(s) for s in right)}({len(right)}) -> {len(combined)}"
        )

        nmappings = sum(len(s.mappings.data) for s in combined)
        for_einsum_text = f"for Einsum {right_einsum}"
        logger.info(f"\tNumber of groups {for_einsum_text}: {len(combined)}")
        # for c in combined:
        #     print(f"\t\t{c.compatibility}")
        logger.info(f"\tNumber of mappings {for_einsum_text}: {nmappings}")
        logger.info(
            f"\tMappings per group {for_einsum_text}: {nmappings / len(combined)}"
        )
        logger.info(
            f"\tLargest left: {max(len(s2.mappings.data) for s in left.values() for s2, _ in s)}"
        )
        logger.info(
            f"\tLargest right: {max(len(s2.mappings.data) for s in right.values() for s2, _ in s)}"
        )

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
    left = PmappingGroup.left_consolidate(left, None, pbar="Final consolidate")
    s_final = PmappingGroup.combine_combineable(left, set())
    assert len(s_final) == 1
    mappings = s_final[0].mappings

    timer.log_total_time()
    # if evaluations_tracker is not None and "Total_latency" in data.columns and "Total_energy" in data.columns:
    #     edp = data["Total_latency"] * data["Total_energy"]
    #     edp_min = edp.min()
    #     evaluations_tracker.add_evaluation(n_evaluations, edp_min)
    #     evaluations_tracker.n_mappings.update(n_mappings)
    #     evaluations_tracker.runtime.update(runtime)

    return mappings


def _check_einsum2pmappings_not_empty(einsum2pmappings, pmappings):
    for einsum_name, einsum_pmappings in einsum2pmappings.items():
        total = sum(len(p.mappings.data) for p in einsum_pmappings)
        n_compatibilities = len(einsum_pmappings)
        logger.info(
            f"Einsum {einsum_name} has {total} pmappings with {n_compatibilities} compatibilities"
        )
        if total == 0:
            if einsum_name in pmappings.einsums_with_pmappings_generated:
                raise ValueError(
                    f"Einsum {einsum_name} has no pmappings. This likely means that "
                    f"no pmappings satisfied constraints for the Einsum. Please check "
                    f"the stats outputs from the MultiEinsumPmappings object."
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

    def __call__(self) -> Mapping:
        return Mapping._from_pmappings(
            row2pmappings(self.row, self.einsum_names, self.rank_variable_bounds),
            rank_variable_bounds=self.rank_variable_bounds,
        )

    def _repr_svg_(self) -> str:
        return self.render()

    def render(self) -> str:
        return self().render()
