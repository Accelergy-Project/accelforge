from collections import defaultdict
import itertools
import logging
import time

from fastfusion.accelerated_imports import pd
from fastfusion.frontend.specification import Specification
from fastfusion.frontend.mapper.metrics import Metrics
from fastfusion.mapper.FFM._join_pmappings.sim import SIM, Compatibility
from fastfusion.mapper.FFM._pmapping_group import PmappingGroup
from fastfusion.mapper.FFM._pmapping_group.df_convention import col2nameloop
from fastfusion.util import parallel, delayed


def paretofy(k, v):
    return SIM(k, PmappingGroup(pd.DataFrame(v).fillna(0)))


prev_time = 0
total_time = defaultdict(int)


def init_print_time():
    global prev_time, total_time
    prev_time = time.time()
    total_time = defaultdict(int)


def print_time(what: str):
    global prev_time
    t = time.time() - prev_time
    logging.info(f"{what}: {t:.2f} seconds")
    total_time[what] += t
    prev_time = time.time()


def print_total_time():
    logging.info(f"\n======== Total time ========")
    for k, v in total_time.items():
        logging.info(f"{k}: {v:.2f} seconds")
    total = sum(total_time.values())
    if total > 60:
        logging.info(f"\nTotal: {total:.2f} seconds ({total/60:.2f} minutes)")
    else:
        logging.info(f"\nTotal: {total:.2f} seconds")
    logging.info(f"============================\n")


class GroupOfSIMsHolder:
    def __init__(self, einsum_name: str, sim_list: list[SIM]):
        self.einsum_name: str = einsum_name
        self.sims: list[SIM] = sim_list
        self.tensor_names: set[str] = set(sim_list[0].tensor_names)

    def __getitem__(self, i):
        return self.sims[i]


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
    sims: dict[str, list[SIM]],
    resource2capacity: dict[str, int],
) -> dict[str, list[SIM]]:

    always_below = set(resource2capacity.keys())
    total_sizes = {}

    for _, einsum_sims in sims.items():
        max_sizes = {}
        for s in einsum_sims:
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

        for name, size in max_sizes.items():
            total_sizes[name] = total_sizes.get(name, 0) + size

    ignore = (
        set(t for t, s in total_sizes.items() if resource2capacity[t] >= s)
        | always_below
    )

    if not ignore:
        return sims

    def remove_unneeded_columns(s: SIM):
        data = s.mappings.data
        keep_cols = []
        for col in data.columns:
            name_nloops = col2nameloop(col)
            if name_nloops is None or name_nloops[0] not in ignore:
                keep_cols.append(col)
        run_pareto = len(keep_cols) < len(data.columns)
        return SIM(
            s.compatibility,
            PmappingGroup(data[keep_cols], skip_pareto=not run_pareto),
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

    new_sims = {}
    for einsum_name, einsum_sims in sims.items():
        new_sims[einsum_name] = list(
            parallel(
                [delayed(remove_unneeded_columns)(s) for s in einsum_sims],
                pbar=f"Removing unneeded reservations for {einsum_name}",
                return_as="generator",
            )
        )
    return new_sims, ignore


def join_sims(
    sims: dict[str, list[SIM]],
    spec: Specification,
    resource2capacity: dict[str, int],
    # Optimality-maintaining optimizations.
    skip_invalid: bool = True,
    combine_reservations: bool = True,
    lookahead_filter: bool = True,
    metrics: Metrics = None,
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
    ignore_reservations = set()
    if drop_valid_reservations:
        sims, ignore_reservations = get_memories_to_track(sims, resource2capacity)

    mixable_ranks = spec.workload.get_mixable_ranks()

    aliased_tensors = spec.workload.get_tensor_copies()

    n_mappings = {}
    runtime = {}
    nbuckets = []

    n_evaluations = 0

    sims = list(sims.items())

    if not skip_invalid:
        lookahead_filter = False

    for einsum_name, s in sims:
        if not s:
            raise ValueError(f"No pmappings for {einsum_name}")
    init_print_time()

    sims = [GroupOfSIMsHolder(*s) for s in sims]

    if not sims:
        raise ValueError("No pmappings to join")

    # ======================================================================
    # Initial consolidate and group all SIMs
    # ======================================================================
    n_mappings["Post Intra-Layer"] = 0
    for i, sim_holder in enumerate(sims):
        cur_tensors = sim_holder.tensor_names
        right_tensors = set.union(set(), *[s.tensor_names for s in sims[i + 1 :]])
        # First Einsum: Remove dead tensors and left consolidate. This is because the
        # first Einsum will have the first pmappigns that are joined from the left
        if i == 0:
            if cur_tensors - right_tensors:
                SIM.remove_dead_tensors(sim_holder.sims, right_tensors)
                for s in sim_holder.sims:
                    s.compatibility = s.compatibility.clear_dead_tensors(right_tensors)
            sim_holder.sims = SIM.left_consolidate(
                sim_holder.sims,
                right_tensors,
                parallelize=False,  # We're not pareto pruning, so parallelization doesn't help.
                pbar=f"Inital consolidate {sim_holder.einsum_name} ({i+1}/{len(sims)})",
            )
            continue

        # All other Einsums: Will be joined from the right. Remove dead tensors, right
        # consolidate, combine, group.
        t0 = time.time()
        left_tensors = set.union(set(), *[s.tensor_names for s in sims[:i]])
        live_tensors = right_tensors
        shared_tensors = left_tensors & sim_holder.tensor_names

        if cur_tensors - (right_tensors | left_tensors):
            SIM.remove_dead_tensors(sim_holder.sims, right_tensors | left_tensors)
            for s in sim_holder.sims:
                s.compatibility = s.compatibility.clear_dead_tensors(
                    right_tensors | left_tensors
                )

        sim_holder.sims = sorted(
            sim_holder.sims, key=lambda x: len(x.mappings.data), reverse=True
        )
        sim_holder.sims = SIM.right_consolidate(
            sim_holder.sims,
            live_tensors,
            shared_tensors,
            parallelize=False,  # We're not pareto pruning, so parallelization doesn't help.
            pbar=f"Inital consolidate {sim_holder.einsum_name} ({i+1}/{len(sims)})",
        )
        sim_holder.sims = SIM.combine_combineable(
            sim_holder.sims,
            left_tensors | right_tensors,
            combine_reservations=combine_reservations,
            pbar_postfix=f" for {sim_holder.einsum_name} ({i+1}/{len(sims)})",
        )
        n_mappings["Post Intra-Layer"] += sum(
            len(s.mappings.data) for s in sim_holder.sims
        )
        sim_holder.sims = SIM.group(sim_holder.sims, left_tensors, drop_tags=True)
        einsum, prev_einsum = sim_holder.einsum_name, sims[i - 1].einsum_name
        runtime[f"{prev_einsum} → {einsum}"] = time.time() - t0
        t0 = time.time()
    print_time(f"Initial consolidate and group")

    n_iterations = 0
    total_iterations = len(sims)

    def grab_sim_holder() -> tuple[dict[Compatibility, list[SIM]], str, set[str]]:
        nonlocal n_iterations
        n_iterations += 1
        holder = sims.pop(0)
        return holder.sims, holder.einsum_name, holder.tensor_names

    if sims:
        left, left_einsum, left_tensors = grab_sim_holder()

    partial_mapping_size = 1
    while sims:
        t0 = time.time()
        # ======================================================================
        # Grab new Einsum from the right. Record logging data and find still
        # tensors that will be live after this Einsum.
        # ======================================================================
        nbuckets.append(len(left))
        # nmappings.append(sum(len(s.mappings.data) for s in left))
        right, right_einsum, right_tensors = grab_sim_holder()
        logging.info(f"Einsum {right_einsum} ({n_iterations}/{total_iterations})")

        partial_mapping_size += 1

        live_tensors = set.union(set(), *[s.tensor_names for s in sims])
        shared_tensors = set(left_tensors) & set(right_tensors)
        live_tensors_with_right = live_tensors | right_tensors

        # ======================================================================
        # Clean up the previously-combined SIMs. Consolidate, combine, group
        # them into buckets.
        # ======================================================================
        # print_time(f"Consolidating")

        left = SIM.combine_combineable(
            left,
            live_tensors | right_tensors,
            combine_reservations=combine_reservations,
        )

        # print_time(f"Combining")
        # Group left and right into buckets
        left = SIM.group(left, right_tensors, drop_tags=True)
        # print_time("Grouping")

        # ======================================================================
        # Remove dead tensors from left and right. This happens after grouping because
        # we only reserve space for shared tensors after they're dead (alive is handled
        # by the normal reservation system). This is in case the tensor lifetime extends
        # beyond the Einsums for which it is used.
        # ======================================================================
        SIM.remove_dead_tensors(
            [s for lr in [left, right] for v in lr.values() for s, _ in v], live_tensors
        )

        DO_PRINT = False
        DELAY = True  # not debugger_active()
        # ======================================================================
        # Merge the left and right buckets.
        # ======================================================================
        combined: list[SIM] = []
        cur_nmappings = 0
        combined_ids: set[tuple[int, int, tuple[tuple[int, int], ...]]] = set()

        for k in left:
            found = False
            if DO_PRINT:
                print(f"Left key {k}")
            for (a, perm_a), (b, perm_b) in itertools.product(
                left[k], right.get(k, [])
            ):
                a: SIM
                b: SIM
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
                if DO_PRINT:
                    print(f"\t{a.compatibility}\n\t<-->\n\t{b.compatibility}")

                compatibility_a = a.compatibility.permute(perm_a)
                compatibility_b = b.compatibility.permute(perm_b)
                try:
                    compatibility_joined = compatibility_a.merge_next(
                        compatibility_b,
                        live_tensors,
                        mixable_ranks,
                    )
                except ValueError as e:  # Incompatible!
                    if DO_PRINT:
                        print(f"\tIncompatible: {e}")
                    continue

                t0 = time.time()

                combined.append(
                    a.merge_next(
                        b,
                        live_tensors,
                        live_tensors_with_right,
                        aliased_tensors,
                        compatibility_left=a.compatibility,
                        compatibility_right=b.compatibility,
                        compatibility_joined=compatibility_joined,
                        resource2capacity=resource2capacity,
                        drop_valid_reservations=drop_valid_reservations,
                        ignore_reservations=ignore_reservations,
                        delay=DELAY,
                    )
                )
                t1 = time.time()
                # print(f'Took {t1 - t0:.2f} seconds to generate {len(combined[-1].mappings.data)} mappings')

                if not DELAY:
                    cur_nmappings += len(a.mappings.data) * len(b.mappings.data)
                if DO_PRINT:
                    s = f"\t-->\n\t{combined[-1].compatibility}"
                    s += f"({len(a.mappings.data)})x({len(b.mappings.data)})"
                    print(s)
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

        # ======================================================================
        # Look ahead to the next Einsum and see if any of our groups will not
        # be able to merge with it. If so, we can drop them immediately.
        # ======================================================================
        if lookahead_filter:
            cur_tensors = left_tensors | right_tensors
            for next_sims in sims:
                next_right_tensors = next_sims.tensor_names
                if not next_right_tensors & cur_tensors:
                    continue
                prev_combined = combined
                combined = SIM.group(combined, next_right_tensors, drop_tags=True)
                next_keys = {
                    c.clear_dead_tensors(
                        cur_tensors
                    ).clear_tile_patterns_and_reservation_indices()
                    for c in next_sims.sims
                }
                for k in list(combined):
                    k_cleared = k.clear_dead_tensors(
                        next_right_tensors
                    ).clear_tile_patterns_and_reservation_indices()
                    if k_cleared not in next_keys:
                        if DO_PRINT:
                            for b, _ in combined[k]:
                                print(
                                    f"\tLOOKAHEAD to {next_sims.einsum_name}: No match for {b.compatibility}"
                                )
                        del combined[k]
                if not combined:
                    SIM.group(prev_combined, next_right_tensors, drop_tags=True)
                    raise_no_match_error()

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
                pbar=f"Merging pmappings for {left_einsum} <--> {right_einsum} ({n_iterations}/{total_iterations})",
                return_as="generator",
            )
            for c, mapping in zip(combined, mappings):
                c.mappings = mapping
                cur_nmappings += c.n_pre_prune_mappings
        print_time("Pmapping merging")

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
        logging.info(
            f"\tCombining {sum(len(s) for s in left)}({len(left)}) x {sum(len(s) for s in right)}({len(right)}) -> {len(combined)}"
        )

        nmappings = sum(len(s.mappings.data) for s in combined)
        for_einsum_text = f"for Einsum {right_einsum}"
        logging.info(f"\tNumber of groups {for_einsum_text}: {len(combined)}")
        # for c in combined:
        #     print(f"\t\t{c.compatibility}")
        logging.info(f"\tNumber of mappings {for_einsum_text}: {nmappings}")
        logging.info(
            f"\tMappings per group {for_einsum_text}: {nmappings / len(combined)}"
        )
        logging.info(
            f"\tLargest left: {max(len(s2.mappings.data) for s in left.values() for s2, _ in s)}"
        )
        logging.info(
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
    left = SIM.left_consolidate(left, None, pbar="Final consolidate")
    s_final = SIM.combine_combineable(left, set(), drop_tags=True)
    assert len(s_final) == 1
    mappings = s_final[0].mappings

    print_total_time()
    # if evaluations_tracker is not None and "Total_latency" in data.columns and "Total_energy" in data.columns:
    #     edp = data["Total_latency"] * data["Total_energy"]
    #     edp_min = edp.min()
    #     evaluations_tracker.add_evaluation(n_evaluations, edp_min)
    #     evaluations_tracker.n_mappings.update(n_mappings)
    #     evaluations_tracker.runtime.update(runtime)

    return mappings


def join_sims_no_skip_invalid(*args, **kwargs):
    return join_sims(*args, skip_invalid=False, **kwargs)


def join_sims_no_combine_reservations(*args, **kwargs):
    args = list(args)
    if len(args[0]) == 16:
        args[0] = {k: v for k, v in list(args[0].items())[:11]}
    if len(args[0]) > 16:
        args[0] = {k: v for k, v in list(args[0].items())[:2]}
    return join_sims(*args, combine_reservations=False, **kwargs)


def join_sims_no_either(*args, **kwargs):
    args = list(args)
    if len(args[0]) == 16:
        args[0] = {k: v for k, v in list(args[0].items())[:11]}
    if len(args[0]) > 16:
        args[0] = {k: v for k, v in list(args[0].items())[:2]}
    return join_sims(*args, skip_invalid=False, combine_reservations=False, **kwargs)
