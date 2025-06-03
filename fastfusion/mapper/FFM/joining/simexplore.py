from collections import defaultdict
import itertools
import time
from typing import TypeAlias, Union
import pandas as pd
from fastfusion.frontend import architecture
from fastfusion.frontend.specification import Specification
from fastfusion.mapper.FFM.joining.sim import SIM, Loop, Compatibility
from fastfusion.mapper.FFM.pareto import VALID, PartialMappings, DecompressData
from fastfusion.util import fzs, parallel, debugger_active


def mapping2sims(einsum_to_result: Compatibility):
    r = {}
    for einsum_name, compat_dict in einsum_to_result.items():
        r[einsum_name] = [paretofy(k, v) for k, v in compat_dict.items()]
    return list(r.values())


def paretofy(k, v):
    return SIM(k, PartialMappings(pd.DataFrame(v).fillna(0)))


def get_possible_translations(
    t: Compatibility,
    pairwise_equivalent_rank_variables: dict[str, set[str]],
    full_equivalent_rank_variables: dict[str, set[str]],
    right_rank_variables: set[str],
):
    # Fused ranks should be transitive, but if a fused loop indexes into two
    # different ranks in the next Einsum, we can't fuse becuase it will tile in
    # multiple directions.
    #
    # The first union checks what loops we CAN fuse with in the next Einsum. The
    # second union checks what loops MUST index into in the next
    #
    # Einsum. If we alias into multiple ranks, we can't fuse. Otherwise, try out
    # each possible rank.
    def translate_loop(l: Loop):
        compatible_rank_variables = (
            set.union(*(full_equivalent_rank_variables[n] for n in l.rank_variable_names))
            & right_rank_variables
        )
        pairwise_compatible_rank_variables = (
            set.union(*(pairwise_equivalent_rank_variables[n] for n in l.rank_variable_names))
            & right_rank_variables
        )
        if len(pairwise_compatible_rank_variables) > 1:
            return
        for n in compatible_rank_variables:
            yield Loop(fzs((n,)), l.bound, l.is_spatial)

    for loops in itertools.product(*map(translate_loop, t.loops)):
        yield t.update(loops=loops)


prev_time = 0
total_time = defaultdict(int)


def init_print_time():
    global prev_time, total_time
    prev_time = time.time()
    total_time = defaultdict(int)


def print_time(what: str):
    global prev_time
    t = time.time() - prev_time
    print(f"{what}: {t:.2f} seconds")
    total_time[what] += t
    prev_time = time.time()


def print_total_time():
    print(f"\n======== Total time ========")
    for k, v in total_time.items():
        print(f"{k}: {v:.2f} seconds")
    total = sum(total_time.values())
    if total > 60:
        print(f"\nTotal: {total:.2f} seconds ({total/60:.2f} minutes)")
    else:
        print(f"\nTotal: {total:.2f} seconds")
    print(f"============================\n")


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



def join_sims(
    sims: dict[str, list[SIM]],
    spec: Specification = None,
    flattened_architecture: list[architecture.Leaf] = None,
    # For profiling.
    evaluations_tracker=None,
    size_scale: float = 1.0,
    return_nmappings_nbuckets: bool = False,
    # Optimality-maintaining optimizations.
    skip_invalid: bool = True,
    combine_reservations: bool = True,
    lookahead_filter: bool = True,
    drop_valid_reservations: bool = True,
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
    # decompress_data = compress(sims)

    resource2capacity = {}
    for l in flattened_architecture:
        if isinstance(l, architecture.Memory):
            resource2capacity[l.name] = l.attributes.size

    pairwise_equivalent_rank_variables = (
        spec.workload.get_pairwise_equivalent_rank_variables()
    )

    aliased_tensors = {"I_n_to_I": "I_I_to_Q_K_V"}

    full_equivalent_rank_variables = make_full_equivalent_rank_variables(
        pairwise_equivalent_rank_variables
    )

    for sim_list in sims.values():
        for s in sim_list:
            if VALID in s.mappings.data:
                s.mappings.data = s.mappings.data[s.mappings.data[VALID] == 1]

    n_mappings = {}
    runtime = {}
    nbuckets = []

    n_evaluations = 0

    sims = list(sims.items())

    if not skip_invalid:
        lookahead_filter = False

    for einsum_name, s in sims:
        print(f"SIM {einsum_name} tensors: {s[0].tensor_names}")
    init_print_time()

    sims = [GroupOfSIMsHolder(*s) for s in sims]

    # ======================================================================
    # Initial consolidate and group all SIMs
    # ======================================================================
    n_mappings["Post Intra-Layer"] = 0
    for i, sim_holder in enumerate(sims):
        right_tensors = set.union(set(), *[s.tensor_names for s in sims[i + 1 :]])
        if i == 0:
            sim_holder.sims = SIM.left_consolidate(
                sim_holder.sims,
                right_tensors,
                pbar=f"Inital consolidate {sim_holder.einsum_name}",
            )
            continue
        t0 = time.time()
        left_tensors = set.union(set(), *[s.tensor_names for s in sims[:i]])
        live_tensors = right_tensors
        shared_tensors = left_tensors & sim_holder.tensor_names
        sim_holder.sims = sorted(
            sim_holder.sims, key=lambda x: len(x.mappings.data), reverse=True
        )
        sim_holder.sims = SIM.right_consolidate(
            sim_holder.sims,
            live_tensors,
            shared_tensors,
            pbar=f"Inital consolidate {sim_holder.einsum_name}",
        )
        sim_holder.sims = SIM.combine_combineable(
            sim_holder.sims,
            left_tensors | right_tensors,
            combine_reservations=combine_reservations,
        )
        n_mappings["Post Intra-Layer"] += sum(
            len(s.mappings.data) for s in sim_holder.sims
        )
        if i > 0:
            sim_holder.sims = SIM.group_right(
                sim_holder.sims, left_tensors, drop_tags=True
            )
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
        right_rank_variables = spec.workload.einsums[right_einsum].rank_variables
        print(f"\nEinsum {right_einsum} ({n_iterations}/{total_iterations})")

        partial_mapping_size += 1

        live_tensors = set.union(set(), *[s.tensor_names for s in sims])
        shared_tensors = set(left_tensors) & set(right_tensors)
        live_tensors_with_right = live_tensors | right_tensors

        # ======================================================================
        # Clean up the previously-combined SIMs. Consolidate, combine, group
        # them into buckets.
        # ======================================================================
        print_time(f"Consolidating")

        left = SIM.combine_combineable(
            left,
            live_tensors | right_tensors,
            combine_reservations=combine_reservations,
        )

        print_time(f"Combining")
        # Group left and right into buckets
        left = SIM.group_left(left, right_tensors, drop_tags=True)
        print_time("Grouping")

        # ======================================================================
        # Remove dead tensors from left and right. This happens after grouping
        # because we only reserve space for shared tensors after it's dead. This
        # is in case the tensor lifetime extends beyond the Einsums for which it
        # is used.
        # ======================================================================
        SIM.remove_dead_tensors(
            [s for lr in [left, right] for v in lr.values() for s in v], live_tensors
        )

        DO_PRINT = False
        DELAY = not debugger_active()

        # ======================================================================
        # Merge the left and right buckets.
        # ======================================================================
        combined: list[SIM] = []
        cur_nmappings = 0
        for k in left:
            found = False
            for k_translated in get_possible_translations(
                k,
                pairwise_equivalent_rank_variables,
                full_equivalent_rank_variables,
                right_rank_variables,
            ):
                for a, b in itertools.product(left[k], right.get(k_translated, [])):
                    if (
                        a.compatibility.tags.are_compatible_with(b.compatibility.tags)
                    ):
                        found = True
                        combined.append(
                            a.merge_next(
                                b,
                                live_tensors,
                                live_tensors_with_right,
                                aliased_tensors,
                                resource2capacity,
                                drop_valid_reservations=drop_valid_reservations,
                                delay=DELAY,
                            )
                        )
                        if not DELAY:
                            cur_nmappings += len(a.mappings.data) * len(b.mappings.data)
                        if DO_PRINT:
                            s = f"\t{a.compatibility} <--> {b.compatibility}"
                            s += f" --> {combined[-1].compatibility}"
                            s += f"({len(a.mappings.data)})x({len(b.mappings.data)})"
                            print(s)
            if DO_PRINT and not found:
                for a in left[k]:
                    print(f"\tNo match for {a.compatibility}")

        if DO_PRINT:
            for k in right:
                if k not in left:
                    for b in right[k]:
                        print(f"\tREVERSE: No match for {b.compatibility}")

        print_time("Bucket merging")

        # ======================================================================
        # Look ahead to the next Einsum and see if any of our buckets will not
        # be able to merge with it. If so, we can drop them immediately.
        # ======================================================================
        if sims and lookahead_filter:
            prev_len = len(combined)
            next_right_tensors = sims[0].tensor_names
            next_right_rank_variables = spec.workload.einsums[
                sims[0].einsum_name
            ].rank_variables
            combined = SIM.group_left(combined, next_right_tensors, drop_tags=True)
            for k in list(combined):
                translations = get_possible_translations(
                    k,
                    pairwise_equivalent_rank_variables,
                    full_equivalent_rank_variables,
                    next_right_rank_variables,
                )
                if not any(kt in sims[0].sims for kt in translations):
                    list(
                        get_possible_translations(
                            k,
                            pairwise_equivalent_rank_variables,
                            full_equivalent_rank_variables,
                            next_right_rank_variables,
                        )
                    )
                    if DO_PRINT:
                        for b in combined[k]:
                            print(f"\tLOOKAHEAD: No match for {b.compatibility}")
                    del combined[k]
            if not combined:
                raise ValueError("No match found for any bucket")
            combined = list(itertools.chain.from_iterable(combined.values()))
            print(
                f"Removed {prev_len - len(combined)}/{prev_len} ({len(combined)/prev_len*100:.2f}% remaining)"
            )
            print_time("Removing mappings that can't be combined later")

        if not combined:
            raise ValueError("No match found for any bucket")

        # ======================================================================
        # If we delayed the mapping merging, do it now.
        # ======================================================================
        if DELAY:
            mappings = parallel(
                [c.mappings for c in combined],
                pbar=f"Merging mappings {left_einsum} <--> {right_einsum}",
                return_as="generator",
            )
            for c, mapping in zip(combined, mappings):
                c.mappings = mapping
                cur_nmappings += c.n_pre_prune_mappings
        print_time("Mapping merging")

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
        print(
            f'Scaled runtime by {cur_nmappings / prev_nmappings}. Runtime: {runtime[f"{prev_einsum} → {einsum}"]:.2f}'
        )

        # ======================================================================
        # Print statements
        # ======================================================================
        print(
            f"\tCombining {sum(len(s) for s in left)}({len(left)}) x {sum(len(s) for s in right)}({len(right)}) -> {len(combined)}"
        )

        nmappings = sum(len(s.mappings.data) for s in combined)
        for_einsum_text = f"for Einsum {right_einsum}"
        print(f"\tNumber of buckets {for_einsum_text}: {len(combined)}")
        print(f"\tNumber of mappings {for_einsum_text}: {nmappings}")
        print(f"\tMappings per bucket {for_einsum_text}: {nmappings / len(combined)}")

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
    # decompress(decompress_data, s_final, prefix=spec.workload.einsum_names)
    assert len(s_final) == 1
    mappings = s_final[0].mappings
    data = mappings.data

    print_total_time()
    if evaluations_tracker is not None:
        edp = data["Latency"] * data["Energy"]
        edp_min = edp.min()
        evaluations_tracker.add_evaluation(n_evaluations, edp_min)
        evaluations_tracker.n_mappings.update(n_mappings)
        evaluations_tracker.runtime.update(runtime)

    if return_nmappings_nbuckets:
        return mappings, n_mappings, nbuckets
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

def compress_sims(sims: dict[str, list[SIM]] | list[tuple[str, SIM]]) -> DecompressData:
    if not isinstance(sims, (dict, list)):
        raise TypeError(f"Expected dict or list, got {type(sims)}")
    if isinstance(sims, dict):
        sims_flattened = (
            (s, einsum_name)
            for einsum_name, sim_list in sims.items()
            for s in sim_list
        )
    else:
        sims_flattened = sims
    return PartialMappings.compress_paretos([(s.mappings, einsum_name) for s, einsum_name in sims_flattened])

simlist: TypeAlias = list[Union[SIM, PartialMappings]]

def decompress_sims(data: PartialMappings, decompress_data: DecompressData, prefix: list[str] = None):
    if not isinstance(data, PartialMappings):
        raise TypeError(f"Expected PartialMappings, got {type(data)}")
    PartialMappings.decompress_paretos([data], decompress_data, prefix)
