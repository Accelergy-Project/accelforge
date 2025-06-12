import itertools
import time
from collections import defaultdict
from numbers import Number

import pandas as pd

from fastfusion.frontend import architecture, Workload
from fastfusion.frontend.specification import Specification
from fastfusion.frontend.workload import Einsum
from fastfusion.frontend.workload.isl import get_rank_variable_bounds
from fastfusion.frontend.workload.symbolic import get_projection_expr, compute_rank_occupancy
from fastfusion.mapper.FFM.joining.mappinginfo import TilePattern
from fastfusion.mapper.FFM.joining.sim import SIM, Loop, Compatibility
from fastfusion.mapper.FFM.pareto import PartialMappings
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
    full_equivalent_rank_variables: dict[str, set[str]],
    right_einsum: Einsum,
    right_rank_var2initial_delta: dict[tuple[str, str], int]
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
    right_rank_variables = right_einsum.rank_variables
    tensor2rank_var2ranks = {}
    for tensor_access in right_einsum.tensor_accesses:
        tensor2rank_var2ranks[tensor_access.name] = tensor_access.rank_variable2ranks

    def rank_vars_index_only_one_rank_per_tensor(rank_vars):
        for tensor, rank_var2ranks in tensor2rank_var2ranks.items():
            all_ranks = set()
            for rank_var in rank_vars:
                if rank_var in rank_var2ranks:
                    all_ranks.update(rank_var2ranks[rank_var])
            if len(all_ranks) > 1:
                return False
        return True
    print('SRC', t.loops)

    def translate_loop(l: Loop):
        compatible_rank_variables = (
            set.union(*(full_equivalent_rank_variables[n] for n in l.rank_variable_names))
            & right_rank_variables
        )
        # TODO: convince that this always works
        if not rank_vars_index_only_one_rank_per_tensor(l.rank_variable_names):
            return

        for n in compatible_rank_variables:
            left_bound = l.bound

            if (l.rank_variable_name, n) in right_rank_var2initial_delta:
                delta = right_rank_var2initial_delta[(l.rank_variable_name, n)]
            else:
                delta = None

            if delta is not None and isinstance(left_bound, TilePattern):
                delta = right_rank_var2initial_delta[(l.rank_variable_name, n)]
                if left_bound.stride == left_bound.initial:
                    yield Loop(fzs((n,)), left_bound.stride, l.is_spatial)
                elif left_bound.stride + delta == left_bound.initial:
                    yield Loop(fzs((n,)), left_bound.stride, l.is_spatial)
                    # yield Loop(fzs((n,)),
                    #            TilePattern(left_bound.stride, left_bound.stride),
                    #            l.is_spatial)
                else:
                    # yield Loop(fzs((n,)),
                    #            TilePattern(left_bound.stride,
                    #                        left_bound.initial-delta),
                    #            l.is_spatial)
                    pass
            elif delta is None and isinstance(left_bound, Number):
                yield Loop(fzs((n,)), left_bound, l.is_spatial)

    for loops in itertools.product(*map(translate_loop, t.loops)):
        new_compatibility = t.update(loops=loops)
        print('DST', new_compatibility.loops)
        yield new_compatibility


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

    resource2capacity = {}
    flattened_architecture = flattened_architecture or spec.get_flattened_architecture()
    for l in flattened_architecture:
        if isinstance(l, architecture.Memory):
            resource2capacity[l.name] = l.attributes.size

    pairwise_equivalent_rank_variables = (
        spec.workload.get_pairwise_equivalent_rank_variables()
    )

    aliased_tensors = spec.workload.get_tensor_copies()

    full_equivalent_rank_variables = make_full_equivalent_rank_variables(
        pairwise_equivalent_rank_variables
    )

    n_mappings = {}
    runtime = {}
    nbuckets = []

    n_evaluations = 0

    sims = list(sims.items())

    if not skip_invalid:
        lookahead_filter = False

    for einsum_name, s in sims:
        if not s:
            raise ValueError(f"No SIMs for {einsum_name}")
        print(f"SIM {einsum_name} tensors: {s[0].tensor_names}")
    init_print_time()

    sims = [GroupOfSIMsHolder(*s) for s in sims]

    if not sims:
        raise ValueError("No SIMs to join")

    right_rank2initial_delta = get_initial_deltas(workload=spec.workload)

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
                full_equivalent_rank_variables,
                spec.workload.einsums[right_einsum],
                right_rank2initial_delta[right_einsum],
            ):
                print(right)
                print(right.get(k_translated, []))
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
        # Look ahead to the next Einsum and see if any of our groups will not
        # be able to merge with it. If so, we can drop them immediately.
        # ======================================================================
        if sims and lookahead_filter:
            prev_len = len(combined)
            next_right_tensors = sims[0].tensor_names
            combined = SIM.group_left(combined, next_right_tensors, drop_tags=True)
            for k in list(combined):
                translations = get_possible_translations(
                    k,
                    full_equivalent_rank_variables,
                    spec.workload.einsums[sims[0].einsum_name],
                    right_rank2initial_delta[right_einsum],
                )
                if not any(kt in sims[0].sims for kt in translations):
                    list(
                        get_possible_translations(
                            k,
                            full_equivalent_rank_variables,
                            spec.workload.einsums[sims[0].einsum_name],
                            right_rank2initial_delta[right_einsum],
                        )
                    )
                    if DO_PRINT:
                        for b in combined[k]:
                            print(f"\tLOOKAHEAD: No match for {b.compatibility}")
                    del combined[k]
            if not combined:
                raise ValueError("No match found for any group")
            combined = list(itertools.chain.from_iterable(combined.values()))
            print(
                f"Removed {prev_len - len(combined)}/{prev_len} ({len(combined)/prev_len*100:.2f}% remaining)"
            )
            print_time("Removing mappings that can't be combined later")

        if not combined:
            raise ValueError("No match found for any group")

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
        print(f"\tNumber of groups {for_einsum_text}: {len(combined)}")
        print(f"\tNumber of mappings {for_einsum_text}: {nmappings}")
        print(f"\tMappings per group {for_einsum_text}: {nmappings / len(combined)}")

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
    # if evaluations_tracker is not None and "metric_Latency" in data.columns and "metric_Energy" in data.columns:
    #     edp = data["metric_Latency"] * data["metric_Energy"]
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


def get_initial_deltas(workload: Workload) -> dict[str,
                                                   dict[tuple[str, str], int]]:
    deltas = {}
    for producer in workload.einsums:
        output_tensors = producer.output_tensors()
        if len(output_tensors) > 1:
            raise ValueError('Does not support more output tensors than one.')

        tensor = next(iter(output_tensors))

        prod_rank2rank_vars = producer.tensor_accesses[tensor].rank2rank_variables

        for consumer in workload.einsums_that_read_tensor(tensor):
            delta_for_pair = deltas.setdefault(consumer.name, {})
            projection = get_projection_expr(consumer, tensor)
            cons_shape = get_rank_variable_bounds(workload, consumer.name)

            cons_rank2rank_vars = consumer.tensor_accesses[tensor].rank2rank_variables
            for cons_rank, cons_rank_vars in cons_rank2rank_vars.items():
                if cons_rank not in prod_rank2rank_vars:
                    continue
                if len(cons_rank_vars) == 1:
                    continue  # Not an affine expr at the consumer
                prod_rank_vars = prod_rank2rank_vars[cons_rank]
                if len(prod_rank_vars) != 1:
                    continue  # Unclear what to do in this case

                prod_rank_var = next(iter(prod_rank_vars))

                for cons_rank_var in cons_rank_vars:
                    original_shape = cons_shape[cons_rank_var]
                    cons_shape[cons_rank_var] = 1
                    delta_for_pair[(prod_rank_var, cons_rank_var)] = (
                        compute_rank_occupancy(projection[cons_rank], cons_shape)
                        - 1
                    )
                    cons_shape[cons_rank_var] = original_shape
    return deltas