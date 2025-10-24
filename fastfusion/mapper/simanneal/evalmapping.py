from collections import defaultdict
import itertools
import time
from fastfusion.accelerated_imports import pd
from fastfusion.mapper.FFM._join_pmappings.sim import SIM, Loop, Compatibility
from fastfusion.mapper.FFM._pmapping_group import PmappingGroup
from fastfusion.mapper.simanneal.mapspaceglobals import MapspaceGlobals
from fastfusion.util.util import fzs


def mapping2sims(einsum_to_result: Compatibility):
    r = {}
    for einsum_name, compat_dict in einsum_to_result.items():
        r[einsum_name] = [paretofy(k, v) for k, v in compat_dict.items()]
    return list(r.values())


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
            set.union(
                *(full_equivalent_rank_variables[n] for n in l.rank_variable_names)
            )
            & right_rank_variables
        )
        pairwise_compatible_rank_variables = (
            set.union(
                *(pairwise_equivalent_rank_variables[n] for n in l.rank_variable_names)
            )
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


def quick_join(
    sims: dict[str, SIM],
    mapspace_globals: MapspaceGlobals,
):
    resource2capacity = mapspace_globals.resource2capacity
    pairwise_equivalent_rank_variables = mapspace_globals.pairwise_equivalent_ranks
    aliased_tensors = mapspace_globals.aliased_tensors
    full_equivalent_rank_variables = mapspace_globals.full_equivalent_ranks

    n_mappings = {}
    runtime = {}
    nbuckets = []

    n_evaluations = 0

    sims = list(sims.items())

    init_print_time()

    sims = [GroupOfSIMsHolder(*s) for s in sims]

    if not sims:
        raise ValueError("No SIMs to join")

    # ======================================================================
    # Initial consolidate and group all SIMs
    # ======================================================================
    for i, sim_holder in enumerate(sims):
        right_tensors = set.union(set(), *[s.tensor_names for s in sims[i + 1 :]])
        if i == 0:
            sim_holder.sims = SIM.left_consolidate(
                sim_holder.sims,
                right_tensors,
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
        )
        sim_holder.sims = SIM.combine_combineable(
            sim_holder.sims,
            left_tensors | right_tensors,
        )
        if i > 0:
            sim_holder.sims = SIM.group_right(
                sim_holder.sims, left_tensors, drop_tags=True
            )
        einsum, prev_einsum = sim_holder.einsum_name, sims[i - 1].einsum_name
        runtime[f"{prev_einsum} → {einsum}"] = time.time() - t0
        t0 = time.time()

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
        right_rank_variables = mapspace_globals.einsum2ranks[right_einsum]

        partial_mapping_size += 1

        live_tensors = set.union(set(), *[s.tensor_names for s in sims])
        shared_tensors = set(left_tensors) & set(right_tensors)
        live_tensors_with_right = live_tensors | right_tensors

        # ======================================================================
        # Clean up the previously-combined SIMs. Consolidate, combine, group
        # them into buckets.
        # ======================================================================

        left = SIM.combine_combineable(
            left,
            live_tensors | right_tensors,
        )

        # Group left and right into buckets
        left = SIM.group_left(left, right_tensors, drop_tags=True)

        # ======================================================================
        # Remove dead tensors from left and right. This happens after grouping
        # because we only reserve space for shared tensors after it's dead. This
        # is in case the tensor lifetime extends beyond the Einsums for which it
        # is used.
        # ======================================================================
        SIM.remove_dead_tensors(
            [s for lr in [left, right] for v in lr.values() for s in v], live_tensors
        )

        # ======================================================================
        # Merge the left and right buckets.
        # ======================================================================
        combined: list[SIM] = []
        for k in left:
            for k_translated in get_possible_translations(
                k,
                pairwise_equivalent_rank_variables,
                full_equivalent_rank_variables,
                right_rank_variables,
            ):
                for a, b in itertools.product(left[k], right.get(k_translated, [])):
                    if a.compatibility.tags.are_compatible_with(b.compatibility.tags):
                        combined.append(
                            a.merge_next(
                                b,
                                live_tensors,
                                live_tensors_with_right,
                                aliased_tensors,
                                resource2capacity,
                                delay=False,
                            )
                        )

        if not combined:
            raise ValueError("No match found for any group")

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
    left = SIM.left_consolidate(left, None)
    s_final = SIM.combine_combineable(left, set(), drop_tags=True)
    assert len(s_final) == 1
    mappings = s_final[0].mappings

    return mappings
