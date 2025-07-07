import itertools

from collections.abc import Iterable, Set

from fastfusion.frontend.specification import Specification
from fastfusion.frontend.workload import TensorName
from fastfusion.mapper.FFM.joining.sim import Compatibility


DO_PRINT = False
def myprint(*args, **kwargs):
    if DO_PRINT:
        print(*args, **kwargs)


def join_compatibilities(
    einsum2compatibilities: dict[str, list[Compatibility]],
    spec: Specification = None,
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
    if len(einsum2compatibilities) == 0:
        raise ValueError("Nothing to join")

    for einsum_name, per_einsum_compats in einsum2compatibilities.items():
        if not per_einsum_compats:
            raise ValueError(f"No compatibility for {einsum_name}")

    compatibilities = [
        (einsum_name, {c.clear_loop_bounds() for c in compatibilities})
        for einsum_name, compatibilities in einsum2compatibilities.items()
    ]

    einsum2tensor_names = {
        einsum_name: spec.workload.einsums[einsum_name].tensor_names
        for einsum_name in einsum2compatibilities
    }

    einsum2important_compatibilities = {}

    # while-loop states
    assert len(compatibilities) == 0
    left_einsum, all_left_compats = compatibilities.pop(0)
    left_tensors = einsum2tensor_names[left_einsum]
    first = True
    while compatibilities:
        # ======================================================================
        # Grab new Einsum from the right. Record logging data and find still
        # tensors that will be live after this Einsum.
        # ======================================================================
        # nmappings.append(sum(len(s.mappings.data) for s in left))
        right_einsum, all_right_compats = compatibilities.pop(0)

        right_tensors = einsum2tensor_names[right_einsum]
        live_tensors = set.union(
            set(),
            (einsum2tensor_names[e] for e, _ in compatibilities)
        )

        grouped_left_compats = group_left(all_left_compats, right_tensors)
        grouped_right_compats = group_right(all_right_compats, left_tensors)

        (
            left_important_compatibilities,
            right_important_compatibilities,
            combined
        ) = combine_left_and_right_compats(
            grouped_left_compats,
            grouped_right_compats,
            live_tensors,
            return_left_compats=first
        )

        if DO_PRINT:
            print_reverse_unmatched(grouped_left_compats, grouped_right_compats)

        if not combined:
            raise ValueError("No match found for any group")

        if first:
            einsum2important_compatibilities[left_einsum] = \
                left_important_compatibilities
        einsum2important_compatibilities[right_einsum] = right_important_compatibilities

        # ======================================================================
        # Update left for the next iteration.
        # =================================================================
        all_left_compats = combined
        left_einsum = right_einsum
        left_tensors |= right_tensors
        first = False

    return einsum2important_compatibilities


def combine_left_and_right_compats(
    grouped_left_compats: dict[Compatibility, Iterable[Compatibility]],
    grouped_right_compats: dict[Compatibility, Iterable[Compatibility]],
    live_tensors: set[TensorName],
    return_left_compats: bool
):
    left_important_compatibilities = set()
    right_important_compatibilities = set()
    combined: list[Compatibility] = []
    for left_key, left_compats in grouped_left_compats:
        myprint(f'Left key {left_key}')

        compatible_right_compats = grouped_right_compats.get(left_key, [])

        if len(compatible_right_compats) == 0:
            if DO_PRINT:
                for l in left_compats:
                    print(f"\tNo match for {l}")
            continue

        for l, r in itertools.product(left_compats, compatible_right_compats):
            if l.tags.are_compatible_with(r.tags):
                if return_left_compats:
                    left_important_compatibilities.add(l)
                right_important_compatibilities.add(r)

                merged = l.merge_next(r, live_tensors)
                combined.append(merged)
                myprint(f"\t{l}\n\t<-->\n\t{r}")
                myprint(f"\t-->\n\t{merged}")

    return left_important_compatibilities, right_important_compatibilities, combined


def print_reverse_unmatched(
    grouped_left_compats,
    grouped_right_compats
):
    for right_key, right_compats in grouped_right_compats.items():
        if right_key not in grouped_left_compats:
            for r in right_compats:
                print(f"\tREVERSE: No match for {r} using {right_key}")


def group_left(
    left_compatibilities: Iterable[Compatibility],
    right_tensors: Set[TensorName],
) -> dict[Compatibility, set[Compatibility]]:
    grouped_compats = {}
    for compat in left_compatibilities:
        key = compat.clear_dead_tensors(right_tensors,
                                        keep_loops=True,
                                        drop_tags=True)
        grouped_compats.get(key, set()).add(compat)
    return grouped_compats



def group_right(
    right_compatibilities: Iterable[Compatibility],
    left_tensors: Set[TensorName],
) -> dict[Compatibility, set[Compatibility]]:
    grouped_compats = {}
    for compat in right_compatibilities:
        key = compat.clear_dead_tensors(left_tensors,
                                        keep_loops=True,
                                        drop_tags=True)
        for per_loop_key in key.all_n_loops():
            grouped_compats.get(per_loop_key, set()).add(compat)
    return grouped_compats