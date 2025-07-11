import itertools

from collections.abc import Iterable, Set

from fastfusion.frontend.specification import Specification
from fastfusion.frontend.workload import EinsumName, TensorName
from fastfusion.mapper.FFM.joining.mappinginfo import Compatibility
from fastfusion.mapper.FFM.joining.sim import SIM


DO_PRINT = False
def myprint(*args, **kwargs):
    if DO_PRINT:
        print(*args, **kwargs)


def remove_unimportant_sims(
    einsum2sims: dict[EinsumName, Iterable[SIM]],
    einsum2important_compats: dict[EinsumName, Iterable[Compatibility]],
) -> dict[EinsumName, list[SIM]]:
    einsum2pruned_sims: dict[EinsumName, list[SIM]] = {}
    for einsum, sims in einsum2sims.items():
        important_compats = einsum2important_compats[einsum]
        pruned_sims = [
            sim for sim in sims
            if sim.compatibility.clear_loop_bounds() in important_compats
        ]
        einsum2pruned_sims[einsum] = pruned_sims
    return einsum2pruned_sims


def sims2untiled_compats(
    einsum2sims: dict[EinsumName, Iterable[SIM]]
) -> dict[EinsumName, set[Compatibility]]:
    return {
        einsum_name: {sim.compatibility.clear_loop_bounds() for sim in sims}
        for einsum_name, sims in einsum2sims.items()
    }


def join_compatibilities(
    einsum2compatibilities: dict[EinsumName, Iterable[Compatibility]],
    spec: Specification = None,
) -> dict[EinsumName, set[Compatibility]]:
    """
    Return dict from Einsum name to compatibilities (without tile shape)
    that will ever contribute to full mappings.

    CONTRACT FOR MAPPINGS GETTING TO THIS POINT: see `simexplore.join_sims`
    """
    if len(einsum2compatibilities) == 0:
        raise ValueError("Nothing to join")

    for einsum_name, per_einsum_compats in einsum2compatibilities.items():
        if not per_einsum_compats:
            raise ValueError(f"No compatibility for {einsum_name}")

    compat2einsum2original\
        : dict[Compatibility, dict[EinsumName, set[Compatibility]]] = {}
    for einsum_name, per_einsum_compats in einsum2compatibilities.items():
        for compat in per_einsum_compats:
            einsum2original = compat2einsum2original.setdefault(compat, {})
            original = einsum2original.setdefault(einsum_name, set())
            original.add(compat)

    compatibilities = list(einsum2compatibilities.items())

    einsum2tensor_names = {
        einsum_name: spec.workload.einsums[einsum_name].tensor_names
        for einsum_name in einsum2compatibilities
    }

    einsum2important_compatibilities = {}

    # while-loop states
    assert len(compatibilities) > 0
    left_einsum, all_left_compats = compatibilities.pop(0)
    left_tensors = einsum2tensor_names[left_einsum]

    while compatibilities:
        right_einsum, all_right_compats = compatibilities.pop(0)

        right_tensors = einsum2tensor_names[right_einsum]
        live_tensors = set.union(
            set(),
            *(einsum2tensor_names[e] for e, _ in compatibilities)
        )

        grouped_left_compats = group_left(all_left_compats, right_tensors)
        grouped_right_compats = group_right(all_right_compats, left_tensors)

        combined = combine_left_and_right_compats(
            compat2einsum2original,
            grouped_left_compats,
            grouped_right_compats,
            live_tensors,
        )

        if DO_PRINT:
            print_reverse_unmatched(grouped_left_compats, grouped_right_compats)

        if not combined:
            raise ValueError("No match found for any group")

        # update while-loop states
        all_left_compats = combined
        left_einsum = right_einsum
        left_tensors |= right_tensors

    einsum2important_compatibilities: dict[EinsumName, set[Compatibility]] = {}
    for compat in combined:
        for einsum, original in compat2einsum2original[compat].items():
            important_compats = \
                einsum2important_compatibilities.setdefault(einsum, set())
            important_compats.update(original)
    return einsum2important_compatibilities


def combine_left_and_right_compats(
    compat2einsum2original: dict[Compatibility, dict[EinsumName, set[Compatibility]]],
    grouped_left_compats: dict[Compatibility, Iterable[Compatibility]],
    grouped_right_compats: dict[Compatibility, Iterable[Compatibility]],
    live_tensors: set[TensorName],
):
    combined: list[Compatibility] = []
    for left_key, left_compats in grouped_left_compats.items():
        myprint(f'Left key {left_key}')

        compatible_right_compats = grouped_right_compats.get(left_key, [])

        if len(compatible_right_compats) == 0:
            if DO_PRINT:
                for l in left_compats:
                    print(f"\tNo match for {l}")
            continue

        for l, r in itertools.product(left_compats, compatible_right_compats):
            if l.tags.are_compatible_with(r.tags):
                merged = l.merge_next(r, live_tensors)
                combined.append(merged)

                einsum2original = compat2einsum2original.setdefault(merged, {})

                left_einsum2original = compat2einsum2original[l]
                right_einsum2original = compat2einsum2original[r]

                einsums = set(left_einsum2original) | set(right_einsum2original)
                for einsum in einsums:
                    einsum2original.setdefault(einsum, set()).update(
                        left_einsum2original.get(einsum, set())
                        |
                        right_einsum2original.get(einsum, set())
                    )

                myprint(f"\t{l}\n\t<-->\n\t{r}")
                myprint(f"\t-->\n\t{merged}")
    return combined


def print_reverse_unmatched(
    grouped_left_compats: dict[Compatibility, Iterable[Compatibility]],
    grouped_right_compats: dict[Compatibility, Iterable[Compatibility]],
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
        grouped_compats.setdefault(key, set()).add(compat)
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
            grouped_compats.setdefault(per_loop_key, set()).add(compat)
    return grouped_compats
