from accelforge.frontend.mapping.mapping import MappingNode


import copy
from collections import defaultdict
import itertools
import logging
from typing import Any, Iterator, List
import uuid

from tqdm import tqdm

import accelforge.frontend.arch as arch
from accelforge.util._frozenset import oset
from accelforge.frontend.mapping import (
    Compute,
    Loop,
    Mapping,
    MappingNode,
    Spatial,
    TensorHolder,
    Temporal,
)
from accelforge.frontend.renames import Rank
from accelforge.frontend.spec import Spec
from accelforge.frontend._workload_isl._isl import get_rank_variable_bounds
from accelforge.frontend._workload_isl._symbolic import (
    Relevant,
    get_rank_variable_relevancy,
    get_stride_and_halo,
    get_stride_and_halo_of_einsum,
    PartiallyRelevant,
)
from accelforge.frontend.workload import (
    TensorName,
    Einsum,
    EinsumName,
    RankVariable,
    Workload,
    SymbolTable,
)
from accelforge.mapper.FFM._make_pmappings.make_pmapping_templates.make_storage_order import (
    get_tensor_choices,
)
from accelforge.mapper.FFM._make_pmappings.make_pmapping_templates.make_reservations import (
    get_reservation_choices,
)
from accelforge.mapper.FFM._make_pmappings.contraints.constraints import (
    MappingConstraints,
    get_constraints,
)
from accelforge.mapper.FFM._make_pmappings.make_pmapping_templates.make_loops import (
    insert_temporal_loops,
    insert_spatial_loops,
)
from accelforge.mapper.FFM._make_pmappings.pmapper_job import (
    Job,
    SameEinsumJobs,
)
from accelforge.mapper.FFM._join_pmappings.compatibility import Compatibility
from accelforge.model._looptree.reuse.symbolic import (
    label_fused_loops,
    quick_insert_reservation_nodes,
)


def unpack_loops_to_rank_variables(mapping: List[MappingNode]):
    mapping_new = []
    for node in mapping:
        if not isinstance(node, Loop) or not isinstance(node.rank_variable, set):
            mapping_new.append(node)
            continue

        for r in sorted(node.rank_variable):
            mapping_new.append(
                type(node)(
                    rank_variable=r,
                    **node.model_dump(exclude={"rank_variable"}, recursive=False),
                )
            )
    return mapping_new


# =================================================================================================
# Iterate over mappings
# =================================================================================================
def place_missing_temporal_loops(
    mapping: List[MappingNode], einsum: Einsum, flattened_arch: list[arch.Leaf]
):
    """
    Adds temporal loops to the mapping to fill in any rank variables that are missing.
    This may occur if there are no points where it'd be helpful to add a non-fused loop,
    so we just need to add one somewhere.
    """
    # If any rank variables are missing, add them as high as possible.

    rank_variables = einsum.rank_variables
    for m in mapping:
        if isinstance(m, Temporal) and not m._fused:
            rank_variables.discard(m.rank_variable)

    # Insert point: Right under the last backing & below any out-of-order fanouts
    fanouts = {}
    fanout = 1
    for node in flattened_arch:
        fanouts[node.name] = (fanout := fanout * node.get_fanout())

    insert_point = 0
    greatest_previous_fanout = 1
    for i in range(len(mapping)):
        if isinstance(mapping[i], TensorHolder):
            if mapping[i]._backing:
                insert_point = i + 1
            cur_fanout = fanouts[mapping[i].component]
            if cur_fanout < greatest_previous_fanout:
                insert_point = i + 1
            greatest_previous_fanout = max(greatest_previous_fanout, cur_fanout)

        # Put it below all the other temporals here in case we're lowering through them
        if isinstance(mapping[i], Temporal) and insert_point == i:
            insert_point = i + 1

    temporals = [Temporal(rank_variable=r) for r in sorted(rank_variables)]

    if insert_point == len(mapping):
        mapping.extend(temporals)
    else:
        for t in temporals:
            mapping.insert(insert_point, t)


def remove_unordered_spatial_temporal_loops(
    mapping: list[MappingNode],
    flattened_arch: list[arch.Leaf],
    einsum: Einsum,
    explore_unordered_spatial_loops: bool = True,
):
    fanout = 1
    fanouts = {}
    for node in flattened_arch:
        fanouts[node.name] = (fanout := fanout * node.get_fanout())

    # TensorHolders' tiles include the full iteration space of temporal loops below
    # them. If all of the following are true:
    # - Any later spatials have <= the fanout of this TensorHolder
    # - There's a temporal loop between the TensorHolder and the Spatial
    # - The temporal/spatial have the same rank variable
    # - The rank variable indexes into the tensor
    #
    # Then the temporal/spatial pair will result in a non-contiguous tile of the
    # iteration space, which is not supported and must be removed.

    tensor2rvs = einsum.tensor2rank_variables
    disallowed_combinations: list[tuple[set[int], set[int]]] = []
    for i, node in enumerate(mapping):
        # Track TensorHolders that have been seen and the rank variables that affect
        # them
        if not isinstance(node, TensorHolder):
            continue

        relevent_rvs = oset.union(*[tensor2rvs[t] for t in node.tensors])

        # Find the last spatial whose component's fanout is <= the TensorHolder's
        # component fanout. This spatial will affect the TensorHolder's tile.
        check_up_to = i
        for j, node2 in enumerate(mapping[i + 1 :]):
            if isinstance(node2, Spatial):
                if fanouts[node.component] >= fanouts[node2.component]:
                    check_up_to = i + j + 1

        # Find all temporal and spatial loops between the TensorHolder and the last
        # spatial that affects it.
        rv2spatial = {}
        rv2temporal = {}
        for node2 in mapping[i + 1 : check_up_to]:
            if isinstance(node2, Spatial) and node2.rank_variable in relevent_rvs:
                rv2spatial.setdefault(node2.rank_variable, []).append(node2)
            if isinstance(node2, Temporal):
                rv2temporal.setdefault(node2.rank_variable, []).append(node2)

        for shared_rv in sorted(oset(rv2spatial) & oset(rv2temporal) & relevent_rvs):
            disallowed_combinations.append(
                (
                    tuple(id(x) for x in rv2spatial[shared_rv]),
                    tuple(id(x) for x in rv2temporal[shared_rv]),
                )
            )

    if not explore_unordered_spatial_loops:
        disallowed_combinations = [x[1:] for x in disallowed_combinations]

    for combo in itertools.product(*disallowed_combinations):
        combo = oset.union(oset(), *combo)
        yield [n for n in mapping if id(n) not in combo]


def pad_with_bottom_loops(mapping: list[MappingNode], einsum: Einsum):
    rank_variables = einsum.rank_variables
    rank_var_to_count = defaultdict(lambda: 0)
    for node in mapping:
        if isinstance(node, Temporal):
            rank_var_to_count[node.rank_variable] += 1

    for rank_var in rank_variables:
        if rank_var_to_count[rank_var] < 2:
            mapping.append(Temporal(rank_variable=rank_var))


def _timeloop_style_even(mapping: list[MappingNode]):
    # Iterate through the mapping. If there are >2 TensorHolder nodes for the same
    # memory, move all below the 2nd to the same level as the 2nd.
    mapping = copy.deepcopy(mapping)
    memory2indices = defaultdict(list)
    i = 0
    while i < len(mapping):
        node = mapping[i]
        if not isinstance(mapping[i], TensorHolder):
            i += 1
            continue
        node: TensorHolder
        seen = memory2indices[node.component]
        mapping[i]._lower = False  # Lowering might re-uneven the reservationsxs

        if len(seen) <= 1:
            seen.append(i)
        else:
            mapping.insert(seen[-1] + 1, mapping.pop(i))
        i += 1
    return mapping


def assert_proper_fusion_labeling(
    mapping: list[MappingNode],
    fusable_tensors: set[TensorName],
    check_loops: bool = True,
):
    tensors = oset()
    for i, t in enumerate(mapping):
        if not isinstance(t, TensorHolder):
            continue

        new = (oset(t.tensors) - tensors) & fusable_tensors

        if new and check_loops:
            for j in range(i):
                if isinstance(mapping[j], Loop):
                    assert mapping[
                        j
                    ]._fused, f"Node {j} is not fused in {' '.join(m.compact_str() for m in mapping)}"
        assert (
            t._backing & fusable_tensors
        ) == new, f"Node {i} backing missing {new - t._backing} in {' '.join(m.compact_str() for m in mapping)}"
        tensors.update(new)
        tensors.update(t.tensors)


def iterate_mappings_no_constraints(
    spec: Spec,
    einsum_name: str,
    flattened_arch: list[arch.Leaf],
    rank_variable_bounds: dict[RankVariable, int],
    job: Job,
) -> Iterator[tuple[Mapping, SymbolTable, arch.Compute, int]]:
    first_memory = None
    for node in flattened_arch:
        if isinstance(node, arch.Memory):
            first_memory = node
            break
    if first_memory is None:
        raise ValueError("No memory found in architecture")

    einsum = spec.workload.einsums[einsum_name]
    symbol_table = {r.name: r.source for r in einsum.renames}
    fusable_tensors = job.fusable_tensors

    ranks_with_tile_pattern = oset(
        r for r, c in job.initial_delta_choices.items() if len(c) > 1
    )
    job.ranks_with_tile_pattern = ranks_with_tile_pattern

    fanouts = {}
    fanout = 1
    for node in flattened_arch:
        fanouts[node.name] = (fanout := fanout * node.get_fanout())

    for mapping, symbol_table, compute in get_tensor_choices(
        einsum_name,
        flattened_arch,
        symbol_table,
        spec,
        first_memory,
        fusable_tensors,
        fanouts,
        spec.mapper.prioritize_reuse_of_unfused_tensors,
    ):
        logging.info(
            "\tGenerated tensor choices: " + ", ".join(m.compact_str() for m in mapping)
        )
        mapping = copy.deepcopy(mapping)
        for mapping, n_orders in insert_temporal_loops(
            mapping,
            einsum,
            first_memory,
            rank_variable_bounds,
            ranks_with_tile_pattern,
            spec.workload,
            spec.mapper._can_lower_outermost_memory,
            flattened_arch,
            spec.mapper.max_fused_loops,
            fanouts,
            fusable_tensors,
            job.intermediate_tensors,
            spec.mapper._let_non_intermediate_tensors_respawn_in_backing_storage,
            spec.mapper.explore_loop_orders,
        ):
            mapping = copy.deepcopy(mapping)
            insert_spatial_loops(
                mapping, einsum, flattened_arch, job.intermediate_tensors
            )
            mapping = unpack_loops_to_rank_variables(mapping)
            if spec.mapper._timeloop_style_even:
                mapping = _timeloop_style_even(mapping)

            place_missing_temporal_loops(mapping, einsum, flattened_arch)
            label_fused_loops(mapping, fusable_tensors)
            assert_proper_fusion_labeling(mapping, fusable_tensors)
            yield mapping, symbol_table, compute, n_orders


def iterate_mappings_constraints(
    spec: Spec,
    einsum_names: list[str] | str,
    flattened_arch: list[arch.Leaf],
    rank_variable_bounds: dict[RankVariable, int],
    tensor_to_relevancy: dict[
        TensorName, dict[RankVariable, Relevant | PartiallyRelevant]
    ],
    job: Job,
) -> Iterator[tuple[Mapping, MappingConstraints, dict[str, str]]]:
    compute_name = flattened_arch[-1].name

    n_yielded = 0

    if isinstance(einsum_names, str):
        einsum_names = [einsum_names]

    for einsum_name in einsum_names:
        logging.info(
            f"Generating pmapping templates for compute {compute_name} Einsums "
            f"{einsum_name}"
        )

        for mapping, symbol_table, compute, n_orders in iterate_mappings_no_constraints(
            spec,
            einsum_name,
            flattened_arch,
            rank_variable_bounds,
            job,
        ):
            mapping, constraints = get_constraints(
                flattened_arch,
                mapping,
                symbol_table,
                einsum_name,
                tensor_to_relevancy,
                is_copy_operation=spec.workload.einsums[einsum_name].is_copy_operation,
            )

            # This goes after the constraints because constraints may remove some loops,
            # giving us fewer that may be reordered.
            for mapping in remove_unordered_spatial_temporal_loops(
                mapping,
                flattened_arch,
                spec.workload.einsums[einsum_name],
                spec.mapper.out_of_order_hierarchy_explore_removing_spatials_for_more_temporals,
            ):
                constraints.remove_missing_targets(mapping)

                mapping.append(
                    Compute(
                        einsum=einsum_name,
                        component=compute_name,
                        component_object=compute,
                    )
                )

                # MAPPING MUST NOT BE MODIFIED AFTER constraints.set_loop_indices
                constraints.set_loop_indices(mapping)

                mapping = Mapping(nodes=[copy.copy(n) for n in mapping])
                mapping._n_loop_orders = n_orders
                yield mapping, constraints, symbol_table
                n_yielded += 1
                if n_yielded >= spec.mapper.max_pmapping_templates_per_einsum:
                    if spec.mapper._only_output_pmapping_with_index is None:
                        return
                    if (
                        isinstance(spec.mapper._only_output_pmapping_with_index, dict)
                        and einsum_name
                        not in spec.mapper._only_output_pmapping_with_index
                    ):
                        return


# =================================================================================================
# Top level
# =================================================================================================
def make_pmapping_templates(job: Job, print_progress: bool = True) -> SameEinsumJobs:
    compute_name = job.flattened_arch[-1].name

    job.tensor_to_relevancy = {
        tensor: get_rank_variable_relevancy(
            job.spec_one_einsum.workload.einsums[job.einsum_name], tensor
        )
        for tensor in job.spec_one_einsum.workload.einsums[job.einsum_name].tensor_names
    }

    mappings_constraints = iterate_mappings_constraints(
        job.spec_one_einsum,
        job.einsum_name,
        job.flattened_arch,
        job.rank_variable_bounds,
        job.tensor_to_relevancy,
        job,
    )
    if print_progress:
        mappings_constraints = tqdm(
            mappings_constraints,
            desc=f"Generating pmapping templates for compute {compute_name} Einsum {job.einsum_name}",
        )

    jobs = SameEinsumJobs()
    only_output_index = job.spec_one_einsum.mapper._only_output_pmapping_with_index
    if isinstance(only_output_index, dict):
        only_output_index = only_output_index.get(job.einsum_name, None)

    for i, (mapping, constraints, symbol_table) in enumerate(mappings_constraints):
        if only_output_index is not None and i != only_output_index:
            continue
        if isinstance(only_output_index, set) and i not in only_output_index:
            continue
        new_job = copy.copy(job)
        new_job.mapping = mapping
        new_job.mapping._template_index = i
        new_job.constraints = constraints
        new_job.job_id = uuid.uuid4()
        new_job.rank_variable_bounds = job.rank_variable_bounds
        mapping_with_reservations = quick_insert_reservation_nodes(new_job)
        new_job.compatibility = Compatibility.from_mapping(
            mapping_with_reservations,
            new_job.fusable_tensors,
            new_job.einsum,
        )
        jobs.append(new_job)

    return jobs
