from fastfusion.frontend.mapping.mapping import MappingNode


import copy
from collections import defaultdict
import itertools
import logging
from typing import Any, Iterator, List
import uuid

from tqdm import tqdm

import fastfusion.frontend.arch as arch
from fastfusion.frontend.mapping import (
    Compute,
    Loop,
    Mapping,
    MappingNode,
    Spatial,
    TensorHolder,
    Temporal,
)
from fastfusion.frontend.spec import Spec
from fastfusion.frontend._workload_isl._isl import get_rank_variable_bounds
from fastfusion.frontend._workload_isl._symbolic import (
    Relevant,
    get_rank_variable_relevancy,
    get_stride_and_halo,
    get_stride_and_halo_of_einsum,
    PartiallyRelevant,
)
from fastfusion.frontend.workload import (
    TensorName,
    Einsum,
    EinsumName,
    RankVariable,
    Workload,
    isl_expression_has_variable,
    SymbolTable,
)
from fastfusion.mapper.FFM._make_pmappings.make_pmapping_templates.make_storage_order import (
    get_tensor_choices,
)
from fastfusion.mapper.FFM._make_pmappings.make_pmapping_templates.make_reservations import (
    get_reservation_choices,
)
from fastfusion.mapper.FFM._make_pmappings.contraints.constraints import (
    MappingConstraints,
    get_constraints,
)
from fastfusion.mapper.FFM._make_pmappings.make_pmapping_templates.make_loops import (
    insert_temporal_loops,
    insert_spatial_loops,
)
from fastfusion.mapper.FFM._make_pmappings.pmapper_job import (
    Job,
    SameEinsumJobs,
)
from fastfusion.model._looptree.reuse.symbolic import label_fused_loops


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

    index_exprs = einsum.indexing_expressions

    # Remove a temporal loop if:
    # - It's between a spatial loop and a storage node above that fanout in the arch
    # - It indexes into one of the same indexing expressions as the spatial loop

    disallowed_combinations: list[tuple[set[int], set[int]]] = []
    for i, node in enumerate(mapping):
        if not isinstance(node, Spatial):
            continue

        last_idx_to_check = _idx_of_lowest_tensor_holder_with_component_above_fanout(
            mapping, i, fanouts, node
        )
        to_check = mapping[i + 1 : last_idx_to_check]
        to_remove = set()
        for n in to_check:
            if isinstance(n, Temporal):
                for expr in index_exprs:
                    if not isl_expression_has_variable(expr, node.rank_variable):
                        continue
                    if not isl_expression_has_variable(expr, n.rank_variable):
                        continue
                    to_remove.add(id(n))
                    break

        if to_remove:
            disallowed_combinations.append((set([id(node)]), to_remove))

    if not explore_unordered_spatial_loops:
        disallowed_combinations = [x[1:] for x in disallowed_combinations]

    for combo in itertools.product(*disallowed_combinations):
        combo = set.union(set(), *combo)
        yield [n for n in mapping if id(n) not in combo]


def _idx_of_lowest_tensor_holder_with_component_above_fanout(
    mapping, start_idx, fanouts, node
):
    """
    Return idx of lowest tensor holder with component above fanout. If none
    found, returns index right under start idx (start_idx + 1).
    """
    for j in range(len(mapping) - 1, start_idx, -1):
        n = mapping[j]
        if (
            isinstance(n, TensorHolder)
            and fanouts[n.component] < fanouts[node.component]
        ):
            return j
    return start_idx + 1


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
    tensors = set()
    for i, t in enumerate(mapping):
        if not isinstance(t, TensorHolder):
            continue

        new = (set(t.tensors) - tensors) & fusable_tensors

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


def get_initial_delta_choices(einsum_name: str, workload: Workload):
    stride_and_halo = get_stride_and_halo(workload)
    einsum = workload.einsums[einsum_name]

    choices = defaultdict(lambda: set([0]))
    consumer_chains = []
    stack = [[(None, einsum)]]
    while stack:
        cur_chain = stack.pop()
        last_tensor, last_einsum = cur_chain[-1]
        for tensor in last_einsum.output_tensor_names:
            einsums_with_tensor_as_input = workload.einsums_with_tensor_as_input(tensor)

            if len(einsums_with_tensor_as_input) == 0:
                consumer_chains.append(cur_chain)

            for next_einsum in einsums_with_tensor_as_input:
                stack.append(cur_chain + [(tensor, next_einsum)])

    for chain in consumer_chains:
        for (_, producer), (tensor, consumer) in zip(
            list(reversed(chain))[1:], reversed(chain)
        ):
            rank_stride_and_halo = stride_and_halo[(consumer.name, tensor)]
            if tensor is None:
                break  # done

            for cons_rank_var in consumer.rank_variables:
                for prod_rank_var in producer.rank_variables:
                    for cons_choice in choices[cons_rank_var]:
                        if (prod_rank_var, cons_rank_var) not in rank_stride_and_halo:
                            continue
                        stride, halo = rank_stride_and_halo[
                            (prod_rank_var, cons_rank_var)
                        ]
                        choices[prod_rank_var].add(cons_choice * stride + halo)

    return choices


def get_ranks_with_tile_pattern(producer_name: EinsumName, workload: Workload):
    initial_choices = get_initial_delta_choices(producer_name, workload)
    return {
        rank_var
        for rank_var in workload.einsums[producer_name].rank_variables
        if len(initial_choices[rank_var]) > 1
    }


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

    ranks_with_tile_pattern = get_ranks_with_tile_pattern(einsum_name, spec.workload)

    einsum = spec.workload.einsums[einsum_name]
    symbol_table = {r.name: r.source for r in einsum.renames}
    fusable_tensors = job.fusable_tensors

    for mapping, symbol_table, compute in get_tensor_choices(
        einsum_name,
        flattened_arch,
        symbol_table,
        spec,
        first_memory,
        fusable_tensors,
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
            spec.mapper.ffm._can_lower_outermost_memory,
            flattened_arch,
            spec.mapper.ffm.max_fused_loops,
        ):
            mapping = copy.deepcopy(mapping)
            insert_spatial_loops(mapping, einsum, flattened_arch)
            mapping = unpack_loops_to_rank_variables(mapping)
            if spec.mapper.ffm._timeloop_style_even:
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
                flattened_arch, mapping, symbol_table, einsum_name, tensor_to_relevancy
            )

            # This goes after the constraints because constraints may remove some loops,
            # giving us fewer that may be reordered.
            for mapping in remove_unordered_spatial_temporal_loops(
                mapping,
                flattened_arch,
                spec.workload.einsums[einsum_name],
                spec.mapper.ffm.out_of_order_hierarchy_explore_removing_spatials_for_more_temporals,
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
                if n_yielded >= spec.mapper.ffm.max_pmapping_templates_per_einsum:
                    return


# =================================================================================================
# Top level
# =================================================================================================
def make_pmapping_templates(job: Job) -> SameEinsumJobs:
    compute_name = job.flattened_arch[-1].name

    job.tensor_to_relevancy = {
        tensor: get_rank_variable_relevancy(
            job.spec.workload.einsums[job.einsum_name], tensor
        )
        for tensor in job.spec.workload.einsums[job.einsum_name].tensor_names
    }

    mappings_constraints = tqdm(
        iterate_mappings_constraints(
            job.spec,
            job.einsum_name,
            job.flattened_arch,
            job.rank_variable_bounds,
            job.tensor_to_relevancy,
            job,
        ),
        desc=f"Generating pmapping templates for compute {compute_name} Einsum {job.einsum_name}",
    )

    stride_and_halo = get_stride_and_halo_of_einsum(
        job.einsum_name, job.spec.workload, job.rank_variable_bounds
    )

    jobs = SameEinsumJobs()
    only_output_pmapping_index = job.spec.mapper.ffm._only_output_pmapping_index
    for i, (mapping, constraints, symbol_table) in enumerate(mappings_constraints):
        if only_output_pmapping_index is not None and i != only_output_pmapping_index:
            continue
        new_job = copy.copy(job)
        new_job.mapping = mapping
        new_job.constraints = constraints
        new_job.job_id = uuid.uuid4()
        new_job.rank_variable_bounds = job.rank_variable_bounds
        new_job.stride_and_halo = stride_and_halo
        new_job.compatibility
        jobs.append(new_job)

    return jobs
