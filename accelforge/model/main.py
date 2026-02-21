from copy import copy, deepcopy
from uuid import uuid4

import pandas as pd

from accelforge.frontend import arch
from accelforge.frontend.arch import Memory
from accelforge.frontend.mapping.mapping import MappingNodeWithChildren
from accelforge.frontend.renames import EinsumName
from accelforge.frontend.spec import Mapping, Spec
from accelforge.frontend.mapping import (
    Compute,
    Reservation,
    Split,
    Nested,
    NodeList,
    TensorHolder,
)
from accelforge.frontend.workload import Workload
from accelforge.frontend._workload_isl._symbolic import (
    get_stride_and_halo_of_einsum,
    get_rank_variable_relevancy,
)
from accelforge.mapper.FFM._pareto_df.df_convention import col_used_in_joining


def evaluate_mapping(
    spec: Spec,
    flattened_arches: dict[(EinsumName, str), list[arch.Leaf]] | None = None,
    evaluated_specs: dict[EinsumName, Spec] | None = None,
):
    """
    Evaluate a mapping.

    Parameters
    ----------
    spec:
        The specification of architecture, workload, and mapping.
    flattened_arches:
        A dictionary of (EinsumName, Compute Name) to lists of architecture nodes. These
        contain the evaluated and flattened architecture node for that particular Einsum
        and compute combination. If provided, then these will be used instead of
        re-parsing the architecture.
    evaluated_specs:
        A dictionary of Einsum names to evaluated specifications. These contain the evaluated
        specification for that particular Einsum. If provided, then these will be used
        instead of re-parsing the specification.
    """
    from accelforge.mapper.FFM._join_pmappings.compatibility import Compatibility
    from accelforge.mapper.FFM._join_pmappings.pmapping_dataframe import (
        PmappingDataframe,
    )
    from accelforge.mapper.FFM._join_pmappings.pmapping_group import PmappingGroup
    from accelforge.mapper.FFM._join_pmappings.join_pmappings import (
        clean_compress_and_join_pmappings,
    )
    from accelforge.mapper.FFM.pmappings import MultiEinsumPmappings
    from accelforge.mapper.FFM._make_pmappings.make_pmappings import (
        get_rank_variable_bounds_for_all_einsums,
    )
    from accelforge.mapper.FFM._make_pmappings.make_pmappings_from_templates.run_model import (
        run_model,
    )
    from accelforge.mapper.FFM._make_pmappings.make_pmappings_from_templates.make_tile_shapes import (
        _calculate_iterations_and_rank_columns, _clean_energy_columns,
    )
    from accelforge.mapper.FFM._make_pmappings.pmapper_job import Job

    assert (evaluated_specs is not None) == (
        flattened_arches is not None
    ), f"Provide either flattened_arches or evaluated_specs, not both."

    original_job = Job(
        metrics=spec.model.metrics,
        rank_variable_bounds=get_rank_variable_bounds_for_all_einsums(spec),
        spec=spec,
    )

    einsum2pmappings = {}
    pmapping_objects = {}
    einsum2jobs = {}
    s = (
        "Spec must not be evaluated before evaluating a mapping. Was "
        "this spec returned by spec.calculate_component_area_energy_latency_leak()?"
    )

    needs_reservations = not bool(spec.mapping.get_nodes_of_type(Reservation))

    assert not getattr(spec, "_evaluated", False), s
    for pmapping in _split_mapping_to_pmappings(spec.mapping, spec.workload):
        einsum_name = pmapping.nodes[-1].einsum
        compute_name = pmapping.nodes[-1].component
        pmapping_id = uuid4()
        job = copy(original_job)

        if flattened_arches is not None:
            flattened_arch = flattened_arches[(einsum_name, compute_name)]
            cur_spec = evaluated_specs[einsum_name]

        else:
            cur_spec = spec.calculate_component_area_energy_latency_leak(
                einsum_name=einsum_name,
                area=False,
            )
            flattened_arch = cur_spec._get_flattened_architecture(
                compute_node=pmapping.nodes[-1].component
            )

        job.spec = cur_spec
        job.einsum_name = pmapping.nodes[-1].einsum
        job.stride_and_halo = get_stride_and_halo_of_einsum(
            job.einsum_name, cur_spec.workload
        )

        pmapping.split_reservations()
        pmapping.split_loop_with_multiple_rank_variables(job.einsum_name)
        pmapping.split_tensor_holders_with_multiple_tensors()
        _add_backing_to_tensor_holders(pmapping)

        job.mapping = pmapping
        job.tensor_to_relevancy = {
            tensor: get_rank_variable_relevancy(
                job.spec.workload.einsums[job.einsum_name], tensor
            )
            for tensor in job.spec.workload.einsums[job.einsum_name].tensor_names
        }
        einsum2jobs[job.einsum_name] = job

        job.flattened_arch = flattened_arch
        job.memories_track_all = [
            m.name for m in flattened_arch if isinstance(m, Memory)
        ]

        job.fusable_tensors = set(
            cur_spec.workload.tensor_names_used_in_multiple_einsums
            & set(job.tensor_to_relevancy)
        )
        einsum = cur_spec.workload.einsums[job.einsum_name]
        rank_variable_to_ranks = {
            t.name: t.rank_variable2ranks for t in einsum.tensor_accesses
        }

        _, df, _, _, tensor2mapping = run_model(
            job, add_reservations=needs_reservations
        )

        # Calculate iteration counts and rank columns
        _clean_energy_columns(df, job.metrics)
        _calculate_iterations_and_rank_columns(
            job.mapping.nodes, job, df, job.rank_variable_bounds
        )
        compatibility = Compatibility.from_mapping(
            job.mapping,
            job.fusable_tensors,
            rank_variable_to_ranks,
        )
        symbol_renames, compatibility = compatibility.make_fused_loop_symbols(
            einsum_name
        )
        for k, v in symbol_renames.items():
            try:
                df[v] = df.pop(k)
            except:
                Compatibility.from_mapping(
                    job.mapping,
                    job.fusable_tensors,
                    rank_variable_to_ranks,
                )
                _calculate_iterations_and_rank_columns(
                    job.mapping.nodes, job, df, job.rank_variable_bounds
                )
                raise

        new_df = {}
        for key, value in df.items():
            if not col_used_in_joining(key):
                key = f"{job.einsum_name}<SEP>{key}"
            # Want usage both for joining & for per-einsum info
            if key.startswith("usage<SEP>"):
                new_df[f"{job.einsum_name}<SEP>{key}"] = value
            new_df[key] = value
        df = new_df
        df[f"{job.einsum_name}<SEP>mapping"] = pmapping_id

        einsum2pmappings[job.einsum_name] = [
            PmappingGroup(
                compatibility,
                PmappingDataframe(pd.DataFrame(df, columns=df.keys(), index=[0]), 1, 1),
            )
        ]
        pmapping_objects[job.einsum_name] = {pmapping_id: job.mapping}

    # Restore the original order
    einsum2pmappings = {
        einsum_name: einsum2pmappings[einsum_name]
        for einsum_name in spec.workload.einsum_names
        if einsum_name in einsum2pmappings
    }

    return clean_compress_and_join_pmappings(
        pmappings=MultiEinsumPmappings(
            spec=spec,
            einsum2pmappings=einsum2pmappings,
            pmapping_objects=pmapping_objects,
            einsum2jobs=einsum2jobs,
            can_combine_multiple_runs=False,
            einsums_with_pmappings_generated=set(spec.workload.einsum_names),
            flattened_arches=flattened_arches,
            evaluated_specs=evaluated_specs,
        ),
        metrics=spec.model.metrics,
        print_progress=False,
    )


def _add_backing_to_tensor_holders(pmapping: Mapping):
    seen_tensors = set()
    for node in pmapping.nodes:
        if isinstance(node, TensorHolder):
            new_tensors = set(node.tensors) - seen_tensors
            node._backing = new_tensors
            seen_tensors.update(new_tensors)


def _split_mapping_worker(node: MappingNodeWithChildren):
    if isinstance(node, Split):
        for subnodes in node.nodes:
            yield from _split_mapping_worker(subnodes)
        return

    assert isinstance(node, Nested), "BUG"

    for n in node.nodes[:-1]:
        assert not isinstance(n, MappingNodeWithChildren), "BUG"

    if not isinstance(node.nodes[-1], MappingNodeWithChildren):
        yield node.nodes
        return

    for subnodes in _split_mapping_worker(node.nodes[-1]):
        yield node.nodes[:-1] + subnodes


def _split_mapping_to_pmappings(mapping: Mapping, workload: Workload):
    """
    A DFS-like algorithm to split a mapping into pmappings at Split nodes.

    DFS has to be modified because the tree has list of nodes for nested nodes
    instead of links to children.
    """
    for nodes in _split_mapping_worker(mapping):
        mapping = Mapping(nodes=deepcopy(nodes))
        _remove_storage_of_unrelevant_tensors(mapping, workload)
        yield mapping


def _remove_storage_of_unrelevant_tensors(pmapping: Mapping, workload: Workload):
    """
    Remove tensors from Storage nodes that are not relevant to the Einsum being
    mapped.
    """
    einsum_name = pmapping.nodes[-1].einsum
    einsum = workload.einsums[einsum_name]
    relevant_tensors = set(t.name for t in einsum.tensor_accesses)

    new_nodes = []
    for node in pmapping.nodes:
        if isinstance(node, TensorHolder):
            node.tensors = [t for t in node.tensors if t in relevant_tensors]
            if node.tensors:
                new_nodes.append(node)
        else:
            new_nodes.append(node)

    pmapping.nodes = new_nodes
