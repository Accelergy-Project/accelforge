from copy import copy
from uuid import uuid4

import pandas as pd

from fastfusion.frontend import arch
from fastfusion.frontend.arch import Memory
from fastfusion.frontend.spec import Mapping, Spec
from fastfusion.frontend.mapping import Compute, Split, Nested, NodeList, TensorHolder
from fastfusion.frontend._workload_isl._symbolic import (
    get_stride_and_halo_of_einsum,
)
from fastfusion.mapper import Metrics
from fastfusion.mapper.FFM._join_pmappings.compatibility import Compatibility
from fastfusion.mapper.FFM._join_pmappings.pmapping_dataframe import PmappingDataframe
from fastfusion.mapper.FFM._join_pmappings.pmapping_group import PmappingGroup
from fastfusion.mapper.FFM import join_pmappings
from fastfusion.mapper.FFM.pmappings import MultiEinsumPmappings
from fastfusion.mapper.FFM._make_pmappings.make_pmappings import get_rank_variable_bounds_for_all_einsums
from fastfusion.mapper.FFM._make_pmappings.make_pmapping_templates.make_pmapping_templates import parse_flattened_arch
from fastfusion.mapper.FFM._make_pmappings.make_pmappings_from_templates.run_model import run_model
from fastfusion.mapper.FFM._make_pmappings.pmapper_job import Job


def evaluate_mapping(spec: Spec):
    print(spec.mapping)
    spec = spec.calculate_component_energy_area(area=False)
    flattened_arches = spec.get_flattened_architecture()
    original_job = Job(
        spec=spec,
        metrics=Metrics.all_metrics(),
        rank_variable_bounds=get_rank_variable_bounds_for_all_einsums(spec),
        flattened_arch=flattened_arches[0],
    )

    resource2capacity = {}
    for flattened_arch in flattened_arches:
        for l in flattened_arch:
            if isinstance(l, arch.Memory):
                resource2capacity[l.name] = l.attributes.size

    einsum2pmappings = {}
    pmapping_objects = {}
    einsum2jobs = {}
    for pmapping in _split_mapping_to_pmappings(spec.mapping):
        pmapping_id = uuid4()
        job = copy(original_job)
        _add_backing_to_tensor_holders(pmapping)
        job.mapping = pmapping
        job.einsum_name = pmapping.nodes[-1].einsum
        einsum2jobs[job.einsum_name] = job

        symbol_table = spec.workload.get_constraint_symbol_table(job.einsum_name)
        flattened_arch = parse_flattened_arch(
            job,
            symbol_table,
        )
        job.flattened_arch = flattened_arch
        job.memories_track_all = [
            m.name for m in flattened_arch if isinstance(m, Memory)
        ]

        job.stride_and_halo = get_stride_and_halo_of_einsum(job.einsum_name, spec.workload)
        _, df, _, _ = run_model(job)
        df[f"{job.einsum_name}<SEP>mapping"] = pmapping_id

        einsum = spec.workload.einsums[pmapping.nodes[-1].einsum]
        rank_variable_to_ranks = {
            t.name: t.rank_variable2ranks for t in einsum.tensor_accesses
        }
        compatibility = Compatibility.from_mapping(pmapping, einsum.tensor_names, rank_variable_to_ranks)

        einsum2pmappings[job.einsum_name] = [
            PmappingGroup(
                compatibility,
                PmappingDataframe(
                    pd.DataFrame(df, columns=df.keys(), index=[0]),
                    1,
                    1
                )
            )
        ]
        pmapping_objects[job.einsum_name] = {pmapping_id: pmapping}

    m = MultiEinsumPmappings(
        einsum2pmappings,
        pmapping_objects,
        resource2capacity,
        einsum2jobs,
        can_combine_multiple_runs=True,
        einsums_with_pmappings_generated=spec.workload.einsum_names,
    )

    return join_pmappings(spec, m)


def _add_backing_to_tensor_holders(pmapping: Mapping):
    seen_tensors = set()
    for node in pmapping.nodes:
        if isinstance(node, TensorHolder):
            new_tensors = set(node.tensors) - seen_tensors
            node._backing = new_tensors
            seen_tensors.update(new_tensors)


def _split_mapping_to_pmappings(mapping: Mapping):
    """
    A DFS-like algorithm to split a mapping into pmappings at Split nodes.

    DFS has to be modified because the tree has list of nodes for nested nodes
    instead of links to children.
    """
    dfs_stack: list[NodeList] = [mapping.nodes]
    cur_pmapping = []

    while dfs_stack:
        # nodes_segment is a list of nested nodes with a Split or Compute at the end.
        nodes_segment = dfs_stack.pop()
        assert isinstance(nodes_segment[-1], (Split, Compute))

        cur_pmapping.append(nodes_segment[:-1])

        last_node = nodes_segment[-1]
        if isinstance(last_node, Split):
            for segment in last_node.nodes:
                assert isinstance(segment, Nested)
                dfs_stack.append(segment.nodes)
        else:
            assert isinstance(last_node, Compute)

            mapping = Mapping()
            mapping.nodes = [n for ns in cur_pmapping for n in ns] + [last_node]
            yield mapping

            cur_pmapping.pop() # Remove the last segment


def _get_compatibility_from_pmapping(pmapping: Mapping) -> Compatibility:
    for node in pmapping.nodes:
        if isinstance(node, TensorHolder):
            for tensor in node.tensors:
                compatibility.add_tensor_holder(tensor, node.component)
    return compatibility