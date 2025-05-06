from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Optional, overload

from bindings.looptree import TemporalTag, SequentialTag, PipelineTemporalTag

import islpy as isl

from pytimeloop.looptree.reuse.isl import IslReuseAnalysisOutput
from pytimeloop.looptree.reuse.summarized import SummarizedAnalysisOutput

from pytimeloop.isl.singular import get_sum_of_pw_qpolynomial
from pytimeloop.isl.sum import sum_with_mask
from pytimeloop.looptree.mapping_utilities import *


@dataclass
class Accesses:
    total_reads: Optional[float]
    total_writes: Optional[float]
    max_per_unit_reads: Optional[float]
    max_per_unit_writes: Optional[float]


class BufferAccesses:
    def __init__(self):
        self.accesses: dict[tuple, Accesses] = {}

    def get_accesses(self, buffer, dspace, einsum) -> Accesses:
        key = (buffer, dspace, einsum)
        if key not in self.accesses:
            self.accesses[key] = Accesses(0, 0, 0, 0)
        return self.accesses[key]

    def items(self):
        return self.accesses.items()

    def items_with_buffer(self, ref_buffer):
        """Returns iterator similar to `items` but only for `ref_buffer`"""
        return (
            ((buffer, dspace, einsum), value)
            for (buffer, dspace, einsum), value in self.accesses.items()
            if buffer == ref_buffer
        )


@overload
def summarize_total_and_per_unit_actions(
    reuse_analysis_result: IslReuseAnalysisOutput
):
    pass
@overload
def summarize_total_and_per_unit_actions(
    reuse_analysis_result: SummarizedAnalysisOutput
):
    pass

def summarize_total_and_per_unit_actions(
    reuse_analysis_result
):
    result = {}
    reads_to_parent = reuse_analysis_result.reads_to_parent
    reads_to_peer = reuse_analysis_result.reads_to_peer
    if isinstance(reuse_analysis_result, IslReuseAnalysisOutput):
        for key, (tags, fill) in reuse_analysis_result.fills.items():
            read_to_parent = reads_to_parent[key][1]
            read_to_peer = reads_to_peer[key][1]

            total_fill = get_sum_of_pw_qpolynomial(fill)
            total_read_to_parent = get_sum_of_pw_qpolynomial(read_to_parent)
            total_read_to_peer = get_sum_of_pw_qpolynomial(read_to_peer)

            max_per_unit_fill = \
                _sum_over_temporal_max_over_spatial(tags, fill)

            n_read_to_parent_dim = read_to_parent.dim(isl.dim_type.in_)
            max_per_unit_read_to_parent = \
                _sum_over_temporal_max_over_spatial(tags[:n_read_to_parent_dim],
                                                    read_to_parent)

            max_per_unit_read_to_peer = \
                _sum_over_temporal_max_over_spatial(tags, read_to_peer)

            result[key] = (total_fill,
                           total_read_to_parent,
                           total_read_to_peer,
                           max_per_unit_fill,
                           max_per_unit_read_to_parent,
                           max_per_unit_read_to_peer)
    elif isinstance(reuse_analysis_result, SummarizedAnalysisOutput):
        for key, (tags, fill) in reuse_analysis_result.fills.items():
            buffer_name = key[0]

            if key in reads_to_parent:
                read_to_parent = reads_to_parent[key][1]
            else:
                read_to_parent = 0

            if key in reads_to_peer:
                read_to_peer = reads_to_peer[key][1]
            else:
                read_to_peer = 0

            total_fill = fill
            total_read_to_parent = read_to_parent
            total_read_to_peer = read_to_peer

            fanout = reuse_analysis_result.fanout[buffer_name]
            total_fanout = reduce(mul, fanout, 1)

            max_per_unit_fill = fill / total_fanout
            max_per_unit_read_to_parent = read_to_parent / total_fanout
            max_per_unit_read_to_peer = read_to_peer / total_fanout

            result[key] = (total_fill,
                           total_read_to_parent,
                           total_read_to_peer,
                           max_per_unit_fill,
                           max_per_unit_read_to_parent,
                           max_per_unit_read_to_peer)
    return result



@overload
def buffer_accesses_from_buffet_actions(
    reuse_analysis_result: IslReuseAnalysisOutput,
    mapping,
    workload,
    is_path=False
) -> BufferAccesses:
    pass
@overload
def buffer_accesses_from_buffet_actions(
    reuse_analysis_result: SummarizedAnalysisOutput,
    mapping,
    workload,
    is_path=False
) -> BufferAccesses:
    pass
# TODO: is_path should be removed and we should accept only regular mappings
def buffer_accesses_from_buffet_actions(
    reuse_analysis_result,
    mapping,
    workload,
    is_path=False
) -> BufferAccesses:
    mapping = mapping['nodes']
    dspace_id_to_name = workload.data_space_id_to_name()
    einsum_id_to_name = workload.einsum_id_to_name()


    parent_buffers = get_parent_buffers(mapping, workload, is_path)

    einsums_with_complete_mappings = \
        get_einsums_with_complete_mappings(mapping, workload, is_path)

    compute_targets = set()
    for compute_node in get_leaves(mapping, is_path):
        assert compute_node["type"] == "compute"
        compute_targets.add(compute_node["target"])

    summarized_actions = \
        summarize_total_and_per_unit_actions(reuse_analysis_result)

    accesses_results = BufferAccesses()
    for (buffer_id, dspace_id, einsum_id), value in summarized_actions.items():
        (
            fill,
            read_to_parent,
            read_to_peer,
            max_per_unit_fill,
            max_per_unit_read_to_parent,
            max_per_unit_read_to_peer
        ) = value

        dspace_name = dspace_id_to_name[dspace_id]
        einsum_name = einsum_id_to_name[einsum_id]
        if einsum_id not in einsums_with_complete_mappings:
            continue

        parent_buffer = parent_buffers[(buffer_id, dspace_id, einsum_id)]
        if parent_buffer is not None:
            accesses = accesses_results.get_accesses(parent_buffer,
                                                     dspace_name,
                                                     einsum_name)
            if dspace_id in workload.tensors_written_by_einsum(einsum_id):
                accesses.total_writes += read_to_parent
                accesses.total_reads += read_to_parent

                # TODO: figure out how to do this per unit
                total_elided_reads = workload.get_tensor_volume(dspace_id)
                accesses.total_reads -= total_elided_reads

                accesses.max_per_unit_reads += max_per_unit_read_to_parent
                accesses.max_per_unit_writes += max_per_unit_read_to_parent
            elif dspace_id in workload.tensors_read_by_einsum(einsum_id):
                accesses.total_reads += read_to_parent

                accesses.max_per_unit_reads += read_to_parent

        # Fills will write into current buffer except for compute (which does
        # not have write action) and top-level buffer
        accesses = accesses_results.get_accesses(buffer_id,
                                                 dspace_name,
                                                 einsum_name)
        if buffer_id not in compute_targets and parent_buffer is not None:
            if dspace_id in workload.tensors_written_by_einsum(einsum_id):
                accesses.total_writes += fill
                accesses.max_per_unit_writes += max_per_unit_fill

                # TODO: figure out how to do this per unit
                total_elided_writes = workload.get_tensor_volume(dspace_id)
                accesses.total_writes -= total_elided_writes
            else:
                accesses.total_writes += fill
                accesses.max_per_unit_writes += max_per_unit_fill

        accesses.total_reads += read_to_peer
        accesses.max_per_unit_reads += max_per_unit_read_to_peer

    return accesses_results


def get_parent_buffers(mapping, workload, is_path):
    parent_buffers = {}
    if is_path:
        paths = [mapping]
    else:
        paths = get_paths(mapping)

    for path in paths:
        leaf = path[-1]
        einsum_name = leaf['einsum']
        if isinstance(einsum_name, int):
            einsum_id = einsum_name
        else:
            einsum_id = workload.einsum_name_to_id()[einsum_name]

        dspace_to_top_buffer = {}
        for node in path:
            if node['type'] == 'storage':
                for dspace in node['dspace']:
                    if isinstance(dspace, int):
                        dspace_id = dspace
                    else:
                        dspace_id = workload.data_space_name_to_id()[dspace]
                    key = (node['target'], dspace_id, einsum_id)
                    if dspace_id in dspace_to_top_buffer:
                        parent_buffers[key] = dspace_to_top_buffer[dspace_id]
                    else:
                        parent_buffers[key] = None
                    dspace_to_top_buffer[dspace_id] = node['target']
            elif node['type'] == 'compute':
                for dspace_id in workload.tensors_read_by_einsum(einsum_id):
                    key = (node['target'], dspace_id, einsum_id)
                    if dspace_id in dspace_to_top_buffer:
                        parent_buffers[key] = dspace_to_top_buffer[dspace_id]
                for dspace_id in workload.tensors_written_by_einsum(einsum_id):
                    key = (node['target'], dspace_id, einsum_id)
                    if dspace_id in dspace_to_top_buffer:
                        parent_buffers[key] = dspace_to_top_buffer[dspace_id]

    return parent_buffers


def _sum_over_temporal_max_over_spatial(tags, actions):
    return sum_with_mask(
        [
            (
                isinstance(t, TemporalTag) or
                isinstance(t, PipelineTemporalTag) or
                isinstance(t, SequentialTag)
            )
            for t in tags
        ],
        actions
    ).max().to_python()
