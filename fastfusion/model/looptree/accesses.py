from dataclasses import dataclass
from typing import Optional, overload

# from bindings.looptree import TemporalTag, SequentialTag, PipelineTemporalTag

import islpy as isl

from fastfusion.model.looptree.reuse.isl import IslReuseAnalysisOutput
from fastfusion.model.looptree.reuse.summarized.symbolic_new import SummarizedAnalysisOutput
from fastfusion.model.looptree.mapping_utilities import get_einsums_with_complete_mappings, get_paths, get_leaves

from fastfusion.frontend.mapping import Mapping, Storage, Compute
from fastfusion.frontend.workload import Workload, get_tensor_size

# from pytimeloop.isl.singular import get_sum_of_pw_qpolynomial
# from pytimeloop.isl.sum import sum_with_mask


@dataclass(eq=True)
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

    def __str__(self):
        return repr(self.accesses)

    def __repr__(self):
        return f'BufferAccesses({repr(self.accesses)})'


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
    if isinstance(reuse_analysis_result, IslReuseAnalysisOutput):
        reads_to_parent = reuse_analysis_result.reads_to_parent
        reads_to_peer = reuse_analysis_result.reads_to_peer
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
        for buffet, buffet_stats in reuse_analysis_result.buffet_stats.items():
            level = buffet.level
            einsum = buffet.einsum

            key = (level, buffet.tensor, einsum)

            total_fill = buffet_stats.total_fills
            total_read_to_parent = buffet_stats.total_reads_to_parent
            total_read_to_peer = buffet_stats.reads_to_peer

            result[key] = (
                buffet_stats.total_fills,
                buffet_stats.total_reads_to_parent,
                buffet_stats.reads_to_peer,
                buffet_stats.max_per_unit_fills,
                buffet_stats.max_per_parent_reads_to_parent,
                0 # TODO: peer-to-peer
            )
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
    workload: Workload,
    is_path=False
) -> BufferAccesses:
    mapping = mapping.nodes

    parent_buffers = get_parent_buffers(mapping, workload, is_path)

    compute_targets = set()
    for compute_node in get_leaves(mapping, is_path):
        assert isinstance(compute_node, Compute)
        compute_targets.add(compute_node.compute)

    summarized_actions = \
        summarize_total_and_per_unit_actions(reuse_analysis_result)

    accesses_results = BufferAccesses()
    for (buffer_id, tensor, einsum), value in summarized_actions.items():
        (
            fill,
            read_to_parent,
            read_to_peer,
            max_per_unit_fill,
            max_per_parent_read_to_parent,
            max_per_unit_read_to_peer
        ) = value

        parent_buffer = parent_buffers[(buffer_id, tensor, einsum)]
        if parent_buffer is not None:
            accesses = accesses_results.get_accesses(parent_buffer,
                                                     tensor,
                                                     einsum)
            if tensor in workload.tensors_written_by_einsum(einsum):
                accesses.total_writes += read_to_parent
                accesses.total_reads += read_to_parent

                # # TODO: figure out how to do this per unit
                # total_elided_reads = get_tensor_size(workload, tensor)
                # accesses.total_reads -= total_elided_reads

                accesses.max_per_unit_reads += max_per_parent_read_to_parent
                accesses.max_per_unit_writes += max_per_parent_read_to_parent
            elif tensor in workload.tensors_read_by_einsum(einsum):
                accesses.total_reads += read_to_parent

                accesses.max_per_unit_reads += max_per_parent_read_to_parent

        # Fills will write into current buffer except for compute (which does
        # not have write action) and top-level buffer
        if buffer_id not in compute_targets and parent_buffer is not None:
            accesses = accesses_results.get_accesses(buffer_id,
                                                    tensor,
                                                    einsum)
            if tensor in workload.tensors_written_by_einsum(einsum):
                accesses.total_writes += fill
                accesses.max_per_unit_writes += max_per_unit_fill

                # # TODO: figure out how to do this per unit
                # total_elided_writes = get_tensor_size(workload, tensor)
                # accesses.total_writes -= total_elided_writes
            else:
                accesses.total_writes += fill
                accesses.max_per_unit_writes += max_per_unit_fill

        accesses.total_reads += read_to_peer
        accesses.max_per_unit_reads += max_per_unit_read_to_peer

    return accesses_results


def get_parent_buffers(mapping: Mapping, workload: Workload, is_path):
    parent_buffers = {}
    if is_path:
        paths = [mapping]
    else:
        paths = get_paths(mapping)

    for path in paths:
        leaf = path[-1]
        einsum = leaf.einsum

        tensor_to_top_buffer = {}
        for node in path:
            if isinstance(node, Storage):
                for tensor in node.tensor:
                    key = (node.memory, tensor, einsum)
                    if tensor in tensor_to_top_buffer:
                        parent_buffers[key] = tensor_to_top_buffer[tensor]
                    else:
                        parent_buffers[key] = None
                    tensor_to_top_buffer[tensor] = node.memory
            elif isinstance(node, Compute):
                for tensor in workload.tensors_read_by_einsum(einsum):
                    key = (node.compute, tensor, einsum)
                    if tensor in tensor_to_top_buffer:
                        parent_buffers[key] = tensor_to_top_buffer[tensor]
                for tensor in workload.tensors_written_by_einsum(einsum):
                    key = (node.compute, tensor, einsum)
                    if tensor in tensor_to_top_buffer:
                        parent_buffers[key] = tensor_to_top_buffer[tensor]

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
