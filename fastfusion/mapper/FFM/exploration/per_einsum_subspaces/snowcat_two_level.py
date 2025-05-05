from bindings.looptree import LooptreeWorkload, LooptreeWorkloadDependencyAnalyzer

from fastfusion.mapper.FFM.exploration.per_einsum_mapper import (
    make_storage,
    make_temporal_fors,
    LinearMapping
)

from pytimeloop.looptree.equivalent_ranks import EquivalentGroups
from pytimeloop.looptree.mapping_utilities import get_intermediate_tensors


def snowcat_two_level(
    workload: LooptreeWorkload,
    einsum_id
):
    analyzer = LooptreeWorkloadDependencyAnalyzer(workload)

    tensors = workload.tensors_read_by_einsum(einsum_id) \
            | workload.tensors_written_by_einsum(einsum_id)
    intermediate_tensors = tensors & get_intermediate_tensors(workload)
    all_ranks = workload.einsum_ospace_dimensions(einsum_id)
    all_ranks = workload.einsum_ospace_dimensions(einsum_id)

    tensor_to_relevant_ranks = {
        tensor: analyzer.einsum_dims_relevant_to_tensor(einsum_id, tensor)
        for tensor in tensors
    }

    tensor_to_relevant_ranks = {
        tensor: analyzer.einsum_dims_relevant_to_tensor(einsum_id, tensor)
        for tensor in tensors
    }

    def off_chip_storage(mapping):
        off_chip_must_retain = tensors - intermediate_tensors
        off_chip_can_retain = intermediate_tensors
        yield from make_storage(
            mapping,
            level=0,
            must_retain_tensors=off_chip_must_retain,
            can_retain_tensors=off_chip_can_retain,
            tensor_to_relevant_ranks=tensor_to_relevant_ranks,
            explore_uneven=False,
            add_split_at_tensors=intermediate_tensors,
            return_retained_tensors=True,
        )

    def glb_temporal_fors(mapping, unfused_tensors):
        for partial_mapping in make_temporal_fors(mapping, all_ranks):
            yield partial_mapping, unfused_tensors

    def glb_storage(mapping, unfused_tensors):
        glb_fused_tensors = intermediate_tensors - unfused_tensors
        yield from make_storage(
            mapping,
            level=1,
            must_retain_tensors=tensors,
            can_retain_tensors=set(),
            tensor_to_relevant_ranks=tensor_to_relevant_ranks,
            explore_uneven=True,
            add_split_at_tensors=glb_fused_tensors,
            return_retained_tensors=True,
        )

    def pe_temporal_fors(mapping, unfused_tensors):
        for partial_mapping in make_temporal_fors(mapping, all_ranks):
            yield partial_mapping, unfused_tensors

    def pe_storage(mapping, unfused_tensors):
        yield from make_storage(
            mapping,
            level=2,
            must_retain_tensors=tensors,
            can_retain_tensors=set(),
            tensor_to_relevant_ranks=tensor_to_relevant_ranks,
            explore_uneven=True,
        )

    def mac(mapping):
        pass

    return [
        lambda: [LinearMapping()],
        off_chip_storage,
        glb_temporal_fors,
        glb_storage,
        pe_temporal_fors,
        pe_storage,
        mac
    ]