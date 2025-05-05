from .subspaces import (
    LinearMapping,
    make_storage,
    make_temporal_fors_with_smallest_tile,
    make_temporal_fors_in_order
)

def make_ffmt_subspaces(tensors,
                        intermediate_tensors,
                        tensor_to_relevant_ranks,
                        einsum_id,
                        workload,
                        refetch_weights: bool=True):

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

    # TODO: this is brittle
    all_ranks = list(sorted(workload.einsum_ospace_dimensions(einsum_id)))
    all_ranks = list(sorted(all_ranks))
    M = all_ranks[0]
    N = all_ranks[1]
    K = all_ranks[2]
    weight_tensor = None
    input_tensor = None
    for tensor_id in workload.tensors_read_by_einsum(einsum_id):
        if tensor_to_relevant_ranks[tensor_id] == {K, N}:
            weight_tensor = tensor_id
        elif tensor_to_relevant_ranks[tensor_id] == {M, K}:
            input_tensor = tensor_id
    assert weight_tensor is not None
    assert input_tensor is not None
    output_tensor = next(iter(workload.tensors_written_by_einsum(einsum_id)))
    non_weight_tensor = tensors - {weight_tensor}

    def fused_temporal_fors(mapping, unfused_tensors):
        if input_tensor in unfused_tensors:
            allowed_fused_ranks = [M, N, K]
        elif output_tensor in unfused_tensors:
            allowed_fused_ranks = [M, N]
        else:
            allowed_fused_ranks = [M, K]
        for partial_mapping in make_temporal_fors_in_order(mapping, allowed_fused_ranks):
            yield partial_mapping, unfused_tensors


    def glb_storage_io(mapping, unfused_tensors):
        glb_fused_tensors = intermediate_tensors - unfused_tensors
        yield from make_storage(
            mapping,
            level=1,
            must_retain_tensors=non_weight_tensor,
            can_retain_tensors=set(),
            must_fully_reuse_tensors=glb_fused_tensors,
            tensor_to_relevant_ranks=tensor_to_relevant_ranks,
            explore_uneven=False,
            add_split_at_tensors=glb_fused_tensors,
            return_retained_tensors=True,
        )

    def intra_temporal_fors(mapping, _):
         for partial_mapping in make_temporal_fors_with_smallest_tile(mapping,
                                                                      {K, N}):
              yield partial_mapping, _ 

    def glb_storage_weights(mapping, _):
         yield from make_storage(
              mapping,
              level=1,
              must_retain_tensors={weight_tensor},
              can_retain_tensors=set(),
              tensor_to_relevant_ranks=tensor_to_relevant_ranks,
              explore_uneven=False,
              return_retained_tensors=True,
         )

    def mac(mapping, _):
            mapping.add_compute(einsum_id, 2)
            yield mapping

    if refetch_weights:
        return [
            lambda: [LinearMapping()],
            off_chip_storage,
            fused_temporal_fors,
            glb_storage_io,
            intra_temporal_fors,
            glb_storage_weights,
            mac
        ]
    else:
        return [
            lambda: [LinearMapping()],
            off_chip_storage,
            glb_storage_weights,
            fused_temporal_fors,
            glb_storage_io,
            intra_temporal_fors,
            mac
        ]