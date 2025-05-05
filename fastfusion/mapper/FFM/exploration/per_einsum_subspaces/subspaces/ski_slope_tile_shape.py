from copy import deepcopy
from functools import partial, reduce
from operator import and_, or_


def infer_smallest_tile_shape(mapping,
                              workload,
                              einsum_id,
                              tensor_to_relevant_ranks,
                              hw_level):
    """
    Set the tile shape of a loop node to 1 (smallest) if the loop is
    relevant to all tensors in storage nodes below the loop.

    Rationale: loop that is relevant to all tensors in storage nodes
    below the loop will not reduce reuse.

    Caution: might sacrifice paralellism.
    """
    tensors = workload.tensors_read_by_einsum(einsum_id)
    tensors |= workload.tensors_written_by_einsum(einsum_id)

    unstored_tensor = set(tensors)
    all_ranks = reduce(
        or_,
        [tensor_to_relevant_ranks[t] for t in tensors],
        set()
    )

    relevant_ranks = partial(
        _ranks_relevant_to_tensors,
        tensor_to_relevant_ranks=tensor_to_relevant_ranks,
        all_ranks=all_ranks
    )

    ranks_relevant_to_unstored = relevant_ranks(unstored_tensor)
    mapping = deepcopy(mapping)
    for node in mapping:
        if node["type"] == "temporal":
            rank = node["rank"]
            if rank in ranks_relevant_to_unstored:
                node["tile_shape"] = 1
        elif node["type"] == "storage" and node["target"] == hw_level:
            tensors_at_node = set(node["dspace"])
            unstored_tensor -= tensors_at_node
            ranks_relevant_to_unstored = relevant_ranks(unstored_tensor)
    yield mapping


def _ranks_relevant_to_tensors(tensors,
                               tensor_to_relevant_ranks,
                               all_ranks):
    return reduce(
        and_,
        [tensor_to_relevant_ranks[t] for t in tensors],
        all_ranks
    )

