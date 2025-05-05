def make_subspace_for_level(
    level_binding: int,
    constraint,
    workload,
    einsum_id
):
    tensors = get_tensors(workload, einsum_id)
    intermediate_tensors = get_intermediate_tensors(workload, einsum_id)
    ranks = get_ranks(workload, einsum_id)

    is_uneven = constraint["storage"]["uneven"]

    # If for certain there are no partially relevant ranks, `exploits_reuse`
    # can be set to False and reuse handled only by placing loops UNDER
    # the storage node.
    #
    # In that case, the ordering of loops is not impactful.
    if not maybe_partially_relevant_ranks(workload, einsum_id):
        unordered = is_uneven
        exploits_reuse = False
    else:
        unordered = False
        exploits_reuse = True


    def spatial_loops(mapping):
        yield from make_spatial_fors(mapping, ranks, unordered=True)

    def temporal_loops(mapping):
        yield from make_temporal_fors(mapping, ranks, unordered=unordered)

    def storage_nodes(mapping):
        yield from make_storage(explore_uneven=is_uneven,
                                exploits_reuse=exploits_reuse)

    return [spatial_loops, temporal_loops, storage_nodes]


def make_subspaces():
    subspaces = []
    for level in something:
        subspaces.extend(make_subspace_for_level())
    return subspaces


def process_constraint(constraint):
    # TODO: identify tile_shape == 1 ranks
    # TODO: detect even or uneven
    pass


def maybe_partially_relevant_ranks(workload, einsum_id):
    """
    Returns True if there may be a partially relevant ranks (false
    positive allowed).
    """
    # TODO: actually implement this
    return True
