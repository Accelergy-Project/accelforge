from .util import get_fused_loops_per_tensor


FFMT_VALID = "FFMT_VALID"
FFMT_INVALID = "FFMT_INVALID"
FFMT_WEIGHT_UNTILED = "FFMT_WEIGHT_UNTILED"
FFMT_WEIGHT_TILED = "FFMT_WEIGHT_TILED"


def get_ffmt_tag(pmapping, intermediate_tensors, non_fused_memory):
    einsum_name = pmapping[-1].einsum_name
    if "Matmul" in einsum_name:
        return get_ffmt_matmul_tag(pmapping,
                                   intermediate_tensors,
                                   non_fused_memory)
    else:
        return get_ffmt_mha_tag(pmapping,
                                intermediate_tensors,
                                non_fused_memory)



def get_ffmt_matmul_tag(pmapping, intermediate_tensors, non_fused_memory):
    tensor_to_n_fused_loops = get_fused_loops_per_tensor(pmapping,
                                                         intermediate_tensors,
                                                         non_fused_memory)


    # TODO: this is *unfused* and *uneven*. Do we need this?
    # unfused = all(n is None for n in tensor_to_n_fused_loops.values())
    # if unfused:
    #     if is_even(tiling, tensor_to_relevant_ranks, skip_tensors=["Filter"]):
    #         return (FFMT_VALID,)
    #     return (FFMT_INVALID,)

    untiled_fused = all(n == 0
                        for t, n in tensor_to_n_fused_loops
                        if t in intermediate_tensors)
    if untiled_fused:
        return (FFMT_VALID, )

    min_weight_idx, max_weight_idx, max_non_weight_idx = float('inf'), 0, 0
    max_weight_idx = 0
    for tensor, n_loops in tensor_to_n_fused_loops.items():
        is_weight = "Filter" in tensor.name
        if is_weight:
            min_weight_idx = min(min_weight_idx, n_loops)
            max_weight_idx = max(max_weight_idx, n_loops)
        else:
            max_non_weight_idx = max(max_non_weight_idx, n_loops)

    weight_untiled = min_weight_idx == 0 and max_weight_idx == 0
    if weight_untiled:
        return (FFMT_VALID, FFMT_WEIGHT_UNTILED)
    elif min_weight_idx >= max_non_weight_idx:
        return (FFMT_VALID, FFMT_WEIGHT_TILED)
    return (FFMT_INVALID,)


def get_ffmt_mha_tag(pmapping, intermediate_tensors, non_fused_memory):
    einsum_name = pmapping[-1].einsum_name
    B, H, M, F, P, G, E, D, C, J = 'bhmfpgedcj'
    EINSUM_NAME_TO_REDUCED_RANK_OUTPUT_RANK = {
        "Q":   [D, E],
        "K":   [D, E],
        "V":   [D, F],
        "QK":  [E, P],
        "AV":  [P, F],
        "Z":   [F, G],
        "FFA": [G, C],
        "FFB": [C, J]
    }

    tensor_to_n_fused_loops = get_fused_loops_per_tensor(pmapping,
                                                         intermediate_tensors,
                                                         non_fused_memory)
    unfused = all(n is None
                  for t, n in tensor_to_n_fused_loops.items()
                  if t in intermediate_tensors)
    if einsum_name not in EINSUM_NAME_TO_REDUCED_RANK_OUTPUT_RANK:
        if unfused:
            return (FFMT_VALID,)
        return (FFMT_INVALID,)

    reduced_rank, output_rank = EINSUM_NAME_TO_REDUCED_RANK_OUTPUT_RANK[einsum_name]

    EINSUM_NAME_TO_INPUT_OUTPUT_TENSORS = {
        "Q":   ["I_I_to_Q_K_V",   "Q_Q_to_QK"],
        "K":   ["I_I_to_Q_K_V",   "K_K_to_QK"],
        "V":   ["I_I_to_Q_K_V",   "V_V_to_AV"],
        "QK":  ["Q_Q_to_QK",      "QK_QK_to_AV"],
        "AV":  ["QK_QK_to_AV",    "AV_AV_to_Z"],
        "Z":   ["AV_AV_to_Z",     "Z_Z_to_FFA"],
        "FFA": ["Z_Z_to_FFA",     "FFA_FFA_to_FFB"],
        "FFB": ["FFA_FFA_to_FFB", "FFB_FFB_to_n"]
    }

    input_tensor, output_tensor = EINSUM_NAME_TO_INPUT_OUTPUT_TENSORS[einsum_name]
    input_output_tensors = {input_tensor, output_tensor}

    min_weight_idx = float('inf')
    max_weight_idx = 0
    max_non_weight_idx = 0
    first, last = True, True
    for tensor, n_loops in tensor_to_n_fused_loops.items():
        if tensor.name == input_tensor and n_loops is not None:
            first = False
        if tensor.name == output_tensor and n_loops is not None:
            last = False

        is_weight = tensor.name not in input_output_tensors
        if is_weight:
            min_weight_idx = min(min_weight_idx, n_loops)
            max_weight_idx = max(max_weight_idx, n_loops)
        else:
            max_non_weight_idx = max(max_non_weight_idx, n_loops)

    unfused = first and last
    if unfused:
        return (FFMT_VALID,)

    FFMT_CANNOT_FUSE = {"K", "V"}
    if einsum_name in FFMT_CANNOT_FUSE:
        return (FFMT_INVALID,)

    prefix_choices = [
        ([B, H], (2, 2))
    ]

    unfused = False
    extra_rank_choices = [
        ([M], (1, 1)),
    ]
    if first and last:
        unfused = True
    elif first:
        if output_rank is not None:
            extra_rank_choices.append((
                [M, output_rank],
                (1, 2)
            ))
        if reduced_rank is not None and output_rank is not None:
            extra_rank_choices.append((
                [M, output_rank, reduced_rank],
                (3, 2)
            ))
        if output_rank is None and reduced_rank is not None:
            extra_rank_choices.append((
                [M, reduced_rank],
                (2, 1)
            ))
    elif last:
        if output_rank is not None:
            extra_rank_choices.append((
                [M, output_rank],
                (1, 2)
            ))
    else:
        if reduced_rank is not None:
            extra_rank_choices.append((
                [M, reduced_rank],
                (2, 1)
            ))

    for prefix_permutation, prefix_storage in prefix_choices:
        for extra_permutation, extra_storage in extra_rank_choices:
            permutation = prefix_permutation + extra_permutation
            input_storage = prefix_storage[0] + extra_storage[0]
            output_storage = prefix_storage[1] + extra_storage[1]
            untiled_weight_idx = len(prefix_permutation)

            check_tensors = [
                TensorStorage(input_tensor, input_storage, 1, "*"),
                TensorStorage(output_tensor, output_storage, 1, "*")
            ]

            if not tiling.matches_permutation(permutation):
                continue
            if not tiling.has_tensor(*check_tensors):
                continue

            # INVARIANCE: at this point, loops[0] must be over batch
            # and loops[1] must be over heads
            if tiling.loops[0].bound != 1:   # TODO: `bound` should be `shape`
                continue
            if tiling.loops[1].bound != 1:
                continue

            weight_untiled = (
                min_weight_idx == untiled_weight_idx
                and
                max_weight_idx == untiled_weight_idx
            )
            if weight_untiled:
                return (FFMT_VALID, FFMT_WEIGHT_UNTILED)
            elif min_weight_idx >= max_non_weight_idx:
                return (FFMT_VALID, FFMT_WEIGHT_TILED)

    return (FFMT_INVALID,)