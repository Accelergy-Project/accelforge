from fastfusion.frontend.mapping import Loop, Temporal
from fastfusion.mapper.FFM.deprecate_maybe.tags import Tags

from .util import get_fused_loops_per_tensor


FFMT_VALID = "FFMT_VALID"
FFMT_WEIGHT_UNTILED = "FFMT_WEIGHT_UNTILED"
FFMT_WEIGHT_TILED = "FFMT_WEIGHT_TILED"


def get_ffmt_tag(compatibility):
    return get_ffmt_matmul_tag(compatibility)
    if "Matmul" in einsum_name:
        return get_ffmt_matmul_tag(compatibility)
    else:
        return get_ffmt_mha_tag(compatibility)


def get_ffmt_matmul_tag(compatibility):
    # FFMT is:
    # - [input | output, weight]
    # If there's >1 fused loop, they must be above the same number of loops
    tensors = [s for s in compatibility.tensors if s.resource_name != "MainMemory"]
    if len(tensors) <= 1:
        return Tags((FFMT_VALID,))

    allowed_n_loops = [
        (0, 0),
        (1, 1),
        (1, 2),
    ]

    # If there's a B or H fused loop, add one to the allowed n_loops
    for rank_var in "b", "h":
        if any(rank_var in l.rank_variable for l in compatibility.loops):
            allowed_n_loops = [(x + 1, y + 1) for x, y in allowed_n_loops]

    if tuple(sorted(s.above_loop_index for s in tensors)) in [
        (0, 0),
        (1, 1),
        (1, 2),
    ]:
        return Tags((FFMT_VALID,))
    raise ValueError()


def get_ffmt_mha_tag(compatibility):
    tensors = [s for s in compatibility.tensors if s.resource_name != "MainMemory"]
    if len(compatibility.loops) == 0:
        return Tags((FFMT_VALID,))

    # Loops have to be in the order (b, h)
    if len(compatibility.loops) == 1:
        return Tags((FFMT_INVALID,))

    if len(set(s.above_loop_index for s in tensors)) > 1:
        raise ValueError()
    return Tags((FFMT_VALID,))

    for tensors in compatibility.tensors:
        if tensor.resource_name == "MainMemory":
            continue
        unique_loops.add(tensor.above_loop_index)

    if len(unique_loops) == 0:
        return Tags()  # unfused is compatible with anything

    untiled_fused = len(unique_loops) == 1 and next(iter(unique_loops)) == 0
    if untiled_fused:
        return Tags((FFMT_VALID,))

    min_weight_idx, max_weight_idx, max_non_weight_idx = float("inf"), 0, 0
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
        return Tags((FFMT_VALID, FFMT_WEIGHT_UNTILED))
    elif min_weight_idx >= max_non_weight_idx:
        return Tags((FFMT_VALID, FFMT_WEIGHT_TILED))
    raise ValueError()


def get_ffmt_mha_tag(pmapping):
    einsum_name = pmapping[-1].einsum_name
    B, H, M, F, P, G, E, D, C, J = "bhmfpgedcj"
    EINSUM_NAME_TO_REDUCED_RANK_OUTPUT_RANK = {
        "Q": [D, E],
        "K": [D, E],
        "V": [D, F],
        "QK": [E, P],
        "AV": [P, F],
        "Z": [F, G],
        "FFA": [G, C],
        "FFB": [C, J],
    }

    rank_var_permutation = []
    for node in pmapping:
        if isinstance(node, Loop):
            if not isinstance(node, Temporal):
                raise RuntimeError(
                    "get_ffmt_mha_tag should not be used for "
                    "anything other than Snowcat"
                )
            rank_var_permutation.append(node.rank_variable)

    tensor_to_n_fused_loops = get_fused_loops_per_tensor(
        pmapping, intermediate_tensors, "MainMemory"
    )
    unfused = all(
        n is None
        for t, n in tensor_to_n_fused_loops.items()
        if t in intermediate_tensors
    )
    if einsum_name not in EINSUM_NAME_TO_REDUCED_RANK_OUTPUT_RANK:
        if unfused:
            return Tags((FFMT_VALID,))
        raise ValueError()

    reduced_rank, output_rank = EINSUM_NAME_TO_REDUCED_RANK_OUTPUT_RANK[einsum_name]

    EINSUM_NAME_TO_INPUT_OUTPUT_TENSORS = {
        "Q": ["I_I_to_Q_K_V", "Q_Q_to_QK"],
        "K": ["I_I_to_Q_K_V", "K_K_to_QK"],
        "V": ["I_I_to_Q_K_V", "V_V_to_AV"],
        "QK": ["Q_Q_to_QK", "QK_QK_to_AV"],
        "AV": ["QK_QK_to_AV", "AV_AV_to_Z"],
        "Z": ["AV_AV_to_Z", "Z_Z_to_FFA"],
        "FFA": ["Z_Z_to_FFA", "FFA_FFA_to_FFB"],
        "FFB": ["FFA_FFA_to_FFB", "FFB_FFB_to_n"],
    }

    input_tensor, output_tensor = EINSUM_NAME_TO_INPUT_OUTPUT_TENSORS[einsum_name]
    input_output_tensors = {input_tensor, output_tensor}

    min_weight_idx = float("inf")
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

    # Rank variable order and the n_loops for (input, output)
    prefix_choices = [([B, H], (2, 2))]

    # Rank variable order and the n_loops for (input, output)
    extra_rank_choices = [
        ([M], (1, 1)),
    ]
    if first:
        if output_rank is not None:
            extra_rank_choices.append(([M, output_rank], (1, 2)))
        if reduced_rank is not None and output_rank is not None:
            extra_rank_choices.append(([M, output_rank, reduced_rank], (3, 2)))
        if output_rank is None and reduced_rank is not None:
            extra_rank_choices.append(([M, reduced_rank], (2, 1)))
    elif last:
        if output_rank is not None:
            extra_rank_choices.append(([M, output_rank], (1, 2)))
    else:
        if reduced_rank is not None:
            extra_rank_choices.append(([M, reduced_rank], (2, 1)))

    for prefix_permutation, prefix_n_loops in prefix_choices:
        for extra_permutation, extra_n_loops in extra_rank_choices:
            permutation = prefix_permutation + extra_permutation
            input_n_loops = prefix_n_loops[0] + extra_n_loops[0]
            output_n_loops = prefix_n_loops[1] + extra_n_loops[1]
            untiled_weight_idx = len(prefix_permutation)

            permutation_matches = True
            for rank_var, ref_rank_var in zip(rank_var_permutation, permutation):
                if rank_var != ref_rank_var:
                    permutation_matches = False
                    break

            if not permutation_matches:
                continue

            if tensor_to_n_fused_loops[input_tensor] != input_n_loops:
                continue
            if tensor_to_n_fused_loops[output_tensor] != output_n_loops:
                continue

            weight_untiled = (
                min_weight_idx == untiled_weight_idx
                and max_weight_idx == untiled_weight_idx
            )
            if weight_untiled:
                return Tags((FFMT_VALID, FFMT_WEIGHT_UNTILED))
            elif min_weight_idx >= max_non_weight_idx:
                return Tags((FFMT_VALID, FFMT_WEIGHT_TILED))

    raise ValueError()
