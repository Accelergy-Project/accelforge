from fastfusion.frontend.mapping import Storage, Iteration


def get_fused_loops_per_tensor(pmapping, intermediate_tensors, non_fused_memory):
    """
    Returns a dictionary mapping tensor to number of fused loops or None
    if unfused (backed in non_fused_memory).
    """
    tensor_to_n_fused_loops = {t: None for t in intermediate_tensors}
    n_loops = 0
    for node in pmapping:
        if isinstance(node, Storage):
            for tensor in node.tensors:
                if (
                    tensor not in intermediate_tensors
                    or tensor in tensor_to_n_fused_loops
                ):
                    continue
                if node.memory == non_fused_memory:
                    tensor_to_n_fused_loops[tensor] = None
                else:
                    tensor_to_n_fused_loops[tensor] = n_loops
        elif isinstance(node, Iteration):
            n_loops += 1

    return tensor_to_n_fused_loops