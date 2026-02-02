from accelforge.frontend.mapping import Reservation, Loop, Mapping


def get_fused_loops_per_tensor(
    pmapping: Mapping, intermediate_tensors, non_fused_memory
):
    """
    Returns a dictionary mapping tensor to number of fused loops or None
    if unfused (backed in non_fused_memory).
    """
    tensor_to_n_fused_loops = {}
    n_loops = 0
    for node in pmapping.nodes:
        if isinstance(node, Reservation):
            tensor = node.tensor
            if tensor not in intermediate_tensors or tensor in tensor_to_n_fused_loops:
                continue
            if node.component == non_fused_memory:
                tensor_to_n_fused_loops[tensor] = None
            else:
                tensor_to_n_fused_loops[tensor] = n_loops
        elif isinstance(node, Loop):
            n_loops += 1
    return tensor_to_n_fused_loops
