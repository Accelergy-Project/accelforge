from fastfusion.frontend.mapping import Storage, Iteration


def get_looptree_tag_mha(pmapping, intermediate_tensors, non_fused_memory):
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

    # Unfused
    if all(t is None for t in tensor_to_n_fused_loops.values()):
        return ("LOOPTREE_VALID",)

    # Fused with one side but not the other. We don't want to interfere with the
    # unfused side, so just go LOOPTREE_VALID. The number of loops will be enforced
    # by the tiling since it must match for the one fused tensor.
    if len(tensor_to_n_fused_loops) == 1:
        return ("LOOPTREE_VALID",)
    
    # Fused with both sides. Make sure that the number of loops is the same.
    unique_loops = set(t for t in tensor_to_n_fused_loops.values() if t is not None)
    if len(unique_loops) > 1:
        return ("LOOPTREE_INVALID",)
    return ("LOOPTREE_VALID", f"FUSED_LOOPS={next(iter(unique_loops))}")