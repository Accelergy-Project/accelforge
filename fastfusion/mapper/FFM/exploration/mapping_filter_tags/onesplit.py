from fastfusion.mapper.FFM.tags import Tags

from .util import get_fused_loops_per_tensor


ONE_SPLIT = 'ONE_SPLIT'
NOT_ONE_SPLIT = 'NOT_ONE_SPLIT'


def get_one_split_tag(pmapping, intermediate_tensors, non_fused_memory) -> Tags:
    tensor_to_n_fused_loops = get_fused_loops_per_tensor(pmapping,
                                                         intermediate_tensors,
                                                         non_fused_memory)

    unfused = all(n is None
                  for t, n in tensor_to_n_fused_loops.items()
                  if t in intermediate_tensors)
    if unfused:
        return Tags()  # unfused is compatible with anything

    # # Fused with one side but not the other. We don't want to interfere with the
    # # unfused side, so just go ONE_SPLIT. The number of loops will be enforced
    # # by the tiling since it must match for the one fused tensor.
    # if len(tensor_to_n_fused_loops) == 1:
    #     return Tags((ONE_SPLIT,))
    
    # Fused with both sides. Make sure that the number of loops is the same.
    unique_loops = set(t for t in tensor_to_n_fused_loops.values() if t is not None)
    if len(unique_loops) > 1:
        return Tags((NOT_ONE_SPLIT,))
    return Tags((ONE_SPLIT, f"FUSED_LOOPS={next(iter(unique_loops))}"))