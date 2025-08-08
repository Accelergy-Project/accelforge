from fastfusion.mapper.FFM.deprecate_maybe.tags import Tags
from fastfusion.mapper.FFM.joining.mappinginfo import Compatibility


ONE_SPLIT = 'ONE_SPLIT'
NOT_ONE_SPLIT = 'NOT_ONE_SPLIT'


def get_one_split_tag(compatibility: Compatibility) -> Tags:
    # TODO
    unique_loops = set()
    for tensor in compatibility.tensors:
        if tensor.resource_name == "MainMemory":
            continue
        unique_loops.add(tensor.above_loop_index)

    if len(unique_loops) == 0:
        return Tags()  # unfused is compatible with anything

    # Fused with both sides. Make sure that the number of loops is the same.
    if len(unique_loops) > 1:
        return Tags(("INVALID",))

    return Tags((ONE_SPLIT, f"FUSED_LOOPS={next(iter(unique_loops))}"))