from collections import defaultdict
from collections.abc import Callable, Set
from itertools import combinations, product

from .linear_mapping import LinearMapping
from pytimeloop.looptree.mapping_utilities import get_last_storage_node

def make_storage(
    mapping: LinearMapping,
    level,
    must_retain_tensors: Set,
    can_retain_tensors: Set,
    tensor_to_relevant_ranks,
    explore_uneven,
    must_fully_reuse_tensors: Set=None,
    add_split_at_tensors: Set=None,
    must_have_terminal_storage: bool=False,
    logfunc: Callable=None,
    return_retained_tensors: bool=False,
    apply_lrp_after_loop_idx: int=None,
    force_at_end: bool=False,
    min_index: int=None,
    max_index: int=None,
):
    if logfunc is None:
        logfunc = lambda msg: None  # do nothing

    if add_split_at_tensors is None:
        add_split_at_tensors = set()

    if apply_lrp_after_loop_idx is None:
        apply_lrp_after_loop_idx = float('inf')

    tensors = must_retain_tensors | can_retain_tensors

    if must_fully_reuse_tensors is None:
        must_fully_reuse_tensors = set()

    # Further mutated mappings copy from original first.
    original = mapping

    if not explore_uneven:
        assert min_index is None, 'Only supported for uneven'
        assert max_index is None, 'Only supported for uneven'
        assert force_at_end == False, 'Only supported for uneven'
        for r in range(len(can_retain_tensors)+1):
            for also_retained_tensors in combinations(can_retain_tensors, r):
                mapping = original.copy()

                retained_tensors = must_retain_tensors | set(also_retained_tensors)
                mapping.add_storage(level, retained_tensors)
                # if any(t in add_split_at_tensors for t in retained_tensors):
                #     mapping.add_sequential()

                if return_retained_tensors:
                    yield mapping, retained_tensors
                else:
                    yield mapping
        return

    tensors = list(sorted(tensors))

    all_tensor_choices = []
    for tensor_id in tensors:
        tensor_must_be_fully_reused = tensor_id in must_fully_reuse_tensors

        relevant_ranks = tensor_to_relevant_ranks[tensor_id]
        tensor_choices = []
        last_is_relevant = True
        
        last_storage_idx = get_last_storage_node(mapping, tensor_id)
        has_storage = last_storage_idx is not None
        if not has_storage:
            min_i = 0
        else:
            min_i = last_storage_idx + 1
        if force_at_end:
            min_i = len(mapping)

        any_irrelevant_loop = False
        for i in range(min_i, len(mapping)):
            if min_index is not None and i < min_index:
                continue
            if max_index is not None and i > max_index:
                continue
            node = mapping[i]
            if node["type"] == "temporal":
                is_relevant = node["rank"] in relevant_ranks

                any_irrelevant_loop = any_irrelevant_loop or not is_relevant

                auto_lower = i > apply_lrp_after_loop_idx
                # auto_lower = False

                if not auto_lower or (last_is_relevant and not is_relevant):
                    tensor_choices.append(i)

                last_is_relevant = is_relevant

                if tensor_must_be_fully_reused and any_irrelevant_loop:
                    break
                
        auto_lower = len(mapping) > apply_lrp_after_loop_idx
        # auto_lower = False

        # Lowest possible storage node
        if (not auto_lower or last_is_relevant) and not (tensor_must_be_fully_reused and any_irrelevant_loop):
            tensor_choices.append(len(mapping))

        if tensor_id in can_retain_tensors:
            tensor_choices.append(None)

        all_tensor_choices.append(tensor_choices)

    for choices in product(*all_tensor_choices):
        # Collect tensors with the same idx
        retained_tensors = set()
        idx_to_tensors = defaultdict(list)
        for idx, tensor in zip(choices, tensors):
            if idx is not None:
                idx_to_tensors[idx].append(tensor)
                retained_tensors.add(tensor)

        mapping = original.copy()
        success = True

        for idx, tensors_at_idx in sorted(idx_to_tensors.items(),
                                          key=lambda pair: pair[0],
                                          reverse=True):
            # if any(t in add_split_at_tensors for t in tensors_at_idx):
            #     mapping.add_sequential(idx)
            mapping.add_storage(level, tensors_at_idx, idx)
            # Check for any irrelevant loops above the backing storage for a tensor
            for t in tensors_at_idx:
                relevant_ranks = tensor_to_relevant_ranks[t]
                for node in mapping[:idx]:
                    if node["type"] == "storage" and t in node["dspace"]:
                        break
                    if node["type"] == "temporal" and node["rank"] not in relevant_ranks:
                        success = False
                        break
        if not success:
            continue

        if must_have_terminal_storage and mapping[-1]["type"] != "storage":
            continue

        assert retained_tensors & must_retain_tensors == must_retain_tensors

        if return_retained_tensors:
            yield mapping, retained_tensors
        else:
            yield mapping