import itertools

import fastfusion.frontend.architecture as architecture
from fastfusion.frontend.mapping import (
    MappingNode,
    Temporal,
    Spatial,
    TensorHolder
)
from fastfusion.frontend.workload.workload import (
    Einsum,
    RankVariableName,
    Workload,
)


# =================================================================================================
# Insert loops
# =================================================================================================
import itertools


def insert_temporal_loops(
    mapping: list[TensorHolder],
    einsum: Einsum,
    first_memory: architecture.Memory,
    rank_variable_bounds: dict[RankVariableName, int],
    ranks_with_tile_pattern: set,
    workload: Workload,
    except_from_imperfect: set,
):
    # First establish insertion points. Insertion points are:
    # - Below the last instance of the first memory
    # - Between any two TensorHolder nodes
    # - After the last TensorHolder node

    split_mapping: list[list[TensorHolder]] = [[]]
    for m in mapping:
        split_mapping.append([m])
        if m.component == first_memory.name:
            while len(split_mapping) > 1:
                split_mapping[0].extend(split_mapping.pop(1))

    # These Einsum properties are recalculated since Einsum is mutable
    # We're pre-computing and reusing for efficiency
    tensor2fully_relevant_rank_vars = einsum.tensor2fully_relevant_rank_variables
    tensor2partially_relevant_rank_vars = (
        einsum.tensor2partially_relevant_rank_variables
    )
    tensor2irrelevant_rank_vars = einsum.tensor2irrelevant_rank_variables
    tensor2rank_vars = einsum.tensor2rank_variables

    intermediate_tensors = einsum.tensor_names & workload.intermediate_tensor_names
    is_fused_loops = True
    seen_tensors = set()
    choices = []
    lowering_choices = []

    for i, prev_tensor_holders in enumerate(split_mapping):
        # =============================================================================
        # Choose what temporal loops to insert between prev_tensor_holders and the next
        # TensorHolder node(s).
        # =============================================================================

        next_tensor_holders = split_mapping[i + 1] if i < len(split_mapping) - 1 else []
        assert sum(len(set(s.tensors) - s._backing) for s in prev_tensor_holders) <= 1
        assert sum(len(s.tensors) for s in next_tensor_holders) <= 1

        rank_variables = einsum.rank_variables
        # rank_variables = {r for r in rank_variables if rank_variable_bounds[r] > 1}
        seen_tensors |= set.union(*(set(t.tensors) for t in prev_tensor_holders), set())
        is_fused_loops = is_fused_loops and len(intermediate_tensors - seen_tensors) > 0
        prev_tensors = set.union(set(), *(set(t.tensors) for t in prev_tensor_holders))

        # Generally we want to only use rank variables that are irrelevant to the
        # previous tensors, else we'd just lower those tensors. However, we can't lower
        # backing TensorHolder nodes because this will add loops to compatibility.

        # No recomputation: If we haven't seen a tensor yet, must only iterate over
        # fully-relevant rank variables.
        for t in intermediate_tensors - seen_tensors:
            rank_variables &= tensor2fully_relevant_rank_vars[t]

        # Optimality-preserving optimizations: We can trivially lower non-backing
        # TensorHolder nodes through fully-relevant loops. Can't do this if the loops
        # are fused because that'd add loops to the compatibility.
        for s in prev_tensor_holders:
            for t in s.tensors:
                if t not in s._backing and not s._must_be_here:
                    rank_variables -= tensor2fully_relevant_rank_vars[t]

        # Optimality-preserving optimization: We can trivially raise TensorHolder nodes
        # through irrelevant unfused loops. Can't do this if the loops are fused because
        # that'd increase the lifetime of the TensorHolder node.
        if not is_fused_loops:
            for s in next_tensor_holders:
                if not s._must_be_here:
                    for t in s.tensors:
                        rank_variables -= tensor2irrelevant_rank_vars[t]

        # Test permutations of partially-relevant rank variables because we'll be
        # lowering through them. Don't permute fully-relevant rank variables because
        # we won't lower through them.
        partially_relevant_to_previous = set.union(
            set(), *(tensor2partially_relevant_rank_vars[t] for t in prev_tensors)
        )
        partially_relevant_choices = list(
            itertools.permutations(rank_variables & partially_relevant_to_previous)
        )
        irrelevant_choices = tuple(
            sorted(rank_variables - partially_relevant_to_previous)
        )
        choices.append([x + irrelevant_choices for x in partially_relevant_choices])

        # =============================================================================
        # Choose whether to lower TensorHolder nodes through partially-relevant loops.
        # =============================================================================

        # Option 1: Previous TensorHolder is backing and the loop(s) are
        # partially-relevant. We want to explore both lowering and non-lowering.
        # Partially-relevant loop becomes fused if we lower.
        prev_has_backing = any(s._backing for s in prev_tensor_holders)
        if prev_has_backing and partially_relevant_to_previous:
            lowering_choices.extend([[True, False]] * len(prev_tensor_holders))

        # Option 2: No backing in previous. Lower all. No cost to lowering. Conditioned
        # on option 1 being false.
        elif not prev_has_backing:
            lowering_choices.extend([[True]] * len(prev_tensor_holders))

        # Option 3: Fused, but all previous TensorHolder are for the first memory. Don't
        # lower. We don't need to reduce memory usage for DRAM.
        elif all(s.component == first_memory.name for s in prev_tensor_holders):
            lowering_choices.extend([[False]] * len(prev_tensor_holders))

        # Option 4: Previous TensorHolder is backing. Don't lower this; needs to be
        # alive for the other Einsum(s).
        elif prev_has_backing:
            lowering_choices.extend([[False]] * len(prev_tensor_holders))

        else:
            raise RuntimeError("BUG")

    # ==================================================================================
    # Iterate over all possible mappings
    # ==================================================================================
    for loop_orders in itertools.product(*choices):
        full_mapping = []
        for prev_tensor_holders, loop_order in zip(split_mapping, loop_orders):
            full_mapping.extend(prev_tensor_holders)
            full_mapping.extend(
                Temporal(rank_variable=r, tile_shape="symbol") for r in loop_order
            )
        tensor_holders = [node for node in full_mapping if isinstance(node, TensorHolder)]
        assert len(lowering_choices) == len(tensor_holders)
        for lowering_choice in itertools.product(*lowering_choices):
            for lower, node in zip(lowering_choice, tensor_holders):
                node._lower = lower

            yield list(full_mapping)


def insert_spatial_loops(
    mapping: list[MappingNode],
    einsum: Einsum,
    arch_flattened: list[architecture.Memory],
):
    nodes_with_fanout = [n for n in arch_flattened if n.spatial.get_fanout() > 1]
    arch_node_names = [n.name for n in arch_flattened]

    # Place spatials above the last instance of the first memory ABOVE each fanout
    for fanout in nodes_with_fanout:
        insertion_point = 0
        for i in range(len(mapping)):
            if not isinstance(mapping[i], TensorHolder):
                continue
            component = mapping[i].component
            if arch_node_names.index(component) < arch_node_names.index(fanout.name):
                insertion_point = i + 1

        rv = einsum.rank_variables
        for fanout_dim in fanout.spatial.fanout:
            for r in rv:
                mapping.insert(
                    insertion_point,
                    Spatial(
                        rank_variable=r,
                        dimension=fanout_dim,
                        across_object=fanout,
                        across=fanout.name,
                        tile_shape="symbol",
                    ),
                )
                
