from accelforge.frontend.renames import TensorName


import itertools
from enum import Enum

import accelforge.frontend.arch as arch
from accelforge.util._frozenset import oset
from accelforge.frontend.mapping import (
    MappingNode,
    Toll,
    Temporal,
    Spatial,
    TensorHolder,
    Loop,
)
from accelforge.frontend.workload import (
    Einsum,
    RankVariable,
    Workload,
)

# =================================================================================================
# Insert loops
# =================================================================================================


class LowerChoice(Enum):
    YES = 0
    NO = 1
    OPTIONAL = 2


def insert_temporal_loops(
    mapping: list[TensorHolder],
    einsum: Einsum,
    first_memory: arch.Memory,
    rank_variable_bounds: dict[RankVariable, int],
    ranks_with_tile_pattern: set,
    workload: Workload,
    _can_lower_outermost_memory: bool,
    flattened_arch: list[arch.Leaf],
    max_fused_loops: int,
    fanouts: dict[str, int],
    fusable_tensors: set[TensorName],
    intermediate_tensors: set[TensorName],
    let_non_intermediate_tensors_respawn_in_backing_storage: bool,
    explore_loop_orders: bool,
):
    # First establish insertion points. Insertion points are:
    # - Below the last instance of the first memory
    # - Between any two TensorHolder nodes
    # - After the last TensorHolder node

    # The following logic is really just to make sure that all the storage nodes for the
    # outermost memory are together at the beginning of the split mapping. After that,
    # each entries in the split mapping has a single TensorHolder.
    split_mapping: list[list[TensorHolder]] = []
    inserted = oset()  # for tracking which element of `mapping` is handled

    # The first group is for persistent tensors
    storage_nodes_above_any_loop = []
    for m in mapping:
        if id(m) in inserted:
            continue
        if m.persistent:
            storage_nodes_above_any_loop.append(m)
            inserted.add(id(m))

    # TODO: we have a misnomer: what we call `first_memory` in code is the one that
    # `_can_lower_outermost_memory` refers to.
    if not _can_lower_outermost_memory:
        for m in mapping:
            if id(m) in inserted:
                continue
            if m.component == first_memory.name:
                storage_nodes_above_any_loop.append(m)
                inserted.add(id(m))
    split_mapping.append(storage_nodes_above_any_loop)

    # All other storage nodes may have loops between them
    for m in mapping:
        if id(m) in inserted:
            continue
        split_mapping.append([m])

    if sum(map(len, split_mapping)) != len(mapping):
        raise RuntimeError("BUG: number of storage nodes post-split != original")

    for i in range(len(split_mapping)-1):
        for j in range(i+1, len(split_mapping)):
            for m_i in split_mapping[i]:
                for m_j in split_mapping[j]:
                    if mapping.index(m_i) > mapping.index(m_j):
                        raise RuntimeError(
                            f"BUG: node {m_i.compact_str()} and "
                            f"{m_j.compact_str()} inverted post-splitting\n"
                            f"Original mapping: {[m.compact_str() for m in mapping]}\n"
                            f"Split mapping: {[[m.compact_str() for m in s] for s in split_mapping]}\n"
                        )

    # These Einsum properties are recalculated since Einsum is mutable
    # We're pre-computing and reusing for efficiency
    tensor2fully_relevant_rank_vars = einsum.tensor2directly_indexing_rank_variables
    tensor2partially_relevant_rank_vars = (
        einsum.tensor2expression_indexing_rank_variables
    )
    for k, v in tensor2partially_relevant_rank_vars.items():
        tensor2partially_relevant_rank_vars[k] = v - tensor2fully_relevant_rank_vars[k]
    tensor2irrelevant_rank_vars = einsum.tensor2irrelevant_rank_variables
    tensors = einsum.tensor_names

    is_fused_loops = True
    seen_tensors = oset()
    choices = []
    lowering_choices: list[tuple[bool, ...]] = []

    def _get_next_storages(i: int, toll_allowed: bool = False) -> list[TensorHolder]:
        for j in range(i + 1, len(split_mapping)):
            assert len(split_mapping[j]) <= 1, f"Mapping: {[m.compact_str() for m in mapping]}"
            # We don't add loops before tolls since they don't reuse things
            if isinstance(split_mapping[j][0], Toll) and not toll_allowed:
                continue
            return split_mapping[j]
        return []

    for i, prev_storages in enumerate(split_mapping):
        # =============================================================================
        # Choose what temporal loops to insert between prev_storages and the next
        # TensorHolder node(s).
        # =============================================================================

        next_storages = _get_next_storages(i)
        next_anything = _get_next_storages(i, toll_allowed=True)

        for s in prev_storages:
            # No tensor holders must mix backing/non-backing tensors.
            assert not s._backing or all(t in s._backing for t in s.tensors)
            # One tensor per holder
            assert len(s.tensors) == 1

        rank_variables = einsum.rank_variables
        # rank_variables = {r for r in rank_variables if rank_variable_bounds[r] > 1}
        seen_tensors |= oset.union(*(oset(t.tensors) for t in prev_storages), oset())
        is_fused_loops = is_fused_loops and len(fusable_tensors - seen_tensors) > 0
        prev_tensors = oset.union(oset(), *(oset(t.tensors) for t in prev_storages))
        next_persistent = oset.union(
            oset(), *(oset(t.tensors) for t in next_storages if t.persistent)
        )

        max_fanout_before = max(
            [fanouts[s2.component] for s in split_mapping[:i] for s2 in s],
            default=float("inf"),
        )
        cur_fanout = oset(fanouts[s2.component] for s2 in prev_storages)
        next_fanout = oset(fanouts[s2.component] for s2 in next_anything)
        if len(cur_fanout) == 0:  # Happens if we're inserting above all storage nodes
            cur_fanout.add(1)
        if len(next_fanout) == 0:  # Happens if we're inserting below all storage nodes
            next_fanout.add(float("inf"))
        # Either it's main memory or we have one entry in the list, so there should only
        # be one
        assert len(cur_fanout) == 1
        assert len(next_fanout) == 1
        cur_fanout = next(iter(cur_fanout))
        next_fanout = next(iter(next_fanout))

        # Can't have loops above persistent tensor holders
        if next_persistent:
            rank_variables &= oset()

        # No recomputation: If we haven't seen a tensor yet, must only iterate over
        # fully-relevant rank variables.
        check_tensors = tensors
        if let_non_intermediate_tensors_respawn_in_backing_storage:
            check_tensors = intermediate_tensors

        for t in check_tensors - seen_tensors:
            rank_variables &= tensor2fully_relevant_rank_vars[t]

        if max_fused_loops == 0 and (fusable_tensors - seen_tensors):
            rank_variables &= oset()

        # The fanout for a prior node may be placed here, so spatial nodes may be moved
        # here
        someone_elses_spatials_may_be_placed_below = (
            next_fanout < cur_fanout and max_fanout_before < cur_fanout
        )

        # If the fanout is about to increase, then spatial loops may be placed below the
        # current node. There may have been constrained temporal loops earlier that need
        # to be placed here, so we won't prohibit any loops. TODO:
        # CONTIGUOUS_ITERATION_SPACE_DISCUSSION: This causes all loops to be added, but
        # really we only need to re-add the ones that may conflict with a spatial loop.
        if someone_elses_spatials_may_be_placed_below:
            pass
        else:
            # Optimality-preserving optimization: Loops below tolls aren't helpful
            # because there is no storage. Ctrl-F for
            # CONTIGUOUS_ITERATION_SPACE_DISCUSSION: Can't do this if we may put another
            # node's spatial loops below this one, because lowering would add move the
            # spatials down, which would constrain the temporals due to spatial-temporal
            # crossing.
            if prev_storages and isinstance(prev_storages[0], Toll):
                rank_variables &= oset()

            # Optimality-preserving optimization: We can trivially lower non-backing
            # TensorHolder nodes through fully-relevant loops. Can't do this if the
            # loops are fused because that'd add loops to the compatibility.
            # CONTIGUOUS_ITERATION_SPACE_DISCUSSION: Can't do this if we may put another
            # node's spatial loops below this one, because lowering would add move the
            # spatials down, which would constrain the temporals due to spatial-temporal
            # crossing.
            for s in prev_storages:
                for t in s.tensors:
                    if t not in s._backing and not s._must_be_here:
                        rank_variables -= tensor2fully_relevant_rank_vars[t]

            # Optimality-preserving optimization: We can trivially raise TensorHolder
            # nodes through irrelevant unfused loops. Can't do this if the loops are
            # fused because that'd increase the lifetime of the TensorHolder node. Can't
            # do this if the irrelevant rank variables are partially-relevant to the
            # previous tensors, since that affects the permutation. See
            # CONTIGUOUS_ITERATION_SPACE_DISCUSSION: Can't do this if we may put another
            # node's spatial loops above this one, because raising would add move the
            # temporals down, which would constrain them due to spatial-temporal
            # crossing.
            if not is_fused_loops:
                for s in next_storages:
                    if not s._must_be_here:
                        for t in s.tensors:
                            rvs = tensor2irrelevant_rank_vars[t]
                            for t2 in prev_tensors:
                                rvs -= tensor2partially_relevant_rank_vars[t2]
                            rank_variables -= rvs

        # =============================================================================
        # Determine whether to lower TensorHolder nodes through partially-relevant
        # loops.
        # =============================================================================
        partially_relevant_to_previous = rank_variables & oset.union(
            oset(), *(tensor2partially_relevant_rank_vars[t] for t in prev_tensors)
        )
        permutable_partially_relevant = oset()

        # NOTE: If the lowering logic for backing TensorHolders is updated & we can
        # lower through >1 loops, then also update label_fused_loops
        for s in prev_storages:
            partially_relevant_to_previous = oset.union(
                oset(), *(tensor2partially_relevant_rank_vars[t] for t in s.tensors)
            )
            partially_relevant_to_previous &= rank_variables
            lowerable_backing = (
                _can_lower_outermost_memory or s.component != first_memory.name
            )

            # Persistent. Must be at the top of the mapping.
            if s.persistent:
                lowering_choices.append((False,))
            # Don't lower our own reservations through someone else's spatial loops.
            elif someone_elses_spatials_may_be_placed_below:
                lowering_choices.append((False,))
            # Processing stage. Lowering doesn't matter. Don't lower.
            elif isinstance(s, Toll):
                lowering_choices.append((False,))
            # Previous is backing and there's partially-relevant rank variables. May
            # want to lower to reduce memory footprint, or raise to reduce number of
            # fused loops.
            elif s._backing and lowerable_backing and partially_relevant_to_previous:
                lowering_choices.append((False, True))
                permutable_partially_relevant |= partially_relevant_to_previous
            # No backing in previous. No cost to lowering. Lower all
            elif not s._backing:
                # print(f'Can lower {s.tensors} in {s.component}')
                lowering_choices.append((True,))
                permutable_partially_relevant |= partially_relevant_to_previous
            # Previous TensorHolder is backing but not lowerable or there are no
            # partially relevant rank vars.
            else:
                lowering_choices.append((False,))

        # =============================================================================
        # Create loop order and lowering choices
        # =============================================================================

        can_lower = any(any(c) for c in lowering_choices)

        # Create canonical loop orders that avoids repeating reuse patterns.
        choices.append(
            list(
                canonical_loop_orders(
                    rank_variables,
                    permutable_partially_relevant,
                    can_lower,
                    explore_loop_orders,
                )
            )
        )

    # ==================================================================================
    # Iterate over all possible mappings
    # ==================================================================================

    # TODO: Optimization: If we can optionally lower a tensor & the loop below it is
    # not something through which we can lower for a given permutation, skip options
    # that lower that tensor because they get the same result as not lowering the
    # tensor.
    n_loop_orders = len(list(itertools.product(*choices)))
    for loop_orders in itertools.product(*choices):
        full_mapping = []
        for prev_storages, loop_order in zip(split_mapping, loop_orders):
            full_mapping.extend(prev_storages)
            full_mapping.extend(Temporal(rank_variable=r) for r in loop_order)

        storages = [node for node in full_mapping if isinstance(node, TensorHolder)]
        assert len(lowering_choices) == len(storages)
        for lowering_choice in itertools.product(*lowering_choices):
            for lower, node in zip(lowering_choice, storages):
                node._lower = lower

            yield list(full_mapping), n_loop_orders


def insert_spatial_loops(
    mapping: list[MappingNode],
    einsum: Einsum,
    flattened_arch: list[arch.Memory],
    intermediate_tensors: set[TensorName],
):
    nodes_with_fanout = [n for n in flattened_arch if n.get_fanout() > 1]
    arch_node_names = [n.name for n in flattened_arch]
    tensor2fully_relevant_rank_vars = einsum.tensor2directly_indexing_rank_variables
    simple_rank_variables = einsum._simple_rank_variables

    for node in nodes_with_fanout:
        # Insert spatials below the lowest storage node whose component is
        # above the fanout in the arch, and below any temporal loops in the
        # same block.
        insertion_point = _idx_below_lowest_tensor_holder_with_component_above_fanout(
            node, mapping, arch_node_names
        )
        while insertion_point < len(mapping) and isinstance(
            mapping[insertion_point], Temporal
        ):
            insertion_point += 1

        # No recomputation: If we haven't seen a tensor yet, must only iterate over
        # fully-relevant rank variables.
        rank_variables = einsum.rank_variables
        for t in intermediate_tensors - _tensors_seen_above_point(
            insertion_point, mapping
        ):
            rank_variables &= tensor2fully_relevant_rank_vars[t]

        for fanout_dim in node.spatial:
            for r in sorted(rank_variables):
                s = Spatial(
                    rank_variable=r,
                    name=fanout_dim.name,
                    component_object=node,
                    component=node.name,
                )
                if insertion_point == len(mapping):
                    mapping.append(s)
                else:
                    mapping.insert(insertion_point, s)


def _tensors_seen_above_point(idx, mapping):
    seen_tensors = oset()
    for i in range(idx):
        node = mapping[i]
        if not isinstance(node, TensorHolder):
            continue
        seen_tensors |= oset(node.tensors)
    return seen_tensors


def _idx_below_lowest_tensor_holder_with_component_above_fanout(
    fanout_node, mapping, arch_node_names
):
    """Return the index right after the lowest TensorHolder whose component
    is above the fanout in the arch. If none found, returns len(mapping)."""
    fanout_arch_idx = arch_node_names.index(fanout_node.name)
    result = 0
    for i in range(len(mapping)):
        if not isinstance(mapping[i], TensorHolder):
            continue
        if arch_node_names.index(mapping[i].component) < fanout_arch_idx:
            result = i + 1
    return result


def label_imperfect_tile_shapes(mapping, einsum: Einsum, job):
    allow_all = job.allow_imperfect_all
    imperfect_spatial = allow_all or job.explore_imperfect_spatial_loops
    imperfect_temporal = allow_all or job.explore_imperfect_temporal_loops
    simple_ranks = einsum._simple_rank_variables

    rv2nodes = {}
    for node in mapping:
        if isinstance(node, Loop) and node.rank_variable in simple_ranks:
            rv2nodes.setdefault(node.rank_variable, []).append(node)

    for _, nodes in rv2nodes.items():
        for i, node in enumerate(nodes):
            # (A) We can get full expressiveness by unlocking options for the PREVIOUS
            # loop, (B) which is also much faster to search. However, if we're the
            # outermost loop for this rank variable, then unlock options for this loop.

            # (A) reasoning: Loop bound = ceil(outer size / this size). Imperfect can
            # relax options, so we choose outer size.
            #
            # (B) reasoning: For spatial loops, we know the fanout once we pick outer
            # size AND inner size. We enumerate inner to outer, so this effectively
            # defers imperfect factorization until we know the exact fanout, and can
            # immediately use it to prune too-large fanouts.
            target = max(i - 1, 0)
            if isinstance(node, Spatial) and imperfect_spatial:
                nodes[target]._may_cause_imperfect = True
            if isinstance(node, Temporal) and imperfect_temporal:
                nodes[target]._may_cause_imperfect = True


def canonical_loop_orders(
    rank_variables: set[RankVariable],
    partially_relevant_to_previous: set[RankVariable],
    can_lower: bool,
    explore_loop_orders: bool,
):
    """Generate loop orders that result in unique reuse patterns."""
    # Only the first partially-relevant rank variable matters is a meaningful
    # choice because lowering only happens through at most one rank var.
    if not partially_relevant_to_previous or not can_lower:
        yield tuple(sorted(rank_variables))
        return

    for first_rank_var in partially_relevant_to_previous:
        rest_of_partially_relevant = partially_relevant_to_previous - {first_rank_var}
        rest_rank_vars = rank_variables - partially_relevant_to_previous
        # Since order does not matter, we choose alphabetical order as canonical.
        yield (
            (first_rank_var,)
            + tuple(sorted(rest_of_partially_relevant))
            + tuple(sorted(rest_rank_vars))
        )
        if not explore_loop_orders:
            return
