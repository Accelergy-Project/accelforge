from accelforge.frontend.renames import TensorName


import copy
import itertools
from enum import Enum
from typing import Iterator

import accelforge.frontend.arch as arch
from accelforge.frontend.arch._flattened_arch import FlattenedArch
from accelforge.util._frozenset import oset
from accelforge.util._setexpressions import InvertibleSet
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
from accelforge.util.indent import print

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
    flattened_arch: FlattenedArch,
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

    # All other storage nodes may have loops between them, unless its a copy op in which
    # case we put back-to-back storage nodes for the same component together (so we
    # don't have both a tensor and a partition of the tensor living in the same memory
    # at one time).
    is_copy_op = einsum.is_copy_operation
    prev_component = None
    for m in mapping:
        if id(m) in inserted:
            continue
        if is_copy_op and prev_component == m.component:
            split_mapping[-1].append(m)
        else:
            split_mapping.append([m])
        prev_component = m.component

    for s in split_mapping:
        # Within each split mapping group, sort by arch levels.
        # This can help create places to put spatial loops
        s.sort(key=lambda tensor_holder: flattened_arch.index(tensor_holder.component))

    if sum(map(len, split_mapping)) != len(mapping):
        raise RuntimeError("BUG: number of storage nodes post-split != original")

    for i in range(len(split_mapping) - 1):
        for j in range(i + 1, len(split_mapping)):
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
            assert (
                len(split_mapping[j]) <= 1 or is_copy_op
            ), f"Mapping: {[m.compact_str() for m in mapping]}"
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

        # These are used to check if fanouts increased and therefore spatial loops
        # may be placed, so we use the maximum.
        assert len(cur_fanout) > 0
        assert len(next_fanout) > 0
        cur_fanout = max(cur_fanout)
        next_fanout = max(next_fanout)

        # Can't have loops above persistent tensor holders
        if next_persistent:
            rank_variables &= oset()

        # Can't have both a tensor and a copied partition of the tensor living in the
        # same memory at one time
        if is_copy_op and any(
            s.component == s2.component
            for s in prev_storages
            for group in split_mapping[i + 1 :]
            for s2 in group
        ):
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
    flattened_arch: FlattenedArch,
    intermediate_tensors: set[TensorName],
    explore_loop_orders: bool,
):
    """
    Yield a copy of `mapping` with spatial loops inserted. One copy is yielded per
    combination of the loop orders of each block of spatial loops.
    """
    tensor2relevant_rank_vars = einsum.tensor2rank_variables

    # Which rank variables go in each block, and where each block goes, are the same
    # for every loop order, so we work them out with one throwaway insertion.
    blocks = _insert_spatial_loops(
        list(mapping), einsum, flattened_arch, intermediate_tensors
    )
    choices = [
        list(
            spatial_loop_orders(
                rank_variables,
                tensor2relevant_rank_vars,
                fanout_dim.tensor_order_options,
                explore_loop_orders,
            )
        )
        for rank_variables, fanout_dim in blocks
    ]

    for block_orders in itertools.product(*choices):
        new_mapping = copy.deepcopy(mapping)
        _insert_spatial_loops(
            new_mapping, einsum, flattened_arch, intermediate_tensors, block_orders
        )
        yield new_mapping


def _insert_spatial_loops(
    mapping: list[MappingNode],
    einsum: Einsum,
    flattened_arch: FlattenedArch,
    intermediate_tensors: set[TensorName],
    block_orders: tuple[tuple[RankVariable, ...], ...] | None = None,
) -> list[tuple[set[RankVariable], arch.Spatial]]:
    """
    Insert one spatial loop per rank variable for each spatial dimension of each fanout
    in the arch, in place. The i-th block of spatial loops is ordered by
    `block_orders[i]`, or alphabetically if `block_orders` is None.

    Returns the rank variables and the fanout dimension of each block, in the order the
    blocks were inserted.
    """
    nodes_with_fanout = [n for n in flattened_arch if n.get_fanout() > 1]
    tensor2fully_relevant_rank_vars = einsum.tensor2directly_indexing_rank_variables
    blocks: list[tuple[set[RankVariable], arch.Spatial]] = []

    for node in nodes_with_fanout:
        # Insert spatials below the lowest storage node whose component is
        # above the fanout in the arch, and below any temporal loops in the
        # same block.
        insertion_point = _idx_below_lowest_tensor_holder_with_component_above_fanout(
            node, mapping, flattened_arch
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
            if block_orders is None:
                order = sorted(rank_variables)
            else:
                order = block_orders[len(blocks)]
            blocks.append((rank_variables, fanout_dim))

            mapping[insertion_point:insertion_point] = [
                Spatial(
                    rank_variable=r,
                    name=fanout_dim.name,
                    component_object=node,
                    component=node.name,
                )
                for r in order
            ]

    return blocks


def _tensors_seen_above_point(idx, mapping):
    seen_tensors = oset()
    for i in range(idx):
        node = mapping[i]
        if not isinstance(node, TensorHolder):
            continue
        seen_tensors |= oset(node.tensors)
    return seen_tensors


def _idx_below_lowest_tensor_holder_with_component_above_fanout(
    fanout_node, mapping, flattened_arch: FlattenedArch
):
    """Return the index right after the lowest TensorHolder whose component
    is above the fanout in the arch. If none found, returns len(mapping)."""
    fanout_arch_idx = flattened_arch.index(fanout_node.name)
    result = 0
    for i in range(len(mapping)):
        if not isinstance(mapping[i], TensorHolder):
            continue
        if flattened_arch.index(mapping[i].component) < fanout_arch_idx:
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


def permitted_tensor_orders(
    tensors: list[TensorName],
    tensor_order_options: list[list[InvertibleSet[TensorName]]],
) -> Iterator[tuple[TensorName, ...]]:
    """The orders of `tensors` that at least one of `tensor_order_options` permits.

    Each option is a list of sets of tensors, from highest in the mapping to lowest, and
    permits an order if the tensors it names appear in the order its sets do. Tensors
    that an option does not name may go anywhere, as may tensors of an option that this
    Einsum does not have -- their set expressions did not evaluate against this Einsum's
    symbol table, so they are left as strings and skipped here, the same way
    `get_tensor_order_constraint` skips tensors that are not in the mapping.

    An option that ends up naming fewer than two of `tensors` permits every order, and
    so leaves the whole constraint permitting every order.
    """
    all_orders = itertools.permutations(tensors)

    # The position that each option constrains each tensor to.
    option_positions: list[dict[TensorName, int]] = []
    for option in tensor_order_options:
        tensor2position = {}
        for position, together_tensors in enumerate(option):
            if not isinstance(together_tensors, InvertibleSet):
                continue
            for tensor in tensors:
                if tensor in together_tensors:
                    tensor2position[tensor] = position
        if len(tensor2position) < 2:
            return all_orders
        option_positions.append(tensor2position)

    if not option_positions:
        return all_orders

    def permits(tensor2position: dict[TensorName, int], order) -> bool:
        positions = [tensor2position[t] for t in order if t in tensor2position]
        return all(p0 <= p1 for p0, p1 in zip(positions, positions[1:]))

    return (
        order
        for order in all_orders
        if any(permits(positions, order) for positions in option_positions)
    )


def spatial_loop_orders(
    rank_variables: set[RankVariable],
    tensor2relevant_rank_vars: dict[TensorName, set[RankVariable]],
    tensor_order_options: list[list[InvertibleSet[TensorName]]],
    explore_loop_orders: bool,
):
    """
    Generate orders of a block of spatial loops that give unique reuse patterns.

    A rank variable relevant to every tensor is relevant to whichever tensor comes
    first, so it is always placed highest in the block. A rank variable irrelevant to
    every tensor tiles nothing, so it is always placed lowest.

    See :py:attr:`accelforge.frontend.arch.Spatial.tensor_order_options`.
    """
    seen = set()
    for tensor_order in permitted_tensor_orders(
        sorted(tensor2relevant_rank_vars), tensor_order_options
    ):
        remaining = oset(rank_variables)
        order = []
        for tensor in tensor_order:
            # These tile `tensor` and reuse the tensor above them, which is the one
            # whose group of rank variables we just placed.
            tiling_this_tensor = remaining & tensor2relevant_rank_vars[tensor]
            # Order within a group does not matter, so alphabetical is canonical.
            order.extend(sorted(tiling_this_tensor))
            remaining -= tiling_this_tensor
        order.extend(sorted(remaining))

        order = tuple(order)
        if order in seen:
            continue
        seen.add(order)
        yield order
        if not explore_loop_orders:
            return
