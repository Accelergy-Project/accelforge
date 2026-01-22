from fastfusion.frontend.renames import TensorName


import itertools
from enum import Enum

import fastfusion.frontend.arch as arch
from fastfusion.frontend.mapping import (
    MappingNode,
    ProcessingStage,
    Temporal,
    Spatial,
    TensorHolder,
)
from fastfusion.frontend.workload import (
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
):
    # First establish insertion points. Insertion points are:
    # - Below the last instance of the first memory
    # - Between any two TensorHolder nodes
    # - After the last TensorHolder node

    # The following logic is really just to make sure that all the storage nodse for the
    # outermost memory are together at the beginning of the split mapping. After that,
    # each entries in the split mapping has a single TensorHolder.
    split_mapping: list[list[TensorHolder]] = [[]]
    for m in mapping:
        split_mapping.append([m])
        if len(split_mapping) > 1 and m.component == first_memory.name:
            split_mapping[-2].extend(split_mapping.pop(-1))
    for i, s in enumerate[list[TensorHolder | Spatial]](split_mapping):
        for m in s:
            if i == 0 and m.component != first_memory.name:
                raise ValueError(
                    "The first TensorHolder in the mapping is not for the outermost "
                    "memory. This isn't known to be invalid, but the code may not "
                    "handle it."
                )
            elif i > 0 and m.component == first_memory.name:
                raise ValueError(
                    "First memory isn't at the top of the hierarchy. This isn't known"
                    "to be invalid, but the code may not handle it."
                )
            elif i == 0 and isinstance(m, Spatial):
                raise ValueError(
                    "Found Spatial node before any TensorHolder. This isn't known to "
                    "be invalid, but the code may not handle it."
                )

    split_mapping = [m for m in split_mapping if m]

    # These Einsum properties are recalculated since Einsum is mutable
    # We're pre-computing and reusing for efficiency
    tensor2fully_relevant_rank_vars = einsum.tensor2directly_indexing_rank_variables
    tensor2partially_relevant_rank_vars = (
        einsum.tensor2expression_indexing_rank_variables
    )
    tensor2irrelevant_rank_vars = einsum.tensor2irrelevant_rank_variables
    tensor2rank_vars = einsum.tensor2rank_variables

    fusable_tensors = (
        einsum.tensor_names & workload.tensor_names_used_in_multiple_einsums
    )
    is_fused_loops = True
    seen_tensors = set()
    choices = []
    lowering_choices: list[tuple[bool, ...]] = []
    fanouts = {}
    fanout = 1
    for node in flattened_arch:
        fanouts[node.name] = (fanout := fanout * node.get_fanout())

    def _get_next_storages(i: int, pstage_allowed: bool = False) -> list[TensorHolder]:
        for j in range(i + 1, len(split_mapping)):
            assert len(split_mapping[j]) <= 1
            # We don't add loops before processing stages
            if isinstance(split_mapping[j][0], ProcessingStage) and not pstage_allowed:
                continue
            return split_mapping[j]
        return []

    prev_fanout = 1
    someone_elses_spatials_may_be_placed_above = False
    for i, prev_storages in enumerate(split_mapping):
        # =============================================================================
        # Choose what temporal loops to insert between prev_storages and the next
        # TensorHolder node(s).
        # =============================================================================

        next_storages = _get_next_storages(i)
        next_anything = _get_next_storages(i, pstage_allowed=True)

        for s in prev_storages:
            # No tensor holders must mix backing/non-backing tensors.
            assert not s._backing or all(t in s._backing for t in s.tensors)
            # One tensor per holder
            assert len(s.tensors) == 1

        rank_variables = einsum.rank_variables
        # rank_variables = {r for r in rank_variables if rank_variable_bounds[r] > 1}
        seen_tensors |= set.union(*(set(t.tensors) for t in prev_storages), set())
        is_fused_loops = is_fused_loops and len(fusable_tensors - seen_tensors) > 0
        prev_tensors = set.union(set(), *(set(t.tensors) for t in prev_storages))
        next_persistent = set.union(
            set(), *(set(t.tensors) for t in next_storages if t.persistent)
        )

        max_fanout_before = max(
            [fanouts[s2.component] for s in split_mapping[:i] for s2 in s],
            default=float("inf"),
        )
        min_fanout_after = min(
            [fanouts[s2.component] for s in split_mapping[i + 1 :] for s2 in s],
            default=0,
        )
        cur_fanout = set(fanouts[s2.component] for s2 in prev_storages)
        next_fanout = set(fanouts[s2.component] for s2 in next_anything)
        if len(next_fanout) == 0:
            next_fanout.add(float("inf"))
        # Either it's main memory or we have one entry in the list, so there should only
        # be one
        assert len(cur_fanout) == 1
        assert len(next_fanout) == 1
        cur_fanout = next(iter(cur_fanout))
        next_fanout = next(iter(next_fanout))

        # Can't have loops above persistent tensor holders
        if next_persistent:
            rank_variables &= set()

        #  The fanout for a prior node may be placed here, so spatial nodes may be moved
        #  here
        someone_elses_spatials_may_be_placed_below = (
            next_fanout > cur_fanout and max_fanout_before > cur_fanout
        )

        # If the fanout is about to increase, then spatial loops may be placed below the
        # current node. There may have been constrained temporal loops earlier that need
        # to be placed here, so we won't prohibit any loops.
        if someone_elses_spatials_may_be_placed_below:
            pass

        # Loops below processing stages aren't helpful because there is no storage.
        # Ctrl-F for CONTIGUOUS_ITERATION_SPACE_DISCUSSION: Can't do this if we may put
        # another node's spatial loops below this one, because lowering would add move
        # the spatials down, which would constrain the temporals due to spatial-temporal
        # crossing.
        if (
            isinstance(prev_storages[0], ProcessingStage)
            and not someone_elses_spatials_may_be_placed_below
        ):
            rank_variables &= set()

        # Generally we want to only use rank variables that are irrelevant to the
        # previous tensors, else we'd just lower those tensors. However, we can't lower
        # backing TensorHolder nodes because this will add loops to compatibility.

        # No recomputation: If we haven't seen a tensor yet, must only iterate over
        # fully-relevant rank variables.
        for t in fusable_tensors - seen_tensors:
            rank_variables &= tensor2fully_relevant_rank_vars[t]

        # Optimality-preserving optimizations: We can trivially lower non-backing
        # TensorHolder nodes through fully-relevant loops. Can't do this if the loops
        # are fused because that'd add loops to the compatibility. Ctrl-F
        # forCONTIGUOUS_ITERATION_SPACE_DISCUSSION: Can't do this if we may put another
        # node's spatial loops below this one, because lowering would add move the
        # spatials down, which would constrain the temporals due to spatial-temporal
        # crossing.
        for s in prev_storages:
            for t in s.tensors:
                if (
                    t not in s._backing
                    and not s._must_be_here
                    and not someone_elses_spatials_may_be_placed_below
                ):
                    rank_variables -= tensor2fully_relevant_rank_vars[t]

        # Optimality-preserving optimization: We can trivially raise TensorHolder nodes
        # through irrelevant unfused loops. Can't do this if the loops are fused because
        # that'd increase the lifetime of the TensorHolder node. Can't do this if the
        # irrelevant rank variables partially-relevant to the previous tensors, since
        # that affects the permutation. See CONTIGUOUS_ITERATION_SPACE_DISCUSSION: Can't
        # do this if we may put another node's spatial loops above this one, because
        # raising would add move the temporals down, which would constrain them due to
        # spatial-temporal crossing. TODO: CONTIGUOUS_ITERATION_SPACE_DISCUSSION: This
        # causes all loops to be added, but really we only need to re-add the ones that
        # may conflict with a spatial loop.
        if not is_fused_loops:
            for s in next_storages:
                if (
                    not s._must_be_here
                    and not someone_elses_spatials_may_be_placed_above
                ):
                    for t in s.tensors:
                        rvs = tensor2irrelevant_rank_vars[t]
                        for t2 in prev_tensors:
                            rvs -= tensor2partially_relevant_rank_vars[t2]
                        rank_variables -= rvs

        partially_relevant_to_previous = set.union(
            set(), *(tensor2partially_relevant_rank_vars[t] for t in prev_tensors)
        )
        partially_relevant_to_previous &= rank_variables

        permutable_partially_relevant = set()

        # =============================================================================
        # Determine whether to lower TensorHolder nodes through partially-relevant loops.
        # =============================================================================
        for s in prev_storages:
            partially_relevant_to_previous = set.union(
                set(), *(tensor2partially_relevant_rank_vars[t] for t in s.tensors)
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
            elif isinstance(s, ProcessingStage):
                lowering_choices.append((False,))
            # Previous is backing and there's partially-relevant rank variables. May
            # want to lower to reduce memory footprint, or raise to reduce number of
            # fused loops.
            elif s._backing and lowerable_backing and partially_relevant_to_previous:
                lowering_choices.append((False, True))
                permutable_partially_relevant |= partially_relevant_to_previous
            # No backing in previous. No cost to lowering. Lower all
            elif not s._backing:
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
                    rank_variables, permutable_partially_relevant, can_lower
                )
            )
        )
        prev_fanout = cur_fanout
        someone_elses_spatials_may_be_placed_above = (
            someone_elses_spatials_may_be_placed_below
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
):
    nodes_with_fanout = [n for n in flattened_arch if n.get_fanout() > 1]
    arch_node_names = [n.name for n in flattened_arch]

    # Place spatials above the first instance of the first memory BELOW each fanout
    for node in nodes_with_fanout:
        insertion_point = 0
        for i in range(len(mapping)):
            if not isinstance(mapping[i], TensorHolder):
                continue
            if arch_node_names.index(mapping[i].component) >= arch_node_names.index(
                node.name
            ):
                insertion_point = i
                break
        else:
            insertion_point = len(mapping)

        rv = einsum.rank_variables
        for fanout_dim in node.spatial:
            for r in rv:
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


def canonical_loop_orders(
    rank_variables: set[RankVariable],
    partially_relevant_to_previous: set[RankVariable],
    can_lower: bool,
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
