from accelforge.frontend.mapping.mapping import MappingNode


import copy
from collections import Counter, defaultdict
import itertools
import logging
from typing import Any, Iterator, List
import uuid

from tqdm import tqdm

import accelforge.frontend.arch as arch
from accelforge.frontend.arch._flattened_arch import FlattenedArch
from accelforge.util._frozenset import fzs, oset
from accelforge.frontend.mapping import (
    Compute,
    Loop,
    Mapping,
    MappingNode,
    Spatial,
    Storage,
    TensorHolder,
    Temporal,
    BindingBlock,
    iter_binding_blocks,
)
from accelforge.frontend.arch.binding_constraints import BindingConstraints
from accelforge.frontend.renames import Rank, RANK_DUPLICATE
from accelforge.util._basetypes import EvalableList
from accelforge.frontend.spec import Spec
from accelforge.frontend._workload_isl._isl import get_rank_variable_bounds
from accelforge.frontend._workload_isl._symbolic import (
    Relevant,
    get_rank_variable_relevancy,
    get_stride_and_halo,
    get_stride_and_halo_of_einsum,
    PartiallyRelevant,
)
from accelforge.frontend.workload import (
    TensorName,
    Einsum,
    EinsumName,
    RankVariable,
    Workload,
    SymbolTable,
)
from accelforge.mapper.FFM._make_pmappings.make_pmapping_templates.make_storage_order import (
    get_tensor_choices,
)
from accelforge.mapper.FFM._make_pmappings.make_pmapping_templates.make_reservations import (
    get_reservation_choices,
)
from accelforge.mapper.FFM._make_pmappings.contraints.constraints import (
    MappingConstraints,
    get_constraints,
)
from accelforge.mapper.FFM._make_pmappings.make_pmapping_templates.make_loops import (
    insert_temporal_loops,
    insert_spatial_loops,
    label_imperfect_tile_shapes,
)
from accelforge.mapper.FFM._make_pmappings.pmapper_job import (
    Job,
    SameEinsumJobs,
)
from accelforge.mapper.FFM._join_pmappings.compatibility import Compatibility
from accelforge.model._looptree.reuse.symbolic import (
    label_fused_loops,
    quick_insert_reservation_nodes,
)


def unpack_loops_to_rank_variables(mapping: List[MappingNode]):
    mapping_new = []
    for node in mapping:
        if not isinstance(node, Loop) or not isinstance(node.rank_variable, set):
            mapping_new.append(node)
            continue

        for r in sorted(node.rank_variable):
            mapping_new.append(
                type(node)(
                    rank_variable=r,
                    **node.model_dump(exclude={"rank_variable"}, recursive=False),
                )
            )
    return mapping_new


# =================================================================================================
# Iterate over mappings
# =================================================================================================
def place_missing_temporal_loops(
    mapping: List[MappingNode], einsum: Einsum, flattened_arch: FlattenedArch
):
    """
    Adds temporal loops to the mapping to fill in any rank variables that are missing.
    This may occur if there are no points where it'd be helpful to add a non-fused loop,
    so we just need to add one somewhere.
    """
    # If any rank variables are missing, add them as high as possible.

    rank_variables = einsum.rank_variables
    for m in mapping:
        if isinstance(m, Temporal) and not m._fused:
            rank_variables.discard(m.rank_variable)

    # Insert point: Right under the last backing & below any out-of-order fanouts
    fanouts = {}
    fanout = 1
    for node in flattened_arch:
        fanouts[node.name] = (fanout := fanout * node.get_fanout())

    insert_point = 0
    greatest_previous_fanout = 1
    for i in range(len(mapping)):
        if isinstance(mapping[i], TensorHolder):
            if mapping[i]._backing:
                insert_point = i + 1
            cur_fanout = fanouts[mapping[i].component]
            if cur_fanout < greatest_previous_fanout:
                insert_point = i + 1
            greatest_previous_fanout = max(greatest_previous_fanout, cur_fanout)

        # Put it below all the other temporals here in case we're lowering through them
        if isinstance(mapping[i], Temporal) and insert_point == i:
            insert_point = i + 1

    temporals = [Temporal(rank_variable=r) for r in sorted(rank_variables)]

    if insert_point == len(mapping):
        mapping.extend(temporals)
    else:
        for t in temporals:
            mapping.insert(insert_point, t)


def remove_unordered_spatial_temporal_loops(
    mapping: list[MappingNode],
    flattened_arch: FlattenedArch,
    einsum: Einsum,
    explore_unordered_spatial_loops: bool = True,
):
    fanout = 1
    fanouts = {}
    for node in flattened_arch:
        fanouts[node.name] = (fanout := fanout * node.get_fanout())

    # TensorHolders' tiles include the full iteration space of temporal loops below
    # them. If all of the following are true:
    # - Any later spatials have <= the fanout of this TensorHolder
    # - There's a temporal loop between the TensorHolder and the Spatial
    # - The temporal/spatial have the same rank variable
    # - The rank variable indexes into the tensor
    #
    # Then the temporal/spatial pair will result in a non-contiguous tile of the
    # iteration space, which is not supported and must be removed.

    tensor2rvs = einsum.tensor2rank_variables
    disallowed_combinations: list[tuple[set[int], set[int]]] = []
    for i, node in enumerate(mapping):
        # Track TensorHolders that have been seen and the rank variables that affect
        # them
        if not isinstance(node, TensorHolder):
            continue

        relevent_rvs = oset.union(*[tensor2rvs[t] for t in node.tensors])

        # Find the last spatial whose component's fanout is <= the TensorHolder's
        # component fanout. This spatial will affect the TensorHolder's tile.
        check_up_to = i
        for j, node2 in enumerate(mapping[i + 1 :]):
            if isinstance(node2, Spatial):
                if fanouts[node.component] >= fanouts[node2.component]:
                    check_up_to = i + j + 1

        # Find all temporal and spatial loops between the TensorHolder and the last
        # spatial that affects it.
        rv2spatial = {}
        rv2temporal = {}
        for node2 in mapping[i + 1 : check_up_to]:
            if isinstance(node2, Spatial) and node2.rank_variable in relevent_rvs:
                rv2spatial.setdefault(node2.rank_variable, []).append(node2)
            if isinstance(node2, Temporal):
                rv2temporal.setdefault(node2.rank_variable, []).append(node2)

        for shared_rv in sorted(oset(rv2spatial) & oset(rv2temporal) & relevent_rvs):
            disallowed_combinations.append(
                (
                    tuple(id(x) for x in rv2spatial[shared_rv]),
                    tuple(id(x) for x in rv2temporal[shared_rv]),
                )
            )

    if not explore_unordered_spatial_loops:
        disallowed_combinations = [x[1:] for x in disallowed_combinations]

    for combo in itertools.product(*disallowed_combinations):
        combo = oset.union(oset(), *combo)
        yield [n for n in mapping if id(n) not in combo]


def pad_with_bottom_loops(mapping: list[MappingNode], einsum: Einsum):
    rank_variables = einsum.rank_variables
    rank_var_to_count = defaultdict(lambda: 0)
    for node in mapping:
        if isinstance(node, Temporal):
            rank_var_to_count[node.rank_variable] += 1

    for rank_var in rank_variables:
        if rank_var_to_count[rank_var] < 2:
            mapping.append(Temporal(rank_variable=rank_var))


def _timeloop_style_even(mapping: list[MappingNode]):
    # Iterate through the mapping. If there are >2 TensorHolder nodes for the same
    # memory, move all below the 2nd to the same level as the 2nd.
    mapping = copy.deepcopy(mapping)
    memory2indices = defaultdict(list)
    i = 0
    while i < len(mapping):
        node = mapping[i]
        if not isinstance(mapping[i], TensorHolder):
            i += 1
            continue
        node: TensorHolder
        seen = memory2indices[node.component]
        mapping[i]._lower = False  # Lowering might re-uneven the reservationsxs

        if len(seen) <= 1:
            seen.append(i)
        else:
            mapping.insert(seen[-1] + 1, mapping.pop(i))
        i += 1
    return mapping


def assert_proper_fusion_labeling(
    mapping: list[MappingNode],
    fusable_tensors: set[TensorName],
    check_loops: bool = True,
):
    tensors = oset()
    for i, t in enumerate(mapping):
        if not isinstance(t, TensorHolder):
            continue

        new = (oset(t.tensors) - tensors) & fusable_tensors

        if new and check_loops:
            for j in range(i):
                if isinstance(mapping[j], Loop):
                    assert mapping[
                        j
                    ]._fused, f"Node {j} is not fused in {' '.join(m.compact_str() for m in mapping)}"
        assert (
            t._backing & fusable_tensors
        ) == new, f"Node {i} backing missing {new - t._backing} in {' '.join(m.compact_str() for m in mapping)}"
        tensors.update(new)
        tensors.update(t.tensors)


def iterate_mappings_no_constraints(
    spec: Spec,
    einsum_name: str,
    flattened_arch: FlattenedArch,
    rank_variable_bounds: dict[RankVariable, int],
    job: Job,
) -> Iterator[tuple[Mapping, SymbolTable, arch.Compute, int]]:
    first_memory = None
    for node in flattened_arch:
        if isinstance(node, arch.Memory):
            first_memory = node
            break
    if first_memory is None:
        raise ValueError("No memory found in architecture")

    einsum = spec.workload.einsums[einsum_name]
    symbol_table = {r.name: r.source for r in einsum.renames}
    fusable_tensors = job.fusable_tensors

    ranks_with_tile_pattern = oset(
        r for r, c in job.initial_delta_choices.items() if len(c) > 1
    )
    job.ranks_with_tile_pattern = ranks_with_tile_pattern

    fanouts = {}
    fanout = 1
    for node in flattened_arch:
        fanouts[node.name] = (fanout := fanout * node.get_fanout())

    for mapping, symbol_table, compute in get_tensor_choices(
        einsum_name,
        flattened_arch,
        symbol_table,
        spec,
        first_memory,
        fusable_tensors,
        fanouts,
        spec.mapper.prioritize_reuse_of_unfused_tensors,
    ):
        logging.info(
            "\tGenerated tensor choices: " + ", ".join(m.compact_str() for m in mapping)
        )
        mapping = copy.deepcopy(mapping)
        for mapping, n_orders in insert_temporal_loops(
            mapping,
            einsum,
            first_memory,
            rank_variable_bounds,
            ranks_with_tile_pattern,
            spec.workload,
            spec.mapper._can_lower_outermost_memory,
            flattened_arch,
            spec.mapper.max_fused_loops,
            fanouts,
            fusable_tensors,
            job.intermediate_tensors,
            spec.mapper._let_non_intermediate_tensors_respawn_in_backing_storage,
            spec.mapper.explore_loop_orders,
        ):
            # insert_spatial_loops copies the mapping for each spatial loop order it
            # yields, so there's no need to copy it here.
            for spatial_mapping in insert_spatial_loops(
                mapping,
                einsum,
                flattened_arch,
                job.intermediate_tensors,
                spec.mapper.explore_loop_orders,
            ):
                spatial_mapping = unpack_loops_to_rank_variables(spatial_mapping)
                if spec.mapper._timeloop_style_even:
                    spatial_mapping = _timeloop_style_even(spatial_mapping)

                place_missing_temporal_loops(spatial_mapping, einsum, flattened_arch)
                label_fused_loops(spatial_mapping, fusable_tensors)
                assert_proper_fusion_labeling(spatial_mapping, fusable_tensors)
                yield spatial_mapping, symbol_table, compute, n_orders


def _template_dedup_key(mapping: Mapping) -> tuple:
    key = []
    storage_group = []
    for node in mapping.nodes:
        if isinstance(node, TensorHolder):
            storage_group.append(node.compact_str())
        else:
            if storage_group:
                key.append(fzs(storage_group))
                storage_group = []
            key.append(node.compact_str())
    if storage_group:
        key.append(fzs(storage_group))
    return tuple(key)


def iterate_mappings_constraints(
    spec: Spec,
    einsum_names: list[str] | str,
    flattened_arch: FlattenedArch,
    rank_variable_bounds: dict[RankVariable, int],
    tensor_to_relevancy: dict[
        TensorName, dict[RankVariable, Relevant | PartiallyRelevant]
    ],
    job: Job,
) -> Iterator[tuple[Mapping, MappingConstraints, dict[str, str]]]:
    compute_name = flattened_arch[-1].name

    n_yielded = 0
    seen_templates = set()

    if isinstance(einsum_names, str):
        einsum_names = [einsum_names]

    for einsum_name in einsum_names:
        logging.info(
            f"Generating pmapping templates for compute {compute_name} Einsums "
            f"{einsum_name}"
        )

        for mapping, symbol_table, compute, n_orders in iterate_mappings_no_constraints(
            spec,
            einsum_name,
            flattened_arch,
            rank_variable_bounds,
            job,
        ):
            mapping, constraints = get_constraints(
                flattened_arch,
                mapping,
                symbol_table,
                einsum_name,
                tensor_to_relevancy,
                is_copy_operation=spec.workload.einsums[einsum_name].is_copy_operation,
            )

            # This goes after the constraints because constraints may remove some loops,
            # giving us fewer that may be reordered.
            for mapping in remove_unordered_spatial_temporal_loops(
                mapping,
                flattened_arch,
                spec.workload.einsums[einsum_name],
                spec.mapper.out_of_order_hierarchy_explore_removing_spatials_for_more_temporals,
            ):
                constraints.remove_missing_targets(mapping)

                mapping.append(
                    Compute(
                        einsum=einsum_name,
                        component=compute_name,
                        component_object=compute,
                    )
                )

                # MAPPING MUST NOT BE MODIFIED AFTER constraints.set_loop_indices
                constraints.set_loop_indices(mapping)
                label_imperfect_tile_shapes(
                    mapping, spec.workload.einsums[einsum_name], job
                )

                mapping = Mapping(nodes=[copy.copy(n) for n in mapping])
                mapping._n_loop_orders = n_orders

                key = _template_dedup_key(mapping)
                if key in seen_templates:
                    continue
                seen_templates.add(key)

                yield mapping, constraints, symbol_table
                n_yielded += 1
                if n_yielded >= spec.mapper.max_pmapping_templates_per_einsum:
                    if spec.mapper._only_output_pmapping_with_index is None:
                        return
                    if (
                        isinstance(spec.mapper._only_output_pmapping_with_index, dict)
                        and einsum_name
                        not in spec.mapper._only_output_pmapping_with_index
                    ):
                        return


# =================================================================================================
# Bindings
# =================================================================================================
def binding_constraints_for(job: Job, component: str) -> BindingConstraints:
    """
    The binding constraints of the memory named `component`. They come from the
    flattened arch, so their tensor set expressions are already evaluated against this
    job's Einsum.
    """
    return job.flattened_arch[component].binding


def _tensor_ranks(tensor: TensorName, job: Job) -> list[Rank]:
    return list(job.einsum.tensor_accesses[tensor].ranks)


def _is_unshared(tensor: TensorName, job: Job) -> bool:
    return tensor not in (job.fusable_tensors or set())


def _allowed_binding_ranks(block: BindingBlock, loop: Spatial, job: Job) -> list[Rank]:
    """
    The ranks that `block`'s binding loop over `loop`'s spatial dimension may name:
    `BindingConstraints.allowed_ranks`, minus RANK_DUPLICATE when `block`'s tensor is
    unshared and this memory's `must_shard_unshared_tensors` names it.
    """
    constraints = binding_constraints_for(job, block.component)
    allowed = constraints.allowed_ranks(
        block.tensor, loop.name, _tensor_ranks(block.tensor, job)
    )
    if _is_unshared(block.tensor, job) and constraints.must_shard_unshared(
        block.tensor
    ):
        allowed = [r for r in allowed if r != RANK_DUPLICATE]
    return allowed


def default_binding_ranks(block: BindingBlock, job: Job) -> list[Rank] | None:
    """
    The locally-optimal binding of `block`'s tensor to its spatial loops: partition the
    tensor along the rank that each spatial loop's rank variable projects into, and
    replicate it (RANK_DUPLICATE) if the loop's rank variable does not index the tensor
    -- unless `must_shard_unshared_tensors` has taken replicating away from it, in which
    case it partitions along a rank there too (see `_allowed_binding_ranks`).

    The binding constraints in force for `block` may forbid that rank for some loop.
    Such a loop instead takes the first rank the constraints do allow it, preferring one
    that no other loop of this binding already partitions -- which spreads the tensor
    over the spatial dimensions rather than piling it onto one rank. The result is a
    valid, deterministic binding rather than a provably optimal one, and it is the
    binding that `max_binding_rank_changes` counts changes against.

    Returns None if no binding of `block` satisfies the constraints.
    """
    tensor = block.tensor
    spatials = block.spatials
    irrelevant_rvs = job.einsum.tensor2irrelevant_rank_variables[tensor]
    rank_variable2ranks = job.einsum.tensor_accesses[tensor].rank_variable2ranks

    ranks: list[Rank] = []
    for loop in spatials:
        allowed = _allowed_binding_ranks(block, loop, job)
        rv = loop.rank_variable
        if rv in irrelevant_rvs:
            preferred = RANK_DUPLICATE
        else:
            rv_ranks = rank_variable2ranks[rv]
            assert len(rv_ranks) == 1
            preferred = next(iter(rv_ranks))

        order = (
            [preferred]
            + [r for r in allowed if r != preferred and r not in ranks]
            + [r for r in allowed if r != preferred and r in ranks]
        )
        for rank in order:
            if rank in allowed:
                ranks.append(rank)
                break
        else:
            return None
    return ranks


def _binding_rank_classes(
    allowed: list[Rank], default: Rank
) -> list[tuple[Rank, ...]]:
    """
    `allowed` grouped into the classes of ranks that a binding loop is modeled
    identically for, the class of `default` first and each class's representative first
    within it.

    A loop that partitions its tensor along a rank other than the locally-optimal one
    hands every instance the same 1/fanout share of the tile and the same scattered
    delivery to serve it, whichever of those ranks it picks, so the mapper models one of
    them and lets it stand for the rest. RANK_DUPLICATE is not one of them: replicating
    puts a whole copy in every instance, which costs capacity that a partition does not
    and saves the delivery that a partition pays for.
    """
    classes = [(default,)]
    if default != RANK_DUPLICATE and RANK_DUPLICATE in allowed:
        classes.append((RANK_DUPLICATE,))
    partitions = [r for r in allowed if r != default and r != RANK_DUPLICATE]
    if partitions:
        classes.append(tuple(partitions))
    return classes


def binding_rank_options(
    block: BindingBlock, job: Job
) -> list[list[tuple[Rank, ...]]] | None:
    """
    The classes of ranks that each binding loop of `block`'s tensor may take. A
    binding loop may partition the tensor along any of the tensor's ranks, or replicate
    it (RANK_DUPLICATE), except as the binding constraints in force for `block`
    restrict it; `_binding_rank_classes` collects the ones that model alike, so a loop
    offers one option per class rather than one per rank. The default choice (see
    `default_binding_ranks`) is the first class of each loop, so the first combination
    of the options is the default binding.

    Returns None if no binding of `block` satisfies the constraints.
    """
    default = default_binding_ranks(block, job)
    if default is None:
        return None
    return [
        _binding_rank_classes(_allowed_binding_ranks(block, loop, job), d)
        for loop, d in zip(block.spatials, default)
    ]


def binding_representatives(
    choice: list[list[tuple[Rank, ...]]]
) -> list[list[Rank]]:
    """The rank each binding loop of `choice` actually binds: its class's first."""
    return [[ranks[0] for ranks in block] for block in choice]


def binding_rank_aliases(
    mapping: Mapping, choice: list[list[tuple[Rank, ...]]]
) -> dict[tuple[TensorName, str, int], tuple[Rank, ...]]:
    """
    Per binding loop, the ranks that the representative `mapping` was bound to stands
    for -- the ranks this template's compatibilities are expanded over once it has been
    modeled. Loops whose class holds one rank are left out: they bind what they bind and
    there is nothing to expand.

    A loop is named by its tensor, the memory holding it, and its position in that
    storage's binding. Position, because a memory with several spatial loops over one
    dimension binds a rank for each of them and the dimension name alone would name them
    all at once; and `Compatibility.from_mapping` walks the same binding in the same
    order, so the position carries over.
    """
    aliases: dict[tuple[TensorName, str, int], tuple[Rank, ...]] = {}
    seen: set[tuple[TensorName, str]] = set()
    for block, classes in zip(iter_binding_blocks(mapping.nodes), choice):
        for i, (loop, ranks) in enumerate(zip(block.spatials, classes)):
            if len(ranks) > 1:
                # A tensor held twice in one memory would make (tensor, memory, index)
                # name two loops at once. Nothing downstream could tell them apart, so
                # say so here rather than expand one loop's ranks onto the other's.
                assert (block.tensor, block.component) not in seen, (
                    f"{block.tensor} is held more than once in "
                    f"{block.component}, so its binding loops cannot be named "
                    f"by (tensor, memory, index)."
                )
                aliases[(block.tensor, block.component, i)] = tuple(ranks)
        seen.add((block.tensor, block.component))
    return aliases


def _shardable_positions(
    blocks: list[BindingBlock],
    constraints: list[BindingConstraints],
    options: list[list[tuple[Rank, ...]]],
    default: list[tuple[Rank, ...]],
    block_of: list[int],
    loop_of: list[int],
    shared: set[TensorName],
    job: Job,
) -> dict[int, tuple[Rank, ...]]:
    """
    The binding loops whose unshared tensor may be sharded instead of replicated, each
    mapped to the class of ranks it would partition along. `options`, `default`,
    `block_of` and `loop_of` are indexed by the flat binding-loop position that
    `iter_binding_choices` works in.

    A loop qualifies when all three of these hold:

    - its tensor is unshared and this memory's `may_shard_unshared_tensors` names it,
      which is what opts the tensor into being explored at all;
    - its rank variable does not index the tensor, so the loop is one the locally-optimal
      binding replicates along. A loop whose rank variable does index the tensor already
      partitions it, and which rank it partitions along is not a question an unshared
      tensor gets asked;
    - the default really is to replicate, and there is a rank to partition along
      instead. `must_shard_unshared_tensors`, `must_duplicate`, `may_duplicate` and
      `dimension_ranks` may each have settled the loop already, and a settled loop has
      nothing left to choose.
    """
    shard_classes: dict[int, tuple[Rank, ...]] = {}
    for position in range(len(options)):
        block = blocks[block_of[position]]
        tensor = block.tensor
        if tensor in shared:
            continue
        if not constraints[block_of[position]].may_shard_unshared(tensor):
            continue
        loop = block.spatials[loop_of[position]]
        irrelevant = job.einsum.tensor2irrelevant_rank_variables[tensor]
        if loop.rank_variable not in irrelevant:
            continue
        if default[position][0] != RANK_DUPLICATE:
            continue
        shards = [c for c in options[position] if c[0] != RANK_DUPLICATE]
        if shards:
            # One class holds every rank the loop could partition along, since they all
            # model alike; see `_binding_rank_classes`.
            shard_classes[position] = shards[0]
    return shard_classes


def iter_binding_choices(
    mapping: Mapping, job: Job
) -> Iterator[list[list[tuple[Rank, ...]]]]:
    """
    Iterate over the bindings of the physically-distributed storages of `mapping` that
    satisfy the binding constraints. Each yielded choice gives, for each block of
    `iter_binding_blocks` in order, the class of ranks each of that block's binding
    loops may bind: the loop binds the class's representative, and the ranks behind it
    are expanded into their own compatibilities once the template has been modeled.
    Ranks that model alike therefore cost one template between them, not one each.

    Which rank a binding loop names is explored only for the shared tensors -- those
    this Einsum passes to or receives from another one -- and only for those their
    memory's `may_explore_binding` admits, within the budget `max_binding_rank_changes`
    gives them. No other Einsum has to agree with the binding of a tensor that only this
    Einsum touches, so there is no rank to negotiate and no budget spent on one.

    An unshared tensor is still asked one question, where its memory's
    `may_shard_unshared_tensors` names it: whether the loops that do not index it
    replicate it, as the locally-optimal binding does, or partition it along one of its
    ranks. That is this Einsum's own tradeoff of capacity against traffic -- a whole
    copy in every physical instance may not fit where a shard would -- so it is not
    budgeted, and it multiplies with the shared tensors' rank changes.

    Yields nothing if the constraints admit no binding of `mapping`, in which case the
    caller must drop the template: there is no binding it could be modeled with.
    """
    blocks = iter_binding_blocks(mapping.nodes)
    block_lengths = [len(b.spatials) for b in blocks]
    constraints = [binding_constraints_for(job, b.component) for b in blocks]

    def split(flat: list[Rank]) -> list[list[Rank]]:
        """Re-split one flat list of per-loop ranks into per-block ranks."""
        out, i = [], 0
        for length in block_lengths:
            out.append(flat[i : i + length])
            i += length
        return out

    # One list of candidate rank classes per binding loop, the default class first.
    per_block = [binding_rank_options(b, job) for b in blocks]
    if any(o is None for o in per_block):
        return
    options: list[list[Rank]] = list(itertools.chain(*per_block))
    default = [o[0] for o in options]

    if not options:
        yield split(default)
        return

    # Each memory bounds how many of its own binding loops may change, and the template
    # as a whole may not change more than the most permissive of them. Budgeting per
    # memory instead would multiply the budgets together across memories, blowing up the
    # number of templates by orders of magnitude on an architecture with several
    # distributed memories.
    budgets = [c.max_binding_rank_changes for c in constraints]
    block_of = [b for b, length in enumerate(block_lengths) for _ in range(length)]
    loop_of = [i for length in block_lengths for i in range(length)]

    # A binding loop is explored only if its tensor is shared with another Einsum and
    # its memory allows that tensor to be explored; the rest keep the default.
    shared = job.fusable_tensors or set()

    def explorable_tensor(block: int) -> bool:
        tensor = blocks[block].tensor
        return tensor in shared and constraints[block].may_explore(tensor)

    explorable = [
        p
        for p in range(len(options))
        if explorable_tensor(block_of[p]) and len(options[p]) > 1
    ]
    # The budget of a memory whose tensor is not explored buys nothing, so the total
    # comes from the memories that are.
    explorable_blocks = {block_of[p] for p in explorable}
    total_budget = min(
        max((budgets[b] for b in explorable_blocks), default=0), len(explorable)
    )

    shard_classes = _shardable_positions(
        blocks, constraints, options, default, block_of, loop_of, shared, job
    )

    def budgeted_changes() -> Iterator[list[tuple[int, tuple[Rank, ...]]]]:
        """
        The rank changes the budget affords the shared tensors, one list of
        (position, rank class) pairs at a time, the empty list of them first.
        """
        if total_budget <= 0:
            yield []
            return
        for n_changes in range(int(total_budget) + 1):
            for positions in itertools.combinations(explorable, n_changes):
                changes_per_block = Counter(block_of[p] for p in positions)
                if any(n > budgets[b] for b, n in changes_per_block.items()):
                    continue
                # [1:] because the default class of a loop is not a change to it.
                for classes in itertools.product(*[options[p][1:] for p in positions]):
                    yield list(zip(positions, classes))

    # Sharding an unshared tensor is not a change to a binding two Einsums have to
    # agree on, so it is not counted against any budget and multiplies with the shared
    # tensors' changes rather than competing with them.
    shardable = sorted(shard_classes)
    for n_sharded in range(len(shardable) + 1):
        for sharded in itertools.combinations(shardable, n_sharded):
            for changes in budgeted_changes():
                choice = list(default)
                for position in sharded:
                    choice[position] = shard_classes[position]
                for position, rank_class in changes:
                    choice[position] = rank_class
                yield split(choice)


def _set_binding(block: BindingBlock, ranks: list[Rank]) -> None:
    """Give `block` a binding that names `ranks`, one per loop of its spatial loops."""
    assert len(ranks) == len(block.spatials), (
        f"Binding of {block.tensor} in {block.component} has {len(ranks)} loops, "
        f"but the storage is bound by {len(block.spatials)} spatial loops."
    )
    binding = EvalableList()
    for loop, rank in zip(block.spatials, ranks):
        binding_spatial = loop.model_copy()
        binding_spatial.rank_variable = rank
        binding_spatial.tile_shape = None
        binding_spatial.initial_tile_shape = None
        binding.append(binding_spatial)
    block.set_binding(binding)


def _default_binding_ranks_or_raise(block: BindingBlock, job: Job) -> list[Rank]:
    ranks = default_binding_ranks(block, job)
    if ranks is None:
        raise ValueError(
            f"No binding of {block.tensor} in {block.component} satisfies "
            f"{block.component}.binding. The memory is bound by "
            f"{len(block.spatials)} spatial loops "
            f"({', '.join(l.name for l in block.spatials)}) and the tensor has ranks "
            f"{_tensor_ranks(block.tensor, job)}."
        )
    return ranks


def apply_binding(
    mapping: Mapping, job: Job, ranks: list[list[Rank]] | None
) -> None:
    """
    Sets the binding of every physically-distributed (memory, tensor) pair of the
    mapping to `ranks`, which is specified in the same order as `iter_binding_blocks`
    would return, in the mapping's :class:`~.Binding` nodes. If `ranks` is None, the
    default binding of each pair is used.
    """
    blocks = iter_binding_blocks(mapping.nodes)
    # A Binding node is keyed by (memory, tensor), so a mapping that held one tensor
    # twice in one memory could not say two things about it. Nothing downstream builds
    # such a mapping; say so here rather than let the second binding overwrite the
    # first.
    keys = [block.key for block in blocks]
    assert len(set(keys)) == len(keys), (
        f"A tensor is held more than once in one physically-distributed memory "
        f"({[k for k in keys if keys.count(k) > 1]}), so its two bindings cannot both "
        f"be written down."
    )
    for i, block in enumerate(blocks):
        block_ranks = (
            _default_binding_ranks_or_raise(block, job) if ranks is None else ranks[i]
        )
        _set_binding(block, block_ranks)


def apply_default_binding_if_unspecified(mapping: Mapping, job: Job) -> None:
    """
    Give every distributed Storage of `mapping` a binding, and complete the ones it
    already has.

    A binding written by hand names the rank each of its loops binds and nothing else:
    a loop's iteration count is that of the spatial loop it pairs with, and there is no
    way to write one down. So a binding that is already there keeps its ranks and takes
    everything else from the spatial loops, exactly as one the mapper builds does. One
    that does not have a loop per spatial loop is left alone, since there is no telling
    which of its loops pairs with which.
    """
    for block in iter_binding_blocks(mapping.nodes):
        binding = block.binding
        if len(binding) == 0:
            ranks = _default_binding_ranks_or_raise(block, job)
        elif len(binding) == len(block.spatials):
            named = [loop.name for loop in binding if loop.name is not None]
            expected = [loop.name for loop in block.spatials]
            if named and named != expected:
                raise ValueError(
                    f"The binding of {block.tensor} in {block.component} binds "
                    f"dimensions {named}, but the spatial loops below it are over "
                    f"{expected}. A binding is positional: its i-th loop pairs with "
                    f"the i-th spatial loop below its Storage."
                )
            ranks = [loop.rank_variable for loop in binding]
        else:
            # A mapping still being built can have a binding that has not caught up
            # with its spatial loops; `validate_binding` leaves that alone too, and
            # `_beautify_loops` is what brings the two back into step.
            continue
        _set_binding(block, ranks)


def validate_binding(mapping: Mapping, job: Job) -> None:
    """
    Raise if any binding of `mapping` violates the binding constraints in force. Called
    on mappings that the mapper did not build, so that a mapping and the constraints it
    is evaluated under cannot silently disagree.
    """
    problems: list[str] = []
    for block in iter_binding_blocks(mapping.nodes):
        binding = block.binding
        # A binding may have fewer loops than the storage has spatial loops if the
        # mapping has not been given one yet; `_beautify_loops` keeps the two in step.
        if len(binding) != len(block.spatials):
            continue
        constraints = binding_constraints_for(job, block.component)
        for problem in constraints.violations(
            block.tensor,
            [loop.name for loop in binding],
            [loop.rank_variable for loop in binding],
            _tensor_ranks(block.tensor, job),
        ):
            problems.append(
                f"The binding of {block.tensor} in {block.component} is invalid: "
                f"{problem}."
            )
    if problems:
        raise ValueError(
            "The mapping violates the binding constraints of the architecture:\n\t"
            + "\n\t".join(problems)
        )


def _copy_mapping(mapping: Mapping) -> Mapping:
    """
    A copy of `mapping` whose nodes may be given a different binding without affecting
    `mapping`. The nodes themselves are shallow copies, matching how templates are built
    in `iterate_mappings_constraints`; `apply_binding` puts each binding in a Binding
    node of the copy's own node list, so the copies do not share bindings.
    """
    new = Mapping(nodes=[copy.copy(n) for n in mapping.nodes])
    new._n_loop_orders = mapping._n_loop_orders
    new._template_index = mapping._template_index
    return new


# =================================================================================================
# Top level
# =================================================================================================
def make_pmapping_templates(job: Job, print_progress: bool = True) -> SameEinsumJobs:
    compute_name = job.flattened_arch[-1].name

    job.tensor_to_relevancy = {
        tensor: get_rank_variable_relevancy(
            job.spec_one_einsum.workload.einsums[job.einsum_name], tensor
        )
        for tensor in job.spec_one_einsum.workload.einsums[job.einsum_name].tensor_names
    }

    mappings_constraints = iterate_mappings_constraints(
        job.spec_one_einsum,
        job.einsum_name,
        job.flattened_arch,
        job.rank_variable_bounds,
        job.tensor_to_relevancy,
        job,
    )
    if print_progress:
        mappings_constraints = tqdm(
            mappings_constraints,
            desc=f"Generating pmapping templates for compute {compute_name} Einsum {job.einsum_name}",
        )

    jobs = SameEinsumJobs()
    mapper = job.spec_one_einsum.mapper
    only_output_index = job.spec_one_einsum.mapper._only_output_pmapping_with_index
    if isinstance(only_output_index, dict):
        only_output_index = only_output_index.get(job.einsum_name, None)

    n_dropped_by_binding_constraints = 0

    for i, (mapping, constraints, symbol_table) in enumerate(mappings_constraints):
        if only_output_index is not None and i != only_output_index:
            continue
        if isinstance(only_output_index, set) and i not in only_output_index:
            continue

        # Each binding of the physically-distributed storages is its own template. The
        # first one is the default binding, which is the only one if bindings are not
        # being explored. The copies are all taken before any of the templates below is
        # processed, because processing one may mutate its nodes.
        binding_choices = list(
            iter_binding_choices(mapping, job)
        )
        if not binding_choices:
            # The constraints admit no binding of this template, so there is no
            # mapping here to model. Counted so that a constraint that removes
            # mappings outright does not look like one that had no effect.
            n_dropped_by_binding_constraints += 1
            continue
        bound_mappings = [mapping] + [
            _copy_mapping(mapping) for _ in binding_choices[1:]
        ]

        for bound_mapping, binding_choice in zip(bound_mappings, binding_choices):
            apply_binding(
                bound_mapping, job, binding_representatives(binding_choice)
            )

            new_job = copy.copy(job)
            new_job.mapping = bound_mapping
            # The ranks this template's binding stands for. Its compatibility names the
            # representative; the siblings get theirs once the template is modeled.
            new_job.binding_rank_aliases = binding_rank_aliases(
                bound_mapping, binding_choice
            )
            new_job.mapping._template_index = i
            new_job.constraints = constraints
            new_job.job_id = uuid.uuid4()
            new_job.rank_variable_bounds = job.rank_variable_bounds
            mapping_with_reservations = quick_insert_reservation_nodes(new_job)
            new_job.compatibility = Compatibility.from_mapping(
                mapping_with_reservations,
                new_job.fusable_tensors,
                new_job.einsum,
            )
            jobs.append(new_job)

    if n_dropped_by_binding_constraints:
        logging.warning(
            f"Binding constraints dropped "
            f"{n_dropped_by_binding_constraints} pmapping template(s) of Einsum "
            f"{job.einsum_name} on compute {compute_name}: no binding of those "
            f"templates satisfies the constraints. {len(jobs)} template(s) remain."
        )

    return jobs
