from accelforge.frontend.mapping.mapping import MappingNode


import copy
from collections import defaultdict
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
    iter_binding_blocks,
)
from accelforge.frontend.renames import Rank, RANK_ANY
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
            mapping = copy.deepcopy(mapping)
            insert_spatial_loops(
                mapping, einsum, flattened_arch, job.intermediate_tensors
            )
            mapping = unpack_loops_to_rank_variables(mapping)
            if spec.mapper._timeloop_style_even:
                mapping = _timeloop_style_even(mapping)

            place_missing_temporal_loops(mapping, einsum, flattened_arch)
            label_fused_loops(mapping, fusable_tensors)
            assert_proper_fusion_labeling(mapping, fusable_tensors)
            yield mapping, symbol_table, compute, n_orders


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
def default_binding_ranks(
    tensor: TensorName, spatials: list[Spatial], job: Job
) -> list[Rank]:
    """
    The locally-optimal, non-duplicating binding of `tensor` to the spatial loops
    `spatials`: partition the tensor along the rank that each spatial loop's rank
    variable projects into, and replicate it (RANK_ANY) if the loop's rank variable
    does not index the tensor.
    """
    irrelevant_rvs = job.einsum.tensor2irrelevant_rank_variables[tensor]
    rank_variable2ranks = job.einsum.tensor_accesses[tensor].rank_variable2ranks
    ranks = []
    for loop in spatials:
        rv = loop.rank_variable
        if rv in irrelevant_rvs:
            ranks.append(RANK_ANY)
        else:
            rv_ranks = rank_variable2ranks[rv]
            assert len(rv_ranks) == 1
            ranks.append(next(iter(rv_ranks)))
    return ranks


def binding_rank_options(
    tensor: TensorName, spatials: list[Spatial], job: Job
) -> list[list[Rank]]:
    """
    The ranks that each binding loop of `tensor` may take. A binding loop may partition
    the tensor along any of the tensor's ranks, or replicate it (RANK_ANY). The
    locally-optimal choice (see `default_binding_ranks`) comes first for each loop, so
    the first combination of the options is the default binding.
    """
    all_ranks = list(job.einsum.tensor_accesses[tensor].ranks) + [RANK_ANY]
    return [
        [default] + [r for r in all_ranks if r != default]
        for default in default_binding_ranks(tensor, spatials, job)
    ]


def iter_binding_choices(
    mapping: Mapping,
    job: Job,
    explore: bool = False,
    max_rank_changes: float | int = 1,
    max_choices: float | int = float("inf"),
) -> Iterator[list[list[Rank]]]:
    """
    Iterate over the bindings of the physically-distributed storages of `mapping`. Each
    yielded choice gives, for each block of `iter_binding_blocks` in order, the rank
    bound by each of that block's binding loops.

    Every choice has one binding loop per spatial loop binding the storage, and the
    binding loops take their iteration counts from those spatial loops (see
    `insert_sympy_symbols`), so bindings differ only in the ranks they bind.

    There are exponentially many bindings, so they are yielded in order of increasing
    number of loops that bind a rank other than the locally-optimal one: the
    locally-optimal binding comes first, then the bindings that change one loop, and so
    on up to `max_rank_changes` changes and `max_choices` bindings in total.
    """
    blocks = iter_binding_blocks(mapping.nodes)
    block_lengths = [len(spatials) for _, spatials in blocks]

    def split(flat: list[Rank]) -> list[list[Rank]]:
        """Re-split one flat list of per-loop ranks into per-block ranks."""
        out, i = [], 0
        for length in block_lengths:
            out.append(flat[i : i + length])
            i += length
        return out

    # One list of candidate ranks per binding loop, the locally-optimal rank first.
    options: list[list[Rank]] = list(
        itertools.chain(
            *[
                binding_rank_options(storage.tensor, spatials, job)
                for storage, spatials in blocks
            ]
        )
    )
    default = [o[0] for o in options]

    if not explore or not options:
        yield split(default)
        return

    n_yielded = 0
    max_rank_changes = min(max_rank_changes, len(options))
    for n_changes in range(int(max_rank_changes) + 1):
        for positions in itertools.combinations(range(len(options)), n_changes):
            # [1:] because the locally-optimal rank of a loop is not a change to it.
            for ranks in itertools.product(*[options[p][1:] for p in positions]):
                if n_yielded >= max_choices:
                    logging.info(
                        f"Stopped after exploring {n_yielded} bindings of a pmapping "
                        f"template. Raise max_binding_options_per_pmapping_template to "
                        f"explore more."
                    )
                    return
                choice = list(default)
                for position, rank in zip(positions, ranks):
                    choice[position] = rank
                n_yielded += 1
                yield split(choice)


def apply_binding(
    mapping: Mapping, job: Job, ranks: list[list[Rank]] | None = None
) -> None:
    """
    Bind the tensor of every physically-distributed storage of `mapping` to the spatial
    loops below it. `ranks` gives the rank bound by each binding loop of each block of
    `iter_binding_blocks`, as yielded by `iter_binding_choices`; if it is None, the
    locally-optimal binding is used. Any binding already on the mapping is replaced.

    A binding loop takes only the spatial dimension and the number of iterations of the
    spatial loop it comes from, not its tile shape. The shape of the bound rank may
    differ from the shape of the rank variable that the spatial loop tiles. Thus,
    the tile shapes are cleared.
    """
    for i, (storage, spatials) in enumerate(iter_binding_blocks(mapping.nodes)):
        tensor = storage.tensor
        block_ranks = (
            default_binding_ranks(tensor, spatials, job) if ranks is None else ranks[i]
        )
        assert len(block_ranks) == len(spatials), (
            f"Binding of {tensor} in {storage.component} has {len(block_ranks)} loops, "
            f"but the storage is bound by {len(spatials)} spatial loops."
        )
        binding = EvalableList()
        for loop, rank in zip(spatials, block_ranks):
            binding_spatial = loop.model_copy()
            binding_spatial.rank_variable = rank
            binding_spatial.tile_shape = None
            binding_spatial.initial_tile_shape = None
            binding.append(binding_spatial)
        storage.binding = binding


def _copy_mapping(mapping: Mapping) -> Mapping:
    """
    A copy of `mapping` whose nodes may be given a different binding without affecting
    `mapping`. The nodes themselves are shallow copies, matching how templates are built
    in `iterate_mappings_constraints`; `apply_binding` replaces the binding list of each
    Storage node rather than mutating it, so the copies do not share bindings.
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

    for i, (mapping, constraints, symbol_table) in enumerate(mappings_constraints):
        if only_output_index is not None and i != only_output_index:
            continue
        if isinstance(only_output_index, set) and i not in only_output_index:
            continue

        # Each binding of the physically-distributed storages is its own template. The
        # first one is the locally-optimal binding, which is the only one if bindings
        # are not being explored. The copies are all taken before any of the templates
        # below is processed, because processing one may mutate its nodes.
        binding_choices = list(
            iter_binding_choices(
                mapping,
                job,
                mapper.explore_bindings,
                mapper.max_binding_rank_changes,
                mapper.max_binding_options_per_pmapping_template,
            )
        )
        bound_mappings = [mapping] + [
            _copy_mapping(mapping) for _ in binding_choices[1:]
        ]

        for bound_mapping, binding_ranks in zip(bound_mappings, binding_choices):
            apply_binding(bound_mapping, job, binding_ranks)

            new_job = copy.copy(job)
            new_job.mapping = bound_mapping
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

    return jobs
