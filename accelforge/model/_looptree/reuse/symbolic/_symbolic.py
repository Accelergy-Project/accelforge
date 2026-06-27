import copy
from accelforge.frontend.arch import Network as NetworkSpec
from accelforge.util import indent
from accelforge.util._frozenset import oset
from accelforge.frontend.mapping import (
    Compute,
    Mapping,
    Toll,
    Reservation,
    Spatial,
    Storage,
    Temporal,
)
from typing import Any, List

from accelforge.frontend import arch
import accelforge.frontend.mapping as mapping_spec
from accelforge.frontend.mapping import (
    Mapping,
    MappingNode,
    Spatial,
    Temporal,
    Storage,
    Reservation,
    Loop,
    TensorHolder,
    Toll,
)
from accelforge.frontend.workload import (
    Workload,
    TensorName,
)
from accelforge.frontend._workload_isl._symbolic import (
    get_projection_expr,
    compute_dense_tile_occupancy,
    Irrelevant,
    Relevant,
    PartiallyRelevant,
)

from accelforge.model._looptree.types import Buffet, Compute, Network
from accelforge.model._looptree.reuse.symbolic.mapping_utils import (
    DataMovementConnections,
    get_tensor_to_backer_id,
)

from accelforge.mapper.FFM._make_pmappings.pmapper_job import Job
from accelforge.util._sympy.broadcast_max import MaxGeqZero
from accelforge.mapper.FFM._pareto_df.df_convention import iterations2col

import sympy
import symengine as se

from ._common import (
    AnalysisInfo,
    loop_stride_and_shape,
    has_parent_tensor_holder,
    SequenceOfRepatedvalues,
    RepeatedValue,
)
from ._stats import ComputeStats, BuffetStats, NetworkStats, SymbolicAnalysisOutput
from ._network import NetworkAnalyzer
from accelforge.util.indent import print

SYMBOL = "symbol"

PRINT_FORMULAS = False


def quick_insert_reservation_nodes(
    job: Job, mapping: Mapping | None = None, tensors: oset[TensorName] | None = None
) -> Mapping:
    if mapping is None:
        mapping = list(job.mapping.nodes)
    else:
        mapping = list(mapping.nodes)
    workload = job.spec_one_einsum.workload

    # TODO: Subclass reservation with TensorReservation or something so that we can
    # track which are for tensors and which are for non-tensor resources.

    info = AnalysisInfo(
        mapping=None,
        workload=workload,
        full_rank_variable_shapes=None,
        all_tensors=None,
        einsum_tensor_to_projection=None,
        tensor_to_relevancy=job.tensor_to_relevancy,
        tensor_to_backer_id=None,
        is_copy_operation=None,
        job=None,
        current_tensor=None,
    )

    fusable_tensors = job.fusable_tensors
    if tensors is not None:
        fusable_tensors = fusable_tensors & tensors

    insert_reservation_nodes(mapping, info, fusable_tensors)
    m = Mapping(nodes=mapping)
    m._n_loop_orders = job.mapping._n_loop_orders
    m._template_index = job.mapping._template_index
    return m


def convert_to_copy(
    mapping: Mapping, workload: Workload
) -> tuple[Mapping, dict[TensorName, int]]:
    # Calculate backer IDs from the ORIGINAL mapping (before deepcopy) so that
    # the id() values match the node objects used in subsequent analysis.
    # get_tensor_to_backer_id is read-only, so this is safe.
    tensor_to_backer_id = get_tensor_to_backer_id(mapping)
    mapping = list(mapping.nodes)

    mapping = copy.deepcopy(mapping)

    first_input_tensor = workload.einsums[mapping[-1].einsum].copy_source_tensor()

    for node in mapping:
        if isinstance(node, TensorHolder):
            if node.tensors:
                node.tensors = [first_input_tensor]
                node._lower = False
        if isinstance(node, Reservation):
            if node.purposes:
                node.purposes = [first_input_tensor]

    to_remove = []
    i = 0
    while i < len(mapping):
        node = mapping[i]
        if isinstance(node, TensorHolder):
            j = i + 1
            while j < len(mapping):
                node2 = mapping[j]
                if (
                    isinstance(node2, TensorHolder)
                    and node.component == node2.component
                ):
                    mapping.pop(j)
                else:
                    j += 1
        if isinstance(node, Reservation):
            j = i + 1
            while j < len(mapping):
                node2 = mapping[j]
                if isinstance(node2, Reservation) and node.resource == node2.resource:
                    mapping.pop(j)
                else:
                    j += 1
        i += 1
    mapping = [node for node in mapping if node not in to_remove]

    return Mapping(nodes=mapping), tensor_to_backer_id


def analyze_reuse_and_add_reservations_to_mapping(
    job: Job,
    add_reservations: bool = True,
) -> SymbolicAnalysisOutput:
    mapping = job.mapping
    workload = job.spec_one_einsum.workload
    einsum_name = mapping.nodes[-1].einsum
    einsum = workload.einsums[einsum_name]

    is_copy_operation = workload.einsums[einsum_name].is_copy_operation
    symbols = insert_sympy_symbols(job.mapping.nodes, job)

    tensors = oset(einsum.tensor_names)

    if is_copy_operation:
        mapping, tensor_to_backer_id = convert_to_copy(mapping, workload)
        tensors &= oset.union(
            *[oset(t.tensors) for t in mapping.nodes if isinstance(t, TensorHolder)]
        )
    else:
        tensor_to_backer_id = get_tensor_to_backer_id(mapping)

    if add_reservations:
        mapping = quick_insert_reservation_nodes(
            job,
            mapping,
            tensors,
        )
        # If it's a copy operation, then we've changed the mapping, so add the
        # reservations to the job separately because we don't want the copy
        # transformations to get applied to the original job's mapping.
        if is_copy_operation:
            job.mapping = quick_insert_reservation_nodes(job)
        # If it's not a copy, then use the reservations that we just made
        else:
            job.mapping = mapping

    einsum_tensor_to_projection = {}
    for t in tensors:
        einsum_tensor_to_projection[(einsum_name, t)] = get_projection_expr(einsum, t)
    tensor_to_relevancy = {t: job.tensor_to_relevancy[t] for t in tensors}
    assert tensors, f"Einsum {einsum_name} has no tensors"

    """
    Note for how this works.

    Spatial loops are weird, because they don't belong at a single point in the loop
    nest. For example:

    - DRAM keep A, B
    - *
    - Reg keep A
    - for n in [0..N)
    - GLB keep B
    - *
    - Compute

    A loop 'spatial-for (Reg) k in [0..K)' would affect the register at the point of the
    first asterisk, but the global buffer at the point of the second asterisk.

    To handle this, we make a separate mapping for each tensor, analyze each, and
    combine the results.

    To anyone who would like to create behavior that simultaneously looks at multiple
    storage nodes for a given memory, note that there will be two challenges to address:

    1. The code currently analyzes one tensor at a time. This could be fixed by
       processing all mapping(s) together, applying loop(s) from each to only the
       appropriate nodes.
    2. The code must analyze one storage node at a time, and there may be temporal and
       spatial nodes between two storage nodes for a given memory, which would separate
       the analysis steps for the storage nodes. This may be addressed by only
       performing such analysis until the outermost storage node for a particular memory
       has been analyzed.
    """

    # The first iteration (tensor=None) runs on the original mapping to record the
    # correct iteration count for every loop.  Per-tensor mappings may reorder
    # spatial/temporal loops for rank variables irrelevant to the tensor. This changes
    # the iteration counts and tile shapes for the loops, so we use the precomputed
    # iteration counts from the original mapping order. The tile shapes don't matter
    # because we only do it for irrelevant loops.

    result = None
    precomputed_iterations = {}

    tensor2mapping = {}
    for tensor in [None] + sorted(tensors):
        rvs = einsum.tensor2rank_variables.get(
            tensor, oset.union(*einsum.tensor2rank_variables.values())
        )
        cur_mapping = mapping._get_single_tensor_mapping(
            tensor,
            job.flattened_arch,
            tensor_rank_variables=rvs,
        )
        info = AnalysisInfo(
            mapping=cur_mapping.nodes,
            workload=workload,
            full_rank_variable_shapes=job.rank_variable_bounds,
            all_tensors=oset([tensor]) if tensor is not None else oset[Any](),
            current_tensor=tensor,
            einsum_tensor_to_projection=einsum_tensor_to_projection,
            tensor_to_relevancy=tensor_to_relevancy,
            tensor_to_backer_id=tensor_to_backer_id,
            is_copy_operation=is_copy_operation,
            job=job,
            data_movement_connections=DataMovementConnections.from_pmapping(
                cur_mapping.nodes
            ),
            precomputed_iterations=precomputed_iterations,
            tensor_rank_variables=einsum.tensor2rank_variables.get(tensor, oset()),
            simple_rank_variables=einsum._simple_rank_variables,
            is_recording_iterations=tensor is None,
        )
        cur_result = analyze_node(0, job.rank_variable_bounds, info)

        if result is None:
            result = cur_result
            # cur_tensor None is used to initialize compute stats. Compute stats is NOT
            # updated by add_buffet_stats_and_symbols so it needs to come from the first
            # result. Also the first result needs to be from cur_tensor None because we
            # populate precomputed iterations in it.
            assert tensor is None
        else:
            result.add_buffet_stats_and_symbols(cur_result)
            result.add_network_stats(cur_result)
        tensor2mapping[tensor] = cur_mapping

    # For copy operations, we mutate the original mapping before doing analysis, so we
    # need to update tensor2mapping using the original mapping.
    if is_copy_operation:
        for t in workload.einsums[einsum_name].tensor_names:
            tensor2mapping[t] = job.mapping._get_single_tensor_mapping(
                t,
                job.flattened_arch,
                tensor_rank_variables=einsum.tensor2rank_variables[t],
            )

    result.symbols = symbols
    result.tensor2mapping = tensor2mapping

    if PRINT_FORMULAS:
        print(f"Mapping:")
        with indent():
            for node in mapping.nodes:
                print(f"{node.compact_str()}")

            print("Per-tensor mapping:")
            for tensor, tensor_mapping in result.tensor2mapping.items():
                print(f"{tensor}")
                with indent():
                    for node in tensor_mapping.nodes:
                        print(f"{node.compact_str()}")

            for buffet, stats in result.buffet_stats.items():
                print(f"Einsum {buffet.einsum} tensor {buffet.tensor} level {buffet.level}")
                print(f"Total reads to parent: {stats.total_reads_to_parent}")
                print(f"Total writes to parent: {stats.total_writes_to_parent}")
                print(
                    f"Max per parent reads to parent: {stats.max_per_parent_reads_to_parent}"
                )
                print(
                    f"Max per parent writes to parent: {stats.max_per_parent_writes_to_parent}"
                )
                print(f"Total reads to peer: {stats.total_reads_to_peer}")
                print(f"Total writes to peer: {stats.total_writes_to_peer}")
                print(f"Max per unit reads to peer: {stats.max_per_unit_reads_to_peer}")
                print(f"Max per unit writes to peer: {stats.max_per_unit_writes_to_peer}")
                print(f"Max occupancy: {stats.max_occupancy}")
                print(f"N loops above: {stats.n_loops_above}")

    return result


class ReservationAnalysisTracker:
    def __init__(self, buffet, node):
        self.buffet: Buffet = buffet
        self.node: TensorHolder = node

        # These are interface (TODO: should be property)
        self.is_fill_level = False
        self.should_stop = False
        self.insert_reservation_under = False
        self.insert_fill_under = False

        # Temporary values
        self.has_filled = False

    def track_temporal_loop(self, relevancy, node):
        self.is_fill_level = False
        self.insert_reservation_under = False
        self.insert_fill_under = False

        if isinstance(relevancy, Irrelevant):
            if not self.has_filled:
                self.is_fill_level = True
                self.has_filled = True

            self.should_stop = True
        elif isinstance(relevancy, Relevant):
            self.should_stop = False
        elif isinstance(relevancy, PartiallyRelevant):
            self.last = True

            if not self.has_filled:
                self.is_fill_level = True
                self.has_filled = True

            self.should_stop = True
            self.insert_reservation_under = True
        else:
            raise ValueError(f"Unknown relevancy {relevancy}")

    def track_compute(self):
        self.should_stop = True
        if not self.has_filled:
            self.is_fill_level = True
            self.has_filled = True

    def track_spatial_loop(self, relevancy, node):
        if node.component != self.buffet.level:
            self.should_stop = True
            if not self.has_filled:
                self.is_fill_level = True
                self.has_filled = True
            return

        self.is_fill_level = False
        self.should_stop = False


def insert_reservation_nodes(
    mapping, info: AnalysisInfo, fusable_tensors: set[TensorName]
):
    trackers: list[ReservationAnalysisTracker] = []
    einsum = info.workload.einsums[mapping[-1].einsum]
    seen_tensors = oset()  # reservation for top-level buffets cannot be lowered

    n_nodes = len(mapping)
    i = 0
    while i < n_nodes:
        node = mapping[i]
        to_remove = []
        if isinstance(node, Reservation):
            pass
        elif isinstance(node, Temporal):
            rank = node.rank_variable
            for tracker in trackers:
                relevancy = info.tensor_to_relevancy[tracker.buffet.tensor]
                tracker.track_temporal_loop(relevancy[rank], node)
        elif isinstance(node, Spatial):
            rank = node.rank_variable
            for tracker in trackers:
                relevancy = info.tensor_to_relevancy[tracker.buffet.tensor]
                tracker.track_spatial_loop(relevancy[rank], node)
        elif isinstance(node, TensorHolder):
            for tracker in trackers:
                tracker.should_stop = True
                tracker.insert_reservation_under = False
            for tensor in node.tensors:
                tensor = TensorName(tensor)
                buffet = Buffet(tensor, mapping[-1].einsum, node.component)
                trackers.append(ReservationAnalysisTracker(buffet, node))
                if not node._lower or tensor not in seen_tensors:
                    seen_tensors.add(tensor)
                    trackers[-1].is_fill_level = True
                    trackers[-1].insert_reservation_under = True
                    trackers[-1].insert_fill_under = True
                    trackers[-1].should_stop = True
        elif isinstance(node, mapping_spec.Compute):
            for tracker in trackers:
                tracker.track_compute()
                tracker.insert_reservation_under = False
        else:
            raise NotImplementedError(f"Unknown node type {type(node)}")

        reservation_insert_below = []
        reservation_insert_above = []
        for j in range(len(trackers) - 1, -1, -1):
            if not trackers[j].should_stop:
                continue
            tracker = trackers.pop(j)
            buffet = tracker.buffet
            node = Reservation(purposes=[buffet.tensor], resource=buffet.level)
            node.persistent = tracker.node.persistent
            node._backing = tracker.node._backing

            if (
                buffet.tensor not in info.tensor_to_reservation_backer_id
                and buffet.tensor in fusable_tensors
            ):
                info.tensor_to_reservation_backer_id[buffet.tensor] = id(node)

            if tracker.insert_reservation_under:
                reservation_insert_below.append(node)
            else:
                reservation_insert_above.append(node)

        # The order of these for loops is important. Reservation must be below fill.
        for node in reservation_insert_below:
            mapping.insert(i + 1, node)
            i += 1
        for node in reservation_insert_above:
            mapping.insert(i, node)
            i += 1

        i += 1
        n_nodes = len(mapping)

    label_fused_loops(mapping, fusable_tensors)


def label_fused_loops(mapping: List[MappingNode], fusable_tensors: set[TensorName]):
    last_backer = -1
    assert isinstance(
        fusable_tensors, set
    ), f"Fusable tensors must be a set, got {type(fusable_tensors)}"

    for i, node in enumerate(mapping):
        if not isinstance(node, TensorHolder) or isinstance(node, Reservation):
            continue
        assert isinstance(node._backing, set)
        if node._backing & fusable_tensors:
            last_backer = i
            # We may lower this node through at most one additional loop
            if node._lower:
                last_backer += 1
    if last_backer == -1 and fusable_tensors:
        raise ValueError(
            f"No backing TensorHolder found in mapping {", ".join(m.compact_str() for m in mapping)}"
        )

    for i, node in enumerate[MappingNode](mapping):
        if isinstance(node, Loop):
            node._fused = i < last_backer

    return mapping


def analyze_node(node_idx, current_shape, info: AnalysisInfo) -> SymbolicAnalysisOutput:
    node = info.mapping[node_idx]
    class2analysis_function = {
        Temporal: analyze_temporal,
        Spatial: analyze_spatial,
        Storage: analyze_storage,
        Reservation: analyze_reservation,
        mapping_spec.Compute: analyze_compute,
        Toll: analyze_toll,
    }
    if type(node) not in class2analysis_function:
        raise TypeError(f"Unknown node type {type(node)}")
    return class2analysis_function[type(node)](node_idx, current_shape, info)


def analyze_temporal(
    node_idx, current_shape, info: AnalysisInfo
) -> SymbolicAnalysisOutput:
    mapping = info.mapping
    node = mapping[node_idx]
    stride_and_shape = loop_stride_and_shape(node, current_shape, node_idx, info)

    result_accumulator = SymbolicAnalysisOutput()

    first_latency = None

    def handle_repeated_value(repeated_shape):
        nonlocal first_latency
        shape_value = repeated_shape.value
        shape_repeats = repeated_shape.repeats

        child_shape = current_shape.copy()
        child_shape[node.rank_variable] = shape_value

        child_result = analyze_node(node_idx + 1, child_shape, info)

        accumulated_buffet_stats = result_accumulator.buffet_stats
        for buffet, stats in child_result.buffet_stats.items():
            relevancy = info.tensor_to_relevancy[buffet.tensor][node.rank_variable]
            is_fully_relevant = isinstance(relevancy, Relevant)
            stats.n_loops_above = stats.n_loops_above + 1
            accumulated_stats = accumulated_buffet_stats.setdefault(
                buffet, BuffetStats.blank()
            )
            accumulated_stats += stats.repeat_temporal(shape_repeats, is_fully_relevant)

        for network, network_stats in child_result.network_stats.items():
            result_accumulator.network_stats[network] = network_stats.repeat(
                shape_repeats
            )

        for einsum, child_steps in child_result.temporal_steps.items():
            if einsum not in result_accumulator.temporal_steps:
                result_accumulator.temporal_steps[einsum] = 0
            result_accumulator.temporal_steps[einsum] += child_steps * shape_repeats

        result_accumulator.max(fanout=child_result.fanout)

        for key in child_result.compute_stats:
            if first_latency is None:
                first_latency = child_result.compute_stats[key].max_latency

            compute_stats = result_accumulator.compute_stats.setdefault(
                key, ComputeStats()
            )
            compute_stats += child_result.compute_stats[key].repeat_temporal(
                shape_repeats
            )
            result_accumulator.compute_stats[key] = compute_stats

    info.last_temporal_node_idx = node_idx

    shape = stride_and_shape.shape
    if isinstance(shape, SequenceOfRepatedvalues):
        for repeated_shape in shape.sequence:
            assert isinstance(repeated_shape, RepeatedValue)
            handle_repeated_value(repeated_shape)
    elif isinstance(shape, RepeatedValue):
        handle_repeated_value(shape)

    if node_idx in info.idxs_to_track_first_latency:
        for compute_stat in result_accumulator.compute_stats.values():
            # Should be the first time we store this value
            assert node_idx not in compute_stat.max_first_latency
            compute_stat.max_first_latency[node_idx] = first_latency

    return result_accumulator


def analyze_spatial(node_idx, current_shape, info: AnalysisInfo):
    mapping = info.mapping
    einsum_name = mapping[-1].einsum
    node: Spatial = mapping[node_idx]
    rank_var = node.rank_variable
    node_dim = node.name
    spatial_component = info.job.flattened_arch[node.component]
    component_spatial_dim = spatial_component.spatial[node_dim]
    stride_and_shape = loop_stride_and_shape(node, current_shape, node_idx, info)

    result_accumulator = SymbolicAnalysisOutput()

    network_analyzer = NetworkAnalyzer(result_accumulator.network_stats)

    def handle_repeated_value(repeated_shape):
        shape_value = repeated_shape.value
        shape_repeats = repeated_shape.repeats

        child_shape = current_shape.copy()
        child_shape[node.rank_variable] = shape_value

        child_result = analyze_node(node_idx + 1, child_shape, info)

        accumulated_buffet_stats = result_accumulator.buffet_stats
        child_stats = list(child_result.buffet_stats.items())
        for i, (buffet, buffet_stats) in enumerate(child_stats):
            stats = buffet_stats
            accumulated_stats = accumulated_buffet_stats.setdefault(
                buffet, BuffetStats.blank()
            )
            relevancy = info.tensor_to_relevancy[buffet.tensor][rank_var]

            # Reuse parent accesses only:
            # - Irrelevant loops
            # - The outermost level that holds the tensor (the one whose parent accesses
            #   will be going through the network)
            last_buffet = True
            for other_buffet, _ in child_stats[i + 1 :]:
                if other_buffet.tensor == buffet.tensor:
                    last_buffet = False
                    break

            reuse_parent_accesses = (
                last_buffet
                and isinstance(relevancy, Irrelevant)
                and buffet.tensor in component_spatial_dim.may_reuse
            )

            stats.n_loops_above = stats.n_loops_above + 1
            accumulated_stats += stats.repeat_spatial(
                shape_repeats, reuse_parent_accesses
            )

        network_analyzer.accumulate_child_result(
            child_result, info, shape_repeats, einsum_name, child_shape, node
        )

        for einsum, child_steps in child_result.temporal_steps.items():
            if einsum not in result_accumulator.temporal_steps:
                result_accumulator.temporal_steps[einsum] = child_steps
            else:
                result_accumulator.temporal_steps[einsum] = MaxGeqZero(
                    result_accumulator.temporal_steps[einsum], child_steps
                )

        my_key = (node.component, einsum_name)
        child_result.fanout.setdefault(my_key, {})

        # Propagate up everything except the current level and dimension
        child_fanout = copy.deepcopy(child_result.fanout)
        target_fanout = child_fanout[my_key].pop(node_dim, 1)
        result_accumulator.max(fanout=child_fanout)

        # Prpoagate current level and dimension * shape_repeats
        child_fanout = child_result.fanout[my_key]
        fanout = result_accumulator.fanout.setdefault(my_key, {})
        fanout.setdefault(node_dim, 0)  # TODO: Assume sympy can just take in 0
        # TODO: If node_dim was missing, the original code would have omitted
        # shape_repeats. Is this correct?
        fanout[node_dim] += target_fanout * shape_repeats

        for key in child_result.compute_stats:
            # TODO: ensure that `ComputeStats()`, which is initialized ONCE, is okay to use here
            compute_stats = result_accumulator.compute_stats.setdefault(
                key, ComputeStats()
            )
            # TODO: If check omitted. This was in the original code, check history if needed.
            compute_stats.combine_spatial(
                child_result.compute_stats[key].repeat_spatial(shape_repeats)
            )

    shape = stride_and_shape.shape
    if isinstance(shape, SequenceOfRepatedvalues):
        for repeated_shape in shape.sequence:
            assert isinstance(repeated_shape, RepeatedValue)
            handle_repeated_value(repeated_shape)
    elif isinstance(shape, RepeatedValue):
        handle_repeated_value(shape)

    return result_accumulator


def analyze_storage(
    node_idx: int,
    current_shape: dict[str, int],
    info: AnalysisInfo,
    propagate_child_results: bool = False,
    count_upward_movement: bool = True,
    count_downward_movement: bool = True,
    count_writes: bool = True,
):
    mapping = info.mapping
    einsum_name = mapping[-1].einsum
    node: TensorHolder = mapping[node_idx]

    if not isinstance(count_upward_movement, dict):
        count_upward_movement = {
            TensorName(t): count_upward_movement for t in node.tensors
        }
    if not isinstance(count_downward_movement, dict):
        count_downward_movement = {
            TensorName(t): count_downward_movement for t in node.tensors
        }

    child_result = analyze_node(node_idx + 1, current_shape, info)
    component_object = info.job.flattened_arch[node.component]

    # Toll -> No initial output write because things just pass through. Keep
    # skip_initial True so that it inherits the value from the child.
    if isinstance(component_object, arch.Toll):
        skip_initial = True
    # Memory -> Skip initial output write is a setting
    elif isinstance(component_object, arch.Memory):
        skip_initial = component_object.skip_initial_output_write
    else:
        raise ValueError(f"Unknown component object: {component_object}")

    skip_initial = skip_initial and not info.is_copy_operation

    for tensor in node.tensors:
        tensor = TensorName(tensor)
        buffet = Buffet(tensor, einsum_name, node.component)

        # Reservations make these, and they go below the storage node, so the buffet
        # stats are already made at this point
        stats = child_result.buffet_stats[buffet]
        backer_id = info.tensor_to_backer_id[tensor]
        is_backing = backer_id == id(node)
        if node.persistent:
            stats.persistent = True
        below_backing = backer_id in [id(m) for m in mapping[:node_idx]]

        projection = info.einsum_tensor_to_projection[(einsum_name, tensor)]

        fills = compute_dense_tile_occupancy(projection, current_shape)

        child = child_result.get_child_buffet_stats(buffet)
        inherit_from_child = propagate_child_results and child is not None

        # ==============================================================================
        # Calculate the total fills and reads to parent. These propagate upward.
        # ==============================================================================

        def inherit_add(attr: str, default_value: Any = fills) -> Any:
            val = getattr(child, attr) if inherit_from_child else default_value
            setattr(stats, attr, val + getattr(stats, attr))

        if has_parent_tensor_holder(tensor, node_idx, info):
            # Initial fetch: If we're below the backing storage, fetch data from above
            # at the beginning.
            if not is_backing and below_backing:
                inherit_add("total_reads_to_parent", fills)
                inherit_add("max_per_parent_reads_to_parent", fills)

            # Data writeback. Do not writeback if it's a copy operation and we're below
            # the backing storage; data only flows upward.

            # Writeback occurs in two cases:
            # - We're at or above the backing storage, so we need to propagate our
            #   results upward to any storage nodes that will need this data.
            # - This is a written tensor, so we need to write back the written data.
            if (
                tensor in info.workload.einsums[einsum_name].output_tensor_names
                or not below_backing
            ):
                inherit_add("total_writes_to_parent")
                inherit_add("max_per_parent_writes_to_parent")

            # For read+write tensors, we skip the first fill because the data will be
            # initialized with a zero value. This only applies where reads from
            # parent actually occur (below the backing store).
            if (
                tensor in info.workload.einsums[einsum_name].output_tensor_names
                and not is_backing
                and below_backing
                and skip_initial
            ):
                inherit_add("total_skipped_first_reads_to_parent")
                inherit_add("min_per_parent_skipped_first_reads_to_parent")

        # ==============================================================================
        # Convert to actions. These are not used used upward; they are used to get
        # energy and latency.
        # ==============================================================================
        workload_bpv = info.job.einsum.tensor_accesses[tensor].bits_per_value
        bits_per_value = component_object.bits_per_value.get(tensor, workload_bpv)
        read_values_per_action = component_object._get_values_per_action(
            "read", tensor, workload_bpv
        )
        read_scale = 1 / read_values_per_action
        if count_writes:
            write_values_per_action = component_object._get_values_per_action(
                "write", tensor, workload_bpv
            )
            write_scale = 1 / write_values_per_action
        else:
            write_scale = 0

        # ==========================
        # Data exchanges with parent
        if count_downward_movement[tensor]:  # Parent -> Me
            stats.total_write_actions += stats.total_reads_to_parent * write_scale
            stats.max_per_unit_write_actions += (
                stats.total_reads_to_parent * write_scale
            )
            stats.total_skipped_first_write_actions += (
                stats.total_skipped_first_reads_to_parent * write_scale
            )
            stats.min_per_unit_skipped_first_write_actions += (
                stats.min_per_parent_skipped_first_reads_to_parent * write_scale
            )

        if count_upward_movement[tensor]:  # Me -> Parent
            # Comment this to have the final writeback to a buffer hit both that buffer and
            # go directly to the parent without incurring another read from the buffer.
            stats.total_read_actions += stats.total_writes_to_parent * read_scale
            stats.max_per_unit_read_actions += stats.total_writes_to_parent * read_scale

        # ========================
        # Data exchanges with peer
        stats.total_read_actions += stats.total_reads_to_peer * read_scale
        stats.total_write_actions += stats.total_reads_to_peer * write_scale

        # =========================
        # Data exchanges with child
        if child is not None:
            if count_downward_movement[tensor]:  # Me -> Child
                stats.total_read_actions += child.total_reads_to_parent * read_scale
                stats.max_per_unit_read_actions += (
                    child.max_per_parent_reads_to_parent * read_scale
                )
                # Skip first read
                if skip_initial:
                    stats.total_skipped_first_read_actions += (
                        child.total_skipped_first_reads_to_parent * read_scale
                    )
                    stats.min_per_unit_skipped_first_read_actions += (
                        child.min_per_parent_skipped_first_reads_to_parent * read_scale
                    )

            if count_upward_movement[tensor]:  # Child -> Me
                stats.total_write_actions += child.total_writes_to_parent * write_scale
                stats.max_per_unit_write_actions += (
                    child.max_per_parent_writes_to_parent * write_scale
                )

    return child_result


def analyze_toll(node_idx, current_shape, info: AnalysisInfo):
    mapping = info.mapping
    einsum_name = mapping[-1].einsum
    node = mapping[node_idx]
    component_object = info.job.flattened_arch[node.component]
    direction = component_object.direction
    count_up = {TensorName(t): direction[t] != "down" for t in node.tensors}
    count_down = {TensorName(t): direction[t] != "up" for t in node.tensors}
    storage_result = analyze_storage(
        node_idx,
        current_shape,
        info,
        propagate_child_results=True,
        count_upward_movement=count_up,
        count_downward_movement=count_down,
        count_writes=False,
    )
    for tensor in node.tensors:
        buffet = Buffet(tensor, einsum_name, node.component)
        stats = storage_result.buffet_stats[buffet]
        stats.max_occupancy = 0
        assert stats.total_write_actions == 0
    return storage_result


def analyze_reservation(node_idx, current_shape, info: AnalysisInfo):
    mapping = info.mapping
    einsum_name = mapping[-1].einsum
    node = mapping[node_idx]
    tensor = TensorName(node.purpose)

    if info.last_temporal_node_idx is not None and id(
        node
    ) == info.tensor_to_reservation_backer_id.get(node.purpose, None):
        info.idxs_to_track_first_latency.add(info.last_temporal_node_idx)

    child_result = analyze_node(node_idx + 1, current_shape, info)

    buffet = Buffet(tensor, einsum_name, node.resource)

    # Reservation nodes are the first to produce stats for a buffet
    assert buffet not in child_result.buffet_stats

    stats = BuffetStats()
    projection = info.einsum_tensor_to_projection[(einsum_name, tensor)]
    component_object = info.job.flattened_arch[node.resource]
    workload_bpv = info.job.einsum.tensor_accesses[tensor].bits_per_value
    bits_per_value = component_object.bits_per_value.get(tensor, workload_bpv)
    stats.max_occupancy = (
        compute_dense_tile_occupancy(projection, current_shape) * bits_per_value
    )
    child_result.buffet_stats[buffet] = stats

    # Reservation nodes are the first to produce stats for a network
    network_node = info.job.spec_one_einsum.arch.find_first_of_type_above(
        NetworkSpec, buffet.level, default=None
    )
    if network_node is not None:
        network = Network(
            tensor,
            einsum_name,
            info.data_movement_connections.get_src(buffet),
            buffet,
            component=network_node.name if network_node else network_node,
        )
        assert network not in child_result.network_stats
        child_result.network_stats[network] = NetworkStats()

    fanout_key = (node.resource, einsum_name)
    if fanout_key not in child_result.fanout:
        child_result.fanout[fanout_key] = {}

    return child_result


def analyze_compute(
    node_idx, current_shape, info: AnalysisInfo
) -> SymbolicAnalysisOutput:
    einsum = info.mapping[-1].einsum
    node = info.mapping[node_idx]

    computes = 0 if info.is_copy_operation else 1

    component_object = info.job.flattened_arch[node.component]
    skip_initial = (
        component_object.skip_initial_output_write and not info.is_copy_operation
    )

    result_accumulator = SymbolicAnalysisOutput()

    compute_key = Compute(einsum, node.component)
    result_accumulator.temporal_steps[einsum] = computes
    result_accumulator.compute_stats[compute_key] = ComputeStats(
        computes,
        computes,
        1,
    )

    if info.is_copy_operation:
        return result_accumulator

    tensors = info.all_tensors if info.current_tensor is None else [info.current_tensor]
    for tensor in tensors:
        buffet = Buffet(tensor, einsum, node.component)
        stats = BuffetStats()
        stats.total_reads_to_parent = 1
        stats.max_per_parent_reads_to_parent = 1
        if tensor in info.workload.einsums[einsum].output_tensor_names:
            stats.total_writes_to_parent = 1
            stats.max_per_parent_writes_to_parent = 1
            if skip_initial:
                stats.total_skipped_first_reads_to_parent = 1
                stats.min_per_parent_skipped_first_reads_to_parent = 1
        stats.max_occupancy = 1
        result_accumulator.buffet_stats[buffet] = stats

        network_node = info.job.spec_one_einsum.arch.find_first_of_type_above(
            NetworkSpec, node.component, default=None
        )
        if network_node is not None:
            network = Network(
                tensor,
                info.job.einsum_name,
                info.data_movement_connections.get_src(buffet),
                buffet,
                component=network_node.name if network_node else network_node,
            )
            result_accumulator.network_stats[network] = NetworkStats()

    return result_accumulator


def insert_sympy_symbols(mapping: list[MappingNode], job: Job):
    loop_idx = 0
    symbols = []
    rank_var_with_initial = oset()
    for i, node in enumerate(mapping):
        if not isinstance(node, Loop):
            continue

        stride_halos = oset()
        for t in job.spec_one_einsum.workload.einsums[job.einsum_name].tensor_names:
            cur_stride_halo = job.stride_and_halo[job.einsum_name, t]
            for (rank, rank_variable), (stride, halo) in cur_stride_halo.items():
                if rank_variable == node.rank_variable:
                    stride_halos.add((stride, halo))

        if len(stride_halos) == 0:
            raise RuntimeError(
                f"{repr(node.rank_variable)} not found in {job.stride_and_halo}"
            )

        # We only explore imperfect for the outermost fused loops
        simple = (
            (len(stride_halos) <= 1 and next(iter(stride_halos)) == (1, 0))
            or node.rank_variable in rank_var_with_initial
            or not node._fused
        )

        # NOTE: initial_tile_shape must be inserted into `symbols` before `stride`
        # because of the order of tile shape exploration.
        # TODO: there has to be a better way to do this.
        if simple:  # Just use the stride!
            node.initial_tile_shape = None
        elif node.initial_tile_shape == SYMBOL:
            rank_var_with_initial.add(node.rank_variable)
            initial_tile_shape = sympy.symbols(
                f"initial{loop_idx}", positive=True, integer=True
            )
            symbols.append(initial_tile_shape)
            node.initial_tile_shape = initial_tile_shape

        # TODO: Check for 0 < shape < 1 for loop bound target
        if job.rank_variable_bounds[node.rank_variable] == 1:
            node.tile_shape = 1
        elif node.tile_shape == SYMBOL:
            stride = sympy.symbols(f"stride{loop_idx}", positive=True, integer=True)
            symbols.append(stride)
            node.tile_shape = stride

        # TODO: sometimes, a mapping is passed into the model twice.
        #       E.g., after calling mapper, the model is called again for more
        #       details.
        #
        # assert (
        #     node.calculated_n_iterations is None
        # ), "Number of iterations is derived from the model. Do not set it!"
        node.calculated_n_iterations = sympy.symbols(
            iterations2col(loop_idx), positive=True, integer=True
        )

        loop_idx += 1

    return symbols
