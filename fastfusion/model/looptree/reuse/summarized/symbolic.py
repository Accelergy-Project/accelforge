from dataclasses import dataclass, field
from typing import Any

import fastfusion.frontend.mapping as mapping_spec
from fastfusion.frontend.mapping import Mapping, Spatial, Temporal, Storage, Reservation, Fill, Iteration, Pattern
from fastfusion.frontend.workload import (
    Workload,
    TensorName,
    get_rank_variable_bounds
)
from fastfusion.frontend.workload.symbolic import (
    get_projection_expr,
    get_rank_variable_relevancy,
    compute_dense_tile_occupancy,
    Irrelevant,
    Relevant,
    PartiallyRelevant
)

from fastfusion.util.sympy.broadcast_max import Max

import sympy


SYMBOL = 'symbol'


@dataclass(eq=True, frozen=True)
class Buffet:
    tensor: TensorName
    einsum: str
    level: str


@dataclass(eq=True, frozen=True)
class Compute:
    einsum: str
    level: str


class Uninitialized:
    def __init__(self):
        pass

    def __str__(self):
        return 'Uninitialized'

    def __repr__(self):
        return 'Uninitialized()'

    def __rmul__(self, other):
        return self * other

    def __mul__(self, other):
        return self

    def __radd__(self, other):
        return self + other

    def __add__(self, other):
        return self


@dataclass
class BuffetStats:
    total_fills: Any = field(default=0)
    max_per_unit_fills: Any = field(default=0)
    reads_to_peer: Any = field(default=0)
    total_reads_to_parent: Any = field(default=0)
    max_per_parent_reads_to_parent: Any = field(default=0)
    occupancy: Any = field(default=0)
    n_loops_above: int = field(default=0)


@dataclass
class ComputeStats:
    total_ops: Any = field(default=0)
    max_per_unit_ops: Any = field(default=0)


@dataclass
class SummarizedAnalysisOutput:
    compute_stats: dict[Compute, ComputeStats] = field(default_factory=dict)

    buffet_stats: dict[Buffet, BuffetStats] = field(default_factory=dict)

    # Mapping [level, einsum] to the fanout
    fanout: dict = field(default_factory=dict)

    temporal_steps: dict = field(default_factory=dict)

    # Elidable reads of output tensor
    elidable_reads: dict = field(default_factory=dict)

    symbols: list = field(default_factory=list)


@dataclass
class AnalysisInfo:
    """Information needed within the analysis step by multiple functions that
    can be computed once at the beginning.
    """
    mapping: Mapping
    workload: Workload
    full_rank_variable_shapes: dict
    all_tensors: set

    einsum_tensor_to_projection: dict
    tensor_to_relevancy: dict

    tensor_to_backing_storage_id: dict[TensorName, int]


def analyze_reuse(
    mapping: Mapping,
    workload: Workload
) -> SummarizedAnalysisOutput:
    mapping = mapping.nodes
    einsum_name = mapping[-1].einsum
    einsum_shape = get_rank_variable_bounds(workload, einsum_name)

    einsum_tensor_to_projection = {}
    einsum = workload.einsums[einsum_name]
    all_tensors = einsum.input_tensors() | einsum.output_tensors()
    for tensor in all_tensors:
        einsum_tensor_to_projection[(einsum_name, tensor)] = \
            get_projection_expr(einsum, tensor)

    tensor_to_relevancy = {
        tensor: get_rank_variable_relevancy(einsum, tensor)
        for tensor in all_tensors
    }

    info = AnalysisInfo(
        mapping=mapping,
        workload=workload,
        full_rank_variable_shapes=get_rank_variable_bounds(workload,
                                                           einsum_name),
        all_tensors=all_tensors,
        einsum_tensor_to_projection=einsum_tensor_to_projection,
        tensor_to_relevancy=tensor_to_relevancy,
        tensor_to_backing_storage_id=get_tensor_to_backing_storage_id(mapping),
    )
    symbols = insert_sympy_symbols(mapping)

    insert_reservation_nodes(mapping, info)

    result = analyze_node(0, einsum_shape, info)
    result.symbols = symbols

    return result


def get_tensor_to_backing_storage_id(mapping: Mapping):
    tensor_to_ids: dict[TensorName, set[int]] = {}
    for node in mapping:
        if isinstance(node, Storage):
            for tensor in node.tensors:
                if tensor in tensor_to_ids:
                    continue
                tensor_to_ids[tensor] = id(node)
    return tensor_to_ids


class ReservationAnalysisTracker:
    def __init__(self, buffet):
        self.buffet: Buffet = buffet
        self.last = False

        # These are interface (TODO: should be property)
        self.is_fill_level = False
        self.is_should_stop = False

        # Temporary values
        self.has_filled = False

    def track_temporal_loop(self, relevancy, node):
        self.is_fill_level = False
        if self.last:
            if not self.has_filled:
                self.is_fill_level = True
                self.has_filled = True

            self.is_should_stop = True
        elif isinstance(relevancy, Irrelevant):
            if not self.has_filled:
                self.is_fill_level = True
                self.has_filled = True

            self.is_should_stop = True
        elif isinstance(relevancy, Relevant):
            self.is_should_stop = False
        elif isinstance(relevancy, PartiallyRelevant):
            self.last = True

            if not self.has_filled:
                self.is_fill_level = True
                self.has_filled = True

            self.is_should_stop = False
        else:
            raise ValueError(f'Unknown relevancy {relevancy}')

    def track_compute(self):
        self.is_should_stop = True
        if not self.has_filled:
            self.is_fill_level = True
            self.has_filled = True

    def track_spatial_loop(self, relevancy, node):
        if node.across != self.buffet.level:
            self.is_should_stop = True
            if not self.has_filled:
                self.is_fill_level = True
                self.has_filled = True
            return

        self.is_fill_level = False
        self.is_should_stop = False


def insert_reservation_nodes(mapping, info: AnalysisInfo):
    trackers: list[ReservationAnalysisTracker] = []
    einsum = info.workload.einsums[mapping[-1].einsum]
    non_intermediate_tensors = einsum.tensors - info.workload.intermediate_tensors
    seen_tensors = set()  # reservation for top-level buffets cannot be lowered
    for i, node in enumerate(mapping):
        insert_offset = 0 # for inserting under storage
        fills = []
        to_remove = []
        if isinstance(node, Temporal):
            rank = node.rank_variable
            for tracker_idx, tracker in enumerate(trackers):
                relevancy = info.tensor_to_relevancy[tracker.buffet.tensor]
                tracker.track_temporal_loop(relevancy[rank], node)

                if tracker.is_fill_level:
                    fills.append(tracker.buffet)

                if tracker.is_should_stop:
                    to_remove.append(tracker_idx)
        elif isinstance(node, Spatial):
            rank = node.rank_variable
            for tracker_idx, tracker in enumerate(trackers):
                relevancy = info.tensor_to_relevancy[tracker.buffet.tensor]
                tracker.track_spatial_loop(relevancy[rank], node)

                if tracker.is_fill_level:
                    fills.append(tracker.buffet)

                if tracker.is_should_stop:
                    to_remove.append(tracker_idx)
        elif isinstance(node, Storage):
            for tensor in node.tensors:
                tensor = TensorName(tensor)
                buffet = Buffet(tensor, mapping[-1].einsum, node.memory)
                trackers.append(ReservationAnalysisTracker(buffet))
                if (
                    not node._lower
                    or
                    (tensor not in seen_tensors and tensor in non_intermediate_tensors)
                ):
                    insert_offset = 1
                    seen_tensors.add(tensor)
                    fills.append(buffet)
                    to_remove.append(len(trackers)-1)
        elif isinstance(node, mapping_spec.Compute):
            for tracker_idx, tracker in enumerate(trackers):
                tracker.track_compute()

                if tracker.is_fill_level:
                    fills.append(tracker.buffet)
                
                if tracker.is_should_stop:
                    to_remove.append(tracker_idx)
        elif isinstance(node, Reservation):
            pass
        elif isinstance(node, Fill):
            pass
        else:
            raise NotImplementedError(f"Unknown node type {type(node)}")

        for tracker_idx in reversed(to_remove):
            tracker = trackers.pop(tracker_idx)
            mapping.insert(
                i+insert_offset,
                Reservation(tensor=tracker.buffet.tensor,
                            memory=tracker.buffet.level)
            )

        for fill in fills:
            mapping.insert(
                i+insert_offset,
                Fill(tensor=fill.tensor,
                     memory=fill.level)
            )


def analyze_node(node_idx, current_shape, info: AnalysisInfo):
    node = info.mapping[node_idx]
    if isinstance(node, Temporal):
        return analyze_temporal(node_idx, current_shape, info)
    elif isinstance(node, Spatial):
        return analyze_spatial(node_idx, current_shape, info)
    elif isinstance(node, Storage):
        return analyze_storage(node_idx, current_shape, info)
    elif isinstance(node, Reservation):
        return analyze_reservation(node_idx, current_shape, info)
    elif isinstance(node, mapping_spec.Compute):
        return analyze_compute(node_idx, current_shape, info)
    elif isinstance(node, Fill):
        return analyze_fill(node_idx, current_shape, info)


def analyze_temporal(node_idx,
                     current_shape,
                     info: AnalysisInfo) -> SummarizedAnalysisOutput:
    mapping = info.mapping
    node = mapping[node_idx]
    stride_and_shape = get_stride_and_tile_shape(node, current_shape, node_idx)

    result_accumulator = SummarizedAnalysisOutput()

    def handle_repeated_value(repeated_shape):
        shape_value = repeated_shape.value
        shape_repeats = repeated_shape.repeats

        child_shape = current_shape.copy()
        child_shape[node.rank_variable] = shape_value

        child_result = analyze_node(node_idx+1, child_shape, info)

        accumulated_buffet_stats = result_accumulator.buffet_stats
        for buffet, buffet_stats in child_result.buffet_stats.items():
            if buffet not in accumulated_buffet_stats:
                accumulated_stats = BuffetStats()
            else:
                accumulated_stats = accumulated_buffet_stats[buffet]

            accumulated_stats.total_reads_to_parent += \
                buffet_stats.total_reads_to_parent * shape_repeats
            accumulated_stats.max_per_parent_reads_to_parent += \
                buffet_stats.max_per_parent_reads_to_parent * shape_repeats
            accumulated_stats.reads_to_peer += \
                buffet_stats.reads_to_peer * shape_repeats
            accumulated_stats.total_fills += \
                buffet_stats.total_fills * shape_repeats
            accumulated_stats.max_per_unit_fills += \
                buffet_stats.max_per_unit_fills * shape_repeats
            accumulated_stats.occupancy = Max(
                accumulated_stats.occupancy, buffet_stats.occupancy
            )

            accumulated_stats.n_loops_above = buffet_stats.n_loops_above + 1

            accumulated_buffet_stats[buffet] = accumulated_stats

        for einsum, child_steps in child_result.temporal_steps.items():
            if einsum not in result_accumulator.temporal_steps:
                result_accumulator.temporal_steps[einsum] = 0
            result_accumulator.temporal_steps[einsum] += child_steps*shape_repeats

        for key, child_fanout in child_result.fanout.items():
            if key not in result_accumulator.fanout:
                result_accumulator.fanout[key] = child_fanout
                continue
            acc_fanout = result_accumulator.fanout[key]

            for dim in child_fanout:
                if dim in acc_fanout:
                    acc_fanout[dim] = Max(acc_fanout[dim], child_fanout[dim])
                else:
                    acc_fanout[dim] = child_fanout[dim]

        for key in child_result.compute_stats:
            if key not in result_accumulator.compute_stats:
                result_accumulator.compute_stats[key] = ComputeStats()
            compute_stats = result_accumulator.compute_stats[key]
            compute_stats.total_ops += \
                child_result.compute_stats[key].total_ops * shape_repeats
            compute_stats.max_per_unit_ops += \
                child_result.compute_stats[key].max_per_unit_ops * shape_repeats

        for tensor, child_elidable in child_result.elidable_reads.items():
            if tensor not in result_accumulator.elidable_reads:
                result_accumulator.elidable_reads[tensor] = 0
            result_accumulator.elidable_reads[tensor] += child_elidable*shape_repeats

    shape = stride_and_shape.shape
    if isinstance(shape, SequenceOfRepatedvalues):
        for repeated_shape in shape.sequence:
            assert isinstance(repeated_shape, RepeatedValue)
            handle_repeated_value(repeated_shape)
    elif isinstance(shape, RepeatedValue):
        handle_repeated_value(shape)

    return result_accumulator


def analyze_spatial(node_idx, current_shape, info: AnalysisInfo):
    mapping = info.mapping
    einsum_name = mapping[-1].einsum
    node = mapping[node_idx]
    rank_var = node.rank_variable
    node_dim = node.dimension
    stride_and_shape = get_stride_and_tile_shape(node, current_shape, node_idx)

    result_accumulator = SummarizedAnalysisOutput()

    def handle_repeated_value(repeated_shape):
        shape_value = repeated_shape.value
        shape_repeats = repeated_shape.repeats

        child_shape = current_shape.copy()
        child_shape[node.rank_variable] = shape_value

        child_result = analyze_node(node_idx+1, child_shape, info)

        accumulated_buffet_stats = result_accumulator.buffet_stats
        for buffet, buffet_stats in child_result.buffet_stats.items():
            if buffet not in accumulated_buffet_stats:
                accumulated_stats = BuffetStats()
            else:
                accumulated_stats = accumulated_buffet_stats[buffet]

            relevancy = info.tensor_to_relevancy[buffet.tensor][rank_var]
            if buffet.level == node.across and isinstance(relevancy, Irrelevant):
                accumulated_stats.total_reads_to_parent = buffet_stats.total_reads_to_parent
                accumulated_stats.max_per_parent_reads_to_parent = buffet_stats.max_per_parent_reads_to_parent
            elif buffet.level == node.across:
                accumulated_stats.total_reads_to_parent += \
                    buffet_stats.total_reads_to_parent * shape_repeats
                accumulated_stats.max_per_parent_reads_to_parent += \
                    buffet_stats.max_per_parent_reads_to_parent * shape_repeats
            else:
                accumulated_stats.total_reads_to_parent += \
                    buffet_stats.total_reads_to_parent * shape_repeats
                accumulated_stats.max_per_parent_reads_to_parent = Max(
                    accumulated_stats.max_per_parent_reads_to_parent,
                    buffet_stats.max_per_parent_reads_to_parent
                )

            accumulated_stats.reads_to_peer = 0  # TODO: peer-to-peer support

            accumulated_stats.total_fills += \
                buffet_stats.total_fills * shape_repeats
            accumulated_stats.max_per_unit_fills = Max(
                accumulated_stats.max_per_unit_fills,
                buffet_stats.max_per_unit_fills
            )
            accumulated_stats.occupancy = Max(
                accumulated_stats.occupancy, buffet_stats.occupancy
            )

            accumulated_stats.n_loops_above = buffet_stats.n_loops_above + 1

            accumulated_buffet_stats[buffet] = accumulated_stats

        for einsum, child_steps in child_result.temporal_steps.items():
            if einsum not in result_accumulator.temporal_steps:
                result_accumulator.temporal_steps[einsum] = child_steps
            else:
                result_accumulator.temporal_steps[einsum] = Max(
                    result_accumulator.temporal_steps[einsum],
                    child_steps
                )

        my_key = (node.across, einsum_name)

        for key in child_result.fanout:
            child_fanout = child_result.fanout[key]

            if key not in result_accumulator.fanout:
                result_accumulator.fanout[key] = child_fanout
                continue
            fanout = result_accumulator.fanout[key]

            if key == my_key:
                for dim in child_fanout:
                    if dim in fanout and dim == node_dim:
                        fanout[dim] += child_fanout[dim]*shape_repeats
                    elif dim in fanout:
                        fanout[dim] = Max(fanout[dim], child_fanout[dim])
                    else:
                        fanout[dim] = child_fanout[dim]
            else:
                for dim in child_fanout:
                    if dim in fanout:
                        fanout[dim] = Max(fanout[dim], child_fanout[dim])
                    else:
                        fanout[dim] = child_fanout[dim]
                        
        result_accumulator.fanout.setdefault(my_key, {}).setdefault(node.dimension, shape_repeats)

        for key in child_result.compute_stats:
            if key not in result_accumulator.compute_stats:
                result_accumulator.compute_stats[key] = ComputeStats()
            compute_stats = result_accumulator.compute_stats[key]
            compute_stats.total_ops += \
                child_result.compute_stats[key].total_ops * shape_repeats
            if compute_stats.max_per_unit_ops == 0:
                compute_stats.max_per_unit_ops = child_result.compute_stats[key].max_per_unit_ops
            else:
                compute_stats.max_per_unit_ops = Max(
                    compute_stats.max_per_unit_ops,
                    child_result.compute_stats[key].max_per_unit_ops
                )

        for tensor, child_elidable in child_result.elidable_reads.items():
            if tensor not in result_accumulator.elidable_reads:
                result_accumulator.elidable_reads[tensor] = 0
            result_accumulator.elidable_reads[tensor] += child_elidable*shape_repeats

    shape = stride_and_shape.shape
    if isinstance(shape, SequenceOfRepatedvalues):
        for repeated_shape in shape.sequence:
            assert isinstance(repeated_shape, RepeatedValue)
            handle_repeated_value(repeated_shape)
    elif isinstance(shape, RepeatedValue):
        handle_repeated_value(shape)

    return result_accumulator


def reduce_dicts(dict1: dict, dict2: dict, reduce_op):
    for key in dict1:
        if key not in dict2:
            dict2[key] = dict1[key]
        else:
            dict2[key] = reduce_op(dict1[key], dict2[key])


def analyze_storage(node_idx, current_shape, info: AnalysisInfo):
    mapping = info.mapping
    einsum_name = mapping[-1].einsum
    node = mapping[node_idx]

    child_result = analyze_node(node_idx+1, current_shape, info)

    for tensor in node.tensors:
        tensor = TensorName(tensor)
        buffet = Buffet(tensor, einsum_name, node.memory)
        buffet_stats = child_result.buffet_stats[buffet]
        buffet_stats.reads_to_peer = 0  # TODO: peer-to-peer support

        if (
            id(node) == info.tensor_to_backing_storage_id[tensor]
            and tensor in info.workload.einsums[einsum_name].output_tensors()
        ):
            projection = info.einsum_tensor_to_projection[(einsum_name, tensor)]
            child_result.elidable_reads[tensor] = \
                compute_dense_tile_occupancy(projection, current_shape)

    return child_result


def analyze_reservation(node_idx, current_shape, info: AnalysisInfo):
    mapping = info.mapping
    einsum_name = mapping[-1].einsum
    node = mapping[node_idx]

    child_result = analyze_node(node_idx+1, current_shape, info)

    tensor = TensorName(node.tensor)
    buffet = Buffet(tensor, einsum_name, node.memory)

    # Reservation nodes are the first to produce stats for a buffet
    assert buffet not in child_result.buffet_stats

    buffet_stats = BuffetStats()
    projection = info.einsum_tensor_to_projection[(einsum_name, tensor)]
    buffet_stats.occupancy = \
        compute_dense_tile_occupancy(projection, current_shape)
    child_result.buffet_stats[buffet] = buffet_stats

    fanout_key = (node.memory, einsum_name)
    if fanout_key not in child_result.fanout:
        child_result.fanout[fanout_key] = {}

    return child_result


def analyze_fill(node_idx, current_shape, info: AnalysisInfo) -> SummarizedAnalysisOutput:
    mapping = info.mapping
    einsum_name = mapping[-1].einsum
    node = mapping[node_idx]

    child_result = analyze_node(node_idx+1, current_shape, info)
    
    tensor = node.tensor
    buffet = Buffet(tensor, mapping[-1].einsum, node.memory)

    buffet_stats = child_result.buffet_stats[buffet]
    projection = info.einsum_tensor_to_projection[(einsum_name, tensor)]
    buffet_stats.total_fills = \
        compute_dense_tile_occupancy(projection, current_shape)
    buffet_stats.max_per_unit_fills = buffet_stats.total_fills
    buffet_stats.total_reads_to_parent = buffet_stats.total_fills
    buffet_stats.max_per_parent_reads_to_parent = \
        buffet_stats.total_reads_to_parent

    return child_result


def analyze_compute(node_idx,
                    current_shape,
                    info: AnalysisInfo) -> SummarizedAnalysisOutput:
    einsum = info.mapping[-1].einsum
    node = info.mapping[node_idx]

    result_accumulator = SummarizedAnalysisOutput()
    result_accumulator.temporal_steps[einsum] = 1
    result_accumulator.compute_stats[Compute(einsum, node.compute)] = ComputeStats(1, 1)

    for tensor in info.all_tensors:
        buffet = Buffet(tensor, einsum, node.compute)
        buffet_stats = BuffetStats()
        buffet_stats.total_fills = 1
        buffet_stats.max_per_unit_fills = 1
        buffet_stats.reads_to_peer = 0
        buffet_stats.total_reads_to_parent = 1
        buffet_stats.max_per_parent_reads_to_parent = 1
        buffet_stats.occupancy = 1
        result_accumulator.buffet_stats[buffet] = buffet_stats

    return result_accumulator


@dataclass
class RepeatedValue[T]:
    value: T
    repeats: int


@dataclass
class SequenceOfRepatedvalues[T]:
    sequence: list[RepeatedValue[T]]


@dataclass
class StrideAndShape:
    stride: any
    shape: any


def get_stride_and_tile_shape(node: Iteration, full_shape, n: int):
    rank = node.rank_variable
    rank_shape = full_shape[rank]

    if node.tile_shape is not None:
        tile_shape = node.tile_shape

        if node.assume_perfect_factor or known_perfect_factor(tile_shape, rank_shape):
            factor = rank_shape / tile_shape
            return StrideAndShape(tile_shape, RepeatedValue(tile_shape, factor))
        else:
            factor = sympy.ceiling(rank_shape / sympy.Min(tile_shape, rank_shape))
            return make_possibly_different_last(tile_shape, factor, rank_shape)
    elif node.loop_bound is not None:
        factor = node.loop_bound

        if node.assume_perfect_factor or known_perfect_factor(factor, rank_shape):
            tile_shape = rank_shape / factor
            return StrideAndShape(tile_shape, RepeatedValue(tile_shape, factor))
        else:
            tile_shape = sympy.ceiling(rank_shape / sympy.Min(rank_shape, factor))
            return make_possibly_different_last(tile_shape, factor, rank_shape)
    
    elif node.tile_pattern is not None:
        stride = node.tile_pattern.stride
        initial_tile_shape = node.tile_pattern.initial_tile_shape
        tile_shape = node.tile_pattern.tile_shape

        if initial_tile_shape is not None:
            middle_shape_factor = sympy.ceiling((rank_shape - initial_tile_shape)/stride)
            # TODO: sometimes last_shape is 0, causing numerical instability
            # Currently, we are sometimes rounding up last shape.
            # last_shape = rank_shape - initial_tile_shape - stride*middle_shape_factor
            # has_last_shape = sympy.ceiling(last_shape/(last_shape+1))
            return StrideAndShape(
                stride,
                SequenceOfRepatedvalues([
                    RepeatedValue(initial_tile_shape, 1),
                    RepeatedValue(stride, middle_shape_factor),
                    # RepeatedValue(last_shape+0.01, has_last_shape)
                ])
            )
        elif tile_shape is not None:
            raise ValueError('Recomputation not yet supported')
            # shape = node["tile_pattern"]["shape"]

            # factor = sympy.ceiling(rank_shape / stride)

            # common_case_factor = sympy.floor((rank_shape - shape)/stride)

            # iterationvar = sympy.symbols(f"iteration{n}")
            # last_shapes = rank_shape - iterationvar*stride
            # last_case_factor = factor - common_case_factor

            # return StrideAndShape(
            #     stride,
            #     SequenceOfRepatedvalues([
            #         RepeatedValue(shape, common_case_factor),
            #         RepeatedValue(last_shapes, last_case_factor)
            #     ])
            # )
    else:
        raise ValueError(f"Neither tile_shape, factor, nor tile_pattern found")


def known_perfect_factor(divisor, full_shape):
    return (
        isinstance(divisor, int) and isinstance(full_shape, int)
        and full_shape % divisor == 1
    )


def make_possibly_different_last(common_tile_shape, factor, full_shape):
    last_shape = full_shape - common_tile_shape*(factor-1)
    all_shapes = SequenceOfRepatedvalues([
        RepeatedValue(common_tile_shape, factor-1),
        RepeatedValue(last_shape, 1)
    ])
    return StrideAndShape(common_tile_shape, all_shapes)


def insert_sympy_symbols(mapping):
    loop_idx = 0
    symbols = []
    for node in mapping:
        if isinstance(node, Spatial) or isinstance(node, Temporal):
            if node.tile_shape == SYMBOL:
                node.tile_shape = sympy.symbols(f'tileshape{loop_idx}', positive=True, integer=True)
                symbols.append(node.tile_shape)
            elif node.loop_bound == SYMBOL:
                node.loop_bound = sympy.symbols(f'loopbound{loop_idx}', positive=True, integer=True)
                symbols.append(node.loop_bound)
            elif node.tile_pattern == SYMBOL:
                node.tile_pattern = Pattern(stride=0)
                node.tile_pattern.stride = sympy.symbols(f'stride{loop_idx}', positive=True, integer=True)
                node.tile_pattern.initial_tile_shape = sympy.symbols(f'initial{loop_idx}', positive=True, integer=True)
                symbols.append(node.tile_pattern.stride)
                symbols.append(node.tile_pattern.initial_tile_shape)
            loop_idx += 1
    return symbols
