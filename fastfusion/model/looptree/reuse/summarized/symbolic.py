import copy
from dataclasses import dataclass, field
from typing import Any

from fastfusion.frontend import arch
import fastfusion.frontend.mapping as mapping_spec
from fastfusion.frontend.arch import ProcessingStage
from fastfusion.frontend.mapping import Mapping, MappingNode, Spatial, Temporal, Storage, Reservation, Iteration, TensorHolder# NOFILL: Fill
from fastfusion.frontend.workload import (
    Workload,
    TensorName,
    get_rank_variable_bounds
)
from fastfusion.frontend.workload._symbolic import (
    get_projection_expr,
    get_rank_variable_relevancy,
    compute_dense_tile_occupancy,
    Irrelevant,
    Relevant,
    PartiallyRelevant
)

from fastfusion.mapper.FFM._make_pmappings.mapper_one_einsum.mapper_job import Job
from fastfusion.util.sympy.broadcast_max import Min, Max

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

# TODO: unsure if this is needed. If the sympy symbol is created with the
# correct assumption (e.g., positive), this should be automatic.
def min_nonzero(a: Any, b: Any) -> Any:
    if a == 0:
        return b
    if b == 0:
        return a
    return Min(a, b)


def max_dict(a: dict[Any, Any], b: dict[Any, Any]) -> dict[Any, Any]:
    new = {**a}
    for key, value in b.items():
        new[key] = Max(new[key], value) if key in new else value
    return new


@dataclass
class BuffetStats:
    total_reads_to_parent: Any = field(default=0)
    total_writes_to_parent: Any = field(default=0)
    max_per_parent_reads_to_parent: Any = field(default=0)
    max_per_parent_writes_to_parent: Any = field(default=0)
    
    total_reads_to_peer: Any = field(default=0)
    total_writes_to_peer: Any = field(default=0)
    max_per_unit_reads_to_peer: Any = field(default=0)
    max_per_unit_writes_to_peer: Any = field(default=0)

    # Skip the first iteration of temporal loops for data that is written
    total_skipped_first_reads_to_parent: Any = field(default=0)
    total_skipped_first_reads_to_peer: Any = field(default=0)
    min_per_parent_skipped_first_reads_to_parent: Any = field(default=0)
    min_per_unit_skipped_first_writes_to_peer: Any = field(default=0)

    max_occupancy: Any = field(default=0)
    n_loops_above: int = field(default=0)

    # These are used to calculate energy and latency
    total_write_actions: Any = field(default=0)
    max_per_unit_write_actions: Any = field(default=0)
    total_read_actions: Any = field(default=0)
    max_per_unit_read_actions: Any = field(default=0)

    total_skipped_first_write_actions: Any = field(default=0)
    min_per_unit_skipped_first_write_actions: Any = field(default=0)
    total_skipped_first_read_actions: Any = field(default=0)
    min_per_unit_skipped_first_read_actions: Any = field(default=0)
    
    def repeat_temporal(self, factor: int, is_fully_relevant: bool) -> "BuffetStats":
        new = copy.copy(self)
        for attr in self.__dict__:
            if not attr.startswith(("total_", "max_", "min_")):
                continue
            if "skipped_first" in attr and not is_fully_relevant:
                continue  # First actions occur once per relevant iteration.
            if attr == "max_occupancy":
                continue  # Max occupancy is not affected by temporal loops above
            setattr(new, attr, getattr(new, attr) * factor)
        return new
    
    def repeat_spatial(self, factor: int, reuse_parent_accesses: bool) -> "BuffetStats":
        new = copy.copy(self)
        for attr in self.__dict__:
            if not attr.startswith(("total_", "max_", "min_")):
                continue
            if "parent" in attr and reuse_parent_accesses:
                continue  # If parent accesses are reused, no need to multiply
            if "per_unit" in attr:
                continue  # Spatial fanout doesn't affect per-unit stats
            if attr == "max_occupancy":
                continue  # Max occupancy is not affected by temporal loops above
            setattr(new, attr, getattr(new, attr) * factor)
        return new
    
    def max(self, **kwargs: Any):
        for key, value in kwargs.items():
            setattr(self, key, Max(getattr(self, key), value))
            
    def min(self, **kwargs: Any):
        for key, value in kwargs.items():
            setattr(self, key, Min(getattr(self, key), value))
    
    def __add__(self, other: "BuffetStats") -> "BuffetStats":
        new = copy.copy(self)
        for attr in self.__dict__:
            if attr.startswith("min_"):
                setattr(new, attr, min_nonzero(getattr(self, attr), getattr(other, attr)))
            elif attr.startswith("max_"):
                setattr(new, attr, Max(getattr(self, attr), getattr(other, attr)))
            elif attr.startswith("total_"):
                setattr(new, attr, getattr(self, attr) + getattr(other, attr))
        return new
    
    def __iadd__(self, other: "BuffetStats") -> "BuffetStats":
        new = self + other
        for key, value in new.__dict__.items():
            setattr(self, key, value)
        return self
    
    def net_total_read_actions(self) -> Any:
        return self.total_read_actions - self.total_skipped_first_read_actions
    
    def net_total_write_actions(self) -> Any:
        return self.total_write_actions - self.total_skipped_first_write_actions
    
    def net_max_per_unit_read_actions(self) -> Any:
        return self.max_per_unit_read_actions - self.min_per_unit_skipped_first_read_actions
    
    def net_max_per_unit_write_actions(self) -> Any:
        return self.max_per_unit_write_actions - self.min_per_unit_skipped_first_write_actions

@dataclass
class ComputeStats:
    total_ops: Any = field(default=0)
    max_per_unit_ops: Any = field(default=0)
    # "max" below refers to the longest latency of any iteration
    max_latency: Any = field(default=0)
    # Mapping from the loop-index (0 at top) to the latency of the first
    # iteration of that loop. "Max" because we may have loops above that and we
    # will take the maximum of the firsts.
    max_first_latency: dict[int, Any] = field(default_factory=dict)

    def repeat_temporal(self, factor: int) -> "ComputeStats":
        new = copy.copy(self)
        new.total_ops *= factor
        new.max_per_unit_ops *= factor
        new.max_latency *= factor
        # NOTE: max_first_latency does not change
        return new

    def repeat_spatial(self, factor: int) -> "ComputeStats":
        new = copy.copy(self)
        new.total_ops *= factor
        return new

    def __add__(self, other: "ComputeStats") -> "ComputeStats":
        new = copy.copy(self)
        new.total_ops += other.total_ops
        new.max_per_unit_ops += other.max_per_unit_ops
        new.max_latency += other.max_latency
        # max_first_latency is only ever updated across loops ABOVE the loop
        # for which we calculated that first latency, so we should MAX
        # new.max_first_latency = max_dict(self.max_first_latency, other.max_first_latency)
        return new

    def combine_temporal(self, other: "ComputeStats"):
        self.total_ops += other.total_ops
        self.max_per_unit_ops += other.max_per_unit_ops
        self.max_latency += other.max_latency
        # max_first_latency is only ever updated across loops ABOVE the loop
        # for which we calculated that first latency, so we should MAX
        # self.max_first_latency = max_dict(self.max_first_latency, other.max_first_latency)

    def combine_spatial(self, other: "ComputeStats"):
        self.total_ops += other.total_ops
        self.max_per_unit_ops = Max(self.max_per_unit_ops, other.max_per_unit_ops)
        self.max_latency = Max(self.max_latency, other.max_latency)
        # max_first_latency is only ever updated across loops ABOVE the loop
        # for which we calculated that first latency, so we should MAX
        # self.max_first_latency = max_dict(self.max_first_latency, other.max_first_latency)


@dataclass
class SummarizedAnalysisOutput:
    compute_stats: dict[Compute, ComputeStats] = field(default_factory=dict)

    buffet_stats: dict[Buffet, BuffetStats] = field(default_factory=dict)

    # Mapping [level, einsum] to the fanout
    fanout: dict = field(default_factory=dict)

    temporal_steps: dict = field(default_factory=dict)

    symbols: list = field(default_factory=list)

    def get_buffet_for_tensor(self, tensor: TensorName) -> Buffet:
        for buffet in self.buffet_stats:
            if buffet.tensor == tensor:
                return buffet
        raise ValueError(f"Buffet for tensor {tensor} not found")

    def max(self, **kwargs: Any):
        for key, value in kwargs.items():
            assert key in [
                'compute_stats',
                'stats',
                'fanout',
                'temporal_steps',
            ]
            previous = getattr(self, key)
            for k, v in value.items():
                previous.setdefault(k, {})
                for k2, v2 in v.items():
                    if k2 in previous[k]:
                        previous[k][k2] = Max(previous[k][k2], v2)
                    else:
                        previous[k][k2] = v2

    def get_child_buffet_stats(self, buffet: Buffet) -> BuffetStats:
        seen = False
        for child_buffet, child_stats in reversed(self.buffet_stats.items()):
            if not seen:
                seen = child_buffet == buffet
                continue
            if child_buffet.tensor == buffet.tensor:
                return child_stats
        return None

    def sum_buffet_stats_per_level(self) -> dict[str, BuffetStats]:
        result: dict[str, BuffetStats] = {}
        for buffet, stats in self.buffet_stats.items():
            result.setdefault(buffet.level, BuffetStats())
            result[buffet.level] += stats
        return result

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
    tensor_to_backer_id: dict[TensorName, int]

    is_copy_operation: TensorName | None

    job: Job

    # We track first latency for these nodes (should be Temporal)
    ids_to_track_first_latency: set[int] = field(default_factory=set)

def quick_insert_reservation_nodes(job: Job) -> list[MappingNode]:
    mapping = list(job.mapping.nodes)
    workload = job.spec.workload
    einsum_name = mapping[-1].einsum

    einsum = workload.einsums[einsum_name]
    all_tensors = einsum.input_tensors() | einsum.output_tensors()

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
    )
    insert_reservation_nodes(mapping, info)
    return Mapping(nodes=mapping)


def convert_to_copy(mapping: list[MappingNode], workload: Workload) -> tuple[list[MappingNode], dict[TensorName, int]]:
    mapping = copy.deepcopy(mapping)

    # Calculate this BEFORE we modify the mapping. We're going to have the copy source
    # tensor moving upward sometimes, and we don't want the backing tensor holder
    tensor_to_backer_id = get_tensor_to_backer_id(mapping)
    
    first_input_tensor = workload.einsums[mapping[-1].einsum].copy_source_tensor()
    
    for node in mapping:
        if isinstance(node, TensorHolder):
            if node.tensors:
                node.tensors = [first_input_tensor]
                node._lower = False

    to_remove = []
    i = 0
    while i < len(mapping):
        node = mapping[i]
        if isinstance(node, TensorHolder):
            j = i+1
            while j < len(mapping):
                node2 = mapping[j]
                if isinstance(node2, TensorHolder) and node.component == node2.component:
                    mapping.pop(j)
                else:
                    j += 1
        i += 1
    mapping = [node for node in mapping if node not in to_remove]
    
    return mapping, tensor_to_backer_id

    
    
def analyze_reuse_and_add_reservations_to_mapping(
    job: Job,
) -> SummarizedAnalysisOutput:
    mapping = job.mapping.nodes
    workload = job.spec.workload
    einsum_name = mapping[-1].einsum
    einsum_shape = get_rank_variable_bounds(workload, einsum_name)
    symbols = insert_sympy_symbols(mapping, job)

    is_copy_operation = workload.einsums[einsum_name].is_copy_operation
    if is_copy_operation:
        mapping, tensor_to_backer_id = convert_to_copy(mapping, workload)
        # We're working with a new mapping at this point, so we need to add reservations
        # to the job mapping.
        job.mapping = quick_insert_reservation_nodes(job)
    else:
        tensor_to_backer_id = get_tensor_to_backer_id(mapping)

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
        tensor_to_backer_id=tensor_to_backer_id,
        is_copy_operation=is_copy_operation,
        job=job,
    )

    insert_reservation_nodes(mapping, info)

    result = analyze_node(0, einsum_shape, info)

    result.symbols = symbols

    return result


def get_tensor_to_backer_id(mapping: Mapping):
    tensor_to_ids: dict[TensorName, set[int]] = {}
    for node in mapping:
        if isinstance(node, TensorHolder):
            for tensor in node.tensors:
                if tensor in tensor_to_ids:
                    continue
                tensor_to_ids[tensor] = id(node)
    return tensor_to_ids


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
            raise ValueError(f'Unknown relevancy {relevancy}')

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

def insert_reservation_nodes(mapping, info: AnalysisInfo):
    trackers: list[ReservationAnalysisTracker] = []
    einsum = info.workload.einsums[mapping[-1].einsum]
    non_intermediate_tensors = einsum.tensor_names - info.workload.intermediate_tensor_names
    seen_tensors = set()  # reservation for top-level buffets cannot be lowered

    n_nodes = len(mapping)
    i = 0
    while i < n_nodes:
        node = mapping[i]
        to_remove = []
        if isinstance(node, Temporal):
            rank = node.rank_variable
            for tracker_idx, tracker in enumerate(trackers):
                relevancy = info.tensor_to_relevancy[tracker.buffet.tensor]
                tracker.track_temporal_loop(relevancy[rank], node)

                if tracker.should_stop:
                    to_remove.append(tracker_idx)
        elif isinstance(node, Spatial):
            rank = node.rank_variable
            for tracker_idx, tracker in enumerate(trackers):
                relevancy = info.tensor_to_relevancy[tracker.buffet.tensor]
                tracker.track_spatial_loop(relevancy[rank], node)

                if tracker.should_stop:
                    to_remove.append(tracker_idx)
        elif isinstance(node, TensorHolder):
            for tensor in node.tensors:
                tensor = TensorName(tensor)
                buffet = Buffet(tensor, mapping[-1].einsum, node.component)
                trackers.append(ReservationAnalysisTracker(buffet, node))
                if (
                    not node._lower
                    or
                    (tensor not in seen_tensors and tensor in non_intermediate_tensors)
                ):
                    seen_tensors.add(tensor)
                    to_remove.append(len(trackers)-1)
                    trackers[-1].is_fill_level = True
                    trackers[-1].insert_reservation_under = True
                    trackers[-1].insert_fill_under = True
        elif isinstance(node, mapping_spec.Compute):
            for tracker_idx, tracker in enumerate(trackers):
                tracker.track_compute()

                if tracker.should_stop:
                    to_remove.append(tracker_idx)
        elif isinstance(node, Reservation):
            pass
        else:
            raise NotImplementedError(f"Unknown node type {type(node)}")

        # NOFILL: fill_insert_below = []
        # NOFILL: fill_insert_above = []
        # NOFILL: for tracker in trackers:
        # NOFILL:     if not tracker.is_fill_level:
        # NOFILL:         continue
        # NOFILL:     buffet = tracker.buffet
        # NOFILL:     if tracker.insert_fill_under:
        # NOFILL:         fill_insert_below.append(node)
        # NOFILL:     else:
        # NOFILL:         fill_insert_above.append(node)

        reservation_insert_below = []
        reservation_insert_above = []
        for tracker_idx in reversed(to_remove):
            tracker = trackers.pop(tracker_idx)
            buffet = tracker.buffet
            node = Reservation(purposes=[buffet.tensor], resource=buffet.level)
            if tracker.insert_reservation_under:
                reservation_insert_below.append(node)
            else:
                reservation_insert_above.append(node)

        # The order of these for loops is important. Reservation must be below fill.
        for node in reservation_insert_below:
            mapping.insert(i+1, node)
        # NOFILL: for node in fill_insert_below:
        # NOFILL:     mapping.insert(i+1, node)
        for node in reservation_insert_above:
            mapping.insert(i, node)
        # NOFILL: for node in fill_insert_above:
        # NOFILL:     mapping.insert(i, node)

        i += 1
        n_nodes = len(mapping)


def analyze_node(node_idx, current_shape, info: AnalysisInfo) -> SummarizedAnalysisOutput:
    node = info.mapping[node_idx]
    class2analysis_function = {
        Temporal: analyze_temporal,
        Spatial: analyze_spatial,
        Storage: analyze_storage,
        Reservation: analyze_reservation,
        mapping_spec.Compute: analyze_compute,
        ProcessingStage: analyze_processing_stage,
    }
    if type(node) not in class2analysis_function:
        raise TypeError(f"Unknown node type {type(node)}")
    return class2analysis_function[type(node)](node_idx, current_shape, info)

def analyze_temporal(node_idx,
                     current_shape,
                     info: AnalysisInfo) -> SummarizedAnalysisOutput:
    mapping = info.mapping
    node = mapping[node_idx]
    stride_and_shape = get_stride_and_tile_shape(node, current_shape, node_idx)

    result_accumulator = SummarizedAnalysisOutput()

    first_latency = None

    def handle_repeated_value(repeated_shape):
        nonlocal first_latency
        shape_value = repeated_shape.value
        shape_repeats = repeated_shape.repeats

        child_shape = current_shape.copy()
        child_shape[node.rank_variable] = shape_value
        
        child_result = analyze_node(node_idx+1, child_shape, info)

        accumulated_buffet_stats = result_accumulator.buffet_stats
        for buffet, stats in child_result.buffet_stats.items():
            relevancy = info.tensor_to_relevancy[buffet.tensor][node.rank_variable]

            accumulated_stats = accumulated_buffet_stats.setdefault(buffet, BuffetStats())
            accumulated_stats += stats.repeat_temporal(shape_repeats, is_fully_relevant=isinstance(relevancy, Relevant))
            accumulated_stats.n_loops_above = stats.n_loops_above + 1
            
        for einsum, child_steps in child_result.temporal_steps.items():
            if einsum not in result_accumulator.temporal_steps:
                result_accumulator.temporal_steps[einsum] = 0
            result_accumulator.temporal_steps[einsum] += child_steps*shape_repeats

        result_accumulator.max(fanout=child_result.fanout)

        for key in child_result.compute_stats:
            if first_latency is None:
                first_latency = child_result.compute_stats[key].max_latency

            compute_stats = result_accumulator.compute_stats.setdefault(key, ComputeStats())
            compute_stats += child_result.compute_stats[key].repeat_temporal(shape_repeats)
            result_accumulator.compute_stats[key] = compute_stats

    if node_idx in info.ids_to_track_first_latency:
        for compute_stat in result_accumulator.compute_stats.values():
            # Should be the first time we store this value
            assert node_idx not in compute_stat.max_first_latency
            compute_stat.max_first_latency[node_idx] = first_latency

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
    node_dim = node.name
    stride_and_shape = get_stride_and_tile_shape(node, current_shape, node_idx)

    result_accumulator = SummarizedAnalysisOutput()

    def handle_repeated_value(repeated_shape):
        shape_value = repeated_shape.value
        shape_repeats = repeated_shape.repeats

        child_shape = current_shape.copy()
        child_shape[node.rank_variable] = shape_value

        child_result = analyze_node(node_idx+1, child_shape, info)

        accumulated_buffet_stats = result_accumulator.buffet_stats
        child_stats = list(child_result.buffet_stats.items())
        for i, (buffet, buffet_stats) in enumerate(child_stats):
            stats = buffet_stats
            accumulated_stats = accumulated_buffet_stats.setdefault(buffet, BuffetStats())
            relevancy = info.tensor_to_relevancy[buffet.tensor][rank_var]

            # Reuse parent accesses only:
            # - Irrelevant loops
            # - The outermost level that holds the tensor (the one whose parent accesses
            #   will be going through the network)
            last_buffet = True
            for other_buffet, _ in child_stats[i+1:]:
                if other_buffet.tensor == buffet.tensor:
                    last_buffet = False
                    break
            reuse_parent_accesses = last_buffet and isinstance(relevancy, Irrelevant)

            accumulated_stats += stats.repeat_spatial(
                shape_repeats,
                reuse_parent_accesses=reuse_parent_accesses
            )
            accumulated_stats.n_loops_above = stats.n_loops_above + 1

        for einsum, child_steps in child_result.temporal_steps.items():
            if einsum not in result_accumulator.temporal_steps:
                result_accumulator.temporal_steps[einsum] = child_steps
            else:
                result_accumulator.temporal_steps[einsum] = Max(
                    result_accumulator.temporal_steps[einsum],
                    child_steps
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
            compute_stats = result_accumulator.compute_stats.setdefault(key, ComputeStats())
            # TODO: If check omitted. This was in the original code, check history if needed.
            compute_stats.combine_spatial(child_result.compute_stats[key].repeat_spatial(shape_repeats))

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

def get_total_to_per_unit(total, max_per_unit):
    if total == 0 and max_per_unit != 0:
        raise ValueError(f"total is 0 but max_per_unit is {max_per_unit}")
    if total == 0:
        return 1
    return max_per_unit / total

def has_parent_tensor_holder(tensor: TensorName, node_idx: int, info: AnalysisInfo) -> bool:
    for node in info.mapping[:node_idx]:
        if isinstance(node, TensorHolder) and tensor in node.tensors:
            return True
    return False

def find_component_object(component: str, flattened_arch: list[arch.Leaf]) -> arch.TensorHolder:
    for node in flattened_arch:
        if node.name == component:
            return node
    raise ValueError(f"Component {component} not found in flattened arch")

def analyze_storage(node_idx, current_shape, info: AnalysisInfo, _propagate_child_results: bool = False):
    mapping = info.mapping
    einsum_name = mapping[-1].einsum
    node = mapping[node_idx]

    child_result = analyze_node(node_idx+1, current_shape, info)

    for tensor in node.tensors:
        tensor = TensorName(tensor)
        buffet = Buffet(tensor, einsum_name, node.component)

        stats = child_result.buffet_stats.setdefault(buffet, BuffetStats())
        backer_id = info.tensor_to_backer_id[tensor]
        is_backing = backer_id == id(node)
        below_backing = backer_id in [id(m) for m in mapping[:node_idx]]

        projection = info.einsum_tensor_to_projection[(einsum_name, tensor)]

        fills = compute_dense_tile_occupancy(projection, current_shape)

        child = child_result.get_child_buffet_stats(buffet)
        inherit_from_child = _propagate_child_results and child is not None

        # ==============================================================================
        # Calculate the total fills and reads to parent. These propagate upward.
        # ==============================================================================
        if has_parent_tensor_holder(tensor, node_idx, info):
            # Initial fetch: If we're below the backing storage, fetch data from above
            # at the beginning.
            if not is_backing and below_backing:
                stats.total_reads_to_parent += child.total_reads_to_parent if inherit_from_child else fills
                stats.max_per_parent_reads_to_parent += child.max_per_parent_reads_to_parent if inherit_from_child else fills

            # Data writeback. Do not writeback if it's a copy operation and we're below
            # the backing storage; data only flows upward.
            
            # Writeback occurs in two cases: 
            # - We're at or above the backing storage, so we need to propagate our
            #   results upward to any storage nodes that will need this data.
            # - This is a written tensor, so we need to write back the written data.
            if tensor in info.workload.tensors_written_by_einsum(einsum_name) or not below_backing:
                stats.total_writes_to_parent += child.total_writes_to_parent if inherit_from_child else fills
                stats.max_per_parent_writes_to_parent += child.max_per_parent_writes_to_parent if inherit_from_child else fills

            # For read+write tensors, we skip the first fill because the data will be
            # initialized with a zero value.
            if tensor in info.workload.tensors_written_by_einsum(einsum_name):
                stats.total_skipped_first_reads_to_parent += child.total_skipped_first_reads_to_parent if inherit_from_child else fills
                stats.min_per_parent_skipped_first_reads_to_parent += child.min_per_parent_skipped_first_reads_to_parent if inherit_from_child else fills

        # ==============================================================================
        # Convert to actions. These are not used used upward; they are used to get
        # energy and latency.
        # ==============================================================================
        component_object = find_component_object(node.component, info.job.flattened_arch)
        datawidth = component_object.attributes.datawidth[tensor]
        bits_per_read = component_object.actions["read"].arguments.bits_per_action
        bits_per_write = component_object.actions["write"].arguments.bits_per_action
        read_scale = datawidth / bits_per_read
        write_scale = datawidth / bits_per_write

        # ==========================
        # Data exchanges with parent
        stats.total_write_actions += stats.total_reads_to_parent * write_scale
        stats.max_per_unit_write_actions += stats.total_reads_to_parent * write_scale

        # Comment this to have the final writeback to a buffer hit both that buffer and
        # go directly to the parent without incurring another read from the buffer.
        stats.total_read_actions += stats.total_writes_to_parent * read_scale
        stats.max_per_unit_read_actions += stats.total_writes_to_parent * read_scale

        stats.total_skipped_first_write_actions += stats.total_skipped_first_reads_to_parent * write_scale
        stats.min_per_unit_skipped_first_write_actions += stats.min_per_parent_skipped_first_reads_to_parent * write_scale

        # ========================
        # Data exchanges with peer
        stats.total_read_actions += stats.total_reads_to_peer * read_scale
        stats.total_write_actions += stats.total_reads_to_peer * write_scale

        # =========================
        # Data exchanges with child
        if child is not None:
            stats.total_read_actions += child.total_reads_to_parent * read_scale
            stats.max_per_unit_read_actions += child.max_per_parent_reads_to_parent * read_scale

            stats.total_write_actions += child.total_writes_to_parent * write_scale
            stats.max_per_unit_write_actions += child.max_per_parent_writes_to_parent * write_scale

            # Skip first read
            stats.total_skipped_first_read_actions += child.total_skipped_first_reads_to_parent * read_scale
            stats.min_per_unit_skipped_first_read_actions += child.min_per_parent_skipped_first_reads_to_parent * read_scale
        
    return child_result


def analyze_processing_stage(node_idx, current_shape, info: AnalysisInfo):
    mapping = info.mapping
    einsum_name = mapping[-1].einsum
    node = mapping[node_idx]
    storage_result = analyze_storage(node_idx, current_shape, info, _propagate_child_results=True)
    for tensor in node.tensors:
        buffet = Buffet(tensor, einsum_name, node.component)
        stats = storage_result.buffet_stats[buffet]
        stats.max_occupancy = 0
    return storage_result

def analyze_reservation(node_idx, current_shape, info: AnalysisInfo):
    mapping = info.mapping
    einsum_name = mapping[-1].einsum
    node = mapping[node_idx]

    child_result = analyze_node(node_idx+1, current_shape, info)

    tensor = TensorName(node.purpose)
    
    buffet = Buffet(tensor, einsum_name, node.resource)

    # Reservation nodes are the first to produce stats for a buffet
    assert buffet not in child_result.buffet_stats

    stats = BuffetStats()
    projection = info.einsum_tensor_to_projection[(einsum_name, tensor)]
    component_object = find_component_object(node.resource, info.job.flattened_arch)
    datawidth = component_object.attributes.datawidth[tensor]
    stats.max_occupancy = \
        compute_dense_tile_occupancy(projection, current_shape) * datawidth
    child_result.buffet_stats[buffet] = stats

    fanout_key = (node.resource, einsum_name)
    if fanout_key not in child_result.fanout:
        child_result.fanout[fanout_key] = {}

    return child_result


def analyze_fill(node_idx, current_shape, info: AnalysisInfo) -> SummarizedAnalysisOutput:
    mapping = info.mapping
    einsum_name = mapping[-1].einsum
    node = mapping[node_idx]

    child_result = analyze_node(node_idx+1, current_shape, info)
    return child_result

    tensor = node.tensor
    buffet = Buffet(tensor, mapping[-1].einsum, node.component)

    stats = child_result.buffet_stats[buffet]
    projection = info.einsum_tensor_to_projection[(einsum_name, tensor)]
    stats.total_fills = \
        compute_dense_tile_occupancy(projection, current_shape)

    stats.max_per_unit_fills = stats.total_fills
    stats.total_reads_to_parent = stats.total_fills
    stats.max_per_parent_reads_to_parent = \
        stats.total_reads_to_parent

    return child_result


def analyze_compute(node_idx,
                    current_shape,
                    info: AnalysisInfo) -> SummarizedAnalysisOutput:
    einsum = info.mapping[-1].einsum
    node = info.mapping[node_idx]
    compute_node: arch.Compute = info.job.flattened_arch[-1]
    
    computes = 0 if info.is_copy_operation else 1

    result_accumulator = SummarizedAnalysisOutput()

    result_accumulator.temporal_steps[einsum] = computes
    result_accumulator.compute_stats[Compute(einsum, node.compute)] = ComputeStats(
        computes,
        computes,
        1 / compute_node.attributes.computes_per_cycle,
        1 / compute_node.attributes.computes_per_cycle
    )
    
    if info.is_copy_operation:
        return result_accumulator

    for tensor in info.all_tensors:
        buffet = Buffet(tensor, einsum, node.compute)
        stats = BuffetStats()
        stats.total_reads_to_parent = 1
        stats.max_per_parent_reads_to_parent = 1
        if tensor in info.workload.tensors_written_by_einsum(einsum):
            stats.total_writes_to_parent = 1
            stats.max_per_parent_writes_to_parent = 1
            stats.total_skipped_first_reads_to_parent = 1
            stats.min_per_parent_skipped_first_reads_to_parent = 1
        stats.max_occupancy = 1
        result_accumulator.buffet_stats[buffet] = stats

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

    stride = node.stride
    initial_tile_shape = node.initial_tile_shape

    if initial_tile_shape is None:
        if node.assume_perfect_factor or known_perfect_factor(stride, rank_shape):
            factor = rank_shape / stride
            return StrideAndShape(stride, RepeatedValue(stride, factor))
        else:
            factor = sympy.ceiling(rank_shape / sympy.Min(stride, rank_shape))
            return make_possibly_different_last(stride, factor, rank_shape)
        
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
    # if node.tile_shape is not None:
    #     tile_shape = node.tile_shape

    #     if node.assume_perfect_factor or known_perfect_factor(tile_shape, rank_shape):
    #         factor = rank_shape / tile_shape
    #         return StrideAndShape(tile_shape, RepeatedValue(tile_shape, factor))
    #     else:
    #         factor = sympy.ceiling(rank_shape / sympy.Min(tile_shape, rank_shape))
    #         return make_possibly_different_last(tile_shape, factor, rank_shape)
    # elif node.loop_bound is not None:
    #     factor = node.loop_bound

    #     if node.assume_perfect_factor or known_perfect_factor(factor, rank_shape):
    #         tile_shape = rank_shape / factor
    #         return StrideAndShape(tile_shape, RepeatedValue(tile_shape, factor))
    #     else:
    #         tile_shape = sympy.ceiling(rank_shape / sympy.Min(rank_shape, factor))
    #         return make_possibly_different_last(tile_shape, factor, rank_shape)
    
    # elif node.tile_pattern is not None:
    #     stride = node.tile_pattern.stride
    #     initial_tile_shape = node.tile_pattern.initial_tile_shape
    #     tile_shape = node.tile_pattern.tile_shape

    #     if initial_tile_shape is not None:
    #         middle_shape_factor = sympy.ceiling((rank_shape - initial_tile_shape)/stride)
    #         # TODO: sometimes last_shape is 0, causing numerical instability
    #         # Currently, we are sometimes rounding up last shape.
    #         # last_shape = rank_shape - initial_tile_shape - stride*middle_shape_factor
    #         # has_last_shape = sympy.ceiling(last_shape/(last_shape+1))
    #         return StrideAndShape(
    #             stride,
    #             SequenceOfRepatedvalues([
    #                 RepeatedValue(initial_tile_shape, 1),
    #                 RepeatedValue(stride, middle_shape_factor),
    #                 # RepeatedValue(last_shape+0.01, has_last_shape)
    #             ])
    #         )


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


def insert_sympy_symbols(mapping: list[MappingNode], job: Job):
    loop_idx = 0
    symbols = []
    for i, node in enumerate(mapping):
        if not isinstance(node, Iteration):
            continue

        stride_halos = set()
        for t in job.spec.workload.einsums[job.einsum_name].tensors:
            for (rank, rank_variable), (stride, halo) in job.stride_and_halo[t].items():
                if rank_variable == node.rank_variable:
                    stride_halos.add((stride, halo))

        simple = len(stride_halos) <= 1 and next(iter(stride_halos)) == (1, 0)

        # TODO: Check for 0 < shape < 1 for loop bound target
        if node.stride == SYMBOL:
            stride = sympy.symbols(f'stride{loop_idx}', positive=True, integer=True)
            symbols.append(stride)
            node.stride = stride
        if simple: # Just use the stride!
            node.initial_tile_shape = None
        else:
            if node.initial_tile_shape == SYMBOL:
                initial_tile_shape = sympy.symbols(f'initial{loop_idx}', positive=True, integer=True)
                symbols.append(initial_tile_shape)
                node.initial_tile_shape = initial_tile_shape

        assert node.calculated_n_iterations is None, \
            "Number of iterations is derived from the model. Do not set it!"
        node.calculated_n_iterations = sympy.symbols(f'n_iterations{loop_idx}', positive=True, integer=True)

        loop_idx += 1

    return symbols
