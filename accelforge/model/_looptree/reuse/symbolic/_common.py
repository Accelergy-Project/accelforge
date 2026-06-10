from dataclasses import dataclass, field
from accelforge.frontend import arch
from accelforge.frontend.mapping import Mapping, TensorHolder, Loop
from typing import Any

from accelforge.frontend.arch._flattened_arch import FlattenedArch
from accelforge.frontend.workload import (
    Workload,
    TensorName,
)
from accelforge.frontend._workload_isl._symbolic import Irrelevant
from accelforge.model._looptree.reuse.symbolic.mapping_utils import (
    DataMovementConnections,
)

from accelforge.mapper.FFM._make_pmappings.pmapper_job import Job

import symengine as se


@dataclass
class AnalysisInfo:
    """Information needed within the analysis step by multiple functions that
    can be computed once at the beginning.
    """

    mapping: Mapping
    workload: Workload
    full_rank_variable_shapes: dict
    all_tensors: set
    current_tensor: TensorName | None

    einsum_tensor_to_projection: dict
    tensor_to_relevancy: dict
    tensor_to_backer_id: dict[TensorName, int]

    is_copy_operation: TensorName | None

    job: Job

    # Rank variables that appear alone (not in an expression) in every tensor access.
    # Eligible for imperfect tile shapes when paired with `_may_cause_imperfect`.
    simple_rank_variables: set = field(default_factory=set)

    tensor_to_reservation_backer_id: dict[TensorName, int] = field(default_factory=dict)

    data_movement_connections: DataMovementConnections = None

    # For a given tensor, we may rearrange irrelevant loops, which nominally would
    # affect the iteration count and tile shapes. However, they're irrelevant, so we can
    # just track the iteration count for the canonical order and that's sufficient.
    precomputed_iterations: dict[int, Any] = field(default_factory=dict)

    # True during the initial pass that records loop iteration counts.
    is_recording_iterations: bool = False

    tensor_rank_variables: set = field(default_factory=set)

    # We track first latency for these nodes (should be Temporal)
    last_temporal_node_idx: int = None
    """
    node idx of the last (above) temporal node
    """
    idxs_to_track_first_latency: set[int] = field(default_factory=set)
    """
    node idxs for which we track first latency
    """


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


def has_parent_tensor_holder(
    tensor: TensorName, node_idx: int, info: AnalysisInfo
) -> bool:
    for node in info.mapping[:node_idx]:
        if isinstance(node, TensorHolder) and tensor in node.tensors:
            return True
    return False


def loop_stride_and_shape(node, current_shape, node_idx, info):
    """Get the stride-and-shape for a loop node.

    During the initial analysis pass (is_recording_iterations), records each
    loop's iteration count into info.precomputed_iterations.

    During per-tensor passes, loops whose rank variable is irrelevant to the
    tensor may have been reordered relative to other loops on the same rank
    variable, giving them a wrong current_shape. For those loops, the
    iteration count is replaced with the value recorded during the initial
    pass.
    """
    tensor = info.current_tensor

    if not info.is_recording_iterations:
        relevancy = info.tensor_to_relevancy[tensor][node.rank_variable]
        if isinstance(relevancy, Irrelevant):
            n_iters = info.precomputed_iterations[id(node)]
            stride = node.tile_shape
            return StrideAndShape(stride, RepeatedValue(stride, n_iters))

    is_simple = node.rank_variable in info.simple_rank_variables

    # For a rank variable with any `_may_cause_imperfect` loop, only the outermost loop
    # is allowed to imperfectly factorize. Only simple rank variables can be imperfectly
    # factorized because the average-tile-shape assumptions break down otherwise.
    def _mine(l):
        return isinstance(l, Loop) and l.rank_variable == node.rank_variable

    loops_before = [n for n in info.mapping[:node_idx] if _mine(n)]
    loops_after = [n for n in info.mapping[node_idx:] if _mine(n)]
    outermost = not loops_before
    imperfect = any(l._may_cause_imperfect for l in loops_after) and outermost
    if imperfect:
        assert is_simple, "Only simple rank variables can have padding"

    stride_and_shape = get_stride_and_tile_shape(
        node,
        current_shape,
        node_idx,
        info,
        assume_perfect_factor=not imperfect,
        is_simple=is_simple,
        use_average_tiles=imperfect,
    )

    if info.is_recording_iterations:
        iterations = stride_and_shape.shape.repeats

        # If imperfect, the residuals net out to ceil(rank_shape / tile_shape) /
        # outer_iters. This only holds because (a) the rank variable is simple, and (b)
        # only the outermost loop on this rankq variable is imperfect.
        pc_iterations = info.precomputed_iterations
        k = id(node)
        if imperfect:
            assert is_simple
            rank_shape = info.full_rank_variable_shapes[node.rank_variable]
            outer_iters = 1
            for prev_node in loops_before:
                outer_iters *= pc_iterations[id(prev_node)]
            iterations = se.ceiling(rank_shape / node.tile_shape) / outer_iters
        pc_iterations[k] = pc_iterations.get(k, 0) + iterations

    return stride_and_shape


@dataclass
class RepeatedValue[T]:
    value: T
    repeats: int


@dataclass
class SequenceOfRepatedvalues[T]:
    sequence: list[RepeatedValue[T]]

    @property
    def repeats(self):
        return sum(rv.repeats for rv in self.sequence)


@dataclass
class StrideAndShape:
    stride: any
    shape: any


def get_stride_and_tile_shape(
    node: Loop,
    full_shape,
    n: int,
    info: AnalysisInfo,
    assume_perfect_factor: bool,
    is_simple: bool,
    use_average_tiles: bool = False,
) -> StrideAndShape:
    rank = node.rank_variable
    rank_shape = full_shape[rank]

    stride = node.tile_shape
    initial_tile_shape = node.initial_tile_shape

    # PERFECT:
    # - Node shape = stride
    # - # Iterations = total shape / stride
    # IMPERFECT:
    # - Node shape = stride
    # - # Iterations = ceil(total shape / stride)

    if is_simple:
        assert initial_tile_shape is None

        perfect = assume_perfect_factor or known_perfect_factor(stride, rank_shape)
        if perfect or use_average_tiles:
            factor = rank_shape / stride
            if type(factor) is float and factor == int(factor):
                factor = int(factor)
            return StrideAndShape(stride, RepeatedValue(stride, factor))
        else:
            raise NotImplementedError("BUG")
            factor = se.ceiling(rank_shape / stride)
            return make_possibly_different_last(stride, factor, rank_shape)

    assert assume_perfect_factor

    if initial_tile_shape is None:
        factor = rank_shape / stride
        if type(factor) is float and factor == int(factor):
            factor = int(factor)
        return StrideAndShape(stride, RepeatedValue(stride, factor))

    middle_shape_factor = se.ceiling((rank_shape - initial_tile_shape) / stride)
    # TODO: sometimes last_shape is 0, causing numerical instability
    # Currently, we are sometimes rounding up last shape.
    # last_shape = rank_shape - initial_tile_shape - stride*middle_shape_factor
    # has_last_shape = sympy.ceiling(last_shape/(last_shape+1))
    return StrideAndShape(
        stride,
        SequenceOfRepatedvalues(
            [
                RepeatedValue(initial_tile_shape, 1),
                RepeatedValue(stride, middle_shape_factor),
                # RepeatedValue(last_shape+0.01, has_last_shape)
            ]
        ),
    )


def known_perfect_factor(divisor, full_shape):
    return (
        isinstance(divisor, int)
        and isinstance(full_shape, int)
        and full_shape % divisor == 1
    )


def make_possibly_different_last(common_tile_shape, factor, full_shape):
    last_shape = full_shape - common_tile_shape * (factor - 1)
    all_shapes = SequenceOfRepatedvalues(
        [RepeatedValue(common_tile_shape, factor - 1), RepeatedValue(last_shape, 1)]
    )
    return StrideAndShape(common_tile_shape, all_shapes)
