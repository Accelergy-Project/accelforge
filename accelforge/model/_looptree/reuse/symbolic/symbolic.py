import copy
from dataclasses import dataclass, field
import itertools
from accelforge.frontend.mapping import (
    Compute,
    Mapping,
    Nested,
    Pipeline,
    Toll,
    Reservation,
    Sequential,
    Spatial,
    Split,
    Storage,
    Temporal,
)
from typing import Any, List

from accelforge.frontend import arch
import accelforge.frontend.mapping as mapping_spec
from accelforge.frontend.mapping import (
    Mapping,
    MappingNode,
    Nested,
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
    isl_expression_has_variable,
)
from accelforge.frontend._workload_isl._isl import get_rank_variable_bounds
from accelforge.frontend._workload_isl._symbolic import (
    get_projection_expr,
    get_rank_variable_relevancy,
    compute_dense_tile_occupancy,
    Irrelevant,
    Relevant,
    PartiallyRelevant,
)

from accelforge.model._looptree.types import Buffet

from accelforge.mapper.FFM._make_pmappings.pmapper_job import Job
from accelforge.util._sympy.broadcast_max import Min, Max
from accelforge.mapper.FFM._pareto_df.df_convention import iterations2col

import sympy

SYMBOL = "symbol"
IMPERFECT = False

PRINT_FORMULAS = False


@dataclass(eq=True, frozen=True)
class Compute:
    einsum: str
    level: str


class Uninitialized:
    def __init__(self):
        pass

    def __str__(self):
        return "Uninitialized"

    def __repr__(self):
        return "Uninitialized()"

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
    assert isinstance(new, dict)
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
    _n_loops_above: int = field(default=0)

    # These are used to calculate energy and latency
    total_write_actions: Any = field(default=0)
    max_per_unit_write_actions: Any = field(default=0)
    total_read_actions: Any = field(default=0)
    max_per_unit_read_actions: Any = field(default=0)

    total_skipped_first_write_actions: Any = field(default=0)
    min_per_unit_skipped_first_write_actions: Any = field(default=0)
    total_skipped_first_read_actions: Any = field(default=0)
    min_per_unit_skipped_first_read_actions: Any = field(default=0)

    persistent: bool = field(default=False)

    @property
    def n_loops_above(self) -> int:
        if self.persistent:
            return -1
        return self._n_loops_above

    @n_loops_above.setter
    def n_loops_above(self, value: int):
        self._n_loops_above = value

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
        """
        Duplicate attributes to account for spatial fanout.

        Paramters
        ---------
        factor:
            duplication factor
        reuse_parent_access:
            whether to reuse accesses to parent (if True, multicast is assumed
            and accesses to parents are not duplicated).
        """
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
                setattr(
                    new, attr, min_nonzero(getattr(self, attr), getattr(other, attr))
                )
            elif attr.startswith("max_"):
                setattr(new, attr, Max(getattr(self, attr), getattr(other, attr)))
            elif attr.startswith("total_"):
                setattr(new, attr, getattr(self, attr) + getattr(other, attr))
            elif getattr(self, attr) is None:
                setattr(new, attr, getattr(other, attr))
            elif getattr(other, attr) is None:
                setattr(new, attr, getattr(self, attr))
            else:
                assert getattr(self, attr) == getattr(
                    other, attr
                ), f"BUG: {attr} is different. self: {getattr(self, attr)} other: {getattr(other, attr)}"
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
        return (
            self.max_per_unit_read_actions
            - self.min_per_unit_skipped_first_read_actions
        )

    def net_max_per_unit_write_actions(self) -> Any:
        return (
            self.max_per_unit_write_actions
            - self.min_per_unit_skipped_first_write_actions
        )


def blank_buffet_stats() -> BuffetStats:
    stats = BuffetStats()
    stats.n_loops_above = None  # Inherit from whoever is added to this
    return stats


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
        new.max_first_latency = max_dict(
            self.max_first_latency, other.max_first_latency
        )  # FIRST LATENCY
        return new

    def combine_temporal(self, other: "ComputeStats"):
        self.total_ops += other.total_ops
        self.max_per_unit_ops += other.max_per_unit_ops
        self.max_latency += other.max_latency
        # max_first_latency is only ever updated across loops ABOVE the loop
        # for which we calculated that first latency, so we should MAX
        self.max_first_latency = max_dict(
            self.max_first_latency, other.max_first_latency
        )  # FIRST LATENCY

    def combine_spatial(self, other: "ComputeStats"):
        self.total_ops += other.total_ops
        self.max_per_unit_ops = Max(self.max_per_unit_ops, other.max_per_unit_ops)
        self.max_latency = Max(self.max_latency, other.max_latency)
        # max_first_latency is only ever updated across loops ABOVE the loop
        # for which we calculated that first latency, so we should MAX
        self.max_first_latency = max_dict(
            self.max_first_latency, other.max_first_latency
        )  # FIRST LATENCY


@dataclass
class SymbolicAnalysisOutput:
    compute_stats: dict[Compute, ComputeStats] = field(default_factory=dict)

    buffet_stats: dict[Buffet, BuffetStats] = field(default_factory=dict)

    # Mapping [level, einsum] to the fanout
    fanout: dict[(Buffet, str), int] = field(default_factory=dict)

    # Mapping [einsum] to the number of temporal steps
    temporal_steps: dict[str, int] = field(default_factory=dict)

    symbols: list[sympy.Symbol] = field(default_factory=list)

    # tensor to the mapping for that particular tensor
    tensor2mapping: dict[TensorName, Mapping] = field(default_factory=dict)

    def get_buffet_for_tensor(self, tensor: TensorName) -> Buffet:
        for buffet in self.buffet_stats:
            if buffet.tensor == tensor:
                return buffet
        raise ValueError(f"Buffet for tensor {tensor} not found")

    def max(self, **kwargs: Any):
        for key, value in kwargs.items():
            assert key in [
                "compute_stats",
                "stats",
                "fanout",
                "temporal_steps",
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
            result.setdefault(buffet.level, blank_buffet_stats())
            result[buffet.level] += stats
        return result

    def add_buffet_stats_and_symbols(self, other: "SymbolicAnalysisOutput"):
        assert not (set(self.buffet_stats) & set(other.buffet_stats)), "BUG"
        self.buffet_stats.update(other.buffet_stats)
        # if self.temporal_steps != other.temporal_steps:
        #     print(f'Temporal steps are different.')
        #     print(f'\tmine:  {self.temporal_steps}')
        #     print(f'\tother: {other.temporal_steps}')
        # assert self.temporal_steps == other.temporal_steps, "BUG"
        self.temporal_steps.update(other.temporal_steps)
        self.symbols.extend([s for s in other.symbols if s not in self.symbols])
        # Assert compute stats are the same
        # assert self.compute_stats == other.compute_stats, "BUG"


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

    tensor_to_reservation_backer_id: dict[TensorName, int] = field(default_factory=dict)

    # We track first latency for these nodes (should be Temporal)
    last_temporal_node_idx: int = None
    """
    node idx of the last (above) temporal node
    """
    idxs_to_track_first_latency: set[int] = field(default_factory=set)
    """
    node idxs for which we track first latency
    """


def quick_insert_reservation_nodes(job: Job) -> list[MappingNode]:
    mapping = list(job.mapping.nodes)
    workload = job.spec.workload

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
    )
    insert_reservation_nodes(mapping, info, job.fusable_tensors)
    m = Mapping(nodes=mapping)
    m._n_loop_orders = job.mapping._n_loop_orders
    return m


def convert_to_copy(
    mapping: list[MappingNode], workload: Workload
) -> tuple[list[MappingNode], dict[TensorName, int]]:
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
        i += 1
    mapping = [node for node in mapping if node not in to_remove]

    return mapping, tensor_to_backer_id


def analyze_reuse_and_add_reservations_to_mapping(
    job: Job,
    add_reservations: bool = True,
) -> SymbolicAnalysisOutput:
    mapping = job.mapping.nodes
    workload = job.spec.workload
    einsum_name = mapping[-1].einsum

    is_copy_operation = workload.einsums[einsum_name].is_copy_operation
    symbols = insert_sympy_symbols(job.mapping.nodes, job)

    if is_copy_operation:
        mapping, tensor_to_backer_id = convert_to_copy(mapping, workload)
    else:
        tensor_to_backer_id = get_tensor_to_backer_id(mapping)

    if add_reservations:
        job.mapping = quick_insert_reservation_nodes(job)
    # print(f'Job mapping: {job.mapping.compact_str()}')
    # for n in job.mapping.nodes:
    #     print(f'\t{n.compact_str()}')

    einsum_tensor_to_projection = {}
    einsum = workload.einsums[einsum_name]
    all_tensors = einsum.tensor_names
    for tensor in all_tensors:
        einsum_tensor_to_projection[(einsum_name, tensor)] = get_projection_expr(
            einsum, tensor
        )
    tensor_to_relevancy = {
        tensor: get_rank_variable_relevancy(einsum, tensor) for tensor in all_tensors
    }
    assert all_tensors, f"Einsum {einsum_name} has no tensors"

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

    A loop spatial-for (Reg) k in [0..K) would affect the register at the point of the
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
    result = None

    tensor2mapping = {}
    index_expressions = set(einsum.indexing_expressions)
    for k, v in job.rank_variable_bounds.items():
        index_expressions.add(f"0 < {k} <= {v}")
    for tensor in all_tensors:
        cur_mapping = job.mapping._get_single_tensor_mapping(
            tensor, job.flattened_arch, index_expressions
        )
        info = AnalysisInfo(
            mapping=cur_mapping.nodes,
            workload=workload,
            full_rank_variable_shapes=job.rank_variable_bounds,
            all_tensors=set([tensor]),
            einsum_tensor_to_projection=einsum_tensor_to_projection,
            tensor_to_relevancy=tensor_to_relevancy,
            tensor_to_backer_id=tensor_to_backer_id,
            is_copy_operation=is_copy_operation,
            job=job,
        )
        cur_result = analyze_node(0, job.rank_variable_bounds, info)
        if result is None:
            result = cur_result
        else:
            result.add_buffet_stats_and_symbols(cur_result)
        tensor2mapping[tensor] = cur_mapping

    result.symbols = symbols
    result.tensor2mapping = tensor2mapping

    if PRINT_FORMULAS:
        print(f"Mapping:")
        for node in mapping:
            print(f"\t{node.compact_str()}")

        for buffet, stats in result.buffet_stats.items():
            print(f"Einsum {buffet.einsum} tensor {buffet.tensor} level {buffet.level}")
            print(f"\tTotal reads to parent: {stats.total_reads_to_parent}")
            print(f"\tTotal writes to parent: {stats.total_writes_to_parent}")
            print(
                f"\tMax per parent reads to parent: {stats.max_per_parent_reads_to_parent}"
            )
            print(
                f"\tMax per parent writes to parent: {stats.max_per_parent_writes_to_parent}"
            )
            print(f"\tTotal reads to peer: {stats.total_reads_to_peer}")
            print(f"\tTotal writes to peer: {stats.total_writes_to_peer}")
            print(f"\tMax per unit reads to peer: {stats.max_per_unit_reads_to_peer}")
            print(f"\tMax per unit writes to peer: {stats.max_per_unit_writes_to_peer}")

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
    non_intermediate_tensors = (
        einsum.tensor_names - info.workload.tensor_names_used_in_multiple_einsums
    )
    seen_tensors = set()  # reservation for top-level buffets cannot be lowered

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
                if not node._lower or (
                    tensor not in seen_tensors and tensor in non_intermediate_tensors
                ):
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
                and buffet.tensor in info.workload.tensor_names_used_in_multiple_einsums
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
    stride_and_shape = get_stride_and_tile_shape(node, current_shape, node_idx, info)

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
            accumulated_stats = accumulated_buffet_stats.setdefault(
                buffet, blank_buffet_stats()
            )
            accumulated_stats += stats.repeat_temporal(
                shape_repeats, is_fully_relevant=is_fully_relevant
            )
            accumulated_stats.n_loops_above = stats.n_loops_above + 1

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
    spatial_component = node.component_object
    component_spatial_dim = spatial_component.spatial[node_dim]
    stride_and_shape = get_stride_and_tile_shape(node, current_shape, node_idx, info)

    result_accumulator = SymbolicAnalysisOutput()

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
                buffet, blank_buffet_stats()
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

            accumulated_stats += stats.repeat_spatial(
                shape_repeats, reuse_parent_accesses=reuse_parent_accesses
            )
            accumulated_stats.n_loops_above = stats.n_loops_above + 1

        for einsum, child_steps in child_result.temporal_steps.items():
            if einsum not in result_accumulator.temporal_steps:
                result_accumulator.temporal_steps[einsum] = child_steps
            else:
                result_accumulator.temporal_steps[einsum] = Max(
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


def find_component_object(
    component: str, flattened_arch: list[arch.Leaf]
) -> arch.TensorHolder:
    for node in flattened_arch:
        if node.name == component:
            return node
    raise ValueError(f"Component {component} not found in flattened arch")


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

    child_result = analyze_node(node_idx + 1, current_shape, info)

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
            # initialized with a zero value.
            if tensor in info.workload.einsums[einsum_name].output_tensor_names:
                inherit_add("total_skipped_first_reads_to_parent")
                inherit_add("min_per_parent_skipped_first_reads_to_parent")

        # ==============================================================================
        # Convert to actions. These are not used used upward; they are used to get
        # energy and latency.
        # ==============================================================================
        component_object = find_component_object(
            node.component, info.job.flattened_arch
        )
        bits_per_value_scale = component_object.bits_per_value_scale[tensor]
        bits_per_value = (
            bits_per_value_scale
            * info.job.einsum.tensor_accesses[tensor].bits_per_value
        )
        read_bits_per_action = component_object.actions["read"].bits_per_action
        read_scale = bits_per_value / read_bits_per_action
        if count_writes:
            write_bits_per_action = component_object.actions["write"].bits_per_action
            write_scale = bits_per_value / write_bits_per_action
        else:
            write_scale = 0

        # ==========================
        # Data exchanges with parent
        if count_downward_movement:  # Parent -> Me
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

        if count_upward_movement:  # Me -> Parent
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
            if count_downward_movement:  # Me -> Child
                stats.total_read_actions += child.total_reads_to_parent * read_scale
                stats.max_per_unit_read_actions += (
                    child.max_per_parent_reads_to_parent * read_scale
                )
                # Skip first read
                stats.total_skipped_first_read_actions += (
                    child.total_skipped_first_reads_to_parent * read_scale
                )
                stats.min_per_unit_skipped_first_read_actions += (
                    child.min_per_parent_skipped_first_reads_to_parent * read_scale
                )

            if count_upward_movement:  # Child -> Me
                stats.total_write_actions += child.total_writes_to_parent * write_scale
                stats.max_per_unit_write_actions += (
                    child.max_per_parent_writes_to_parent * write_scale
                )

    return child_result


def analyze_toll(node_idx, current_shape, info: AnalysisInfo):
    mapping = info.mapping
    einsum_name = mapping[-1].einsum
    node = mapping[node_idx]
    component_object = find_component_object(node.component, info.job.flattened_arch)
    storage_result = analyze_storage(
        node_idx,
        current_shape,
        info,
        propagate_child_results=True,
        count_upward_movement=component_object.direction != "down",
        count_downward_movement=component_object.direction != "up",
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
    component_object = find_component_object(node.resource, info.job.flattened_arch)
    bits_per_value_scale = component_object.bits_per_value_scale[tensor]
    bits_per_value = (
        bits_per_value_scale * info.job.einsum.tensor_accesses[tensor].bits_per_value
    )
    stats.max_occupancy = (
        compute_dense_tile_occupancy(projection, current_shape) * bits_per_value
    )
    child_result.buffet_stats[buffet] = stats

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

    result_accumulator = SymbolicAnalysisOutput()

    result_accumulator.temporal_steps[einsum] = computes
    result_accumulator.compute_stats[Compute(einsum, node.component)] = ComputeStats(
        computes,
        computes,
        1,
    )

    if info.is_copy_operation:
        return result_accumulator

    for tensor in info.all_tensors:
        buffet = Buffet(tensor, einsum, node.component)
        stats = BuffetStats()
        stats.total_reads_to_parent = 1
        stats.max_per_parent_reads_to_parent = 1
        if tensor in info.workload.einsums[einsum].output_tensor_names:
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


def get_stride_and_tile_shape(node: Loop, full_shape, n: int, info: AnalysisInfo):
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
    if IMPERFECT and initial_tile_shape is None:
        factor = sympy.ceiling(rank_shape / stride)
        stride_avg = stride / sympy.ceiling(rank_shape / stride)
        return StrideAndShape(stride_avg, RepeatedValue(stride, factor))

    if initial_tile_shape is None:
        if node._assume_perfect_factor or known_perfect_factor(stride, rank_shape):
            factor = rank_shape / stride
            return StrideAndShape(stride, RepeatedValue(stride, factor))
        else:
            factor = sympy.ceiling(rank_shape / sympy.Min(stride, rank_shape))
            return make_possibly_different_last(stride, factor, rank_shape)

    middle_shape_factor = sympy.ceiling((rank_shape - initial_tile_shape) / stride)
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
    # if node.tile_shape is not None:
    #     tile_shape = node.tile_shape

    #     if node._assume_perfect_factor or known_perfect_factor(tile_shape, rank_shape):
    #         factor = rank_shape / tile_shape
    #         return StrideAndShape(tile_shape, RepeatedValue(tile_shape, factor))
    #     else:
    #         factor = sympy.ceiling(rank_shape / sympy.Min(tile_shape, rank_shape))
    #         return make_possibly_different_last(tile_shape, factor, rank_shape)
    # elif node.loop_bound is not None:
    #     factor = node.loop_bound

    #     if node._assume_perfect_factor or known_perfect_factor(factor, rank_shape):
    #         tile_shape = rank_shape / factor
    #         return StrideAndShape(tile_shape, RepeatedValue(tile_shape, factor))
    #     else:
    #         tile_shape = sympy.ceiling(rank_shape / sympy.Min(rank_shape, factor))
    #         return make_possibly_different_last(tile_shape, factor, rank_shape)

    # elif node.tile_pattern is not None:
    #     stride = node.tile_pattern.tile_shape
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


def insert_sympy_symbols(mapping: list[MappingNode], job: Job):
    loop_idx = 0
    symbols = []
    rank_var_with_initial = set()
    for i, node in enumerate(mapping):
        if not isinstance(node, Loop):
            continue

        stride_halos = set()
        for t in job.spec.workload.einsums[job.einsum_name].tensor_names:
            for (rank, rank_variable), (stride, halo) in job.stride_and_halo[t].items():
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
