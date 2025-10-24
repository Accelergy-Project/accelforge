from collections.abc import Sequence
from dataclasses import dataclass, field
import logging
from numbers import Number
from typing import Any, Callable, Optional
from uuid import UUID, uuid4

import fastfusion.frontend.arch as arch
from fastfusion.frontend.mapping import (
    Iteration,
    Mapping,
    Reservation,
    Spatial,
    TilePattern,
)
from fastfusion.frontend.specification import Specification
from fastfusion.frontend.workload._symbolic import Relevant, PartiallyRelevant
from fastfusion.frontend.workload.workload import (
    EinsumName,
    RankVariableName,
    TensorName,
    Workload,
    RankName,
)

from fastfusion.frontend.mapper import Metrics
from fastfusion.mapper.FFM._join_pmappings.compatibility import (
    Compatibility,
    TensorReservation,
)
from fastfusion.mapper.FFM._make_pmappings.contraints.constraints import (
    MappingConstraints,
    ConstraintLambda,
)
from fastfusion.util.util import expfmt, fzs
from fastfusion.util.itertools import first
from fastfusion.frontend.mapping import Reservation as ReservationNode


def make_compatibility(
    mapping: Mapping,
    fusable_tensors: set[TensorName],
    workload: Workload,
    rank_variable_bounds: dict[RankVariableName, int],
    stride_and_halo,
) -> Compatibility:

    einsum = workload.einsums[mapping.nodes[-1].einsum]
    rank_variable_to_ranks = {
        t.name: t.rank_variable2ranks for t in einsum.tensor_accesses
    }
    return Compatibility.from_mapping(mapping, fusable_tensors, rank_variable_to_ranks)

    # einsum = workload.einsums[mapping.nodes[-1].einsum]
    # fused_slice = mapping.get_fused_slice(fusable_tensors)
    # fused_loops: list[Iteration] = []
    # loop_idx2reservations: dict[int, list[ReservationNode]] = {}
    # for node in fused_slice.nodes:
    #     if isinstance(node, Iteration):
    #         fused_loops.append(node)
    #     elif isinstance(node, ReservationNode):
    #         loop_idx2reservations.setdefault(len(fused_loops), []).append(node)
    #     elif isinstance(node, arch.TensorHolder):
    #         continue
    #     else:
    #         raise ValueError(f"Unexpected node type: {type(node)}")

    # compatibility_reservations = []
    # for above_loop_index, reservation_nodes in loop_idx2reservations.items():
    #     for reservation in reservation_nodes:
    #         tensor = reservation.purpose
    #         rank_var2ranks = einsum.tensor_accesses[tensor].rank_variable2ranks
    #         tensor_loops = []
    #         for loop in fused_loops[:above_loop_index]:
    #             ranks = rank_var2ranks[loop.rank_variable]
    #             if len(ranks) > 1:
    #                 raise NotImplementedError('co-iteration of ranks with '
    #                                           'one rank var.')
    #             if len(ranks) == 0:
    #                 raise NotImplementedError('recomputation')

    #             rank = first(ranks)
    #             tensor_loops.append(Loop(rank, None, isinstance(loop, Spatial)))

    #         compatibility_reservations.append(
    #             TensorReservation(
    #                 name=reservation.purpose,
    #                 loops=tuple(tensor_loops),
    #                 resource_name=reservation.resource,
    #                 size=None,
    #             )
    #         )

    # compatibility = Compatibility(tensors=fzs(compatibility_reservations))
    # return compatibility, dict(
    #     einsum=einsum,
    #     fused_loops=fused_loops,
    #     rank_variable_bounds=rank_variable_bounds,
    #     loop_idx2reservations=loop_idx2reservations,
    #     stride_and_halo=stride_and_halo,
    # )


# def update_compatibility_with_tile_shapes(compatibility, tile_shapes, tensor2size, einsum,fused_loops, rank_variable_bounds, loop_idx2reservations, stride_and_halo):
#     tile_shape_idx = 0
#     null_loop_indices: set[int] = set()
#     loops: list[tuple[str, int | TilePattern]] = []
#     for loop_idx, loop in enumerate(fused_loops):
#         rank_variable = loop.rank_variable

#         cur_tile_shape = tile_shapes[tile_shape_idx]

#         prev_size = rank_variable_bounds[rank_variable]
#         if loop_idx > 0:
#             prev_loop = first(
#                 (l for l in loops[loop_idx-1::-1] if l[0] == rank_variable),
#                 None
#             )
#             if prev_loop is not None:
#                 prev_rank_var, prev_bound = prev_loop
#                 assert prev_rank_var == rank_variable
#                 if isinstance(prev_bound, TilePattern):
#                     prev_size = prev_bound.stride
#                 elif isinstance(prev_bound, Number):
#                     prev_size = prev_bound
#                 else:
#                     raise RuntimeError('BUG')

#         if prev_size == cur_tile_shape:
#             null_loop_indices.add(loop_idx)

#         if loop.tile_shape is not None:
#             loops.append((rank_variable, cur_tile_shape))
#         elif loop.tile_pattern is not None:
#             loops.append((
#                 rank_variable,
#                 TilePattern(cur_tile_shape, tile_shapes[tile_shape_idx+1])
#             ))

#         tile_shape_idx += 1

#     tensors = []
#     for n_loops, reservations_at_level in loop_idx2reservations.items():
#         for reservation in reservations_at_level:
#             tensor = reservation.purpose
#             tensor_stride_and_halo = stride_and_halo[tensor]
#             rank_var2ranks = einsum.tensor_accesses[tensor].rank_variable2ranks

#             tensor_loops = []
#             for loop_idx, (rank_variable, rank_var_bound) in enumerate(loops[:n_loops]):
#                 if loop_idx in null_loop_indices:
#                     continue

#                 ranks = rank_var2ranks[rank_variable]
#                 if len(ranks) > 1:
#                     raise NotImplementedError('co-iteration of ranks with one rank var.')
#                 if len(ranks) == 0:
#                     raise NotImplementedError('recomputation')

#                 rank = first(ranks)

#                 stride, halo = tensor_stride_and_halo[(rank, rank_variable)]

#                 if isinstance(rank_var_bound, Number):
#                     if halo == 0:
#                         rank_bound = int(rank_var_bound*stride)
#                     else:
#                         rank_bound = TilePattern(
#                             int(rank_var_bound*stride),
#                             int((rank_var_bound-1)*stride + halo)
#                         )
#                 elif isinstance(rank_var_bound, TilePattern):
#                     rank_var_stride = rank_var_bound.stride
#                     rank_var_initial = rank_var_bound.initial
#                     rank_stride = rank_var_stride*stride
#                     rank_initial = (rank_var_initial-1)*stride + halo
#                     if rank_stride == rank_initial:
#                         rank_bound = int(rank_stride)  # regular tile
#                     else:
#                         rank_bound = TilePattern(int(rank_stride),
#                                                     int(rank_initial))

#                 tensor_loops.append(Loop(rank, rank_bound, isinstance(loop, Spatial)))

#             tensors.append(TensorReservation(
#                 reservation.purpose,
#                 tuple(tensor_loops),
#                 reservation.resource,
#                 size=tensor2size[reservation.purpose]
#             ))
#     compatibility = Compatibility(tensors=fzs(tensors))
#     return compatibility, null_loop_indices


@dataclass
class Job:
    spec: Specification
    metrics: Metrics
    rank_variable_bounds: dict[RankVariableName, int]

    job_id: UUID = field(default_factory=uuid4)

    stride_and_halo: (
        dict[TensorName, dict[tuple[RankName, RankVariableName], tuple[int, int]]]
        | None
    ) = None
    mapping: Mapping | None = None
    constraints: MappingConstraints | None = None
    fusable_tensors: set[TensorName] | None = None
    flattened_arch: list[arch.Leaf] | None = None

    einsum_name: EinsumName | None = None
    """If the Job is for a single einsum, this is the einsum name."""

    _compatibility: Compatibility | None = None
    memories_track_all: list[str] | None = None
    memories_track_pmappings_only: list[str] | None = None
    no_drop_reservations_for: set[str] | None = None
    time_limit: float | int = float("inf")
    memory_limit: float | int = float("inf")
    messages: list[str] = field(default_factory=list)
    pmapping_keep_rates: dict[str, float] = field(default_factory=dict)
    tensor_to_relevancy: (
        dict[TensorName, dict[RankVariableName, Relevant | PartiallyRelevant]] | None
    ) = None

    total_pmappings: int = 1
    valid_pmappings: int = 1
    evaluated_pmappings: int = 0

    _update_compatibility_with_tile_shapes_args: dict[str, Any] | None = None

    @property
    def compatibility(self) -> Compatibility:
        if self._compatibility is None:
            self._make_compatibility_and_updater()
        return self._compatibility

    @compatibility.setter
    def compatibility(self, compatibility: Compatibility):
        self._compatibility = compatibility

    def update_compatibility_with_tile_shapes(
        self, tile_shapes: Sequence[Number], tensor2size: dict
    ) -> Callable[[Sequence[Number], dict], Compatibility]:
        if self._update_compatibility_with_tile_shapes_args is None:
            self._make_compatibility_and_updater()
        return update_compatibility_with_tile_shapes(
            self._compatibility,
            tile_shapes=tile_shapes,
            tensor2size=tensor2size,
            **self._update_compatibility_with_tile_shapes_args,
        )

    def _make_compatibility_and_updater(self):
        from fastfusion.model.looptree.reuse.summarized.symbolic import (
            quick_insert_reservation_nodes,
        )

        with_reservations = quick_insert_reservation_nodes(self)
        self._compatibility = make_compatibility(
            with_reservations,
            self.fusable_tensors,
            self.spec.workload,
            self.rank_variable_bounds,
            self.stride_and_halo,
        )

    @property
    def is_copy_operation(self) -> bool:
        return self.spec.workload.einsums[self.einsum_name].is_copy_operation

    @classmethod
    def make_job(
        cls,
        **kwargs,
    ) -> "Job":
        defaults = {
            "spec": None,
            "mapping": None,
            "workload": None,
            "architecture": None,
        }
        kwargs = {**defaults, **kwargs}
        return cls(**kwargs)

    def pretty_str(self) -> str:
        constraints = self.constraints.get_all_constraints()
        node2constraints: dict[int, list[ConstraintLambda]] = {}
        for constraint in constraints:
            for target_index in constraint._target_node_indices:
                l = node2constraints.setdefault(target_index, [])
                l.append(constraint)

        # Reservations are added after mapping generation so it messes up the indexing
        mapping = [n for n in self.mapping.nodes if not isinstance(n, ReservationNode)]

        s = ""
        s += "=" * 80 + "\n"
        s += f"Mapper job with ID {self.job_id}\n"
        s += f"Einsum name: {self.einsum_name}\n"
        s += f"Rank variable bounds: {self.rank_variable_bounds}\n"
        s += f"Compute node name: {self.flattened_arch[-1].name}\n"
        s += f"Mapping:\n"
        for i, node in enumerate(mapping):
            cur_constraints = sorted(
                constraints.index(c) for c in node2constraints.get(i, [])
            )
            s += f"\t{i} {node.compact_str()} constrained by {cur_constraints}\n"
        s += self.constraints.pretty_str()
        s += f"Messages:\n"
        for m in self.messages:
            s += f"\t{m}\n"

        s += f"Total pmappings: {self.total_pmappings}\n"
        s += f"Valid pmappings: {self.valid_pmappings}\n"
        s += f"One in {expfmt(self.total_pmappings / self.valid_pmappings)} pmappings is valid\n"
        s += f"Number of pmappings evaluated: {self.evaluated_pmappings}\n"
        s += f"One in {expfmt(self.evaluated_pmappings / self.total_pmappings)} pmappings was evaluated\n"
        s += f"Pmapping elimination reasons:\n"
        for cause, keep_rate in self.pmapping_keep_rates.items():
            s += f"\t{cause} kept one in {expfmt(1/keep_rate)} pmappings\n"
        s += "=" * 80 + "\n"
        return s

    def set_total_pmappings(self, n_pmappings: int):
        self.total_pmappings = n_pmappings

    def log_porp_pmappings_kept(
        self,
        cause: str,
        porp_kept: float,
        out_of: int = None,
    ):
        if out_of is not None:
            n_kept = porp_kept * out_of + (self.total_pmappings - out_of)
            porp_kept = n_kept / self.total_pmappings

        self.pmapping_keep_rates.setdefault(cause, 1)
        self.pmapping_keep_rates[cause] *= porp_kept

    def log_message(self, message: str):
        self.messages.append(message)
        logging.info(message)

    def __copy__(self) -> "Job":
        new = self.__class__(**self.__dict__)
        new.messages = self.messages.copy()
        new.pmapping_keep_rates = self.pmapping_keep_rates.copy()
        return new


class SameSpecJobs(list[Job]):
    @property
    def spec(self) -> Specification:
        return first(self).spec

    @property
    def rank_variable_bounds(self) -> dict[RankVariableName, int]:
        return first(self).rank_variable_bounds

    @property
    def metrics(self) -> Metrics:
        return first(self).metrics


class SameEinsumJobs(SameSpecJobs):
    def check_invariance(self):
        all_einsums = set(job.einsum_name for job in self)
        if len(all_einsums) > 1:
            raise RuntimeError("broken invariance: not all Einsums are equal.")

    @property
    def fusable_tensors(self) -> set[TensorName]:
        return first(self).fusable_tensors

    @property
    def einsum_name(self) -> set[EinsumName]:
        return first(self).einsum_name

    @property
    def rank_variable_bounds(self) -> dict[RankVariableName, int]:
        return first(self).rank_variable_bounds

    @property
    def stride_and_halo(
        self,
    ) -> dict[tuple[str, str], dict[tuple[str, str], tuple[int, int]]]:
        return first(self).stride_and_halo

    @property
    def is_copy_op(self) -> bool:
        return first(self).is_copy_operation


class SameCompatibilityJobs(SameEinsumJobs):
    """Jobs with the same compatibility before tile shape exploration."""

    def check_invariance(self):
        all_compatibilities = set(job.compatibility for job in self)
        if len(all_compatibilities) > 1:
            raise RuntimeError(
                "broken invariance: " "not all compatibilities are equal."
            )

    @property
    def compatibility(self) -> Compatibility:
        return first(self).compatibility

    @property
    def update_compatibility_with_tile_shapes(
        self,
    ) -> Callable[[Sequence[Number], dict], Compatibility]:
        return first(self).update_compatibility_with_tile_shapes

    def split(self) -> list["SameCompatibilityJobs"]:
        return [SameCompatibilityJobs([j]) for j in self]
