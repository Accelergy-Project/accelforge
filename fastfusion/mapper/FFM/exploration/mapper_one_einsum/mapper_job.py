from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Number
from typing import Callable
from uuid import UUID

import fastfusion.frontend.architecture as architecture
from fastfusion.frontend.mapping import (
    Iteration,
    Mapping,
    ModelOnlyNode,
    Spatial,
)
from fastfusion.frontend.specification import Specification
from fastfusion.frontend.workload.workload import (
    EinsumName,
    RankVariableName,
    TensorName,
    Workload,
    RankName,
)

from fastfusion.mapper import metrics
from fastfusion.mapper.FFM.joining.mappinginfo import (
    Compatibility,
    Loop,
    TilePattern,
    TensorReservation,
)
from fastfusion.mapper.FFM.exploration.contraints.constraints import (
    MappingConstraints,
)
from fastfusion.mapper.FFM.tags import Tags
from fastfusion.model.looptree.reuse.summarized.symbolic import (
    quick_insert_reservation_nodes,
)
from fastfusion.util.util import fzs
from fastfusion.util.itertools import first
from fastfusion.frontend.mapping import Reservation as ReservationNode


def make_compatibility(
    mapping: Mapping,
    tagger,
    intermediate_tensors: set[TensorName],
    workload: Workload,
    rank_variable_bounds: dict[RankVariableName, int],
    stride_and_halo,
) -> Compatibility:
    einsum = workload.einsums[mapping.nodes[-1].einsum]
    fused_slice = mapping.get_fused_slice(intermediate_tensors)
    fused_loops: list[Iteration] = []
    loop_idx2reservations: dict[int, list[ReservationNode]] = {}
    for node in fused_slice.nodes:
        if isinstance(node, Iteration):
            fused_loops.append(node)
        elif isinstance(node, ReservationNode):
            loop_idx2reservations.setdefault(len(fused_loops), []).append(node)
        elif isinstance(node, ModelOnlyNode):
            continue
        elif isinstance(node, TensorHolder):
            continue
        else:
            raise ValueError(f"Unexpected node type: {type(node)}")

    compatibility_reservations = []
    for above_loop_index, reservation_nodes in loop_idx2reservations.items():
        for reservation in reservation_nodes:
            tensor = reservation.purpose
            rank_var2ranks = einsum.tensor_accesses[tensor].rank_variable2ranks
            tensor_loops = []
            for loop_idx, loop in enumerate(fused_loops[:above_loop_index]):
                ranks = rank_var2ranks[loop.rank_variable]
                if len(ranks) > 1:
                    raise NotImplementedError('co-iteration of ranks with '
                                              'one rank var.')
                if len(ranks) == 0:
                    raise NotImplementedError('recomputation')

                rank = first(ranks)
                tensor_loops.append(Loop(rank, None, isinstance(loop, Spatial)))

            compatibility_reservations.append(
                TensorReservation(
                    name=reservation.purpose,
                    loops=tuple(tensor_loops),
                    resource_name=reservation.resource,
                    size=None,
                )
            )

    compatibility = Compatibility(
        n_loops=len(fused_loops),
        tensors=fzs(compatibility_reservations),
    )

    def update_compatibility_with_tile_shapes(tile_shapes, tensor2size):
        tile_shape_idx = 0
        null_loop_indices: set[int] = set()
        loops: list[tuple[str, int | TilePattern]] = []
        for loop_idx, loop in enumerate(fused_loops):
            rank_variable = loop.rank_variable

            cur_tile_shape = tile_shapes[tile_shape_idx]

            prev_size = rank_variable_bounds[rank_variable]
            if loop_idx > 0:
                prev_loop = first(
                    (l for l in loops[loop_idx-1::-1] if l[0] == rank_variable),
                    None
                )
                if prev_loop is not None:
                    prev_rank_var, prev_bound = prev_loop
                    assert prev_rank_var == rank_variable
                    if isinstance(prev_bound, TilePattern):
                        prev_size = prev_bound.stride
                    elif isinstance(prev_bound, Number):
                        prev_size = prev_bound
                    else:
                        raise RuntimeError('BUG')

            if prev_size == cur_tile_shape:
                null_loop_indices.add(loop_idx)

            if loop.tile_shape is not None:
                loops.append((rank_variable, cur_tile_shape))
            elif loop.tile_pattern is not None:
                loops.append((
                    rank_variable,
                    TilePattern(cur_tile_shape, tile_shapes[tile_shape_idx+1])
                ))

            tile_shape_idx += 1

        tensors = []
        for n_loops, reservations_at_level in loop_idx2reservations.items():
            for reservation in reservations_at_level:
                tensor = reservation.purpose
                tensor_stride_and_halo = stride_and_halo[tensor]
                rank_var2ranks = einsum.tensor_accesses[tensor].rank_variable2ranks

                tensor_loops = []
                for loop_idx, (rank_variable, rank_var_bound) in enumerate(loops[:n_loops]):
                    if loop_idx in null_loop_indices:
                        continue

                    ranks = rank_var2ranks[rank_variable]
                    if len(ranks) > 1:
                        raise NotImplementedError('co-iteration of ranks with one rank var.')
                    if len(ranks) == 0:
                        raise NotImplementedError('recomputation')

                    rank = first(ranks)

                    stride, halo = tensor_stride_and_halo[(rank, rank_variable)]

                    if isinstance(rank_var_bound, Number):
                        if halo == 0:
                            rank_bound = int(rank_var_bound*stride)
                        else:
                            rank_bound = TilePattern(
                                int(rank_var_bound*stride),
                                int((rank_var_bound-1)*stride + halo)
                            )
                    elif isinstance(rank_var_bound, TilePattern):
                        rank_var_stride = rank_var_bound.stride
                        rank_var_initial = rank_var_bound.initial
                        rank_stride = rank_var_stride*stride
                        rank_initial = (rank_var_initial-1)*stride + halo
                        if rank_stride == rank_initial:
                            rank_bound = int(rank_stride)  # regular tile
                        else:
                            rank_bound = TilePattern(int(rank_stride),
                                                     int(rank_initial))

                    tensor_loops.append(Loop(rank, rank_bound, isinstance(loop, Spatial)))

                tensors.append(TensorReservation(
                    reservation.purpose,
                    tuple(tensor_loops),
                    reservation.resource,
                    size=tensor2size[reservation.purpose]
                ))
        compat = Compatibility(n_loops=max([0] + [len(s.loops) for s in tensors]),
                               tensors=fzs(tensors))
        # tags = tagger(compat)
        # compat = compat.update(tags=tags)
        return compat, null_loop_indices
    return compatibility, update_compatibility_with_tile_shapes


@dataclass
class Job:
    spec: Specification
    tagger: Callable[[Mapping], Tags]
    metrics: metrics.Metrics
    job_id: UUID
    rank_variable_bounds: dict[RankVariableName, int]
    stride_and_halo: dict[TensorName, dict[tuple[RankName, RankVariableName], tuple[int, int]]] | None = None
    mapping: Mapping | None = None
    constraints: MappingConstraints | None = None
    intermediate_tensors: set[TensorName] | None = None
    flattened_arch: list[architecture.Leaf] | None = None
    einsum_name: EinsumName | None = None
    tensor2compatibilties: dict[TensorName, set[Compatibility]] | None = None
    tensor2boundless_compatibilities: dict[TensorName, set[Compatibility]] | None = None
    except_from_imperfect: set = frozenset()
    _compatibility: Compatibility | None = None
    _update_compatibility_with_tile_shapes: Callable[[Sequence[Number], dict], Compatibility] | None = None
    memories_track_all: list[str] | None = None
    memories_track_pmappings_only: list[str] | None = None

    @property
    def compatibility(self) -> Compatibility:
        if self._compatibility is None:
            self._make_compatibility_and_updater()
        return self._compatibility

    @property
    def update_compatibility_with_tile_shapes(self) -> Callable[[Sequence[Number], dict], Compatibility]:
        if self._update_compatibility_with_tile_shapes is None:
            self._make_compatibility_and_updater()
        return self._update_compatibility_with_tile_shapes

    def _make_compatibility_and_updater(self):
        with_reservations = quick_insert_reservation_nodes(
            self.mapping, self.spec.workload
        )
        self._compatibility, self._update_compatibility_with_tile_shapes = \
            make_compatibility(with_reservations,
                               self.tagger,
                               self.intermediate_tensors,
                               self.spec.workload,
                               self.rank_variable_bounds,
                               self.stride_and_halo)
            
    @property
    def is_copy_operation(self) -> bool:
        return self.spec.workload.einsums[self.einsum_name].is_copy_operation


class SameSpecJobs(list[Job]):
    @property
    def spec(self) -> Specification:
        return first(self).spec

    @property
    def rank_variable_bounds(self) -> dict[RankVariableName, int]:
        return first(self).rank_variable_bounds

    @property
    def tagger(self) -> Callable[[Mapping], Tags]:
        return first(self).tagger

    @property
    def metrics(self) -> metrics.Metrics:
        return first(self).metrics


class SameEinsumJobs(SameSpecJobs):
    def check_invariance(self):
        all_einsums = set(job.einsum_name for job in self)
        if len(all_einsums) > 1:
            raise RuntimeError('broken invariance: not all Einsums are equal.')

    @property
    def intermediate_tensors(self) -> set[TensorName]:
        return first(self).intermediate_tensors

    @property
    def einsum_name(self) -> set[EinsumName]:
        return first(self).einsum_name

    @property
    def rank_variable_bounds(self) -> dict[RankVariableName, int]:
        return first(self).rank_variable_bounds
    
    @property
    def stride_and_halo(self) -> dict[tuple[str, str], dict[tuple[str, str], tuple[int, int]]]:
        return first(self).stride_and_halo

    @property
    def is_copy_op(self) -> bool:
        return first(self).is_copy_operation


class SameCompatibilityJobs(SameEinsumJobs):
    """Jobs with the same compatibility before tile shape exploration."""
    def check_invariance(self):
        all_compatibilities = set(job.compatibility for job in self)
        if len(all_compatibilities) > 1:
            raise RuntimeError('broken invariance: '
                               'not all compatibilities are equal.')

    @property
    def compatibility(self) -> Compatibility:
        return first(self).compatibility

    @property
    def update_compatibility_with_tile_shapes(self) -> Callable[[Sequence[Number], dict], Compatibility]:
        return first(self).update_compatibility_with_tile_shapes
    
    def split(self) -> list["SameCompatibilityJobs"]:
        return [SameCompatibilityJobs([j]) for j in self]
