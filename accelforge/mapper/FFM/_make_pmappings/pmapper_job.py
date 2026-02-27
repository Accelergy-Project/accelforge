from collections.abc import Sequence
from dataclasses import dataclass, field
import logging
from numbers import Number
from typing import Any, Callable
from uuid import UUID, uuid4

import accelforge.frontend.arch as arch
from accelforge.frontend.mapping import (
    Mapping,
)
from accelforge.frontend.spec import Spec
from accelforge.frontend._workload_isl._symbolic import Relevant, PartiallyRelevant
from accelforge.frontend.workload import (
    Einsum,
    EinsumName,
    RankVariable,
    SymbolTable,
    TensorName,
    Workload,
    Rank,
)

from accelforge.frontend.mapper import Metrics
from accelforge.mapper.FFM._join_pmappings.compatibility import (
    Compatibility,
)
from accelforge.mapper.FFM._make_pmappings.contraints.constraints import (
    MappingConstraints,
    _ConstraintLambda,
)
from accelforge.util.parallel import _expfmt
from accelforge.util._itertools import first
from accelforge.frontend.mapping import Reservation as ReservationNode


def make_compatibility(
    mapping: Mapping,
    fusable_tensors: set[TensorName],
    workload: Workload,
) -> Compatibility:

    einsum = workload.einsums[mapping.nodes[-1].einsum]
    rank_variable_to_ranks = {
        t.name: t.rank_variable2ranks for t in einsum.tensor_accesses
    }
    return Compatibility.from_mapping(mapping, fusable_tensors, rank_variable_to_ranks)


@dataclass
class Job:
    spec_one_einsum: Spec | None
    metrics: Metrics
    rank_variable_bounds: dict[RankVariable, int]

    job_id: UUID = field(default_factory=uuid4)

    stride_and_halo: (
        dict[
            tuple[EinsumName, TensorName],
            dict[tuple[Rank, RankVariable], tuple[int, int]],
        ]
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
    ignored_resources: set[str] | None = None
    time_limit: float | int = float("inf")
    memory_limit: float | int = float("inf")
    messages: list[str] = field(default_factory=list)
    pmapping_keep_rates: dict[str, float] = field(default_factory=dict)
    tensor_to_relevancy: (
        dict[TensorName, dict[RankVariable, Relevant | PartiallyRelevant]] | None
    ) = None

    n_total_pmappings: int = 1
    n_valid_pmappings: int = 1
    n_evaluated_pmappings: int = 0

    symbol_table: SymbolTable | None = None

    initial_delta_choices: dict[RankVariable, frozenset[int]] | None = None

    @property
    def einsum(self) -> Einsum:
        return self.spec_one_einsum.workload.einsums[self.einsum_name]

    @property
    def compatibility(self) -> Compatibility:
        if self._compatibility is None:
            self._make_compatibility_and_updater()
        return self._compatibility

    @compatibility.setter
    def compatibility(self, compatibility: Compatibility):
        self._compatibility = compatibility

    def _make_compatibility_and_updater(self):
        from accelforge.model._looptree.reuse.symbolic import (
            quick_insert_reservation_nodes,
        )

        with_reservations = quick_insert_reservation_nodes(self)
        self._compatibility = make_compatibility(
            with_reservations,
            self.fusable_tensors,
            self.spec_one_einsum.workload,
        )

    @property
    def is_copy_operation(self) -> bool:
        return self.spec_one_einsum.workload.einsums[self.einsum_name].is_copy_operation

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
        node2constraints: dict[int, list[_ConstraintLambda]] = {}
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

        s += f"Total pmappings: {self.n_total_pmappings}\n"
        s += f"Valid pmappings: {self.n_valid_pmappings}\n"
        s += f"One in {_expfmt(self.n_total_pmappings / self.n_valid_pmappings)} pmappings is valid\n"
        s += f"Number of pmappings evaluated: {self.n_evaluated_pmappings}\n"
        s += f"One in {_expfmt(self.n_evaluated_pmappings / self.n_total_pmappings)} pmappings was evaluated\n"
        s += f"Pmapping elimination reasons:\n"
        for cause, keep_rate in self.pmapping_keep_rates.items():
            s += f"\t{cause} kept one in {_expfmt(1/keep_rate)} pmappings\n"
        s += "=" * 80 + "\n"
        return s

    def set_total_pmappings(self, n_pmappings: int):
        self.n_total_pmappings = n_pmappings

    def log_porp_pmappings_kept(
        self,
        cause: str,
        porp_kept: float,
        out_of: int = None,
    ):
        if out_of is not None:
            n_kept = porp_kept * out_of + (self.n_total_pmappings - out_of)
            porp_kept = n_kept / self.n_total_pmappings

        if any(x == 0 for x in self.pmapping_keep_rates.values()):
            return

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
    def spec(self) -> Spec:
        return first(self).spec

    @property
    def rank_variable_bounds(self) -> dict[RankVariable, int]:
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
    def rank_variable_bounds(self) -> dict[RankVariable, int]:
        return first(self).rank_variable_bounds

    @property
    def stride_and_halo(
        self,
    ) -> dict[
        tuple[EinsumName, TensorName],
        dict[tuple[Rank, RankVariable], tuple[int, int]],
    ]:
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

    def split(self) -> list["SameCompatibilityJobs"]:
        return [SameCompatibilityJobs([j]) for j in self]
