import copy
from dataclasses import dataclass, field
from typing import Any

from accelforge.frontend.mapping import Compute, Mapping
from accelforge.frontend.workload import TensorName

from accelforge.model._looptree.types import Buffet, Compute, Network

from accelforge.util._frozenset import oset
from accelforge.util._sympy.broadcast_max import (
    MaxGeqZero,
    MinGeqZero,
    min_nonzero,
    max_dict,
)

import sympy


@dataclass
class NetworkStats:
    total_hops: Any = field(default=0)
    max_hops: Any = field(default=0)

    def repeat(self, n_repeats):
        new = copy.copy(self)
        if n_repeats == 1:
            return new
        if type(n_repeats) is float and n_repeats == int(n_repeats):
            n_repeats = int(n_repeats)
        new.total_hops = new.total_hops * n_repeats
        return new

    def combine(self, other: "NetworkStats"):
        self.total_hops += other.total_hops
        self.max_hops = max(self.max_hops, other.max_hops)


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

    # NOTE: anything other than min_, max_, or total_ must default to
    # None. There are asserts that check this.
    persistent: bool = field(default=None)

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
        if factor == 1:
            return new
        if type(factor) is float and factor == int(factor):
            factor = int(factor)  # sympy Symbol * int is 4× faster than * float
        for k, v in new.__dict__.items():
            if not k.startswith(("total_", "max_", "min_")):
                continue
            if "skipped_first" in k and not is_fully_relevant:
                continue  # First actions occur once per relevant iteration.
            if k == "max_occupancy":
                continue  # Max occupancy is not affected by temporal loops above
            new.__dict__[k] = v * factor
        return new

    def repeat_spatial(self, factor: int, reuse_parent_accesses: bool) -> "BuffetStats":
        """
        Repeat buffet stats due to spatial loop `factor` number of times.

        For accesses to parent, the amount of repetition is `factor` if `reuse_parent_access`
        is False; otherwise, there is no repetition.
        """
        new = copy.copy(self)
        if factor == 1:
            return new
        if type(factor) is float and factor == int(factor):
            factor = int(factor)
        for k, v in new.__dict__.items():
            if not k.startswith(("total_", "max_", "min_")):
                continue
            if "parent" in k and reuse_parent_accesses:
                continue  # If parent accesses are reused, no need to multiply
            if "per_unit" in k:
                continue  # Spatial fanout doesn't affect per-unit stats
            if k == "max_occupancy":
                continue  # Max occupancy is not affected by temporal loops above
            new.__dict__[k] = v * factor
        return new

    def max(self, **kwargs: Any):
        for key, value in kwargs.items():
            setattr(self, key, MaxGeqZero(getattr(self, key), value))

    def min(self, **kwargs: Any):
        for key, value in kwargs.items():
            setattr(self, key, MinGeqZero(getattr(self, key), value))

    def __add__(self, other: "BuffetStats") -> "BuffetStats":
        new = copy.copy(self)
        for k, v in self.__dict__.items():
            other_v = other.__dict__[k]
            if k.startswith("min_"):
                new.__dict__[k] = min_nonzero(v, other_v)
            elif k.startswith("max_"):
                new.__dict__[k] = MaxGeqZero(v, other_v)
            elif k.startswith("total_"):
                new.__dict__[k] = v + other_v
            elif v is None:
                new.__dict__[k] = other_v
            else:
                if v is None:
                    new.__dict__[k] = other_v
                else:
                    assert (
                        v == other_v
                    ), f"BUG: {k} is different. self: {v} other: {other_v}"
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

    @classmethod
    def blank(cls):
        stats = cls()
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
        if factor == 1:
            return new
        if type(factor) is float and factor == int(factor):
            factor = int(factor)
        new.total_ops = new.total_ops * factor
        new.max_per_unit_ops = new.max_per_unit_ops * factor
        new.max_latency = new.max_latency * factor
        # NOTE: max_first_latency does not change
        return new

    def repeat_spatial(self, factor: int) -> "ComputeStats":
        new = copy.copy(self)
        if factor == 1:
            return new
        if type(factor) is float and factor == int(factor):
            factor = int(factor)
        new.total_ops = new.total_ops * factor
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
        self.max_per_unit_ops = MaxGeqZero(
            self.max_per_unit_ops, other.max_per_unit_ops
        )
        self.max_latency = MaxGeqZero(self.max_latency, other.max_latency)
        # max_first_latency is only ever updated across loops ABOVE the loop
        # for which we calculated that first latency, so we should MAX
        self.max_first_latency = max_dict(
            self.max_first_latency, other.max_first_latency
        )  # FIRST LATENCY


@dataclass
class SymbolicAnalysisOutput:
    compute_stats: dict[Compute, ComputeStats] = field(default_factory=dict)

    buffet_stats: dict[Buffet, BuffetStats] = field(default_factory=dict)

    network_stats: dict[Network, NetworkStats] = field(default_factory=dict)

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
                        previous[k][k2] = MaxGeqZero(previous[k][k2], v2)
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
            result.setdefault(buffet.level, BuffetStats.blank())
            result[buffet.level] += stats
        return result

    def add_buffet_stats_and_symbols(self, other: "SymbolicAnalysisOutput"):
        assert not (oset(self.buffet_stats) & oset(other.buffet_stats)), "BUG"
        self.buffet_stats.update(other.buffet_stats)
        # if self.temporal_steps != other.temporal_steps:
        #     print(f'Temporal steps are different.')
        #     print(f'\tmine:  {self.temporal_steps}')
        #     print(f'\tother: {other.temporal_steps}')
        # assert self.temporal_steps == other.temporal_steps, "BUG"
        self.temporal_steps.update(other.temporal_steps)
        self.symbols.extend([s for s in other.symbols if s not in self.symbols])

    def add_network_stats(self, other: "SymbolicAnalysisOutput"):
        assert not (oset(self.network_stats) & oset(other.network_stats)), "BUG"
        self.network_stats.update(other.network_stats)
