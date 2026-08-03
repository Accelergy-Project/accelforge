import copy
import operator
from dataclasses import dataclass, field
from typing import Any

from accelforge.frontend.mapping import Compute, Mapping
from accelforge.frontend.workload import TensorName

from accelforge.model._looptree.types import Buffet, Compute, Network

from accelforge.util._frozenset import oset
from accelforge.util._sympy.broadcast_max import (
    max_nonzero,
    min_nonzero,
    max_dict,
)

import sympy
from accelforge.util.indent import print


@dataclass
class NetworkStats:
    total_hops: Any = field(default=0)
    """Total number of hops overall. Useful to calculate energy."""
    max_hops: Any = field(default=0)
    """Longest hops among all routes."""
    max_traffic: dict[int | str, Any] = field(default_factory=dict)
    """Maximum traffic occuring on any single link along a dimension."""

    def repeat(self, n_repeats):
        new = copy.copy(self)
        if n_repeats == 1:
            return new
        if type(n_repeats) is float and n_repeats == int(n_repeats):
            n_repeats = int(n_repeats)
        new.total_hops = new.total_hops * n_repeats
        return new


class ActionCounts(dict):
    """Per-action-name counts. Missing actions count as 0 so `d[action] += x`
    works for actions a component lacks."""

    def __missing__(self, key):
        return 0


def _scale(value: Any, factor: Any) -> Any:
    if isinstance(value, ActionCounts):
        return ActionCounts({k: v * factor for k, v in value.items()})
    return value * factor


def _combine(a: Any, b: Any, op) -> Any:
    if isinstance(a, ActionCounts) or isinstance(b, ActionCounts):
        return ActionCounts({k: op(a[k], b[k]) for k in oset(a) | oset(b)})
    return op(a, b)


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

    # These are used to calculate energy and latency. Keyed by action name and
    # split by which side of the storage the data moves on: exchanges with the
    # parent (fills in, drains out) versus exchanges with the child or peers.
    # By construction, skipped-first writes are parent-side and skipped-first
    # reads child-side. total_actions and max_per_unit_actions are the sums.
    total_actions_to_parent: ActionCounts = field(default_factory=ActionCounts)
    total_actions_to_child: ActionCounts = field(default_factory=ActionCounts)
    max_per_unit_actions_to_parent: ActionCounts = field(default_factory=ActionCounts)
    max_per_unit_actions_to_child: ActionCounts = field(default_factory=ActionCounts)
    total_skipped_first_actions_to_parent: ActionCounts = field(
        default_factory=ActionCounts
    )
    total_skipped_first_actions_to_child: ActionCounts = field(
        default_factory=ActionCounts
    )
    min_per_unit_skipped_first_actions_to_parent: ActionCounts = field(
        default_factory=ActionCounts
    )
    min_per_unit_skipped_first_actions_to_child: ActionCounts = field(
        default_factory=ActionCounts
    )

    # NOTE: anything other than min_, max_, or total_ must default to
    # None. There are asserts that check this.
    persistent: bool = field(default=None)

    # Number of temporal iterations above this buffet's storage node.
    iterations_above: Any = field(default=1)

    @property
    def total_actions(self) -> ActionCounts:
        return _combine(
            self.total_actions_to_parent, self.total_actions_to_child, operator.add
        )

    @property
    def max_per_unit_actions(self) -> ActionCounts:
        return _combine(
            self.max_per_unit_actions_to_parent,
            self.max_per_unit_actions_to_child,
            operator.add,
        )

    @property
    def total_skipped_first_actions(self) -> ActionCounts:
        return _combine(
            self.total_skipped_first_actions_to_parent,
            self.total_skipped_first_actions_to_child,
            operator.add,
        )

    @property
    def min_per_unit_skipped_first_actions(self) -> ActionCounts:
        return _combine(
            self.min_per_unit_skipped_first_actions_to_parent,
            self.min_per_unit_skipped_first_actions_to_child,
            operator.add,
        )

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
            new.__dict__[k] = _scale(v, factor)
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
            # If parent accesses are reused, no need to multiply. Action count
            # dicts always scale.
            if "parent" in k and "actions" not in k and reuse_parent_accesses:
                continue
            if "per_unit" in k:
                continue  # Spatial fanout doesn't affect per-unit stats
            if k == "max_occupancy":
                continue  # Max occupancy is not affected by temporal loops above
            new.__dict__[k] = _scale(v, factor)
        return new

    def max(self, **kwargs: Any):
        for key, value in kwargs.items():
            setattr(self, key, max_nonzero(getattr(self, key), value))

    def min(self, **kwargs: Any):
        for key, value in kwargs.items():
            setattr(self, key, min_nonzero(getattr(self, key), value))

    def __add__(self, other: "BuffetStats") -> "BuffetStats":
        new = copy.copy(self)
        for k, v in self.__dict__.items():
            other_v = other.__dict__[k]
            if k.startswith("min_"):
                new.__dict__[k] = _combine(v, other_v, min_nonzero)
            elif k.startswith("max_"):
                new.__dict__[k] = _combine(v, other_v, max_nonzero)
            elif k.startswith("total_"):
                new.__dict__[k] = _combine(v, other_v, operator.add)
            elif k == "iterations_above" and v is not None and other_v is not None:
                new.__dict__[k] = max_nonzero(v, other_v)
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

    def net_total_actions(self, action: str | None = None) -> Any:
        if action is not None:
            return self.total_actions[action] - self.total_skipped_first_actions[action]
        return ActionCounts({a: self.net_total_actions(a) for a in self.total_actions})

    def net_max_per_unit_actions(self, action: str | None = None) -> Any:
        if action is not None:
            return (
                self.max_per_unit_actions[action]
                - self.min_per_unit_skipped_first_actions[action]
            )
        return ActionCounts(
            {a: self.net_max_per_unit_actions(a) for a in self.max_per_unit_actions}
        )

    def net_max_per_unit_actions_to_parent(self, action: str | None = None) -> Any:
        if action is not None:
            return (
                self.max_per_unit_actions_to_parent[action]
                - self.min_per_unit_skipped_first_actions_to_parent[action]
            )
        return ActionCounts(
            {
                a: self.net_max_per_unit_actions_to_parent(a)
                for a in self.max_per_unit_actions_to_parent
            }
        )

    def net_max_per_unit_actions_to_child(self, action: str | None = None) -> Any:
        if action is not None:
            return (
                self.max_per_unit_actions_to_child[action]
                - self.min_per_unit_skipped_first_actions_to_child[action]
            )
        return ActionCounts(
            {
                a: self.net_max_per_unit_actions_to_child(a)
                for a in self.max_per_unit_actions_to_child
            }
        )

    @classmethod
    def blank(cls):
        stats = cls()
        stats.n_loops_above = None  # Inherit from whoever is added to this
        stats.iterations_above = None
        return stats


@dataclass
class ComputeStats:
    total_ops: Any = field(default=0)
    max_per_unit_ops: Any = field(default=0)
    # "max" below refers to the longest latency of any iteration
    max_latency: Any = field(default=0)

    def repeat_temporal(self, factor: int) -> "ComputeStats":
        new = copy.copy(self)
        if factor == 1:
            return new
        if type(factor) is float and factor == int(factor):
            factor = int(factor)
        new.total_ops = new.total_ops * factor
        new.max_per_unit_ops = new.max_per_unit_ops * factor
        new.max_latency = new.max_latency * factor
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
        return new

    def combine_temporal(self, other: "ComputeStats"):
        self.total_ops += other.total_ops
        self.max_per_unit_ops += other.max_per_unit_ops
        self.max_latency += other.max_latency

    def combine_spatial(self, other: "ComputeStats"):
        self.total_ops += other.total_ops
        self.max_per_unit_ops = max_nonzero(
            self.max_per_unit_ops, other.max_per_unit_ops
        )
        self.max_latency = max_nonzero(self.max_latency, other.max_latency)


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
                        previous[k][k2] = max_nonzero(previous[k][k2], v2)
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
