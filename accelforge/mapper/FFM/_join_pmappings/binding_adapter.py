"""
Adapting one binding of a tensor into another.

Two Einsums that agree on everything about a tensor they share *but* the binding of
its backing storage -- which rank each of the spatial loops below that storage
partitions the tensor along -- describe two different physical layouts of the same
data across the memory's instances. They cannot simply be fused: whichever way the
producing side left the data, the consuming side expects it somewhere else.

They can, however, be *adapted*. The data is shuffled among the physical instances
of the memory that holds it until it is laid out the way the consuming side wants,
and the mapper is charged the latency and energy of that shuffle. That shuffle is
ordinary traffic over the network below the memory, so this module costs it with the
very same :class:`~accelforge.model._looptree.reuse.symbolic._network.TopologyModel`
that costs every other transfer over that network, fed the same way:

* the *source* is the memory, physically distributed exactly as the model already
  knows it to be;
* the *destinations* are one per iteration of each binding loop, spaced by the
  memory's physical stride along that loop's dimension times whatever the binding
  loops inside it already fan out by. This is where the adapter parts company with
  the analyzer, which is delivering to the consumers below a spatial loop and spaces
  them by the mapping's fanout; an adapter delivers to the memory's own instances, so
  it spaces them by the memory's own stride;
* the *bound rank of the source* is the rank the left side's binding loop names, and
  the *relevancy at the destination* is what the right side's binding loop asks for:
  a rank of its own if it partitions along one (each destination wants its own
  slice, a unicast), and ``Irrelevant`` if it replicates instead (every destination
  wants the whole thing, a multicast).

Feeding it that way makes the answers fall out of the model that already exists.
Along a dimension where the two bindings name the same rank, or where the left side
replicated, :func:`~...._binding.binding_is_local` reports the data as already local
and the model charges nothing -- correctly, since an instance that holds a whole copy
holds any slice of it, and one that holds the right slice already need not move it.
Along a dimension where the left side partitioned by some other rank, the model
charges its scattered delivery: an all-gather where the right side wants the whole
tile, and an all-to-all shuffle where it wants a different slice.

The cost is charged once per tile, i.e. once for every iteration of the fused loops
above the tensor's reservation, since the producing side refills the buffer on each
of them.

An adapter as a step of the join
--------------------------------

Nothing about an adapter comes from the workload, so there are no pmappings of one to
make ahead of time the way there are for an Einsum -- but the joiner treats it as
though there were. Every adapter is built up front, by looking at the compatibilities
of the two sides about to be joined and asking which bindings of the right side each
group of the left side could be moved into; applying one to a group is a join like any
other, giving a new group with a compatibility of its own and a cost added to its
pmappings. Joining ``A + adapter + B`` is then joining ``A + adapter``, whose answer
is A laid out the way B wants it, and joining *that* with B down the ordinary path,
which finds the two sides agreeing about the binding and needs to know nothing about
adapters at all.

Building the adapters first is what makes them cheap. An adapter is applied once per
group of the left side rather than once per pair of groups the joiner might try, and
the result is shared by every one of those pairs. And within one adapter everything
but the volume is settled before a single row is looked at: how far the data has to
travel, what a hop costs, and the network's own latency expression all depend only on
how many times each binding loop iterates, so rows that vary only in their tile
shapes -- the overwhelming majority -- are charged by one vectorized expression.

What an adaptation is charged for, and what it is not
----------------------------------------------------

The volume it moves is what one instance of the memory ends up holding once the data
has been moved, which is the volume the network model is written in terms of. The
adapter is applied to the left side alone, so it reads the left side's occupancy of
the tensor and converts it: the two sides hold the very same tile above the very same
fused loops, and differ only in how many instances share it. See
:meth:`BindingAdaptation._rebinding_factor`.

One thing an adaptation is *not* charged for: how much of the memory it needs. A
binding decides how much of a tensor one instance holds -- a replicated one holds a
whole copy where a partitioned one holds a share -- so the two sides of an adaptation
may disagree about the tensor's occupancy, and the joiner carries the producing side's
figure forward as it does for any shared tensor. The movement is charged; the
occupancy is the left side's.
"""

import itertools
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import sympy

from accelforge._accelerated_imports import pandas as pd
from accelforge.frontend import arch
from accelforge.frontend._workload_isl._symbolic import Irrelevant, Relevant
from accelforge.frontend.arch._flattened_arch import FlattenedArch
from accelforge.frontend.renames import RANK_DUPLICATE, TensorName
from accelforge.frontend.spec import Spec
from accelforge.mapper.FFM._join_pmappings.compatibility import (
    Compatibility,
    Loop,
    TensorReservation,
)
from accelforge.mapper.FFM._pareto_df.df_convention import add_to_col, tensor2col
from accelforge.util import delayed, oset, parallel
from accelforge.util._sympy.broadcast_max import Max, MaxGeqZero, max_nonzero

logger = logging.getLogger(__name__)

_HOP_MODEL: dict[tuple, tuple] = {}
"""
What one unit of volume of an adaptation costs, keyed by everything it depends on:
the memory and the network the data moves over, the two bindings loop for loop, and
how many times each of those loops iterates. Nothing else enters into it -- not the
tile shapes, not the occupancies, not which Einsums are being joined -- so two
adapters that move data the same way share an answer however far apart in the join
they were made. Held here rather than per adapter for that reason, and rebuilt per
process, since what it holds is a compiled expression.
"""

_TARGET_TOKENS = itertools.count()
"""Hands each :class:`AdapterTarget` something to be keyed by in :data:`_HOP_MODEL`.
Not the memory's name: two specs of one session may well name the same memory and
give it a different network, or the same network a different energy. Not `id` either,
which is reused as soon as a target is collected. A token travels with a target into
a worker process, so the workers of one join share its entries and no two targets
ever share an entry."""

_VOLUME = sympy.Symbol("adapter_volume_per_destination")
"""Stands in for the per-destination volume while an adaptation's latency is built, so
that one expression covers every row: hops and traffic are both linear in the volume,
and the longest route does not depend on it at all."""


class AdapterUnavailable(Exception):
    """
    Raised when a rebinding that the architecture allows cannot be costed. The mapper
    turns this into "these two do not fuse", the same answer it gives when adaptation
    is switched off, rather than quietly charging nothing for real data movement.
    """


def _as_number(value, what: str) -> float:
    """
    `value` as a plain number. The topology model's helpers return sympy numbers where
    they take a max or a min, and a sympy number multiplied through a dataframe column
    gives an object-typed column rather than a numeric one.
    """
    try:
        return float(value)
    except TypeError:
        raise AdapterUnavailable(
            f"{what} came out as {value!r}, which is not a number, so the cost of "
            f"the adaptation cannot be worked out"
        ) from None


def _as_numbers(values: dict, what: str) -> dict:
    """A per-spatial-dimension quantity with every entry as a plain number."""
    return {dim: _as_number(value, what) for dim, value in values.items()}


@dataclass(frozen=True)
class AdapterTarget:
    """
    One physically-distributed memory and the network below it: everything about the
    architecture that costing a rebinding of one of its tensors needs.

    The components are the flattened-arch ones, whose expressions are already
    evaluated and whose ``_physical_spatial`` says how the memory's instances are
    laid out -- which is exactly what the topology model asks a source component for.
    """

    memory: arch.Memory
    network: arch.Network
    size: int | float
    """The memory's size in bits. The mapper carries a tensor's occupancy as a
    fraction of it, so this is what turns that fraction back into bits."""
    may_adapt: tuple[Any, ...]
    """This memory's `BindingConstraints`, one per flattened arch that holds the
    memory. Each was evaluated against its own Einsum, so they may not agree, and a
    tensor is adaptable only if every Einsum that touches it says so."""
    token: int = field(default_factory=lambda: next(_TARGET_TOKENS), repr=False)
    """What :data:`_HOP_MODEL` keys this target by."""

    def allows(self, tensor: TensorName) -> bool:
        """
        Whether `tensor` may be re-laid-out in this memory.

        Only the Einsums that touch `tensor` get a say. The constraints were evaluated
        per-Einsum, so an Einsum that has never heard of the tensor resolves a set
        expression like ``All`` to a set without it -- which is silence, not a refusal.
        An Einsum's `full_space` is the tensors it could have named.
        """
        asked = 0
        for constraints in self.may_adapt:
            space = getattr(constraints.may_adapt, "full_space", None)
            if space is not None and tensor not in space:
                continue
            asked += 1
            if not constraints.may_adapt_binding(tensor):
                return False
        return asked > 0


class BindingAdaptation:
    """
    The rebinding of one tensor of one memory, from the binding the left side of a
    join leaves it in to the one the right side wants, and what that costs.

    The hop counts are worked out once, from the architecture and the two bindings,
    and shared with every other adaptation that moves data the same way (see
    :data:`_HOP_MODEL`); only the volume and the number of tiles vary from row to row,
    and both scale the answer linearly.
    """

    def __init__(
        self,
        tensor: TensorName,
        target: AdapterTarget,
        left: TensorReservation,
        right: TensorReservation,
        tiles_columns: tuple[str, ...],
        symbol_table: dict,
    ):
        self.tensor = tensor
        self.target = target
        self.loops: tuple[tuple[Loop, Loop], ...] = tuple(
            zip(left.physical_spatial_loops, right.physical_spatial_loops)
        )
        self.tiles_columns = tiles_columns
        """Columns whose product is how many times the tile is refilled, and so how
        many times the adaptation happens."""
        self._symbol_table = symbol_table
        self._iterations = self._iteration_terms()
        self._check_architecture()
        if all(not isinstance(n, str) for n in self._iterations):
            # Every binding loop says how many times it runs without having to look at
            # a row, so the whole hop model is settled here and now and every row of
            # every group this adapter is ever applied to is charged from it.
            self._per_unit_volume(tuple(self._iterations))

    def __repr__(self):
        def binding(index):
            return "/".join(
                f"{pair[index].spatial_dim}:{pair[index].rank_name}"
                for pair in self.loops
            )

        return (
            f"BindingAdaptation({self.tensor} in {self.target.memory.name}: "
            f"{binding(0)} -> {binding(1)} over {self.target.network.name})"
        )

    # =========================================================================
    # The hop model
    # =========================================================================
    def _iteration_terms(self) -> list[str | int | float]:
        """
        How many iterations each binding loop runs, as either a number or the name of
        the column holding it. Taken from the left side, which is the side the adapter
        is charged against and so the only one whose columns are in hand; the join
        equates the two sides' counts, so either would say the same thing.
        """
        terms = []
        for left, _ in self.loops:
            n = left.tile_pattern.calculated_n_iterations
            if n is None:
                raise AdapterUnavailable(
                    f"binding loop {left} of {self.tensor} does not say how many "
                    f"times it iterates, so how far its data has to move is unknown"
                )
            terms.append(n)
        return terms

    def _check_architecture(self) -> None:
        """Fail now, rather than in a worker process partway through a join, if the
        memory's instances are not laid out along the dimensions the binding binds."""
        memory = self.target.memory
        for _, right in self.loops:
            if right.spatial_dim is None:
                raise AdapterUnavailable(
                    f"binding loop {right} of {self.tensor} does not say which "
                    f"spatial dimension it binds over"
                )
            if not memory._has_physical_dim(right.spatial_dim):
                raise AdapterUnavailable(
                    f"{memory.name} has no physical spatial dimension "
                    f"{right.spatial_dim}, so how its instances are laid out along "
                    f"{right.spatial_dim} is unknown"
                )
        size = self.target.size
        if size is None or size != size or size == float("inf"):
            raise AdapterUnavailable(
                f"{memory.name} has size {size}, so the fraction of it that "
                f"{self.tensor} occupies cannot be turned back into bits"
            )
        network = self.target.network
        declared = {action.name for action in network.actions}
        # A dimension the network gives no `hop_<dim>` action of its own falls
        # back to the plain `hop`, so that one has to be there unless every
        # dimension this binding moves over names its own.
        needed = oset(
            network._hop_action_name(right.spatial_dim) for _, right in self.loops
        )
        missing = sorted(name for name in needed if name not in declared)
        if missing:
            raise AdapterUnavailable(
                f"network {network.name} has no "
                f"{' or '.join(repr(name) for name in missing)} action, so moving "
                f"data over it cannot be charged"
            )
        for name in needed:
            if network.actions[name].energy is None:
                raise AdapterUnavailable(
                    f"the {name!r} action of network {network.name} has no energy"
                )

    def required_columns(self) -> list[str]:
        """The columns of the left side's pmappings that costing this adaptation
        reads."""
        columns = [tensor2col(self.tensor), *self.tiles_columns]
        columns += [n for n in self._iterations if isinstance(n, str)]
        return columns

    def validate(self, available_columns) -> None:
        """
        Raise :class:`AdapterUnavailable` unless every column this adaptation reads is
        one the left side carries.
        """
        missing = [c for c in self.required_columns() if c not in available_columns]
        if missing:
            raise AdapterUnavailable(
                f"the pmappings being adapted do not carry {sorted(missing)}, so how "
                f"much of {self.tensor} has to move, and how far, is unknown"
            )

    def _hops(self, iterations: tuple[int | float, ...]) -> tuple[dict, dict, dict]:
        """
        ``(hops_per_unit_volume, max_hops, link_traffic_per_unit_volume)`` of one
        adaptation, given how many times each binding loop iterates. All three are
        per spatial dimension, since a network's links need not be alike in every
        direction and each dimension may charge its own ``hop_<dim>`` action.

        The binding loops are walked innermost-first, exactly as
        :class:`~accelforge.model._looptree.reuse.symbolic._network.NetworkAnalyzer`
        walks the spatial loops they pair with: each level's own transfer cost is
        added to the level below it repeated once per iteration, and the volume that
        one destination of the next level up holds grows by the loops inside it that
        partition the tensor rather than replicate it.
        """
        # Imported here rather than at the top of the module: the model package pulls
        # in the mapper, which is what is importing this.
        from accelforge.model._looptree.reuse.symbolic._network import (
            get_topology_model,
        )

        model = get_topology_model(self.target.network.topology)
        memory = self.target.memory

        # Per dimension, how many ways the binding loops already walked fan it out.
        # Times the memory's physical stride, that is how far apart the destinations of
        # the level being walked are.
        fanout_below: dict[str, int | float] = {}
        traffic: dict[str, Any] = {}
        total: dict[str, Any] = {}
        # How much one destination of the level being walked holds, in multiples of
        # what one instance ends up holding.
        volume = 1
        max_hops: dict[str, Any] = {}

        for (left, right), n in zip(reversed(self.loops), reversed(iterations)):
            dim = right.spatial_dim
            cost = model.per_loop_transfer_cost(
                left.rank_name,
                (
                    Irrelevant()
                    if right.rank_name == RANK_DUPLICATE
                    else Relevant(right.rank_name)
                ),
                shape_repeats=n,
                last_fanout=(
                    memory._get_physical_stride_along(dim) * fanout_below.get(dim, 1)
                ),
                volume=volume,
                src_component=memory,
                dim_name=dim,
            )
            # The levels already walked sit below this one, so their hops are
            # crossed once per iteration of it, whichever dimension they were on.
            total = {d: v * n for d, v in total.items()}
            total[dim] = cost.total_cost + total.get(dim, 0)
            max_hops[dim] = model.accumulate_max_hops(
                self.target.network.name, dim, cost.max_hops
            )
            traffic[dim] = cost.max_traffic + traffic.get(dim, 0)

            fanout_below[dim] = fanout_below.get(dim, 1) * n
            if right.rank_name != RANK_DUPLICATE:
                volume = volume * n

        return (
            _as_numbers(total, "the hop count of the adaptation"),
            _as_numbers(max_hops, "the longest route of the adaptation"),
            _as_numbers(traffic, "the busiest link of the adaptation"),
        )

    def _replication(
        self, side: int, iterations: tuple[int | float, ...]
    ) -> dict[str, float]:
        """
        How many times over one side's binding replicates the tensor along each
        spatial dimension: the product of the iteration counts of its binding loops
        that bind no rank at all. The same quantity as
        :func:`~accelforge.model._looptree.reuse.symbolic._binding.replicated_binding_iterations`,
        which is what divided the occupancies this adapter reads.
        """
        replication: dict[str, float] = {}
        for pair, n in zip(self.loops, iterations):
            if pair[side].rank_name != RANK_DUPLICATE:
                continue
            dim = pair[1].spatial_dim
            replication[dim] = replication.get(dim, 1) * n
        return replication

    def _rebinding_factor(self, iterations: tuple[int | float, ...]) -> float:
        """
        What the left side's occupancy of the tensor has to be multiplied by to give
        what one instance holds once the data has been moved, which is the volume the
        hop model is written in terms of.

        Both sides hold the same tile of the same tensor below the same fused loops,
        so their occupancies differ only in how many instances share it -- and that,
        per :func:`~accelforge.model._looptree.reuse.symbolic._symbolic.analyze_storage`,
        is the fanout below the memory divided by the greater of the memory's own
        stride and how many times the binding replicates. The fanouts are the same on
        both sides, since the join equates the binding loops' iteration counts, so
        they cancel and only the replication is left.
        """
        left = self._replication(0, iterations)
        right = self._replication(1, iterations)
        factor = 1.0
        for dim in {**left, **right}:
            stride = self.target.memory._get_physical_stride_along(dim)
            what = f"how many instances of {self.target.memory.name} share {self.tensor}"
            factor *= _as_number(max_nonzero(right.get(dim, 1), stride), what)
            factor /= _as_number(max_nonzero(left.get(dim, 1), stride), what)
        return factor

    # =========================================================================
    # Turning hops into energy and latency
    # =========================================================================
    def _energy_per_unit_volume(self, hops: dict) -> Any:
        """
        What one unit of volume's worth of hops costs, the way
        :func:`~accelforge.model._looptree.energy.gather_actions` and
        :func:`~accelforge.model._looptree.energy.compute_energy_from_actions` cost
        every other hop: each dimension's hop count, scaled by the component's
        ``actions_scale``, times the energy of the hop action that dimension
        resolves to.
        """
        network = self.target.network
        scale = getattr(network, "actions_scale", 1)
        return sum(
            count * scale * network.actions[network._hop_action_name(dim)].energy
            for dim, count in hops.items()
        )

    def _latency_expression(self, max_hops: dict, traffic: dict) -> Any:
        """
        What the adaptation's traffic takes to move, in terms of :data:`_VOLUME`,
        costed the way the model costs every other transfer over the network. A
        network is latency-bound by its longest route, at the ``hop`` action's
        ``latency`` per hop -- what
        :func:`~accelforge.model._looptree.latency.memory.communication_latency`
        charges for winding data down and back up -- and bandwidth-bound by the
        traffic over its most congested link, at the ``hop`` action's ``throughput``
        -- what :func:`~accelforge.model._looptree.latency.memory.component_latency`
        charges its ``hop`` actions. Whichever of the two binds is what the shuffle
        takes.

        Both terms are worked out per dimension and combined the way those two
        functions combine them. A route crosses this dimension's links and then
        the next one's, so the wind-up latencies add. The links themselves are
        separate wires carrying their traffic side by side, so the bandwidth term
        is whichever dimension takes longest -- with a single ``hop`` action
        every dimension divides by the same throughput and this is the busiest
        link, as it was before there were any others.
        """
        network = self.target.network
        scale = getattr(network, "actions_scale", 1)

        latency_bound = sum(
            hops * network.actions[network._hop_action_name(dim)].latency
            for dim, hops in max_hops.items()
        )

        bandwidth_bound = MaxGeqZero(
            *(
                dim_traffic
                * _VOLUME
                * scale
                / network.actions[network._hop_action_name(dim)].throughput
                for dim, dim_traffic in traffic.items()
            )
        )

        return Max(latency_bound, bandwidth_bound)

    def _per_unit_volume(self, iterations: tuple[int | float, ...]):
        """
        ``(energy, latency_of_volume, rebinding_factor)`` for one adaptation of one
        unit of volume, where a unit is what one instance ends up holding.

        The latency is a callable of the volume, since a network may well be
        bandwidth-bound in one row and latency-bound in the next. The factor turns the
        left side's occupancy into that unit; it belongs here because it, too, depends
        on nothing but how many times each binding loop iterates.
        """
        key = self._key(iterations)
        if key not in _HOP_MODEL:
            hops, max_hops, traffic = self._hops(iterations)
            latency = self._latency_expression(max_hops, traffic)
            _HOP_MODEL[key] = (
                self._energy_per_unit_volume(hops),
                # Not the mapper's `_lambdify_type_check`: that one insists on the
                # single-precision the pmapping columns are stored in, and a volume in
                # bits has more significant digits than that.
                sympy.lambdify((_VOLUME,), latency, "numpy"),
                self._rebinding_factor(iterations),
            )
        return _HOP_MODEL[key]

    def _key(self, iterations: tuple[int | float, ...]) -> tuple:
        """This adaptation's entry in :data:`_HOP_MODEL`."""
        return (
            self.target.token,
            tuple(
                (left.rank_name, right.rank_name, right.spatial_dim)
                for left, right in self.loops
            ),
            iterations,
        )

    # =========================================================================
    # Charging a dataframe
    # =========================================================================
    def _occupancy(self, df: pd.DataFrame):
        """
        What one instance of the memory holds of the tensor before the adaptation, in
        actions of the network. The mapper carries a tensor's occupancy as a fraction
        of the memory it is backed in, so the memory's size turns it back into bits,
        and the network's ``bits_per_action`` turns bits into actions the way
        :meth:`~accelforge.model._looptree.reuse.symbolic._network.NetworkAnalyzer._get_data_volume`
        does.
        """
        column = tensor2col(self.tensor)
        if column not in df.columns:
            raise AdapterUnavailable(
                f"the pmappings being adapted do not carry the occupancy of "
                f"{self.tensor}, so how much data has to move is unknown"
            )
        bits = df[column].astype(float) * self.target.size
        bits_per_action = self.target.network.bits_per_action
        return bits if not bits_per_action else bits / bits_per_action

    def _tiles(self, df: pd.DataFrame):
        """How many times the adaptation happens: once per tile the producing side
        leaves in the memory, i.e. once per iteration of the fused loops above the
        tensor's reservation."""
        tiles = pd.Series(1.0, index=df.index)
        for column in self.tiles_columns:
            tiles = tiles * df[column].astype(float)
        return tiles

    def cost(self, df: pd.DataFrame) -> tuple[Any, Any]:
        """
        ``(energy, latency)`` of adapting this tensor, for every row of the left side's
        pmappings.

        Rows are grouped by how many times each binding loop iterates, because that
        is what decides how far the data has to travel; within a group only the
        volume differs, and the hop counts scale with it. Rows that vary in nothing
        but their tile shapes are therefore one group, charged by one expression.
        """
        occupancy = self._occupancy(df)
        tiles = self._tiles(df)
        energy = pd.Series(0.0, index=df.index)
        latency = pd.Series(0.0, index=df.index)

        terms = self._iterations
        columns = [t for t in terms if isinstance(t, str)]

        def charge(rows, iterations: tuple):
            per_volume_energy, latency_of_volume, factor = self._per_unit_volume(
                iterations
            )
            volume = occupancy[rows] * factor
            energy.loc[rows] = volume * tiles[rows] * per_volume_energy
            latency.loc[rows] = tiles[rows] * latency_of_volume(volume.to_numpy())

        if not columns:
            charge(df.index, tuple(terms))
            return energy, latency

        for values, group in df.groupby(columns, sort=False):
            if not isinstance(values, tuple):
                values = (values,)
            per_column = dict(zip(columns, values))
            charge(
                group.index,
                tuple(per_column[t] if isinstance(t, str) else t for t in terms),
            )
        return energy, latency


ENERGY_COLUMNS = ("Total<SEP>energy", "Total<SEP>dynamic_energy")
LATENCY_COLUMN = "Total<SEP>latency"


def charge_adaptations(
    df: pd.DataFrame, adaptations: Iterable[BindingAdaptation]
) -> None:
    """
    Add what `adaptations` cost to the objectives of a dataframe, in place.

    Several tensors being re-laid-out at once are charged one after the other rather
    than at the same time, which is how the joiner treats the latency of two Einsums
    as well: it sums them.
    """
    adaptations = list(adaptations)
    if not adaptations or df.empty:
        return
    energy = pd.Series(0.0, index=df.index)
    latency = pd.Series(0.0, index=df.index)
    for adaptation in adaptations:
        adaptation_energy, adaptation_latency = adaptation.cost(df)
        energy = energy + adaptation_energy
        latency = latency + adaptation_latency

    for column in ENERGY_COLUMNS:
        if column in df.columns:
            add_to_col(df, column, energy)
            break
    if LATENCY_COLUMN in df.columns:
        add_to_col(df, LATENCY_COLUMN, latency)


class BindingAdapter:
    """
    One step of a join: the left side of it, laid out the way one binding of the right
    side wants it.

    Built from the two sides' compatibilities alone, before any pmapping is looked at,
    and then applied to each group of the left bucket it leads out of. Applying it is
    itself a join -- a new group, with a compatibility of its own and a cost added to
    its pmappings -- which is what lets the joiner go on to join the result with the
    right side by the ordinary path, knowing nothing about adapters.
    """

    def __init__(
        self,
        source: Compatibility,
        target: Compatibility,
        adaptations: Iterable[BindingAdaptation],
    ):
        self.source = source
        """The compatibility of the groups this adapter is applied to."""
        self.target = target
        """The compatibility whose binding it moves them into: a key of the right side
        of the join, whose tile patterns the joiner has already cleared, so nothing but
        its bindings may be read from it."""
        self.adaptations: tuple[BindingAdaptation, ...] = tuple(adaptations)
        """One per tensor the two sides disagree about, charged one after the other."""

    def __repr__(self):
        return f"BindingAdapter({', '.join(repr(a) for a in self.adaptations)})"

    @property
    def tensors(self) -> list[TensorName]:
        return [a.tensor for a in self.adaptations]

    def adapt(self, compatibility: Compatibility) -> Compatibility:
        """
        `compatibility`, bound the way this adapter leaves the data. From here on that
        really is where the data is, so that is what the join carries forward.

        Applies to the compatibility of a group as readily as to the permuted one the
        adapter was built from: a binding loop is not one of the loops a permutation
        moves.
        """
        return compatibility.update_binding_to_match(self.target)

    def required_columns(self) -> list[str]:
        return list(oset(c for a in self.adaptations for c in a.required_columns()))

    def can_charge(self, available_columns: Iterable[str]) -> bool:
        """
        Whether a group carrying `available_columns` has enough in it to be charged
        for this adapter. One that does not is left unadapted, which is the answer the
        mapper gives with no adapter at all: these two do not fuse.
        """
        available_columns = oset(available_columns)
        try:
            for adaptation in self.adaptations:
                adaptation.validate(available_columns)
        except AdapterUnavailable as e:
            logger.info(f"Cannot adapt a binding, so it will not be joined: {e}")
            return False
        return True

    def charge(self, df: pd.DataFrame) -> None:
        """
        Add what this adapter costs to the objectives of `df`, in place.

        Several tensors being re-laid-out at once are charged one after the other rather
        than at the same time, which is how the joiner treats the latency of two Einsums
        as well: it sums them.
        """
        adaptations = list(self.adaptations)
        if not adaptations or df.empty:
            return
        energy = pd.Series(0.0, index=df.index)
        latency = pd.Series(0.0, index=df.index)
        for adaptation in adaptations:
            adaptation_energy, adaptation_latency = adaptation.cost(df)
            energy = energy + adaptation_energy
            latency = latency + adaptation_latency

        for column in ENERGY_COLUMNS:
            if column in df.columns:
                add_to_col(df, column, energy)
                break
        if LATENCY_COLUMN in df.columns:
            add_to_col(df, LATENCY_COLUMN, latency)


def flattened_arches_of(pmappings) -> list[FlattenedArch]:
    """
    The flattened architectures a set of pmappings was modeled against. Taken from the
    pmappings themselves when they carry them, and otherwise from the jobs that made
    them, since the two entry points into joining supply one or the other.
    """
    if pmappings.flattened_arches:
        return list(pmappings.flattened_arches.values())
    arches = []
    for jobs in (pmappings.einsum2jobs or {}).values():
        # The mapper keeps a list of jobs per Einsum; the model, which has one mapping
        # to evaluate, keeps the single job itself.
        for job in jobs if isinstance(jobs, (list, tuple)) else [jobs]:
            if job.flattened_arch is not None:
                arches.append(job.flattened_arch)
    return arches


class BindingAdapters:
    """
    Which rebindings the architecture allows, and what they cost.

    Built once for a join from the flattened architectures the pmappings were modeled
    against, then asked, at each step of the join, for every adapter that leads from a
    bucket of the left side into a binding of the right.
    """

    def __init__(self, spec: Spec, flattened_arches: Iterable[FlattenedArch]):
        self.targets: dict[str, AdapterTarget] = {}
        self.enabled = False

        by_memory: dict[str, list] = {}
        for flattened_arch in flattened_arches:
            for node in flattened_arch:
                if not isinstance(node, arch.Memory) or not node._is_distributed():
                    continue
                by_memory.setdefault(node.name, []).append((flattened_arch, node))

        for name, found in by_memory.items():
            constraints = tuple(node.binding for _, node in found)
            if not any(c.may_adapt for c in constraints):
                # Nothing in this memory may be adapted, so there is no need to look
                # for a network to charge or to complain when there is none.
                continue
            flattened_arch, memory = found[0]
            network = flattened_arch.find_first_of_type_below(
                arch.Network, memory.name, default=None
            )
            if network is None:
                raise ValueError(
                    f"Memory {name} allows the binding of {sorted(memory.binding.may_adapt)} "
                    f"to be adapted between Einsums, but there is no network below it "
                    f"to move the data over. Adapting a binding means shuffling data "
                    f"among the memory's physical instances, which has to be charged "
                    f"to a network."
                )
            self.targets[name] = AdapterTarget(
                memory=memory,
                network=network,
                size=memory.size,
                may_adapt=constraints,
            )
            self.enabled = True

        arch_variables = dict(spec.arch.variables) if spec.arch.variables else {}
        self._symbol_table = {
            **arch_variables,
            **dict(spec.variables or {}),
            "variables": spec.variables,
            "arch_variables": spec.arch.variables,
        }

    def __bool__(self) -> bool:
        return self.enabled

    # =========================================================================
    # One adapter
    # =========================================================================
    def adapter(
        self, source: Compatibility, target: Compatibility
    ) -> BindingAdapter | None:
        """
        The adapter that moves `source` into `target`'s binding, or None if it has no
        work to do or cannot be built at all.

        `source` is the compatibility of a group of the left side of a join, symbols
        and all; `target` is a key of the right side, of which nothing but the
        bindings is read.
        """
        adaptations = self.adaptations(source, target)
        if not adaptations:
            return None
        return BindingAdapter(source, target, adaptations)

    def adaptations(
        self,
        left: Compatibility,
        right: Compatibility,
        available_columns: Iterable[str] | None = None,
    ) -> list[BindingAdaptation] | None:
        """
        The adaptations that turn `left`'s bindings into `right`'s, or None if any
        binding they disagree about may not be adapted -- because the architecture
        forbids it, or because there is not enough in hand to charge for it. Either
        way the answer the mapper gives is the one it gives with no adapter at all:
        these two do not fuse.

        An empty list means the two already agree, i.e. this is an ordinary join.
        """
        try:
            return self._adaptations(left, right, available_columns)
        except AdapterUnavailable as e:
            logger.info(f"Cannot adapt a binding, so it will not be joined: {e}")
            return None

    def _adaptations(
        self,
        left: Compatibility,
        right: Compatibility,
        available_columns: Iterable[str] | None,
    ) -> list[BindingAdaptation] | None:
        result = []
        right_tensors = {(t.name, t.resource_name): t for t in right.tensors}
        for left_tensor in sorted(left.tensors):
            right_tensor = right_tensors.get(
                (left_tensor.name, left_tensor.resource_name)
            )
            if right_tensor is None:
                # Nothing on the right binds it, so there is nothing to disagree with.
                continue
            if not left_tensor.binding_spatial_dim_matches(right_tensor):
                return None
            if not left_tensor.binding_differs_from(right_tensor):
                continue
            target = self.targets.get(left_tensor.resource_name)
            if target is None or not target.allows(left_tensor.name):
                return None
            adaptation = BindingAdaptation(
                tensor=left_tensor.name,
                target=target,
                left=left_tensor,
                right=right_tensor,
                tiles_columns=_tiles_columns(left, left_tensor),
                symbol_table=self._symbol_table,
            )
            if available_columns is not None:
                adaptation.validate(available_columns)
            result.append(adaptation)
        return result

    # =========================================================================
    # A whole step of the join
    # =========================================================================
    def adapt_buckets(
        self,
        left: dict[Compatibility, list],
        right: Mapping[Compatibility, Any],
        print_progress: bool = True,
    ) -> dict[Compatibility, list]:
        """
        `left`, with a group added under every key of `right` that one of its groups
        can be moved into.

        The joiner pairs a left bucket with the right bucket of the same key and
        nothing else, so this is all it takes for an adapted group to be joined: the
        groups this adds are already the left side of an ordinary join, laid out the
        way the right side wants and charged for having been moved. A group is
        adapted, never replaced -- the binding it was modeled with may well be one
        some other bucket of the right side wants as it is.
        """
        todo = self._make_adapters(left, right)
        if not todo:
            return left

        n_adapters = len(oset(id(adapter) for adapter, _, _ in todo))
        logger.info(
            f"Made {n_adapters} binding adapter(s), between them taking {len(todo)} "
            f"group(s) of the left side into a binding of the right"
        )
        adapted = parallel(
            [delayed(group.merge_adapter)(adapter) for adapter, group, _ in todo],
            pbar=f"Adapting bindings for {len(todo)} groups" if print_progress else None,
        )
        for (adapter, _, permutation), group in zip(todo, adapted):
            left.setdefault(adapter.target, []).append((group, permutation))
        return left

    def _make_adapters(
        self, left: Mapping[Compatibility, list], right: Mapping[Compatibility, Any]
    ) -> list[tuple[BindingAdapter, Any, Any]]:
        """
        Every adapter the two sides call for, made from their compatibilities alone
        before a single pmapping is looked at, paired with the group it is to be
        applied to and that group's permutation.

        Two buckets are adaptable into one another exactly when they are equal with
        the bindings set aside, which is the only way to find the pairs: the joiner's
        own bucket keys are what they agree and disagree on. Groups of one bucket that
        describe the same thing share an adapter, since what an adapter costs is
        settled by the compatibilities and the architecture.
        """
        by_binding: dict[Compatibility, list[Compatibility]] = defaultdict(list)
        for key in right:
            by_binding[key.clear_bound_ranks()].append(key)

        made: dict[tuple[Compatibility, Compatibility], BindingAdapter | None] = {}
        todo = []
        for key, groups in left.items():
            targets = [t for t in by_binding.get(key.clear_bound_ranks(), ()) if t != key]
            for target in targets:
                for group, permutation in groups:
                    source = permutation.apply(group.compatibility)
                    if (source, target) not in made:
                        made[(source, target)] = self.adapter(source, target)
                    adapter = made[(source, target)]
                    if adapter is None:
                        continue
                    if not adapter.can_charge(group.mappings.data.columns):
                        continue
                    todo.append((adapter, group, permutation))
        return todo


def _tiles_columns(
    compatibility: Compatibility, tensor: TensorReservation
) -> tuple[str, ...]:
    """
    The iteration-count columns of the fused loops above `tensor`'s reservation. The
    producing side refills the buffer once per iteration of them, and each refill has
    to be adapted again.
    """
    columns = []
    for loop in compatibility.loops[: tensor.above_loop_index]:
        n = loop.tile_pattern.calculated_n_iterations
        if isinstance(n, str):
            columns.append(n)
        elif n is not None and n != 1:
            raise AdapterUnavailable(
                f"fused loop {loop} above {tensor.name} runs {n} times but does not "
                f"say so in a column, so how many times its tile is re-laid-out "
                f"cannot be read off the pmappings"
            )
    return tuple(columns)
