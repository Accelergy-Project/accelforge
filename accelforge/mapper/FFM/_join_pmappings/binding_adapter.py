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

One thing an adaptation is *not* charged for: how much of the memory it needs. A
binding decides how much of a tensor one instance holds -- a replicated one holds a
whole copy where a partitioned one holds a share -- so the two sides of an adaptation
may disagree about the tensor's occupancy, and the joiner carries the producing side's
figure forward as it does for any shared tensor. The movement is charged; the
occupancy is the left side's.
"""

import logging
from dataclasses import dataclass
from typing import Any, Iterable

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
from accelforge.util._basetypes import EvalableList
from accelforge.util._eval_expressions import eval_expression
from accelforge.util._sympy.broadcast_max import Max, MaxGeqZero, Min

logger = logging.getLogger(__name__)

RIGHT_MERGE_SUFFIX = "_RIGHT_MERGE"
"""The suffix `PmappingDataframe.merge_next` gives the right side's copy of a column
that both sides have. The right side's occupancy of the adapted tensor is what one
instance ends up holding, so that is the column the volume comes from."""

_VOLUME = sympy.Symbol("adapter_volume_per_destination")
"""Stands in for the per-destination volume while the network's `total_latency`
expression is evaluated, so that one evaluation covers every row: hops and traffic
are both linear in the volume, and the longest route does not depend on it at all."""


# `max`, `min` and `sum` as a component's `total_latency` expression sees them. Spelled
# out here rather than imported from
# :mod:`accelforge.model._looptree.latency.memory`, whose versions build symengine
# expressions: an adapter's latency is a sympy expression of :data:`_VOLUME`, and the
# two libraries do not mix.
def _sum(*args):
    if len(args) == 1 and hasattr(args[0], "__iter__"):
        args = tuple(args[0])
    return sum(args) if args else 0


def _max(*args):
    if len(args) == 1 and hasattr(args[0], "__iter__"):
        args = tuple(args[0])
    return Max(*args)


def _min(*args):
    if len(args) == 1 and hasattr(args[0], "__iter__"):
        args = tuple(args[0])
    return Min(*args)


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

    The hop counts are worked out once, from the architecture and the two bindings;
    only the volume and the number of tiles vary from row to row, and both scale the
    answer linearly.
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
        self._cache: dict[tuple, tuple] = {}
        self._iterations = self._iteration_terms()
        self._check_architecture()

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
        the column holding it. Taken from the right side, whose binding is the one the
        data ends up in; the join equates the two sides' counts, so either would do.
        """
        terms = []
        for _, right in self.loops:
            n = right.tile_pattern.calculated_n_iterations
            if n is None:
                raise AdapterUnavailable(
                    f"binding loop {right} of {self.tensor} does not say how many "
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
        if "hop" not in {action.name for action in network.actions}:
            raise AdapterUnavailable(
                f"network {network.name} has no 'hop' action, so moving data over it "
                f"cannot be charged"
            )
        if network.actions["hop"].energy is None:
            raise AdapterUnavailable(
                f"the 'hop' action of network {network.name} has no energy"
            )

    def required_columns(self) -> list[str]:
        """The columns of a joined dataframe that costing this adaptation reads."""
        columns = [tensor2col(self.tensor), *self.tiles_columns]
        columns += [n for n in self._iterations if isinstance(n, str)]
        return columns

    def validate(self, available_columns) -> None:
        """
        Raise :class:`AdapterUnavailable` unless every column this adaptation reads
        will be there once the two sides are merged.
        """
        missing = [c for c in self.required_columns() if c not in available_columns]
        if missing:
            raise AdapterUnavailable(
                f"the pmappings being joined do not carry {sorted(missing)}, so how "
                f"much of {self.tensor} has to move, and how far, is unknown"
            )

    def _hops(self, iterations: tuple[int | float, ...]) -> tuple[Any, Any, Any]:
        """
        ``(hops_per_unit_volume, max_hops, max_link_traffic_per_unit_volume)`` of one
        adaptation, given how many times each binding loop iterates.

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
        total = 0
        # How much one destination of the level being walked holds, in multiples of
        # what one instance ends up holding.
        volume = 1
        max_hops = 0

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
            total = cost.total_cost + total * n
            max_hops = model.accumulate_max_hops(self.target.network.name, cost.max_hops)
            traffic[dim] = cost.max_traffic + traffic.get(dim, 0)

            fanout_below[dim] = fanout_below.get(dim, 1) * n
            if right.rank_name != RANK_DUPLICATE:
                volume = volume * n

        link_traffic = MaxGeqZero(*traffic.values()) if traffic else 0
        return (
            _as_number(total, "the hop count of the adaptation"),
            _as_number(max_hops, "the longest route of the adaptation"),
            _as_number(link_traffic, "the busiest link of the adaptation"),
        )

    # =========================================================================
    # Turning hops into energy and latency
    # =========================================================================
    def _energy_per_unit_volume(self, hops) -> Any:
        """
        What one unit of volume's worth of hops costs, the way
        :func:`~accelforge.model._looptree.energy.gather_actions` and
        :func:`~accelforge.model._looptree.energy.compute_energy_from_actions` cost
        every other hop: the hop count, scaled by the component's ``actions_scale``,
        times the energy of its ``hop`` action.
        """
        network = self.target.network
        return (
            hops
            * getattr(network, "actions_scale", 1)
            * network.actions["hop"].energy
        )

    def _latency_expression(self, hops, max_hops, traffic) -> Any:
        """
        The network's own ``total_latency`` expression, evaluated with this
        adaptation's hops and traffic, in terms of :data:`_VOLUME`. This is the same
        expression, and the same ``max_hops`` / ``max_link_traffic`` keywords, that
        :func:`~accelforge.model._looptree.latency.memory.component_latency` gives a
        network when the model runs it.
        """
        network = self.target.network
        scale = getattr(network, "actions_scale", 1)
        actions = EvalableList()
        for action in network.actions:
            action = action.model_copy()
            action._set_n_calls((hops * _VOLUME * scale) if action.name == "hop" else 0)
            actions.append(action)

        dump = network.shallow_model_dump(include_None=True)
        dump["actions"] = actions
        dump["max_hops"] = max_hops
        dump["max_link_traffic"] = traffic * _VOLUME
        symbol_table = {
            **self._symbol_table,
            **dump,
            "max": _max,
            "min": _min,
            "sum": _sum,
        }
        return eval_expression(
            network.total_latency,
            symbol_table,
            attr_name="total_latency",
            location=network.name,
        )

    def _per_unit_volume(self, iterations: tuple[int | float, ...]):
        """``(energy, latency_of_volume)`` for one adaptation of one unit of volume.
        The latency is a callable of the volume, since a network may well be
        bandwidth-bound in one row and latency-bound in the next."""
        if iterations not in self._cache:
            hops, max_hops, traffic = self._hops(iterations)
            latency = self._latency_expression(hops, max_hops, traffic)
            self._cache[iterations] = (
                self._energy_per_unit_volume(hops),
                # Not the mapper's `_lambdify_type_check`: that one insists on the
                # single-precision the pmapping columns are stored in, and a volume in
                # bits has more significant digits than that.
                sympy.lambdify((_VOLUME,), latency, "numpy"),
            )
        return self._cache[iterations]

    # =========================================================================
    # Charging a joined dataframe
    # =========================================================================
    def _volume(self, df: pd.DataFrame):
        """
        The volume one destination ends up holding, in actions of the network. The
        mapper carries a tensor's occupancy as a fraction of the memory it is backed
        in, so the memory's size turns it back into bits, and the network's
        ``bits_per_action`` turns bits into actions the way
        :meth:`~accelforge.model._looptree.reuse.symbolic._network.NetworkAnalyzer._get_data_volume`
        does.
        """
        column = tensor2col(self.tensor)
        right = column + RIGHT_MERGE_SUFFIX
        column = right if right in df.columns else column
        if column not in df.columns:
            raise AdapterUnavailable(
                f"the joined pmappings do not carry the occupancy of {self.tensor}, "
                f"so how much data has to move is unknown"
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
        ``(energy, latency)`` of adapting this tensor, for every row of a joined
        dataframe.

        Rows are grouped by how many times each binding loop iterates, because that
        is what decides how far the data has to travel; within a group only the
        volume differs, and the hop counts scale with it.
        """
        volume = self._volume(df)
        tiles = self._tiles(df)
        energy = pd.Series(0.0, index=df.index)
        latency = pd.Series(0.0, index=df.index)

        terms = self._iterations
        columns = [t for t in terms if isinstance(t, str)]

        def charge(rows, iterations: tuple):
            per_volume_energy, latency_of_volume = self._per_unit_volume(iterations)
            energy.loc[rows] = volume[rows] * tiles[rows] * per_volume_energy
            latency.loc[rows] = tiles[rows] * latency_of_volume(volume[rows].to_numpy())

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
    Add what `adaptations` cost to the objectives of a joined dataframe, in place.

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

    Built once for a join from the flattened architectures the pmappings were
    modeled against, then asked, for every pair of compatibilities the join
    considers, whether their bindings can be adapted into one another.
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

    def adaptations(
        self,
        left: Compatibility,
        right: Compatibility,
        available_columns: Iterable[str] = (),
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
        available_columns: Iterable[str],
    ) -> list[BindingAdaptation] | None:
        result = []
        right_tensors = {t.name: t for t in right.tensors}
        for left_tensor in sorted(left.tensors):
            right_tensor = right_tensors.get(left_tensor.name)
            if right_tensor is None:
                # Nothing on the right binds it, so there is nothing to disagree with.
                continue
            if not left_tensor.binding_lines_up_with(right_tensor):
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
            adaptation.validate(available_columns)
            result.append(adaptation)
        return result


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
