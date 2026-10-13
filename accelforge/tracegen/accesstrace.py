"""
Trace which tensor elements a LoopTree mapping touches at each timestep.

The entry point is :func:`trace_accesses`, which walks a
:class:`~accelforge.frontend.mapping.Mapping` and returns an :class:`AccessTrace`:
a record of ``(timestep, tensor element)`` pairs, plus the timestep spans over which
each memory level holds one tile. Feeding that to
:func:`accelforge.plotting.accesstrace.plot_access_trace` gives the classic
"iteration space vs. data space" picture, from which reuse, tiling, and fusion
behavior are directly readable.
"""

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import sympy

from accelforge._accelerated_imports import numpy as np

from accelforge.frontend.mapping import (
    Compute,
    Loop,
    Mapping,
    MappingNode,
    Nested,
    Sequential,
    Spatial,
    Split,
    Storage,
    Temporal,
)
from accelforge.frontend.workload import (
    Einsum,
    EinsumName,
    Rank,
    RankVariable,
    TensorName,
    Workload,
)

__all__ = [
    "AccessTrace",
    "TensorAccessTrace",
    "trace_accesses",
]


DEFAULT_MAX_POINTS = 2_000_000


# ======================================================================================
# Result types
# ======================================================================================
@dataclass
class TensorAccessTrace:
    """
    The accesses that one Einsum makes to one tensor, as a flat list of
    ``(timestep, element)`` pairs.

    All arrays share a length: entry ``i`` of every array describes the same access.
    Duplicate accesses within a single timestep are removed, so an element that is
    read many times in one timestep appears once.
    """

    tensor: TensorName
    """ The name of the tensor being accessed. """
    einsum: EinsumName
    """ The name of the Einsum making the accesses. """
    is_output: bool
    """ Whether this Einsum writes the tensor (``True``) or reads it (``False``). """

    timestep: "np.ndarray"
    """ Integer array of timesteps at which each access happens. """
    element: "np.ndarray"
    """ Integer array of flattened (row-major) tensor element indices. """
    coordinates: "np.ndarray"
    """ ``(n_accesses, n_ranks)`` integer array of per-rank tensor coordinates. """

    ranks: tuple[Rank, ...]
    """ The names of the tensor's ranks, in the order used by ``coordinates``. """
    tensor_shape: tuple[int, ...]
    """ The size of each of the tensor's ranks, in the order used by ``coordinates``. """

    @property
    def n_accesses(self) -> int:
        """The number of distinct (timestep, element) accesses in this trace."""
        return int(self.timestep.size)

    @property
    def tensor_size(self) -> int:
        """The number of elements in the tensor."""
        return int(np.prod(self.tensor_shape)) if self.tensor_shape else 1

    def __repr__(self) -> str:
        kind = "writes" if self.is_output else "reads"
        return (
            f"TensorAccessTrace({self.einsum} {kind} {self.tensor}"
            f"[{'x'.join(map(str, self.tensor_shape))}], "
            f"{self.n_accesses} accesses)"
        )


@dataclass
class AccessTrace:
    """
    A full record of which tensor elements a mapping accesses at each timestep.

    A *timestep* is one iteration of the innermost temporal loop, i.e. one step of
    the sequential schedule that the LoopTree describes. Spatial loops do not advance
    the timestep, so spatially-parallel accesses share an x-coordinate.
    """

    traces: list[TensorAccessTrace] = field(default_factory=list)
    """ One entry per (Einsum, tensor) pair that the mapping touches. """
    n_timesteps: int = 0
    """ The total number of timesteps in the mapping. """
    tensor_shapes: dict[TensorName, tuple[int, ...]] = field(default_factory=dict)
    """ The shape of each traced tensor. """
    tensor_ranks: dict[TensorName, tuple[Rank, ...]] = field(default_factory=dict)
    """ The rank names of each traced tensor, ordered to match ``tensor_shapes``. """
    einsum_timespans: dict[EinsumName, tuple[int, int]] = field(default_factory=dict)
    """ ``{einsum: (first_timestep, last_timestep + 1)}`` for each Einsum. """
    tile_windows: dict[str, dict[TensorName, list[tuple[int, int]]]] = field(
        default_factory=dict
    )
    """
    ``{memory_level: {tensor: [(first_timestep, last_timestep + 1), ...]}}``: the
    timestep spans over which one tile of the tensor stays resident in that memory
    level. A memory level is the ``component`` of a
    :class:`~accelforge.frontend.mapping.Storage` node, and a new span starts whenever a
    loop outside that node advances.
    """
    truncated: bool = False
    """ Whether the trace was cut short by the ``max_timesteps`` argument. """

    @property
    def tensors(self) -> list[TensorName]:
        """The names of all traced tensors, in the order they are first accessed."""
        seen = {}
        for t in self.traces:
            seen.setdefault(t.tensor, None)
        return list(seen)

    @property
    def einsums(self) -> list[EinsumName]:
        """The names of all traced Einsums, in the order they are first executed."""
        return list(self.einsum_timespans)

    @property
    def memory_levels(self) -> list[str]:
        """
        The names of the storage components the mapping holds tensors in, outermost
        first.
        """
        return list(self.tile_windows)

    def for_tensor(self, tensor: TensorName) -> list[TensorAccessTrace]:
        """Return every :class:`TensorAccessTrace` that touches ``tensor``."""
        return [t for t in self.traces if t.tensor == tensor]

    def tile_lifetime(self, memory_level: str, tensor: TensorName) -> list[tuple[int, int]]:
        """
        The timestep spans over which one tile of ``tensor`` stays resident in
        ``memory_level``. Empty if that level never holds the tensor.
        """
        level = self.tile_windows.get(str(memory_level), {})
        return list(level.get(TensorName(tensor), []))

    def to_dataframe(self):
        """
        Return a tidy ``pandas.DataFrame`` with one row per access and columns
        ``einsum``, ``tensor``, ``is_output``, ``timestep``, and ``element``.
        """
        from accelforge._accelerated_imports import pandas as pd

        frames = [
            pd.DataFrame(
                {
                    "einsum": t.einsum,
                    "tensor": t.tensor,
                    "is_output": t.is_output,
                    "timestep": t.timestep,
                    "element": t.element,
                }
            )
            for t in self.traces
        ]
        if not frames:
            return pd.DataFrame(
                columns=["einsum", "tensor", "is_output", "timestep", "element"]
            )
        return pd.concat(frames, ignore_index=True)

    def __repr__(self) -> str:
        return (
            f"AccessTrace({len(self.traces)} tensor traces, "
            f"{self.n_timesteps} timesteps"
            f"{', truncated' if self.truncated else ''})"
        )


# ======================================================================================
# Public entry point
# ======================================================================================
def trace_accesses(
    spec_or_mapping,
    workload: Workload | None = None,
    tensors: Iterable[TensorName] | None = None,
    einsums: Iterable[EinsumName] | None = None,
    max_timesteps: int | None = None,
    max_points: int = DEFAULT_MAX_POINTS,
) -> AccessTrace:
    """
    Trace which tensor elements a LoopTree mapping accesses at each timestep.

    Parameters
    ----------
    spec_or_mapping:
        Either a :class:`~accelforge.frontend.spec.Spec` (whose ``mapping`` and
        ``workload`` are used), a :class:`~accelforge.mapper.FFM.Mappings` result
        (whose single mapping is used), or a
        :class:`~accelforge.frontend.mapping.Mapping`. If a bare ``Mapping`` is
        given, ``workload`` is required.
    workload:
        The workload the mapping targets. Required only when ``spec_or_mapping``
        is a bare ``Mapping``.
    tensors:
        If given, only trace these tensors.
    einsums:
        If given, only trace these Einsums. Timesteps are still numbered as if all
        Einsums ran, so traces stay aligned on the x-axis.
    max_timesteps:
        If given, trace only the first this-many timesteps. Useful for peeking at
        the start of a schedule too large to enumerate in full.
    max_points:
        Safety limit on the number of iteration-space points enumerated for a single
        Einsum. Raises :class:`ValueError` if exceeded; raise the limit or use
        ``max_timesteps`` to see a prefix instead.

    Returns
    -------
    AccessTrace
        The traced accesses, and the tile-residency spans of every memory level the
        mapping stores tensors in. Pass to
        :func:`accelforge.plotting.accesstrace.plot_access_trace` to visualize.
    """
    mapping, workload = _resolve_mapping_and_workload(spec_or_mapping, workload)

    tensor_filter = None if tensors is None else set(map(str, tensors))
    einsum_filter = None if einsums is None else set(map(str, einsums))
    cache = _Cache(workload)

    state = _Layout()
    n_timesteps, _ = _measure(mapping, cache, state)
    if max_timesteps is not None:
        budget = max(1, int(max_timesteps))
        n_timesteps = min(_clip(mapping, state, budget), budget)

    leaves: list[_Leaf] = []
    holders: list[_Held] = []
    _emit(mapping, state, (), 0, leaves, holders)

    traces: list[TensorAccessTrace] = []
    einsum_timespans: dict[EinsumName, tuple[int, int]] = {}
    # Coordinates are collected first and flattened to element indices afterwards, so
    # that every Einsum touching a tensor agrees on the flattening.
    pending: list[tuple[TensorName, EinsumName, bool, np.ndarray, np.ndarray]] = []

    for leaf in leaves:
        if einsum_filter is not None and leaf.einsum not in einsum_filter:
            continue

        einsum = workload.einsums[leaf.einsum]
        timestep, rank_var_coords = _enumerate_leaf(leaf, cache, max_points)
        if max_timesteps is not None:
            # Clipping loop counts gets close; this makes the cut exact.
            keep = timestep < n_timesteps
            timestep = timestep[keep]
            rank_var_coords = {rv: c[keep] for rv, c in rank_var_coords.items()}
        if timestep.size:
            span = einsum_timespans.get(leaf.einsum)
            new_span = (int(timestep.min()), int(timestep.max()) + 1)
            einsum_timespans[leaf.einsum] = (
                new_span
                if span is None
                else (min(span[0], new_span[0]), max(span[1], new_span[1]))
            )

        for access in einsum.tensor_accesses:
            if tensor_filter is not None and access.name not in tensor_filter:
                continue
            coords = _project(cache, leaf.einsum, access.name, rank_var_coords)
            t, coords = _dedup(timestep, coords)
            pending.append(
                (TensorName(access.name), leaf.einsum, access.output, t, coords)
            )

    tensor_ranks: dict[TensorName, tuple[Rank, ...]] = {}
    tensor_shapes: dict[TensorName, tuple[int, ...]] = {}
    for tensor, _einsum, _out, _t, coords in pending:
        shape_dict = workload.get_tensor_shape(tensor)
        ranks = tuple(shape_dict.keys())
        shape = np.array(list(shape_dict.values()), dtype=np.int64)
        # Projections with halos (e.g. convolution) can reach outside the nominal
        # box, so grow it to cover everything actually touched.
        if coords.size:
            shape = np.maximum(shape, coords.max(axis=0) + 1)
        prev = tensor_shapes.get(tensor)
        tensor_shapes[tensor] = (
            tuple(int(x) for x in shape)
            if prev is None
            else tuple(int(x) for x in np.maximum(shape, np.array(prev)))
        )
        tensor_ranks[tensor] = ranks

    for tensor, einsum, is_output, t, coords in pending:
        shape = tensor_shapes[tensor]
        element = _ravel(coords, shape)
        traces.append(
            TensorAccessTrace(
                tensor=tensor,
                einsum=einsum,
                is_output=is_output,
                timestep=t,
                element=element,
                coordinates=coords,
                ranks=tensor_ranks[tensor],
                tensor_shape=shape,
            )
        )

    return AccessTrace(
        traces=traces,
        n_timesteps=int(n_timesteps),
        tensor_shapes=tensor_shapes,
        tensor_ranks=tensor_ranks,
        einsum_timespans=dict(
            sorted(einsum_timespans.items(), key=lambda kv: kv[1])
        ),
        tile_windows=_storage_windows(holders, int(n_timesteps), tensor_filter),
        truncated=state.truncated,
    )


def _resolve_mapping_and_workload(spec_or_mapping, workload):
    mapping = spec_or_mapping
    if hasattr(mapping, "workload") and hasattr(mapping, "mapping"):  # a Spec
        workload = workload if workload is not None else mapping.workload
        mapping = mapping.mapping
    elif callable(getattr(mapping, "mapping", None)):  # a Mappings result
        mapping = mapping.mapping()

    if not isinstance(mapping, MappingNode):
        raise TypeError(
            f"Expected a Spec, Mappings, or Mapping. Got {type(spec_or_mapping)}."
        )
    if workload is None:
        raise ValueError(
            "A workload is required when tracing a bare Mapping. Pass a Spec, or "
            "supply the workload= argument."
        )
    if not mapping.get_nodes_of_type(Compute):
        raise ValueError(
            "The mapping has no Compute nodes, so there is nothing to trace. Did you "
            "forget to load a mapping into the Spec?"
        )
    return mapping, workload


# ======================================================================================
# Walking the LoopTree
# ======================================================================================
class _Cache:
    """
    Per-trace scratch space. Workloads and Einsums are Pydantic models and are not
    hashable, so the memoized lookups they feed are cached here by name instead.
    """

    def __init__(self, workload: Workload):
        self.workload = workload
        self._bounds: dict[EinsumName, dict[RankVariable, int]] = {}
        self._projections: dict[tuple[EinsumName, TensorName], Any] = {}

    def bounds(self, einsum: EinsumName) -> dict[RankVariable, int]:
        """The extent of each of ``einsum``'s rank variables."""
        if einsum not in self._bounds:
            from accelforge.frontend._workload_isl._isl import get_rank_variable_bounds

            self._bounds[einsum] = dict(
                get_rank_variable_bounds(self.workload, einsum)
            )
        return self._bounds[einsum]

    def projection(self, einsum: EinsumName, tensor: TensorName):
        """
        ``((rank, (rank_variable, ...), callable), ...)`` for projecting ``einsum``'s
        rank variables into ``tensor``'s coordinates.
        """
        key = (einsum, tensor)
        if key not in self._projections:
            from accelforge.frontend._workload_isl._symbolic import get_projection_expr

            exprs = get_projection_expr(self.workload.einsums[einsum], tensor)
            self._projections[key] = tuple(
                (
                    rank,
                    tuple(RankVariable(str(s)) for s in sorted(expr.free_symbols, key=str)),
                    sympy.lambdify(sorted(expr.free_symbols, key=str), expr, "numpy"),
                )
                for rank, expr in exprs.items()
            )
        return self._projections[key]


@dataclass
class _LoopLevel:
    """One enclosing loop, with everything needed to place its iterations in time."""

    loop: Loop
    n_iterations: int
    stride: int
    """ How many timesteps one iteration of this loop consumes. Zero for Spatial. """


@dataclass
class _Held:
    """A Storage node together with the loops that enclose it."""

    node: Storage
    levels: tuple[_LoopLevel, ...]
    """ Only the loops *outside* the node, i.e. the ones that change its tile. """
    t0: int
    """ The timestep at which the node's first tile arrives. """
    duration: int
    """ How many timesteps one tile stays resident. """


@dataclass
class _Leaf:
    """A Compute node together with the loop nest that encloses it."""

    einsum: EinsumName
    levels: tuple[_LoopLevel, ...]
    t0: int
    """ The timestep at which this leaf's first iteration runs. """
    n_timesteps: int
    """ How many timesteps this leaf spans, counting gaps left by sibling branches. """


def _terminal_child(node: Nested) -> MappingNode | None:
    """The single child of a ``Nested`` that continues the tree, if there is one."""
    for child in node.nodes:
        if isinstance(child, (Nested, Split, Compute)):
            return child
    return None


def _loops_of(node: Nested) -> list[Loop]:
    return [child for child in node.nodes if isinstance(child, Loop)]


def _measure(node: MappingNode, cache: _Cache, state: "_Layout") -> tuple[int, str | None]:
    """
    Bottom-up pass: record each loop's iteration count and each node's duration.

    Returns ``(n_timesteps, representative_einsum)``. The representative Einsum names
    the iteration space that the enclosing loops are bound to; loops fused across
    Einsums iterate the same number of times for all of them.
    """
    if isinstance(node, Compute):
        einsum = EinsumName(node.einsum)
        if einsum not in cache.workload.einsum_names:
            raise ValueError(
                f"Compute node targets Einsum '{einsum}', which is not in the "
                f"workload (which has {list(cache.workload.einsum_names)})."
            )
        state.sizes[id(node)] = 1
        return 1, einsum

    if isinstance(node, Split):
        # Sequential branches run one after another; Pipeline/Parallel branches
        # overlap, so the split lasts as long as its longest branch.
        sizes, einsum = [], None
        for child in node.nodes:
            size, child_einsum = _measure(child, cache, state)
            sizes.append(size)
            einsum = einsum or child_einsum
        total = sum(sizes) if isinstance(node, Sequential) else max(sizes, default=0)
        state.sizes[id(node)] = total
        return total, einsum

    if isinstance(node, Nested):
        child = _terminal_child(node)
        if child is None:
            state.sizes[id(node)] = 0
            return 0, None
        size, einsum = _measure(child, cache, state)
        for loop in reversed(_loops_of(node)):
            n = _loop_n_iterations(loop, einsum, cache)
            state.n_iterations[id(loop)] = n
            if isinstance(loop, Temporal):
                size *= n
        state.sizes[id(node)] = size
        return size, einsum

    # Storage / Reservation / TextBox: no iteration of their own.
    state.sizes[id(node)] = 0
    return 0, None


def _clip(node: MappingNode, state: "_Layout", budget: int) -> int:
    """
    Shrink loop iteration counts so that ``node`` fits in ``budget`` timesteps,
    keeping inner loops intact so the result is a genuine prefix of the schedule.
    Returns the clipped duration of ``node``.
    """
    if state.sizes[id(node)] <= budget:
        return state.sizes[id(node)]
    state.truncated = True

    if isinstance(node, Split):
        sequential = isinstance(node, Sequential)
        total, longest = 0, 0
        for child in node.nodes:
            size = _clip(child, state, max(1, budget - total) if sequential else budget)
            longest = max(longest, (total if sequential else 0) + size)
            total += size
        size = total if sequential else longest
    elif isinstance(node, Nested):
        child = _terminal_child(node)
        size = _clip(child, state, budget) if child is not None else 0
        for loop in reversed(_loops_of(node)):
            if not isinstance(loop, Temporal):
                continue
            allowed = max(1, budget // max(size, 1))
            state.n_iterations[id(loop)] = min(state.n_iterations[id(loop)], allowed)
            size *= state.n_iterations[id(loop)]
    else:
        size = state.sizes[id(node)]

    state.sizes[id(node)] = size
    return size


def _emit(
    node: MappingNode,
    state: "_Layout",
    levels: tuple[_LoopLevel, ...],
    t0: int,
    leaves: list[_Leaf],
    holders: list[_Held],
) -> None:
    """
    Top-down pass: give every Compute its loop levels and starting timestep, and every
    Storage node the loops that enclose it.
    """
    if isinstance(node, Compute):
        leaves.append(
            _Leaf(
                einsum=EinsumName(node.einsum),
                levels=levels,
                t0=t0,
                n_timesteps=state.sizes[id(node)],
            )
        )
        return

    if isinstance(node, Split):
        sequential = isinstance(node, Sequential)
        offset = 0
        for child in node.nodes:
            if isinstance(child, Storage):
                # A Storage directly under a Split spans the whole Split.
                holders.append(_Held(child, levels, t0, state.sizes[id(node)]))
                continue
            _emit(child, state, levels, t0 + offset, leaves, holders)
            if sequential:
                offset += state.sizes[id(child)]
        return

    if isinstance(node, Nested):
        child = _terminal_child(node)
        if child is None:
            return
        # A loop's stride is the duration of everything nested inside it, so measure the
        # subtree at each depth from the inside out. `inner[i]` is how long the subtree
        # at or below loop `i` runs; `inner[0]` is the whole Nested.
        loops = _loops_of(node)
        inner = [state.sizes[id(child)]] * (len(loops) + 1)
        for i in reversed(range(len(loops))):
            n = state.n_iterations[id(loops[i])]
            inner[i] = inner[i + 1] * (n if isinstance(loops[i], Temporal) else 1)
        # Spatial iterations happen at the same time as one another, so they get no
        # stride.
        my_levels = tuple(
            _LoopLevel(
                loop,
                state.n_iterations[id(loop)],
                inner[i + 1] if isinstance(loop, Temporal) else 0,
            )
            for i, loop in enumerate(loops)
        )

        # Loops and Storage nodes interleave inside a Nested, and the order of
        # `node.nodes` is the nesting order: the loops listed before a Storage node are
        # the ones outside it, so they are the ones that swap its tile.
        depth = 0
        for sub in node.nodes:
            if isinstance(sub, Loop):
                depth += 1
            elif isinstance(sub, Storage):
                holders.append(
                    _Held(sub, levels + my_levels[:depth], t0, inner[depth])
                )
        _emit(child, state, levels + my_levels, t0, leaves, holders)


@dataclass
class _Layout:
    """Scratch state shared by the measure/clip/emit passes."""

    sizes: dict[int, int] = field(default_factory=dict)
    n_iterations: dict[int, int] = field(default_factory=dict)
    truncated: bool = False


def _storage_windows(
    holders: list[_Held], n_timesteps: int, tensor_filter: set[str] | None
) -> dict[str, dict[TensorName, list[tuple[int, int]]]]:
    """
    Group the Storage nodes by memory level and tensor, and turn each one into the
    timestep spans over which a single tile of it stays resident.

    A tile is swapped out exactly when a loop outside the Storage node advances, so one
    tile lasts as long as the subtree beneath the node, and the spans are the iterations
    of the enclosing temporal loops. Spatial loops do not advance the timestep, so the
    copies they make share a span.
    """
    windows: dict[str, dict[TensorName, set[tuple[int, int]]]] = {}
    for held in holders:
        if held.node.component is None or held.duration <= 0:
            continue
        tensors = [
            TensorName(t)
            for t in held.node.tensors
            if tensor_filter is None or str(t) in tensor_filter
        ]
        if not tensors:
            continue

        outer = [lvl for lvl in held.levels if lvl.stride]
        starts = np.full(1, held.t0, dtype=np.int64)
        if outer:
            idx = np.indices(
                tuple(lvl.n_iterations for lvl in outer), dtype=np.int64
            ).reshape(len(outer), -1)
            starts = held.t0 + sum(idx[i] * lvl.stride for i, lvl in enumerate(outer))
        spans = {
            (int(t), min(int(t) + held.duration, n_timesteps))
            for t in np.unique(starts)
            if t < n_timesteps
        }

        level = windows.setdefault(str(held.node.component), {})
        for tensor in tensors:
            level.setdefault(tensor, set()).update(spans)

    return {
        name: {tensor: sorted(spans) for tensor, spans in level.items()}
        for name, level in windows.items()
    }


def _loop_rank_variable(loop: Loop, einsum: EinsumName) -> RankVariable | None:
    """The rank variable that ``loop`` iterates for ``einsum``, if any."""
    rv = loop.rank_variable
    if isinstance(rv, (set, frozenset)):
        # A loop fused across Einsums records each Einsum's own rank variable.
        per_einsum = getattr(loop, "_einsum_to_rank_variable", None) or {}
        if einsum in per_einsum:
            return RankVariable(per_einsum[einsum])
        if len(rv) == 1:
            return RankVariable(next(iter(rv)))
        return None
    return None if rv is None else RankVariable(rv)


def _as_int(value, what: str, node) -> int:
    """Coerce a mapping attribute to an int, with a message pointing at the node."""
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{what} of {node} is {value!r}, which is not a number.")
    if isinstance(value, sympy.Expr):
        if not value.is_number:
            raise ValueError(
                f"{what} of {node} is the symbolic expression {value}. Trace a "
                f"concrete mapping (e.g. one returned by the mapper or parsed from "
                f"YAML with numeric tile shapes) instead."
            )
        value = float(value)
    value = float(value)
    if value != int(value):
        raise ValueError(f"{what} of {node} is {value}, which is not an integer.")
    return int(value)


def _tile_shapes(loop: Loop, node_desc: str) -> tuple[int, int]:
    """Return ``(stride, initial_tile_shape)`` for ``loop``."""
    stride = _as_int(loop.tile_shape, "tile_shape", node_desc)
    initial = loop.initial_tile_shape
    if initial is None or initial == "symbol":
        initial = stride
    else:
        initial = _as_int(initial, "initial_tile_shape", node_desc)
    return stride, initial


def _loop_n_iterations(loop: Loop, einsum: EinsumName | None, cache: _Cache) -> int:
    n = loop.calculated_n_iterations
    if n is not None and n != "symbol":
        try:
            return _as_int(n, "calculated_n_iterations", loop.compact_str())
        except ValueError:
            pass  # Fall through to computing it from the tile shape.

    bounds = cache.bounds(einsum) if einsum is not None else {}
    rv = _loop_rank_variable(loop, einsum) if einsum is not None else None
    if rv is None or rv not in bounds:
        raise ValueError(
            f"Cannot determine the number of iterations of loop "
            f"'{loop.compact_str()}' for Einsum '{einsum}'. Evaluate the mapping "
            f"first (e.g. with spec.evaluate_mapping()) so that iteration counts are "
            f"filled in."
        )
    stride, initial = _tile_shapes(loop, loop.compact_str())
    shape = bounds[rv]
    if shape <= initial:
        return 1
    return 1 + -(-(shape - initial) // stride)  # 1 + ceil((shape - initial) / stride)


# ======================================================================================
# Enumerating the iteration space of one leaf
# ======================================================================================
def _tile_offsets(loop: Loop, n_iterations: int, node_desc: str):
    """
    Return ``(offsets, extents)``: where each tile of ``loop`` starts, and how many
    indices it covers.

    Tile 0 covers ``[0, initial)``; tile ``i > 0`` covers
    ``[initial + (i - 1) * stride, ... + stride)``. When ``initial == stride`` this
    reduces to the usual ``i * stride``.
    """
    stride, initial = _tile_shapes(loop, node_desc)
    i = np.arange(n_iterations, dtype=np.int64)
    offsets = np.where(i == 0, 0, initial + (i - 1) * stride)
    extents = np.where(i == 0, initial, stride)
    return offsets, extents


def _enumerate_leaf(leaf: _Leaf, cache: _Cache, max_points: int):
    """
    Enumerate every operation-space point under ``leaf``.

    Returns ``(timestep, rank_var_coords)``, where ``timestep`` is a 1-D array and
    ``rank_var_coords`` maps each rank variable to a 1-D array of the same length.
    """
    einsum: Einsum = cache.workload.einsums[leaf.einsum]
    bounds = cache.bounds(leaf.einsum)
    rank_vars = [rv for rv in einsum.rank_variables]

    # --- Per-loop tile offsets, and which rank variable each loop drives ------------
    loop_rank_vars, loop_offsets, loop_extents = [], [], []
    for level in leaf.levels:
        rv = _loop_rank_variable(level.loop, leaf.einsum)
        rv = rv if rv in bounds else None
        offsets, extents = _tile_offsets(
            level.loop, level.n_iterations, level.loop.compact_str()
        )
        loop_rank_vars.append(rv)
        loop_offsets.append(offsets)
        loop_extents.append(extents)

    loop_iters = [int(level.n_iterations) for level in leaf.levels]
    n_loop_points = int(np.prod(loop_iters)) if loop_iters else 1

    # A rank variable with no loop over it is fully covered by every compute; one with
    # loops is covered up to the extent of its innermost loop.
    innermost = {}
    for i, rv in enumerate(loop_rank_vars):
        if rv is not None:
            innermost[rv] = i
    residual_shape = [
        int(loop_extents[innermost[rv]].max()) if rv in innermost else int(bounds[rv])
        for rv in rank_vars
    ]
    n_residual = int(np.prod(residual_shape)) if residual_shape else 1

    total = n_loop_points * n_residual
    if total > max_points:
        raise ValueError(
            f"Tracing Einsum '{leaf.einsum}' would enumerate {total:,} operation-space "
            f"points, which exceeds max_points={max_points:,}. Use a smaller workload, "
            f"pass max_timesteps= to trace a prefix, or raise max_points."
        )

    # --- Loop indices, timesteps, and per-rank-variable tile bases -------------------
    loop_idx = (
        np.indices(tuple(loop_iters), dtype=np.int64).reshape(len(loop_iters), -1)
        if loop_iters
        else np.zeros((0, 1), dtype=np.int64)
    )

    timestep = np.full(n_loop_points, leaf.t0, dtype=np.int64)
    for i, level in enumerate(leaf.levels):
        if level.stride:
            timestep = timestep + loop_idx[i] * level.stride

    base = {rv: np.zeros(n_loop_points, dtype=np.int64) for rv in rank_vars}
    extent = {rv: np.full(n_loop_points, bounds[rv], dtype=np.int64) for rv in rank_vars}
    for i, rv in enumerate(loop_rank_vars):
        if rv is None:
            continue
        base[rv] = base[rv] + loop_offsets[i][loop_idx[i]]
        if innermost[rv] == i:
            extent[rv] = loop_extents[i][loop_idx[i]]

    # --- Expand each tile into its individual elements -------------------------------
    residual = (
        np.indices(tuple(residual_shape), dtype=np.int64).reshape(len(rank_vars), -1)
        if rank_vars
        else np.zeros((0, 1), dtype=np.int64)
    )

    timestep = np.repeat(timestep, n_residual)
    valid = np.ones(total, dtype=bool)
    coords = {}
    for i, rv in enumerate(rank_vars):
        delta = np.tile(residual[i], n_loop_points)
        coord = np.repeat(base[rv], n_residual) + delta
        # Drop points past the end of a partial tile or past the rank's bound.
        valid &= delta < np.repeat(extent[rv], n_residual)
        valid &= coord < bounds[rv]
        coords[rv] = coord

    timestep = timestep[valid]
    coords = {rv: c[valid] for rv, c in coords.items()}
    return timestep, coords


# ======================================================================================
# Projecting operation-space points into tensor coordinates
# ======================================================================================
def _project(
    cache: _Cache, einsum: EinsumName, tensor: TensorName, rank_var_coords: dict
):
    """Map operation-space points to ``(n_points, n_ranks)`` tensor coordinates."""
    n_points = len(next(iter(rank_var_coords.values()), np.zeros(0, dtype=np.int64)))
    columns = []
    for _rank, symbols, fn in cache.projection(einsum, tensor):
        missing = [s for s in symbols if s not in rank_var_coords]
        if missing:
            raise ValueError(
                f"Projection of Einsum '{einsum}' into tensor '{tensor}' uses "
                f"rank variable(s) {missing}, which are not in the Einsum's iteration "
                f"space."
            )
        value = fn(*[rank_var_coords[s] for s in symbols])
        value = np.broadcast_to(np.asarray(value), (n_points,))
        columns.append(np.rint(value).astype(np.int64))
    if not columns:
        return np.zeros((n_points, 0), dtype=np.int64)
    return np.stack(columns, axis=1)


def _dedup(timestep: "np.ndarray", coords: "np.ndarray"):
    """Collapse repeated accesses to the same element within the same timestep."""
    if timestep.size == 0:
        return timestep, coords
    # Pack (timestep, coords) into one integer key so np.unique stays on the fast path.
    key = timestep.astype(np.int64)
    for i in range(coords.shape[1]):
        span = int(coords[:, i].max()) + 1 if coords.shape[0] else 1
        key = key * max(span, 1) + coords[:, i]
    _, keep = np.unique(key, return_index=True)
    keep.sort()
    return timestep[keep], coords[keep]


def _ravel(coords: "np.ndarray", shape: Sequence[int]) -> "np.ndarray":
    """Flatten per-rank coordinates into row-major element indices."""
    if coords.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    if coords.shape[1] == 0:
        return np.zeros(coords.shape[0], dtype=np.int64)
    return np.ravel_multi_index(
        tuple(coords[:, i] for i in range(coords.shape[1])), tuple(shape)
    ).astype(np.int64)
