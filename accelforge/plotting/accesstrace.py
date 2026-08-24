"""
Plot which tensor elements a LoopTree mapping touches at each timestep.

The x-axis is the timestep (one iteration of the innermost temporal loop) and the
y-axis is the tensor element being accessed, so the shape of the scatter *is* the
mapping's data movement: horizontal streaks are elements held across many timesteps,
diagonal sweeps are streaming accesses, and repeated blocks are refetches.

Passing ``memory_level=`` additionally shades, behind the points, the tile that level
holds at each timestep, which turns the same picture into a read on that level's
capacity: a wide block is a long-lived tile, and a tall one is a large tile. Blocks show
*live* residency, i.e. they stop at the tile's last use rather than running to the end of
whatever the mapping reserved.
"""

from collections.abc import Iterable

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from accelforge._accelerated_imports import numpy as np

from accelforge.tracegen.accesstrace import AccessTrace, TensorAccessTrace, trace_accesses

__all__ = ["plot_access_trace"]


# Categorical slots, assigned in fixed order and never cycled. The first three are
# validated for all-pairs separation (which is what a scatter needs); past that we
# lean on the read/write marker shapes and the legend to carry identity.
_SERIES_COLORS = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)
_OTHER_COLOR = "#8a8a85"

_READ_MARKER = "o"
_WRITE_MARKER = "s"

# A neutral block, so that shading a memory level never competes with the series colors
# for attention.
_TILE_COLOR = "#dcdcd4"


def plot_access_trace(
    trace,
    workload=None,
    tensors: Iterable[str] | None = None,
    color_by: str = "einsum",
    rank: str | None = None,
    memory_level: str | None = None,
    level_color: str = _TILE_COLOR,
    marker_size: float | None = None,
    ncols: int = 1,
    ax=None,
    figsize=None,
    **trace_kwargs,
):
    """
    Plot timestep (x) against accessed tensor element (y) for a LoopTree mapping.

    One subplot is drawn per tensor, sharing the timestep axis, so that fusion and
    reuse across tensors line up vertically.

    Parameters
    ----------
    trace:
        An :class:`~accelforge.tracegen.accesstrace.AccessTrace`, or anything
        :func:`~accelforge.tracegen.accesstrace.trace_accesses` accepts (a
        :class:`~accelforge.frontend.spec.Spec`, a mapper result, or a
        :class:`~accelforge.frontend.mapping.Mapping`).
    workload:
        The workload, required only when ``trace`` is a bare ``Mapping``.
    tensors:
        Which tensors to plot, and in what order. Defaults to every traced tensor.
    color_by:
        ``"einsum"`` colors accesses by which Einsum made them; ``"access"`` colors
        by read vs. write. Use ``"access"`` when many Einsums share one tensor.
    rank:
        Plot the coordinate along this tensor rank on the y-axis instead of the
        flattened element index. Only applies to tensors that have this rank.
    memory_level:
        Shade, in a background block, the tile of each tensor that is resident in this
        memory level (a storage component of the architecture, e.g. ``"GlobalBuffer"``).
        One block covers one tile: its width is how long the tile is *live* -- from its
        first touch to its last, not the whole span the mapping reserves for it -- and
        its height is which elements it holds, so the blocks show directly what a level
        is holding while the points show what is being touched. Tensors that this level
        never holds are left unshaded.
    level_color:
        The background color of the ``memory_level`` blocks.
    marker_size:
        Marker size in points. Chosen from the data density if not given.
    ncols:
        Number of subplot columns.
    ax:
        Draw into this existing axes instead of making a figure. Only valid when a
        single tensor is being plotted.
    figsize:
        Figure size. Scaled to the number of subplots if not given.
    **trace_kwargs:
        Passed to :func:`~accelforge.tracegen.accesstrace.trace_accesses` when
        ``trace`` is not already an ``AccessTrace`` (e.g. ``max_timesteps=``,
        ``einsums=``, ``max_points=``).

    Returns
    -------
    (fig, axes)
        The figure and a list of the axes drawn into.
    """
    if color_by not in ("einsum", "access"):
        raise ValueError(f"color_by must be 'einsum' or 'access'. Got {color_by!r}.")

    if not isinstance(trace, AccessTrace):
        trace = trace_accesses(trace, workload=workload, tensors=tensors, **trace_kwargs)
    elif trace_kwargs:
        raise TypeError(
            f"Got {sorted(trace_kwargs)}, but `trace` is already an AccessTrace. Pass "
            f"those arguments to trace_accesses() instead."
        )

    names = list(trace.tensors) if tensors is None else [str(t) for t in tensors]
    missing = [t for t in names if not trace.for_tensor(t)]
    if missing:
        raise ValueError(
            f"No traced accesses for tensor(s) {missing}. The trace covers "
            f"{trace.tensors}."
        )
    if not names:
        raise ValueError("The trace contains no tensor accesses to plot.")

    if memory_level is not None:
        memory_level = str(memory_level)
        if not any(trace.tile_lifetime(memory_level, t) for t in names):
            raise ValueError(
                f"Memory level {memory_level!r} holds none of the tensors being plotted "
                f"({names}). The mapping stores tensors in {trace.memory_levels}."
            )

    color_of = _color_assignment(trace, color_by)

    # --- Figure layout ---------------------------------------------------------------
    if ax is not None:
        if len(names) != 1:
            raise ValueError(
                f"ax= draws a single tensor, but {len(names)} were requested. Drop ax= "
                f"or pass tensors=['{names[0]}']."
            )
        fig, axes = ax.get_figure(), [ax]
    else:
        ncols = max(1, min(int(ncols), len(names)))
        nrows = -(-len(names) // ncols)
        figsize = figsize or (6.5 * ncols, 2.0 * nrows + 0.6)
        fig, grid = plt.subplots(
            nrows, ncols, figsize=figsize, sharex=True, squeeze=False
        )
        axes = list(grid.flat)
        for extra in axes[len(names) :]:
            extra.set_visible(False)
        axes = axes[: len(names)]

    x_max = max(trace.n_timesteps, 1)

    for axis, tensor in zip(axes, names):
        subtraces = trace.for_tensor(tensor)
        _plot_one(
            axis,
            tensor,
            subtraces,
            color_of,
            color_by,
            rank,
            marker_size,
            x_max,
            [] if memory_level is None else trace.tile_lifetime(memory_level, tensor),
            level_color,
        )

    # `sharex` hides tick labels on every row but the last, which leaves a short
    # column dangling. Re-enable them on whichever axes actually sits at the bottom.
    bottom = (
        {0}
        if ax is not None
        else {max(i for i in range(len(names)) if i % ncols == c) for c in range(ncols)}
    )
    for i in bottom:
        axes[i].set_xlabel("Timestep")
        axes[i].tick_params(labelbottom=True)

    handles = _legend_handles(trace, names, color_of, color_by)
    if memory_level is not None:
        handles.append(
            Patch(
                facecolor=level_color,
                edgecolor=_darken(level_color),
                linewidth=0.4,
                label=f"tile in {memory_level}",
            )
        )
    if len(handles) > 1:
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=min(len(handles), 4),
            frameon=False,
            fontsize="small",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.99 - 0.03 * -(-len(handles) // 4)))
    elif ax is None:
        fig.tight_layout()

    if trace.truncated:
        fig.text(
            0.99,
            0.005,
            f"truncated to the first {trace.n_timesteps} timesteps",
            ha="right",
            va="bottom",
            fontsize="x-small",
            alpha=0.7,
        )

    return fig, axes


# ======================================================================================
# Internals
# ======================================================================================
def _color_assignment(trace: AccessTrace, color_by: str) -> dict:
    """Map each series key to a fixed categorical slot."""
    if color_by == "access":
        return {False: _SERIES_COLORS[0], True: _SERIES_COLORS[1]}
    return {
        einsum: (_SERIES_COLORS[i] if i < len(_SERIES_COLORS) else _OTHER_COLOR)
        for i, einsum in enumerate(trace.einsums)
    }


def _series_key(subtrace: TensorAccessTrace, color_by: str):
    return subtrace.is_output if color_by == "access" else subtrace.einsum


def _y_values(subtrace: TensorAccessTrace, rank: str | None):
    """The y-coordinate of each access, and the label and extent of the axis."""
    if rank is not None and rank in subtrace.ranks:
        i = subtrace.ranks.index(rank)
        return subtrace.coordinates[:, i], f"{rank}", subtrace.tensor_shape[i]
    label = "Element" if len(subtrace.ranks) != 1 else str(subtrace.ranks[0])
    return subtrace.element, label, subtrace.tensor_size


def _marker_size(axis, n_x: int, n_y: int, n_points: int) -> float:
    """Pick a marker size that fills the axes without smearing into a solid block."""
    bbox = axis.get_window_extent()
    dpi = axis.get_figure().dpi or 100.0
    width_pt = max(bbox.width, 1.0) * 72.0 / dpi
    height_pt = max(bbox.height, 1.0) * 72.0 / dpi
    per_cell = min(width_pt / max(n_x, 1), height_pt / max(n_y, 1))
    # Sparse traces get a visible dot; dense ones shrink so structure stays legible.
    return float(np.clip(per_cell * 0.9, 0.8, 6.0 if n_points < 20_000 else 3.0))


def _darken(color, factor: float = 0.82):
    """A slightly darker shade of ``color``, for outlining a fill of it."""
    return tuple(channel * factor for channel in mcolors.to_rgb(color))


def _tile_blocks(
    subtraces: list[TensorAccessTrace], windows: list[tuple[int, int]], rank: str | None
):
    """
    One rectangle per contiguous run of elements resident in a memory level, per window.

    The tile a level holds over a window is exactly the set of elements touched while it
    is resident, so the blocks are read straight off the trace rather than recomputed
    from tile shapes. Splitting each tile into its contiguous runs keeps a tile that is
    strided in the flattened element index from shading the gaps it leaves.

    A window is what the mapping *reserves*; a tile is only *live* within it from its
    first touch to its last, so each block is clipped to that span. The reservation of a
    Storage node that sits above a Sequential split, for instance, covers every branch,
    but a tensor only one branch touches is live for only that branch.
    """
    if not windows or not subtraces:
        return []
    timestep = np.concatenate([t.timestep for t in subtraces])
    y = np.concatenate([_y_values(t, rank)[0] for t in subtraces])
    order = np.argsort(timestep, kind="stable")
    timestep, y = timestep[order], y[order]

    blocks = []
    for start, end in windows:
        lo, hi = np.searchsorted(timestep, (start, end))
        if hi <= lo:
            continue  # This level reserves room here, but nothing touches the tensor.
        # `timestep` is sorted, so the ends of the slice are the live span. A tile is
        # allocated and freed as a unit, so all of its runs share that span.
        live_start, live_end = int(timestep[lo]), int(timestep[hi - 1]) + 1
        resident = np.unique(y[lo:hi])
        runs = np.split(resident, np.flatnonzero(np.diff(resident) > 1) + 1)
        blocks.extend(
            Rectangle(
                (live_start - 0.5, float(run[0]) - 0.5),
                live_end - live_start,
                float(run[-1] - run[0]) + 1,
            )
            for run in runs
        )
    return blocks


def _plot_one(
    axis,
    tensor: str,
    subtraces: list[TensorAccessTrace],
    color_of: dict,
    color_by: str,
    rank: str | None,
    marker_size: float | None,
    x_max: int,
    windows: list[tuple[int, int]],
    level_color,
):
    y_label, y_extent = "Element", 1
    n_points = sum(t.n_accesses for t in subtraces)

    blocks = _tile_blocks(subtraces, windows, rank)
    if blocks:
        # Below the gridlines (which set_axisbelow puts at 0.5) and the points.
        axis.add_collection(
            PatchCollection(
                blocks,
                facecolor=level_color,
                edgecolor=_darken(level_color),
                linewidths=0.4,
                zorder=0,
                rasterized=len(blocks) > 2_000,
            )
        )

    for subtrace in subtraces:
        y, y_label, y_extent = _y_values(subtrace, rank)
        size = marker_size or _marker_size(axis, x_max, y_extent, n_points)
        axis.scatter(
            subtrace.timestep,
            y,
            s=size**2,
            c=color_of[_series_key(subtrace, color_by)],
            marker=_WRITE_MARKER if subtrace.is_output else _READ_MARKER,
            linewidths=0.5 if size >= 4 else 0,
            edgecolors="none" if size < 4 else axis.get_facecolor(),
            rasterized=n_points > 50_000,
        )

    axis.set_ylabel(y_label)
    axis.set_title(
        f"{tensor}  [{'x'.join(str(d) for d in subtraces[0].tensor_shape)}]",
        loc="left",
        fontsize="medium",
    )
    axis.set_xlim(-0.5, x_max - 0.5)
    axis.set_ylim(-0.5, max(y_extent - 0.5, 0.5))
    axis.grid(True, lw=0.4, alpha=0.25)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def _legend_handles(trace: AccessTrace, names, color_of: dict, color_by: str):
    """
    Build legend entries for exactly the series on screen. Read vs. write always gets
    its own marker shape, so identity never rests on color alone.
    """
    seen, handles = [], []
    for tensor in names:
        for subtrace in trace.for_tensor(tensor):
            key = (_series_key(subtrace, color_by), subtrace.is_output)
            if key in seen:
                continue
            seen.append(key)
            label = (
                ("write" if subtrace.is_output else "read")
                if color_by == "access"
                else f"{subtrace.einsum} ({'write' if subtrace.is_output else 'read'})"
            )
            handles.append(
                Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker=_WRITE_MARKER if subtrace.is_output else _READ_MARKER,
                    markersize=5,
                    color=color_of[_series_key(subtrace, color_by)],
                    label=label,
                )
            )
    return handles
