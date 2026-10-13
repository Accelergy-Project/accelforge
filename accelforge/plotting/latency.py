"""Latency timelines showing how per-component latency overlaps across Einsums."""

import math
from bisect import insort
from collections import defaultdict, namedtuple
from copy import deepcopy

import matplotlib.pyplot as plt

from accelforge.frontend.arch.spatialable import Spatialable
from accelforge.mapper.FFM import Mappings
from accelforge.mapper.FFM._pareto_df.df_convention import (
    col2commlatency,
    col2complatency,
    col2total_latency,
)
from accelforge.util import oset

_Block = namedtuple("_Block", ["einsum", "start", "end"])
_Bar = namedtuple("_Bar", ["component", "level", "einsum", "start", "end", "shared"])


def _latency_timeline(
    spec, mappings: Mappings = None
) -> tuple[list[_Block], list[_Bar], float]:
    """
    Lay the latencies of a mapping's Einsums out on a shared timeline. X axis is time,
    and Y axis is component and loop level.

    Uses spec.mapping, or the single mapping in a `Mappings` object returned by the
    mapper if `mappings` is given.

    Returns (blocks, bars, total). Bars with a None component are the per-Einsum latency
    that is not tracked per-component (compute, networks, and communication). The total
    equals the model's Total<SEP>latency.
    """
    from accelforge.mapper.FFM._join_pmappings.join_pmappings import (
        clean_compress_and_join_pmappings,
    )
    from accelforge.model.main import _model_pmappings

    flattened_arches = evaluated_specs = None
    spec = deepcopy(spec)
    # The timeline lays Einsums out one after another.
    for s in (s for n in spec.arch.get_nodes_of_type(Spatialable) for s in n.spatial):
        s._allow_multi_einsum_parallelism = False
    if mappings is not None:
        if len(mappings.data) != 1:
            raise ValueError(
                f"mappings holds {len(mappings.data)} mappings; pass exactly one"
            )
        spec.model.metrics = spec.mapper.info_metrics
        spec.mapping = mappings.data.iloc[0]["Total<SEP>mapping"](_for_model=True)
        flattened_arches = mappings.flattened_arches
        evaluated_specs = mappings.evaluated_specs

    pmappings = _model_pmappings(spec, flattened_arches, evaluated_specs)
    einsums = list(pmappings.einsum2pmappings)
    groups = [pmappings.einsum2pmappings[e][0] for e in einsums]
    n = len(groups)
    tensors = [g.compatibility.tensor_names for g in groups]

    # Per-Einsum latency outside the per-level columns, and per-(component, level)
    # latency, one level per column. The blocks run sequentially, so each join is a
    # sequential split at the number of loops the joined Einsums share.
    splits = oset()
    for i in range(1, n):
        past = oset.union(oset(), *tensors[:i])
        splits.add(groups[i].compatibility.shared_loop_index(past & tensors[i]) + 1)

    privates = []
    inclusives = []
    excls = []
    comms = []  # per Einsum: {(direction, level): wind-up/down latency}
    for group in groups:
        row = group.mappings.data.iloc[0]
        privates.append(
            sum(
                row[c]
                for c in group.mappings.data.columns
                if (k := col2total_latency(c)) is not None and not k.left
            )
        )
        inclusive = defaultdict(dict)
        comm_latency = {}
        for col in group.mappings.data.columns:
            # Component latency is upward-inclusive (lower nloops includes higher)
            if (key := col2complatency(col)) is not None:
                inclusive[key.name][key.nloops] = row[col]
            # Communication latencies are per-level and, if any sequential loop happens,
            # repeated per iteration of any loops
            elif (key := col2commlatency(col)) is not None:
                iters = group.compatibility.per_loop_calculated_n_iterations
                comm_latency[key] = row[col]
                for i, it in enumerate(iters[: key.nloops]):
                    if any(i < split <= key.nloops for split in splits):
                        multiplier = row[it] if isinstance(it, str) else it
                        comm_latency[key] *= multiplier

        # Each level's column includes the levels below it, so differences against
        # the next deeper level recover the per-level latency.
        excl = {}
        for c, by_level in inclusive.items():
            levels = sorted(by_level)
            deeper = dict(zip(levels, levels[1:]))
            excl[c] = {
                l: v - by_level.get(deeper.get(l), 0) for l, v in by_level.items()
            }
        inclusives.append(dict(inclusive))
        comms.append(comm_latency)
        excls.append(excl)
    # Outermost-first architecture order, so lanes read top-down like the arch.
    arch_order = list(
        dict.fromkeys(
            m.name for e in einsums for m in pmappings.einsum2jobs[e].flattened_arch
        )
    )
    # Port-split names like "GlobalBuffer (read)" sort by their base component.
    components = sorted(
        oset(c for e in excls for c in e),
        key=lambda c: (arch_order.index(c.split(" (")[0]), c),
    )

    # keep[i]: deepest loop index at which Einsum i's reservations are shared
    # with another Einsum's. Deeper latency is private to block i.
    # shared_after[i]: deepest loop index still shared after Einsum i. Deeper
    # latency must finish by the end of block i; shallower may finish later.
    keeps, shared_after = [], []
    backings = {}
    for i, group in enumerate(groups):
        live = oset.union(oset(), *tensors[i + 1 :])
        past = oset.union(oset(), *tensors[:i])
        keeps.append(group.compatibility.shared_loop_index(live | (past & tensors[i])))
        for t, l in group.compatibility.get_backing_levels().items():
            backings[t] = min(backings.get(t, l), l)
        shared_after.append(
            max((backings[t] for t in live if t in backings), default=-1) - 1
        )
    shared_after[-1] = -2

    # Exact block ends, mirroring the join: each Einsum maxes levels deeper than
    # keep into its own width; shared busy columns sum across Einsums, a missing
    # level falling back to the next-deeper column; levels deeper than
    # shared_after max into the running total.
    def at_or_below(cols, level):
        deeper = [l for l in cols if l >= level]
        return cols[min(deeper)] if deeper else 0

    total = 0.0
    ends = []
    winds = []  # per block: wind-up/down added at its end
    shared_busy = defaultdict(dict)  # component -> {level: busy summed across Einsums}
    shared_winds = {}  # (direction, level) -> wind-up/down maxed across Einsums
    for i in range(n):
        width = privates[i]
        for cols in inclusives[i].values():
            done = [l for l in cols if l > keeps[i]]
            if done:
                width = max(width, cols[min(done)])
        # Wind-up/down of unshared fused loops serializes with the busy time.
        wind = sum(v for k, v in comms[i].items() if k.nloops > keeps[i])
        total += width + wind
        for component, cols in inclusives[i].items():
            kept = {l: v for l, v in cols.items() if l <= keeps[i]}
            mine = shared_busy[component]
            shared_busy[component] = {
                l: at_or_below(mine, l) + at_or_below(kept, l)
                for l in set(mine) | set(kept)
            }
        # Shared fused loops' wind-up/down is summed across the Einsums filling
        # them (wind-up latency is serialized), then added when the loop is freed.
        for key, v in comms[i].items():
            if key.nloops <= keeps[i]:
                shared_winds[key] = shared_winds.get(key, 0) + v
        # A freed loop's wind-up/down serializes, so it is paid before the busy time.
        for key in list(shared_winds):
            if key.nloops > shared_after[i]:
                paid = shared_winds.pop(key)
                total += paid
                wind += paid
        for cols in shared_busy.values():
            done = [l for l in cols if l > shared_after[i]]
            if done:
                total = max(total, cols[min(done)])
            for l in done:
                del cols[l]
        ends.append(total)
        winds.append(wind)

    def floor(level, i):
        return max((ends[j] for j in range(i) if shared_after[j] < level), default=0.0)

    def finish_by(level, i):
        # Busy time must finish before the wind-up/down at the end of the block
        # that frees its level.
        j = next((j for j in range(i, n) if shared_after[j] < level), n - 1)
        return ends[j] - winds[j]

    busy = defaultdict(list)  # component -> sorted (start, end) of placed bars

    def place(component, lo, amount, finish):
        # Earliest gap in the component's schedule that fits, from lo onward.
        start = lo
        for s, e in busy[component]:
            if s >= start + amount:
                break
            start = max(start, e)
        if start + amount > finish:
            # The model lets busy time fill slack anywhere before the finish,
            # even when this component was already busy then.
            start = finish - amount
        insort(busy[component], (start, start + amount))
        return start

    blocks, bars = [], []
    for i, einsum in enumerate(einsums):
        block_start = ends[i - 1] if i else 0.0
        blocks.append(_Block(einsum, block_start, ends[i]))
        if privates[i] > 0:
            bars.append(
                _Bar(None, None, einsum, block_start, block_start + privates[i], False)
            )
        # Wind-up/down paid at this block's end serializes after its busy time.
        if winds[i] > 0:
            bars.append(_Bar(None, None, einsum, ends[i] - winds[i], ends[i], False))

    # Latency deeper than keep is private to its Einsum's block; shared latency
    # may fill slack anywhere while its level stays shared (down to where that
    # sharing starts, until it maxes into the total at the block that frees its
    # level). Private latency has a fixed home, so place it first and let shared
    # latency fill the gaps that remain.
    for shared in [False, True]:
        for i, einsum in enumerate(einsums):
            for component in components:
                for level in sorted(excls[i].get(component, {}), reverse=True):
                    amount = excls[i][component][level]
                    if (level <= keeps[i]) != shared or amount <= 0:
                        continue
                    if shared:
                        start = place(
                            component, floor(level, i), amount, finish_by(level, i)
                        )
                    else:
                        start = place(
                            component, blocks[i].start, amount, ends[i] - winds[i]
                        )
                    bars.append(
                        _Bar(component, level, einsum, start, start + amount, shared)
                    )

    result = clean_compress_and_join_pmappings(
        pmappings=pmappings,
        metrics=spec.model.metrics,
        print_progress=False,
        for_model=True,
    )
    expected = result.data["Total<SEP>latency"].iloc[0]
    assert math.isclose(total, expected, rel_tol=1e-6), (
        f"Latency timeline total {total} does not match the model's "
        f"Total<SEP>latency {expected}"
    )
    return blocks, bars, total


def plot_latency(
    spec, mappings: Mappings = None, ax: plt.Axes = None
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots a latency timeline. Time is on the X axis, and component on the Y axis.
    Component latencies may be divided across multiple shared loop levels, and shared
    latencies may overlap across Einsums. Private latencies do not get a loop level.

    Parameters
    ----------
    spec:
        The spec to plot.
    mappings:
        The mapping to plot. Must hold exactly one mapping. If not given, uses
        spec.mapping.
    ax:
        The axes to plot on. If not given, creates a new figure and axes.

    Returns
    -------
    fig:
        The figure containing the plot.
    ax:
        The axes containing the plot.
    """
    blocks, bars, total = _latency_timeline(spec, mappings)

    # Components from top to bottom with each component's shared levels deepest
    # at the bottom, so bars within a block stair upward toward shared levels.
    # Private latency is lumped into one lane per component below its shared
    # levels; the Other lane is at the very bottom.
    def lane(bar):
        return (bar.component, bar.level if bar.shared else "private")

    lanes = defaultdict(oset)
    for bar in bars:
        if bar.component is not None:
            lanes[bar.component].add(lane(bar)[1])
    lane_order = [
        (c, l)
        for c in lanes
        for l in sorted(lanes[c], key=lambda l: (l == "private", l != "private" and l))
    ]
    if any(bar.component is None for bar in bars):
        lane_order.append((None, "private"))
    y = {lane: len(lane_order) - 1 - i for i, lane in enumerate(lane_order)}

    colors = {}
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for component, _ in lane_order:
        colors.setdefault(component, cycle[len(colors) % len(cycle)])

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(max(6, 2 * len(blocks)), 1.5 + 0.4 * len(lane_order))
        )
    else:
        fig = ax.get_figure()

    for bar in bars:
        ax.barh(
            y[lane(bar)],
            bar.end - bar.start,
            left=bar.start,
            height=0.8,
            color=colors[bar.component],
            edgecolor="black",
            linewidth=0.5,
        )
    for block in blocks:
        ax.axvline(block.end, linestyle=":", color="gray", linewidth=1)
        # Skip labels of blocks too narrow to hold them.
        if block.end - block.start >= 0.01 * len(block.einsum) * total:
            ax.text((block.start + block.end) / 2, -1, block.einsum, ha="center")

    def label(component, level):
        if component is None:
            return "Other"
        if len(lanes[component]) == 1:
            return component
        if level == "private":
            return f"{component} (private)"
        return f"{component} (above loop {level})"

    ax.set_yticks([y[l] for l in lane_order], [label(*l) for l in lane_order])
    ax.set_ylim(-1.5, len(lane_order) - 0.5)
    ax.set_xlim(0, total * 1.02)
    ax.set_xlabel("Time")
    return fig, ax
