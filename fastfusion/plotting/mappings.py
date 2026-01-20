from collections.abc import Iterable, Set

import matplotlib.axes as axes
import matplotlib.pyplot as plt
import pandas as pd

from fastfusion.mapper.FFM import Mappings


def plot_mappings_latency(
    mappings: Iterable[Mappings] | Mappings,
):
    raise NotImplementedError()


def plot_mappings_energy(
    mappings: Iterable[Mappings] | Mappings,
    separate_by: Set[str] = None,
    labels=None,
    ax: axes.Axes = None,
):
    """
    Plot the result(s) of mapper or model call(s).

    Parameters
    ----------
    mappings:
        A mapping to plot or an iterable of mappings to plot.
    separate_by:
        A subset of {"einsum", "component", "action"}. E.g., if
        "einsum" is in `separate_by`, then the bar chart will be
        a stacked bar chart with different Einsums as components
        of the bar.
    ax:
        A matplotlib axes to use instead of a newly generated one.
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_ylabel("Energy (pJ)")

    separate_by = separate_by if separate_by is not None else {}
    mappings = [mappings] if isinstance(mappings, Mappings) else list(mappings)
    labels = labels if labels is not None else [""] * len(mappings)
    assert len(labels) == len(mappings)

    name2color = {}
    for label, df in zip(labels, (m.data for m in mappings)):
        bars = [f"{label}-mapping{i}" for i in range(len(df))]

        energy_colnames = [c for c in df.columns if "energy" in c and "Total" not in c]
        bar_components = _get_bar_components(energy_colnames, separate_by)

        names = []
        colors = []
        heights = []
        last_height = 0
        for name, constituents in bar_components:
            if name not in name2color:
                color = plt.cm.tab10(len(name2color))
                name2color[name] = color
            else:
                color = name2color[name]
            height = 0
            for colname in constituents:
                col = df[colname]
                height += col
            height += last_height

            names.append(name)
            colors.append(color)
            heights.append(height)

            last_height = height

        names = reversed(names)
        colors = reversed(colors)
        heights = reversed(heights)
        for name, color, height in zip(names, colors, heights):
            ax.bar(bars, height=height, label=name, color=color)

    ax.legend()


def _get_bar_components(colnames, separate_by):
    if separate_by is None:
        yield "", colnames
        return
    separate_by = list(separate_by)

    split_colnames = [c.split("<SEP>") for c in colnames]
    split_colnames = [
        [einsum, component, action]
        for (einsum, energy_literal, component, action) in split_colnames
    ]
    transposed_colnames = zip(*split_colnames)
    df = pd.DataFrame(
        {k: v for k, v in zip(["einsum", "component", "action"], transposed_colnames)}
    )
    for group, subdf in df.groupby(by=separate_by):
        constituents = [
            f"{einsum}<SEP>energy<SEP>{component}<SEP>{action}"
            for einsum, component, action in zip(
                subdf.einsum, subdf.component, subdf.action
            )
        ]
        yield group, constituents
