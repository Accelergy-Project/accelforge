from collections.abc import Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from accelforge.mapper.FFM import Mappings
from accelforge.mapper.FFM._pareto_df.df_convention import col2energy, col2action
from accelforge.util._base_analysis_types import VerboseActionKey


def plot_latency_comparison(
    mappings: Iterable[Mappings] | Mappings,
    labels=None,
):
    """
    Plot latency comparison of multiple mappings.

    Parameters
    ----------
    mappings:
        A mapping to plot or an iterable of mappings to plot.
    labels:
        Labels to use for each Mapping class in `mappings`.
    """
    fig, ax = _plot_column_comparison(mappings, labels, "Total<SEP>energy")
    ax.set_ylabel("Latency (s)")
    return fig, ax


def plot_action_breakdown(
    mappings: Iterable[Mappings] | Mappings,
    separate_by: Sequence[str],
    stack_by: Sequence[str] = None,
    labels: Iterable[str] = None,
):
    """
    Plot actions breakdown.

    Parameters
    ----------
    mappings:
        A mapping to plot or an iterable of mappings to plot. Each mapping will
        be plotted in a new subplot.
    labels:
        Labels to use for each Mapping class in `mappings`.
    separate_by:
        A list that has elements in {"einsum", "tensor", "component", "action"}.
        Different bars will be created based on `separate_by`.
        The order from left to right will determine grouping of the breakdown.
    stack_by:
        A list that has elements in {"einsum", "tensor", "component", "action"}.
        Different components in a stacked bar will be created based on `stack_by`.
        By default, will stack actions.
    """
    if stack_by is None:
        stack_by = ["action"]
    fig, axes = _plot_breakdown(
        mappings, labels, separate_by, stack_by, "action", col2action
    )
    axes[0].set_ylabel("Actions")
    return fig, axes


def plot_energy_breakdown(
    mappings: Iterable[Mappings] | Mappings,
    separate_by: Sequence[str],
    stack_by: Sequence[str] = None,
    labels: Iterable[str] = None,
):
    """
    Plot energy breakdown.

    Parameters
    ----------
    mappings:
        A mapping to plot or an iterable of mappings to plot. Each mapping will
        be plotted in a new subplot.
    labels:
        Labels to use for each Mapping class in `mappings`.
    separate_by:
        A list that has elements in {"einsum", "tensor", "component", "action"}.
        Different bars will be created based on `separate_by`.
        The order from left to right will determine grouping of the breakdown.
    stack_by:
        A list that has elements in {"einsum", "tensor", "component", "action"}.
        Different components in a stacked bar will be created based on `stack_by`.
    """
    fig, axes = _plot_breakdown(
        mappings, labels, separate_by, stack_by, "energy", col2energy
    )
    axes[0].set_ylabel("Energy (pJ)")
    return fig, axes


def _plot_breakdown(mappings, labels, separate_by, stack_by, col_keyword: str, keyer):
    mappings = [mappings] if isinstance(mappings, Mappings) else list(mappings)
    n_axes = sum(map(len, (m.data for m in mappings)))

    fig, axes = plt.subplots(1, n_axes, sharey=True)
    if n_axes == 1:
        axes = [axes]

    labels = (
        labels + "-" if labels is not None else [f"{i}-" for i in range(len(mappings))]
    )
    assert len(labels) == len(mappings)

    if len(separate_by) == 0:
        raise ValueError("Missing categories by which to breakdown energy")

    idx = 0
    for label, df in zip(labels, (m.data for m in mappings)):
        colnames = [c for c in df.columns if col_keyword in c and "Total" not in c]
        bar_components = list(
            _get_bar_components(colnames, keyer, separate_by, stack_by)
        )

        for j, (_key, row) in enumerate(df.iterrows()):
            ax = axes[idx]
            idx += 1

            ax.set_title(f"{label}mapping{j}")

            bars = []
            label2heights = {}
            for name, constituents in bar_components:
                bars.append(name)
                for stack_name, subconstituents in constituents:
                    if not stack_name in label2heights:
                        label2heights[stack_name] = []
            for label in label2heights:
                label2heights[label] = [0] * len(bars)

            for name, constituents in bar_components:
                bar_i = bars.index(name)
                for stack_name, subconstituents in constituents:
                    heights = label2heights[stack_name]

                    height = 0
                    for colname in subconstituents:
                        col = df[colname].iloc[0]
                        height += col
                    heights[bar_i] = height
                    assert len(heights) == len(bars)

            for label, heights in label2heights.items():
                ax.bar(bars, height=heights, label=label)
                ax.set_xticklabels(bars, rotation=90)

    for ax in axes:
        ax.legend()
    return fig, axes


def plot_energy_comparison(mappings: Iterable[Mappings] | Mappings, labels=None):
    """
    Plot energy comparison of multiple mappings.

    Parameters
    ----------
    mappings:
        A mapping to plot or an iterable of mappings to plot.
    labels:
        Labels to use for each Mapping class in `mappings`.
    """
    fig, ax = _plot_column_comparison(mappings, labels, "Total<SEP>energy")
    ax.set_ylabel("Energy (pJ)")
    return fig, ax


def _plot_column_comparison(mappings, labels, colname):
    fig, ax = plt.subplots()

    mappings = [mappings] if isinstance(mappings, Mappings) else list(mappings)
    labels = labels + "-" if labels is not None else [""] * len(mappings)
    assert len(labels) == len(mappings)

    for label, df in zip(labels, (m.data for m in mappings)):
        bars = [f"{label}mapping{i}" for i in range(len(df))]
        heights = df[colname]
        ax.bar(bars, heights)

    return fig, ax


def _get_bar_components(colnames, keyer, separate_by, stack_by=None):
    if not stack_by:
        stack_by = []

    split_colnames = []
    for c in colnames:
        key = keyer(c)
        if not isinstance(key, VerboseActionKey):
            continue
        split_colnames.append([key.einsum, key.level, key.tensor, key.action, c])
    transposed_colnames = zip(*split_colnames)
    df = pd.DataFrame(
        {
            k: v
            for k, v in zip(
                ["einsum", "component", "tensor", "action", "colname"],
                transposed_colnames,
            )
        }
    )

    result = []
    for group, subdf in df.groupby(by=separate_by):
        group = ", ".join(group)
        if not stack_by:
            result.append((group, [(None, subdf["colname"])]))
        else:
            finer_separation = []
            for subgroup, stack_df in subdf.groupby(by=stack_by):
                stack_df = stack_df.sort_values(by="colname")
                subgroup = ", ".join(subgroup)
                finer_separation.append((subgroup, stack_df["colname"]))
            result.append((group, finer_separation))
    return result
