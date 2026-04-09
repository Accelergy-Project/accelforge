from collections.abc import Iterable

import matplotlib.pyplot as plt
from accelforge._accelerated_imports import pandas as pd

from accelforge.util._base_analysis_types import VerboseActionKey
from accelforge.mapper.FFM import Mappings


def _plot_breakdown(
    mappings,
    labels,
    separate_by,
    stack_by,
    col_keyword: str,
    keyer,
    ax=None,
    hide_zeros=False,
):
    all_data = [m.data for m in mappings]

    if ax is not None:
        fig = ax.get_figure()
        axes = [ax]
    else:
        n_axes = sum(map(len, all_data))
        fig, axes = plt.subplots(1, n_axes, sharey=True, figsize=(n_axes * 3, 4))
        if n_axes == 1:
            axes = [axes]

    if labels is not None:
        labels = [l + "-" for l in labels]
    else:
        labels = [f"{i}-" for i in range(len(all_data))]
    assert len(labels) == len(all_data)

    if len(separate_by) == 0:
        raise ValueError("Missing categories by which to breakdown energy")

    idx = 0
    for label, df in zip(labels, all_data):
        colnames = [c for c in df.columns if col_keyword in c and "Total" not in c]
        bar_components = list(
            _get_bar_components(colnames, keyer, separate_by, stack_by)
        )

        for j, (_key, row) in enumerate(df.iterrows()):
            cur_ax = axes[min(idx, len(axes) - 1)]
            idx += 1

            if ax is None:
                cur_ax.set_title(f"{label}m{j}")

            # Collect names of bars and initialize label2hieghts
            bars = []
            # label2heights maps labels (values of stack_by) to a list of equal
            # length with bars. Each element is a bar height.
            label2heights = {}
            for name, constituents in bar_components:
                bars.append(name)
                for stack_name, subconstituents in constituents:
                    if not stack_name in label2heights:
                        label2heights[stack_name] = []
            for label in label2heights:
                label2heights[label] = [0] * len(bars)

            # Collect the bar heights from constituents
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

            if hide_zeros:
                nonzero = set()
                for heights in label2heights.values():
                    for i, h in enumerate(heights):
                        if h != 0:
                            nonzero.add(i)
                if nonzero:
                    bars = [bars[i] for i in nonzero]
                    for lbl in label2heights:
                        label2heights[lbl] = [label2heights[lbl][i] for i in nonzero]

            # Stack the bar heights in reverse order
            cur_heights = [0] * len(bars)
            for label, heights in reversed(list(label2heights.items())):
                for i in range(len(bars)):
                    cur_heights[i] += heights[i]
                    heights[i] = cur_heights[i]

            for label, heights in label2heights.items():
                cur_ax.bar(bars, height=heights, label=label)
                cur_ax.set_xticks(bars, labels=bars, rotation=90)

    for a in axes:
        handles, labels = a.get_legend_handles_labels()
        if handles:
            a.legend()
    return fig, axes


def _plot_column_comparison(mappings, labels, colname):
    all_data = [m.data for m in mappings]
    n_bars = sum(map(len, all_data))

    if labels is not None:
        labels = [l + "-" for l in labels]
    else:
        labels = [f"{i}-" for i in range(len(mappings))]
    assert len(labels) == len(mappings)

    fig, ax = plt.subplots(figsize=(n_bars, 4))

    for label, df in zip(labels, all_data):
        bars = [f"{label}m{i}" for i in range(len(df))]
        heights = df[colname]
        ax.bar(bars, heights)
        ax.set_xticks(bars, labels=bars, rotation=90)

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
    for group, subdf in df.groupby(by=separate_by, sort=False):
        group = ", ".join(group) if isinstance(group, tuple) else str(group)
        if not stack_by:
            result.append((group, [(None, subdf["colname"])]))
        else:
            finer_separation = []
            for subgroup, stack_df in subdf.groupby(by=stack_by, sort=False):
                subgroup = (
                    ", ".join(subgroup)
                    if isinstance(subgroup, tuple)
                    else str(subgroup)
                )
                finer_separation.append((subgroup, stack_df["colname"]))
            result.append((group, finer_separation))
    return result


def first_arg_maybe_iterable(f):
    """
    Canonicalizes the first argument, which may be one element or an iterable,
    into a list.
    """

    def new_f(*args, **kwargs):
        mappings = args[0]
        mappings = mappings if isinstance(mappings, Iterable) else [mappings]
        return f(mappings, *args[1:], **kwargs)

    return new_f


def get_title(label: str, mapping_idx: int):
    return f"{label}m{mapping_idx}"
