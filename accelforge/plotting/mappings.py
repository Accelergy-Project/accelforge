from collections.abc import Iterable, Sequence
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt

from accelforge.mapper.FFM import Mappings
from accelforge.mapper.FFM._pareto_df.df_convention import (
    col2energy,
    col2action,
    col2memory_usage,
    USAGE,
    MEMORY,
)
from accelforge.plotting._common import (
    _plot_column_comparison,
    _plot_breakdown,
    first_arg_maybe_iterable,
    get_title,
)


@first_arg_maybe_iterable
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
    fig, ax = _plot_column_comparison(mappings, labels, "Total<SEP>latency")
    ax.set_ylabel("Latency (s)")
    return fig, ax


@first_arg_maybe_iterable
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


@first_arg_maybe_iterable
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


@first_arg_maybe_iterable
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


@first_arg_maybe_iterable
def plot_memory_usage_breakdown(
    mappings: Iterable[Mappings] | Mappings,
    memory_levels: set[str]=None,
    labels: Iterable[str]=None,
):
    tensor2color = {}
    if memory_levels is None:
        memory_levels = set()
        for mapper_result in mappings:
            for c in mapper_result.data.columns:
                if not USAGE/MEMORY in c:
                    continue
                memory, tensor, einsum = col2memory_usage(c)
                memory_levels.add(memory)
    memory_levels = list(memory_levels)

    tensor2color = {}
    for mapper_result in mappings:
        for c in mapper_result.data.columns:
            if not USAGE/MEMORY in c:
                continue
            memory, tensor, einsum = col2memory_usage(c)
            if tensor not in tensor2color:
                tensor2color[tensor] = mpl.colormaps["tab10"](len(tensor2color))

    if labels is None:
        labels = [str(i) for i in range(len(mappings))]
    assert len(labels) == len(mappings)

    n_total_mappings = sum(map(lambda m: len(m.data), mappings))
    fig, axes = plt.subplots(len(memory_levels), n_total_mappings, squeeze=False)
    col = 0
    lines_labels = []
    for label, mapper_result in zip(labels, mappings):
        for mapping_idx, mapping in mapper_result.data.iterrows():
            for row, level in enumerate(memory_levels):
                ax = axes[row, col]

                einsum2tensor2usage = defaultdict(dict)
                for c in mapper_result.data.columns:
                    if not USAGE/MEMORY in c:
                        continue
                    memory, tensor, einsum = col2memory_usage(c)
                    if memory != level:
                        continue
                    einsum2tensor2usage[einsum][tensor] = mapping[c]

                for einsum, tensor2usage in einsum2tensor2usage.items():
                    running_usage = sum(tensor2usage.values())
                    for tensor, usage in tensor2usage.items():
                        ax.bar(
                            einsum,
                            running_usage,
                            label=tensor,
                            color=tensor2color[tensor]
                        )
                        running_usage -= usage
                lines_labels.append(ax.get_legend_handles_labels())
                ax.set_title(f"{get_title(label, mapping_idx)}--{level}")
            col += 1

    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

    # grab unique labels
    unique_labels = set(labels)

    # assign labels and legends in dict
    legend_dict = dict(zip(labels, lines))

    # query dict based on unique labels
    unique_lines = [legend_dict[x] for x in unique_labels]

    fig.legend(unique_lines, unique_labels)

    return fig, axes
