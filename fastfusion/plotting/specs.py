from collections.abc import Iterable

import matplotlib.axes as axes
import matplotlib.pyplot as plt

from fastfusion.frontend.spec import Spec


def plot_area(
    specs: Iterable[Spec], labels: Iterable[str] = None, ax: axes.Axes = None
):
    """
    Plot area of one or more specs.

    Parameters
    ----------
    specs:
        An iterable of specifications.
    labels:
        An iterable of the same length as `specs` to use as labels in the plot.
    ax:
        An matplotlib Axes to use. A new one is created by default.
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_ylabel("Area (m^2)")

    if labels is None:
        labels = [f"spec-{i}" for i in range(len(specs))]
    assert len(labels) == len(specs)

    component2color = {}
    for i, (label, spec) in enumerate(zip(labels, specs)):
        heights = []
        colors = []
        names = []
        height = 0
        for component, area in spec.arch.per_component_total_area.items():
            height += area
            if component not in component2color:
                color = plt.cm.tab10(len(component2color))
                component2color[component] = color
            else:
                color = component2color[component]
            heights.append(height)
            colors.append(color)
            names.append(component)

        heights = reversed(heights)
        colors = reversed(colors)
        names = reversed(names)
        for height, color, name in zip(heights, colors, names):
            ax.bar(i, height=height, label=name, color=color)

    ax.set_xticks(range(len(specs)), labels)

    ax.legend()
