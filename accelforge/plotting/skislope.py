from collections.abc import Iterable
from typing import Any

import matplotlib.pyplot as plt

import accelforge as af
from accelforge.frontend.workload import EinsumName


# Keys in result.data
GLB_USAGE = "reservation<SEP>GlobalBuffer<SEP>-1<SEP>right"
ACCESSES = "Total<SEP>energy"


def generate_ski_slope(
    workload_fname,
    einsum_names: Iterable[EinsumName] = None,
    jinja_parse_data: dict[str, Any] = None,
    ax = None,
    **plot_kwargs,
):
    spec = af.Spec.from_yaml(
        workload_fname,
        af.examples.arches.snowcat,
        jinja_parse_data=jinja_parse_data,
    )
    spec.mapper.metrics = af.Metrics.ENERGY | af.Metrics.RESOURCE_USAGE
    result = spec.map_workload_to_arch(einsum_names=einsum_names)
    result = result.data[[GLB_USAGE, ACCESSES]]
    result.sort_values(GLB_USAGE, inplace=True)
    snowcat_capacity = spec.arch.nodes["GlobalBuffer"].size
    usage = result[GLB_USAGE]*snowcat_capacity
    return plot_step(usage.tolist(), result[ACCESSES].tolist(), ax, **plot_kwargs)


def plot_step(xs, ys, ax=None, **plot_kwargs):
    """
    Plot a staircase pattern (flat then drop) from sorted (x, y) points.

    Parameters:
        points: list of (x, y) tuples, sorted by x
        ax: optional matplotlib axis
        plot_kwargs: passed to plt.plot()
    """
    if ax is None:
        ax = plt.gca()

    stair_x = []
    stair_y = []

    for i in range(len(xs) - 1):
        x0, y0 = xs[i], ys[i]
        x1, y1 = xs[i + 1], ys[i + 1]

        # Horizontal segment
        stair_x.extend([x0, x1])
        stair_y.extend([y0, y0])

        # Vertical drop
        stair_x.append(x1)
        stair_y.append(y1)

    # Plot
    line = ax.plot(stair_x, stair_y, **plot_kwargs)

    ax.set_xscale("log")
    ax.set_yscale("log")

    xmax = ax.get_xlim()[1]
    last_x = stair_x[-1]
    last_y = stair_y[-1]
    ax.plot([last_x, xmax], [last_y, last_y], color=line[0].get_color(), **plot_kwargs)

    ax.set_xlim(left=None, right=xmax)

    ax.set_xlabel("On-Chip Memory Size (bits)")
    ax.set_ylabel("Lowest-Attainable Off-Chip Accesses (bits)")

    return ax
