from typing import Iterable
from numbers import Number

import matplotlib.pyplot as plt

from accelforge.mapper.FFM import Mappings


def plot_roofline(
    bandwidth: Number,
    computational_throughput: Number,
    min_computational_intensity: Number = 0,
    max_computational_intensity: Number = None,
):
    """
    Plot a roofline model.

    Parameters
    ----------
    bandwidth:
        The memory bandwidth to use when generating the roofline.
    computational_throughput:
        The peak computational throughput to use when generating the roofline.
    min_computational_intensity:
        The minimum computational intensity to include in the x-axis.
    max_computational_intensity:
        The maximum computational intensity to include in the x-axis.
    """
    fig, ax = plt.subplots()

    roofline_transition = _roofline_transition(bandwidth, computational_throughput)
    if max_computational_intensity is None:
        max_computational_intensity = 2 * roofline_transition

    ax.plot(
        [min_computational_intensity, roofline_transition],
        [min_computational_intensity * bandwidth, computational_throughput],
        color="black",
    )
    ax.plot(
        [roofline_transition, max_computational_intensity],
        [computational_throughput, computational_throughput],
        color="black",
    )

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

    return fig, ax


def _roofline_transition(bandwidth, computation_throughput):
    return computation_throughput / bandwidth
