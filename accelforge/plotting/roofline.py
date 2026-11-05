"""
Roofline (2D and (3D) plots generated from an architecture description.

The 3D roofline Gerogiannis et al., "DECA: A Near-Core LLM Decompression Accelerator
Grounded on a 3D Roofline Model" (MICRO '25).

In 3D roofline, throughput is bounded by three rates: the memory, a secondary
compute unit (e.g., vector hardware), and a primary compute unit (e.g., matrix
hardware)::

    primary computes/s = min(BW * CI_mem, SECONDARY * CI_secondary, PRIMARY)

where CI_mem is computes per byte moved to/from memory and CI_secondary is
primary computes per secondary compute.

Sets of Einsums may be plotted as points. Each set is treated as fused, only the
set's input and output tensors (not tensors produced and consumed inside the set)
contribute to data movement, and each tensor is moved once.
"""

import argparse
import math
from collections.abc import Iterable, Mapping
from numbers import Number

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from accelforge.frontend.arch import Compute, Memory, Spatialable
from accelforge.frontend.renames import EinsumName
from accelforge.frontend.spec import Spec
from accelforge.util._eval_expressions import EvaluationError

# Region colors: primary compute, secondary compute, memory
_PRIMARY_COLOR = "#2a78d6"
_SECONDARY_COLOR = "#eb6834"
_MEMORY_COLOR = "#1baf7a"
_POINT_COLOR = "#1a1a19"


def plot_roofline(
    spec: Spec,
    einsum_sets: Iterable[EinsumName | Iterable[EinsumName]] = None,
    labels: Iterable[str] = None,
    memory: str = None,
    compute: str = None,
    ax=None,
    log_scale: bool = True,
):
    """
    Plot a 2D roofline for the architecture in `spec`, optionally with one point per
    fused set of Einsums.

    Parameters
    ----------
    spec:
        The specification. Its architecture sets the memory bandwidth and peak compute
        throughput. If `einsum_sets` is given, it must also include a workload.
    einsum_sets:
        Sets of Einsum names. Each set is treated as fused: only its input and output
        tensors contribute to data movement. A bare Einsum name is a set of one. All
        computes in the set are counted against `compute`.
    labels:
        Labels for each set. Defaults to the Einsum names joined by "+".
    memory:
        Name of the memory whose bandwidth bounds performance. Defaults to the outermost
        memory.
    compute:
        Name of the compute unit whose throughput bounds performance. Defaults to the
        compute unit with the highest peak throughput.
    ax:
        A matplotlib Axes to use. A new one is created by default.
    log_scale:
        Whether to use log-scaled axes.

    Returns
    -------
    fig, ax
    """
    einsum_sets, labels = _check_einsum_sets(spec, einsum_sets, labels)
    evaluated = _evaluate(spec)
    memory = _get_memory(evaluated, memory)
    compute = _get_compute(evaluated, compute, default="fastest")
    bandwidth = _peak_bandwidth(evaluated, memory)
    peak = _peak_compute(evaluated, compute)

    points = []
    for einsum_set, label in zip(einsum_sets, labels):
        n_bytes = _bytes_moved(evaluated, einsum_set)
        intensity = evaluated.workload.n_computes(einsum_set) / n_bytes
        points.append((label, intensity, min(peak, bandwidth * intensity)))

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ridge = peak / bandwidth
    intensities = [p[1] for p in points if p[1] > 0]
    max_intensity = max([2 * ridge, *(2 * i for i in intensities)])
    if log_scale:
        min_intensity = min([ridge / 100, *(i / 2 for i in intensities)])
    else:
        min_intensity = 0

    ax.plot(
        [min_intensity, ridge, max_intensity],
        [min_intensity * bandwidth, peak, peak],
        color=_POINT_COLOR,
        linewidth=2,
    )
    ax.axvline(ridge, color="gray", linestyle=":", linewidth=1)
    _label_line(ax, f"{memory}-bound", ridge, peak, "left")
    _label_line(ax, f"{compute}-bound", ridge, peak, "right")

    for label, x, y in points:
        ax.scatter([x], [y], s=48, color=_PRIMARY_COLOR, zorder=3)
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(4, -12),
            fontsize=8,
        )

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlim(min_intensity, max_intensity)
    ax.set_xlabel(f"Arithmetic intensity (computes / {memory} byte)")
    ax.set_ylabel(f"Attainable throughput ({compute} computes / s)")
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

    return fig, ax


def plot_roofline_3d(
    spec: Spec,
    einsum_sets: Iterable[EinsumName | Iterable[EinsumName]] = None,
    labels: Iterable[str] = None,
    memory: str = None,
    primary_compute: str = None,
    secondary_compute: str = None,
    einsum_to_compute: Mapping[EinsumName, str] = None,
    ax=None,
    resolution: int = 60,
    x_max: Number = None,
    y_max: Number = None,
):
    """
    Plot a 3D roofline (DECA, Section 4.1) for the architecture in `spec`,
    optionally with one point per fused set of Einsums.

    The x axis is primary computes per byte moved to/from `memory`, the y axis is
    primary computes per secondary compute, and the z axis is attainable primary
    computes per second. The surface is colored by the bounding factor.

    Parameters
    ----------
    spec:
        The specification. Its architecture must have at least two compute units. If
        `einsum_sets` is given, it must also include a workload.
    einsum_sets:
        Sets of Einsum names. Each set is treated as fused: only its input and output
        tensors contribute to data movement. A bare Einsum name is a set of one. A set
        with no secondary computes is drawn at the top of the y axis.
    labels:
        Labels for each set. Defaults to the Einsum names joined by "+".
    memory:
        Name of the memory whose bandwidth bounds performance. Defaults to the outermost
        memory.
    primary_compute:
        Name of the primary (e.g., matrix) compute unit. Defaults to the compute unit
        with the highest peak throughput.
    secondary_compute:
        Name of the secondary (e.g., vector) compute unit. Defaults to the other compute
        unit if the architecture has exactly two.
    einsum_to_compute:
        Maps Einsum names to the compute unit (`primary_compute` or
        `secondary_compute`) that runs them. Einsums not listed run on whichever of the
        two is enabled for them, preferring `primary_compute` if both are.
    ax:
        A matplotlib 3D Axes to use. A new one is created by default.
    resolution:
        Number of grid points along each of the x and y axes for the surface.
    x_max, y_max:
        Upper limits of the x and y axes. By default, twice the ridge point or enough
        to include every point. Set these to compare plots on the same axes.

    Returns
    -------
    fig, ax
    """
    einsum_sets, labels = _check_einsum_sets(spec, einsum_sets, labels)
    evaluated = _evaluate(spec)
    memory = _get_memory(evaluated, memory)
    primary_compute = _get_compute(evaluated, primary_compute, default="fastest")
    secondary_compute = _get_compute(
        evaluated, secondary_compute, default="other", exclude=primary_compute
    )
    if primary_compute == secondary_compute:
        raise ValueError(
            f"primary_compute and secondary_compute must differ, but both are "
            f"{primary_compute}."
        )
    bandwidth = _peak_bandwidth(evaluated, memory)
    primary_peak = _peak_compute(evaluated, primary_compute)
    secondary_peak = _peak_compute(evaluated, secondary_compute)

    einsum_to_compute = dict(einsum_to_compute or {})
    for einsum, compute in einsum_to_compute.items():
        if einsum not in spec.workload.einsum_names:
            raise ValueError(f"einsum_to_compute has Einsum {einsum} not in workload.")
        if compute not in (primary_compute, secondary_compute):
            raise ValueError(
                f"einsum_to_compute maps {einsum} to {compute}, which is neither the "
                f"primary ({primary_compute}) nor secondary ({secondary_compute}) "
                f"compute unit."
            )

    points = []
    for einsum_set, label in zip(einsum_sets, labels):
        n_computes = {primary_compute: 0, secondary_compute: 0}
        for einsum in einsum_set:
            compute = einsum_to_compute.get(einsum) or _compute_for_einsum(
                spec, einsum, primary_compute, secondary_compute
            )
            n_computes[compute] += evaluated.workload.n_computes(einsum)
        n_primary = n_computes[primary_compute]
        x = n_primary / _bytes_moved(evaluated, einsum_set)
        n_secondary = n_computes[secondary_compute]
        y = n_primary / n_secondary if n_secondary else math.inf
        z = min(bandwidth * x, secondary_peak * y, primary_peak)
        points.append((label, x, y, z))

    finite = [p for p in points if math.isfinite(p[2])]
    if x_max is None:
        x_max = max([2 * primary_peak / bandwidth, *(1.1 * p[1] for p in points)])
    if y_max is None:
        y_max = max([2 * primary_peak / secondary_peak, *(1.1 * p[2] for p in finite)])

    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        # Draw points in call order so they are not hidden by the surface
        ax = fig.add_subplot(projection="3d", computed_zorder=False)
    else:
        fig = ax.get_figure()

    xs, ys = np.meshgrid(
        np.linspace(0, x_max, resolution), np.linspace(0, y_max, resolution)
    )
    bounds = np.stack(
        [bandwidth * xs, secondary_peak * ys, np.full_like(xs, primary_peak)]
    )
    region_colors = np.array(
        [
            plt.matplotlib.colors.to_rgba(c, alpha=0.75)
            for c in (_MEMORY_COLOR, _SECONDARY_COLOR, _PRIMARY_COLOR)
        ]
    )
    ax.plot_surface(
        xs,
        ys,
        bounds.min(axis=0),
        facecolors=region_colors[bounds.argmin(axis=0)],
        linewidth=0,
        antialiased=False,
        shade=False,
    )

    for label, x, y, z in points:
        if y > y_max:
            label += f" (no {secondary_compute} computes)"
            y = y_max
        ax.scatter([x], [y], [z], s=48, color=_POINT_COLOR, depthshade=False)
        ax.plot([x, x], [y, y], [0, z], color=_POINT_COLOR, linewidth=1, linestyle=":")
        ax.text(x, y, z, f" {label}", fontsize=8, color=_POINT_COLOR)

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_zlim(0, primary_peak * 1.05)
    # The memory-bound face (z = bandwidth * x) faces -x and the secondary-bound face
    # (z = secondary_peak * y) faces -y. Viewing from the -x, -y side keeps both facing
    # the camera for any slope. Matplotlib's default (azim=-60) views from +x, where
    # the memory-bound face can be edge-on and invisible.
    ax.view_init(elev=25, azim=-135)
    ax.set_xlabel(f"{primary_compute} computes / {memory} byte")
    ax.set_ylabel(f"{primary_compute} computes / {secondary_compute} compute")
    ax.set_zlabel(f"Attainable {primary_compute} computes / s", labelpad=10)
    ax.legend(
        handles=[
            Patch(color=_MEMORY_COLOR, label=f"{memory}-bound"),
            Patch(color=_SECONDARY_COLOR, label=f"{secondary_compute}-bound"),
            Patch(color=_PRIMARY_COLOR, label=f"{primary_compute}-bound"),
        ],
        loc="upper left",
    )

    return fig, ax


def _check_einsum_sets(
    spec: Spec,
    einsum_sets: Iterable[EinsumName | Iterable[EinsumName]] | None,
    labels: Iterable[str] | None
) -> tuple[list[list[EinsumName]], list[str]]:
    if einsum_sets is None:
        if labels is not None:
            raise ValueError("labels were given without einsum_sets.")
        return [], []

    einsum_sets = [[s] if isinstance(s, str) else list(s) for s in einsum_sets]
    if einsum_sets and not spec.workload.einsums:
        raise ValueError(
            "einsum_sets were given, but the spec has no workload. Include a workload "
            "in the spec to plot Einsums."
        )

    workload_einsums = set(spec.workload.einsum_names)
    for i, einsum_set in enumerate(einsum_sets):
        if not einsum_set:
            raise ValueError(f"einsum_sets[{i}] is empty.")
        unknown = [e for e in einsum_set if e not in workload_einsums]
        if unknown:
            raise ValueError(
                f"einsum_sets[{i}] has Einsum(s) {unknown} that are not in the "
                f"workload. Einsums in the workload are "
                f"{sorted(workload_einsums)}."
            )

    if labels is None:
        labels = ["+".join(s) for s in einsum_sets]
    labels = list(labels)
    if len(labels) != len(einsum_sets):
        raise ValueError(
            f"Got {len(labels)} labels for {len(einsum_sets)} einsum_sets."
        )
    return einsum_sets, labels


def _evaluate(spec: Spec) -> Spec:
    """
    Evaluates the architecture's throughputs. Architectures may depend on the Einsum
    (e.g., a buffer sized by the weight tensor), so this uses the first Einsum for which
    the architecture evaluates.
    """
    first_error = None
    for einsum_name in spec.workload.einsum_names or [None]:
        try:
            return spec.calculate_component_costs(
                einsum_name=einsum_name,
                area=False,
                energy=False,
                latency=False,
                leak=False,
            )
        except EvaluationError as e:
            first_error = first_error or e
    if not spec.workload.einsums:
        first_error.add_note(
            "The architecture may depend on the workload. If so, include a workload "
            "in the spec."
        )
    raise first_error


def _instances(spec: Spec, name: str) -> int:
    """Number of instances of the named component, including its own spatial fanout."""
    for leaf, parents in spec.arch.iterate_hierarchically():
        if leaf.name == name:
            return math.prod(
                p.get_fanout() for p in [*parents, leaf] if isinstance(p, Spatialable)
            )
    raise ValueError(f"Component {name} not found in the architecture.")


def _get_memory(spec: Spec, name: str | None) -> str:
    memories = [m.name for m in spec.arch.get_nodes_of_type(Memory)]
    if not memories:
        raise ValueError("The architecture has no Memory.")
    if name is None:
        return memories[0]
    if name not in memories:
        raise ValueError(f"Memory {name} not found. Memories are {memories}.")
    return name


def _get_compute(
    spec: Spec, name: str | None, default: str, exclude: str = None
) -> str:
    computes = [c.name for c in spec.arch.get_nodes_of_type(Compute)]
    if name is not None:
        if name not in computes:
            raise ValueError(f"Compute {name} not found. Compute units are {computes}.")
        return name
    if default == "fastest":
        if not computes:
            raise ValueError("The architecture has no Compute.")
        return max(computes, key=lambda c: _peak_compute(spec, c))
    others = [c for c in computes if c != exclude]
    if len(others) != 1:
        raise ValueError(
            f"Can not pick a secondary compute unit: expected exactly one compute unit "
            f"other than {exclude}, but found {others}. Set secondary_compute."
        )
    return others[0]


def _peak_compute(spec: Spec, name: str) -> Number:
    """Peak computes per second of all instances of the named compute unit."""
    node = spec.arch.find(name)
    return node.actions["compute"].throughput * _instances(spec, name)


def _peak_bandwidth(spec: Spec, name: str) -> Number:
    """Peak read bandwidth in bytes per second of all instances of the named memory."""
    read = spec.arch.find(name).actions["read"]
    bandwidth = read.throughput * read.bits_per_action * _instances(spec, name) / 8
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError(
            f"Memory {name} has bandwidth {bandwidth} bytes/s; a roofline needs a "
            f"finite, positive bandwidth. Choose a different memory or set a read "
            f"throughput for {name}."
        )
    return bandwidth


def _bytes_moved(spec: Spec, einsum_set: list[EinsumName]) -> float:
    """Bytes of the input and output tensors of a fused set of Einsums."""
    workload = spec.workload
    tensors = workload.input_tensors(einsum_set) | workload.output_tensors(einsum_set)
    total_bits = 0
    for tensor in tensors:
        access = next(
            a
            for e in einsum_set
            for a in workload.einsums[e].tensor_accesses
            if a.name == tensor
        )
        total_bits += workload.get_tensor_size(tensor) * access.bits_per_value
    if total_bits == 0:
        raise ValueError(f"Einsums {einsum_set} move no data.")
    return total_bits / 8


def _compute_for_einsum(
    spec: Spec, einsum: EinsumName, primary: str, secondary: str
) -> str:
    evaluated = spec._spec_eval_expressions(einsum_name=einsum)
    for compute in (primary, secondary):
        if evaluated.arch.find(compute).enabled is True:
            return compute
    raise ValueError(
        f"Neither {primary} nor {secondary} is enabled for Einsum {einsum}. Set "
        f"einsum_to_compute to choose one."
    )


def _label_line(ax, text, ridge, peak, side):
    ax.annotate(
        text,
        (ridge, peak),
        textcoords="offset points",
        xytext=(6 if side == "right" else -6, -16),
        ha="left" if side == "right" else "right",
        fontsize=8,
        color="gray",
    )


def _parse_einsum_set(s: str) -> list[str]:
    return [e.strip() for e in s.split(",") if e.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot a 2D roofline or 3D roofline from an architecture spec."
    )
    parser.add_argument(
        "yaml", nargs="+", help="Spec YAML files (architecture and optional workload)."
    )
    parser.add_argument(
        "-e",
        "--einsums",
        action="append",
        type=_parse_einsum_set,
        default=None,
        help="Comma-separated fused set of Einsums to plot. May be repeated.",
    )
    parser.add_argument("--3d", dest="three_d", action="store_true")
    parser.add_argument("--memory")
    parser.add_argument("--compute", help="Compute unit (primary compute for --3d).")
    parser.add_argument("--secondary-compute", help="Secondary compute for --3d.")
    parser.add_argument("--linear", action="store_true", help="Linear 2D axes.")
    parser.add_argument("-o", "--output", default="roofline.png")
    args = parser.parse_args(argv)

    spec = Spec.from_yaml(*args.yaml)
    if args.three_d:
        fig, _ = plot_roofline_3d(
            spec,
            args.einsums,
            memory=args.memory,
            primary_compute=args.compute,
            secondary_compute=args.secondary_compute,
        )
    else:
        fig, _ = plot_roofline(
            spec,
            args.einsums,
            memory=args.memory,
            compute=args.compute,
            log_scale=not args.linear,
        )
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
