import sys
import os
from IPython.display import display, Markdown

THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(THIS_SCRIPT_DIR)
sys.path.append(
    os.path.join(
        THIS_SCRIPT_DIR,
        "..",
        "..",
        "examples",
        "arches",
        "compute_in_memory",
    )
)

from _load_spec import get_spec as _get_spec
import accelforge as af
import matplotlib.pyplot as plt

from accelforge.util import parallel, delayed


def display_markdown(markdown):
    display(Markdown(markdown))


def get_spec(
    name: str, add_dummy_main_memory: bool = False, n_macros: int = 1
) -> af.Spec:
    return _get_spec(
        name, add_dummy_main_memory=add_dummy_main_memory, n_macros=n_macros
    )


def display_important_variables(name: str):
    result = []
    result.append(f"Some of the important variables for {name}:\n")

    def pfmat(key, value, note=""):
        result.append(f"- *{key}*: {value} {note if note else ''}")

    s: af.Spec = get_spec(name)
    s.calculate_component_area_energy_latency_leak(
        einsum_name=s.workload.einsums[0].name
    )

    def getvalue(key):
        return s.variables.get(key, s.arch.variables.get(key, None))

    for v in [
        ("array_wordlines", "rows in the array"),
        ("array_bitlines", "columns in the array"),
        (
            "array_parallel_inputs",
            "input slice(s) consumed in each cycle.",
        ),
        (
            "array_parallel_weights",
            "weights slice(s) used for computation in each cycle.",
        ),
        ("array_parallel_outputs", "partial sums produced in each cycle."),
        ("tech_node", "m"),
        ("adc_resolution", "bit(s)"),
        ("dac_resolution", "bit(s)"),
        ("n_adc_per_bank", "ADC(s)"),
        ("supported_input_bits", "bit(s)"),
        ("supported_output_bits", "bit(s)"),
        ("supported_weight_bits", "bit(s)"),
        ("bits_per_cell", "bit(s)"),
        (
            "cim_unit_width_cells",
            "adjacent cell(s) in a wordline store bit(s) in one weight slice and process one input & output slice together",
        ),
        (
            "cim_unit_depth_cells",
            "adjacent cell(s) in a bitline operate in separate cycles",
        ),
        "cell_config",
        ("cycle_period", "second(s)"),
    ]:
        if isinstance(v, tuple):
            pfmat(v[0], getvalue(v[0]), v[1])
        else:
            pfmat(v, s.variables.get(v, None))

    display_markdown("\n".join(result))


import matplotlib.pyplot as plt


def run_with_variables(
    arch_name: str,
    variable_overrides: dict = None,
    workload_bits: int = None,
    tensor_bits: dict = None,
    add_dummy_main_memory: bool = True,
) -> af.mapper.FFM.Mappings:
    """Convenience wrapper: get spec, override variables, and run.

    Prefer using get_spec() + spec.variables.X = Y + spec.map_workload_to_arch(print_progress=False) directly
    for more control.
    """
    spec = get_spec(arch_name, add_dummy_main_memory=add_dummy_main_memory)
    if variable_overrides:
        for k, v in variable_overrides.items():
            if hasattr(spec.variables, k):
                setattr(spec.variables, k, v)
            if (
                hasattr(spec.arch, "variables")
                and spec.arch.variables
                and k in spec.arch.variables
            ):
                spec.arch.variables[k] = v
    if workload_bits is not None:
        for einsum in spec.workload.einsums:
            for ta in einsum.tensor_accesses:
                ta.bits_per_value = workload_bits
    if tensor_bits is not None:
        for einsum in spec.workload.einsums:
            for ta in einsum.tensor_accesses:
                if ta.name in tensor_bits:
                    ta.bits_per_value = tensor_bits[ta.name]
    spec.mapper.metrics = af.mapper.Metrics.ENERGY
    return spec.map_workload_to_arch(print_progress=False)


def get_area_breakdown(arch_name: str, variable_overrides: dict = None) -> dict:
    """Get area breakdown for an architecture.

    Args:
        arch_name: Architecture name
        variable_overrides: Dict of {variable_name: value} to override

    Returns:
        Dict of {component_name: area_m2}
    """
    spec = get_spec(arch_name)
    if variable_overrides:
        for k, v in variable_overrides.items():
            if hasattr(spec.variables, k):
                setattr(spec.variables, k, v)
            if (
                hasattr(spec.arch, "variables")
                and spec.arch.variables
                and k in spec.arch.variables
            ):
                spec.arch.variables[k] = v
    evaluated = spec.calculate_component_area_energy_latency_leak()
    return {k: float(v) for k, v in evaluated.arch.per_component_total_area.items()}


def combine_areas(area_dict: dict, groups: dict) -> dict:
    """Combine component areas into named groups.

    Args:
        area_dict: Dict of {component_name: area}
        groups: Dict of {group_name: [component_names]}

    Returns:
        Dict of {group_name: combined_area}
    """
    result = {}
    for group_name, components in groups.items():
        result[group_name] = sum(area_dict.get(c, 0) for c in components)
    return result


def combine_energies(energy_dict: dict, groups: dict) -> dict:
    """Combine component energies into named groups.

    Args:
        energy_dict: Dict of {component_name: energy}
        groups: Dict of {group_name: [component_names]}

    Returns:
        Dict of {group_name: combined_energy}
    """
    result = {}
    for group_name, components in groups.items():
        result[group_name] = sum(float(energy_dict.get(c, 0)) for c in components)
    return result


def bar_stacked(
    data: dict[dict[str, float]],
    xlabel: str,
    ylabel: str,
    title: str,
    ax: plt.Axes,
):
    """Create a stacked bar chart from nested dictionary data.

    Args:
        data: Nested dict where outer keys are x-axis categories,
              inner keys are stack categories, values are heights
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        title: Chart title
        ax: Matplotlib axes to plot on
    """
    import numpy as np

    # Get all categories
    x_categories = list(data.keys())
    stack_categories = list(
        set(k for inner_dict in data.values() for k in inner_dict.keys())
    )

    # Prepare data for stacking
    x_pos = np.arange(len(x_categories))
    bottoms = np.zeros(len(x_categories))

    # Plot each stack category
    for stack_cat in stack_categories:
        heights = [data[x_cat].get(stack_cat, 0) for x_cat in x_categories]
        ax.bar(x_pos, heights, label=stack_cat, bottom=bottoms)
        bottoms += heights

    # Set labels and formatting
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_categories, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)


def bar_comparison(
    data_dict: dict[str, dict[str, float]],
    xlabel: str,
    ylabel: str,
    title: str,
    ax: plt.Axes,
):
    """Create grouped bar chart comparing multiple datasets.

    Args:
        data_dict: Dict where keys are series names (e.g., "Modeled", "Expected"),
                   values are dicts mapping category to value
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        title: Chart title
        ax: Matplotlib axes to plot on
    """
    import numpy as np

    # Get categories (use first dataset's keys)
    categories = list(next(iter(data_dict.values())).keys())
    series_names = list(data_dict.keys())

    # Set up bar positions
    x = np.arange(len(categories))
    width = 0.8 / len(series_names)  # Total width divided by number of series

    # Plot each series
    for i, series_name in enumerate(series_names):
        offset = (i - len(series_names) / 2 + 0.5) * width
        values = [data_dict[series_name][cat] for cat in categories]
        ax.bar(x + offset, values, width, label=series_name)

    # Set labels and formatting
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)


def descending_sort(d: dict):
    """Sort a dict by value descending, return list of (key, value) tuples."""
    return sorted(d.items(), key=lambda x: -x[1])


def display_dnn_results(result: af.mapper.FFM.Mappings, title: str = "DNN Results"):
    """Display summary, 2x2 breakdown plot, and interactive per-einsum slider.

    Args:
        result: Result from the accelforge mapper.
        title: Title for the plots.
    """
    from accelforge.plotting.mappings import (
        plot_energy_breakdown,
        plot_action_breakdown,
        plot_latency_breakdown,
    )
    from ipywidgets import interact, IntSlider

    from IPython.display import display, HTML

    summary_rows = "".join(
        f"<tr><td>{m}</td><td>{v}</td></tr>"
        for m, v in [
            ("Total computes", f"{result.n_computes():,.0f}"),
            ("Total energy", f"{result.energy():.4e} J"),
            ("Total latency", f"{result.latency():.4e} s"),
            ("TOPS", f"{2 / result.per_compute().latency() / 1e12:.4f}"),
            ("TOPS/W", f"{2 / result.per_compute().energy() / 1e12:.1f}"),
        ]
    )
    energy_rows = "".join(
        f"<tr><td>{k}</td><td>{v * 1e15:.2f}</td></tr>"
        for k, v in descending_sort(result.per_compute().energy(per_component=True))
        if v > 0
    )
    display(
        HTML(
            f"<h2>{title}</h2>"
            '<div style="display: flex; gap: 2em;">'
            '<div><b>Summary</b><table border="1" style="border-collapse:collapse">'
            "<tr><th>Metric</th><th>Value</th></tr>" + summary_rows + "</table></div>"
            '<div><b>Energy per Compute</b><table border="1" style="border-collapse:collapse">'
            "<tr><th>Component</th><th>fJ/MAC</th></tr>"
            + energy_rows
            + "</table></div>"
            "</div>"
        )
    )

    # 2x2 grid: energy stacked + latency per einsum, total and per-compute
    _, axes = plt.subplots(2, 2, figsize=(16, 10))

    plot_energy_breakdown(
        [result],
        separate_by=["einsum"],
        stack_by=["component"],
        ax=axes[0, 0],
        hide_zeros=True,
    )
    axes[0, 0].set_ylabel("Energy (J)")

    plot_energy_breakdown(
        [result.per_compute(per_einsum=True)],
        separate_by=["einsum"],
        stack_by=["component"],
        ax=axes[0, 1],
        hide_zeros=True,
    )
    axes[0, 1].set_ylabel("Energy per Compute (J)")

    latency = result.latency(per_einsum=True)
    einsums = list(latency.keys())
    axes[1, 0].bar(einsums, latency.values())
    axes[1, 0].set_xticks(range(len(einsums)), labels=einsums, rotation=90)
    axes[1, 0].set_ylabel("Latency (s)")

    latency_pc = result.per_compute(per_einsum=True).latency(per_einsum=True)
    einsums_pc = list(latency_pc.keys())
    axes[1, 1].bar(einsums_pc, latency_pc.values())
    axes[1, 1].set_xticks(range(len(einsums_pc)), labels=einsums_pc, rotation=90)
    axes[1, 1].set_ylabel("Latency per Compute (s)")

    plt.tight_layout()
    plt.show()

    # Interactive per-einsum slider
    einsum_names = result.einsum_names

    def show_einsum(idx):
        name = einsum_names[idx]
        cur_result = result.per_compute(per_einsum=True).access(
            name, col_idx=0, keep_key_index=True
        )

        _, axes = plt.subplots(1, 3, figsize=(18, 5))

        plot_energy_breakdown(
            [cur_result],
            separate_by=["component"],
            stack_by=[],
            ax=axes[0],
            hide_zeros=True,
        )
        axes[0].set_ylabel("Energy per Compute")

        plot_latency_breakdown(
            [cur_result], separate_by=["component"], ax=axes[1], hide_zeros=True
        )
        axes[1].set_ylabel("Latency per Compute")

        plot_action_breakdown(
            [cur_result],
            separate_by=["component", "action"],
            ax=axes[2],
            hide_zeros=True,
        )
        axes[2].set_ylabel("Actions per Compute")
        axes[2].set_yscale("log")

        plt.suptitle(
            f"Einsum: {name} ({result.n_computes(name):.0f} computes)", fontsize=14
        )
        plt.tight_layout()
        plt.show()

        # Side-by-side tables via HTML
        energy_rows = "".join(
            f"<tr><td>{k}</td><td>{v:.4e} J</td></tr>"
            for k, v in descending_sort(cur_result.energy(per_component=True))
            if v != 0
        )
        latency_rows = "".join(
            f"<tr><td>{k}</td><td>{v:.4e} s</td></tr>"
            for k, v in descending_sort(cur_result.latency(per_component=True))
            if v != 0
        )
        action_rows = "".join(
            f"<tr><td>{k}</td><td>{v:.4f}</td></tr>"
            for k, v in descending_sort(cur_result.actions(per_component=True))
            if v != 0
        )
        from IPython.display import display, HTML

        display(
            HTML(
                '<div style="display: flex; gap: 2em;">'
                '<div><b>Energy/compute</b><table border="1" style="border-collapse:collapse">'
                "<tr><th>Component</th><th>Energy</th></tr>"
                + energy_rows
                + "</table></div>"
                '<div><b>Latency/compute</b><table border="1" style="border-collapse:collapse">'
                "<tr><th>Component</th><th>Latency</th></tr>"
                + latency_rows
                + "</table></div>"
                '<div><b>Actions/compute</b><table border="1" style="border-collapse:collapse">'
                "<tr><th>Component</th><th>Actions</th></tr>"
                + action_rows
                + "</table></div>"
                "</div>"
            )
        )

    interact(
        show_einsum,
        idx=IntSlider(
            min=0,
            max=len(einsum_names) - 1,
            step=1,
            description="Einsum:",
        ),
    )


def bar(
    data: dict[str, float],
    xlabel: str,
    ylabel: str,
    title: str,
    ax: plt.Axes,
):
    """Create a simple bar chart from a dictionary.

    Args:
        data: Dict mapping category names to values
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        title: Chart title
        ax: Matplotlib axes to plot on
    """
    import numpy as np

    categories = list(data.keys())
    values = list(data.values())

    x = np.arange(len(categories))
    ax.bar(x, values)

    # Set labels and formatting
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
