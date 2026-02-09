import sys
import os
from IPython.display import display, Markdown

THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(THIS_SCRIPT_DIR)
sys.path.append(
    os.path.join(THIS_SCRIPT_DIR, "..", "..", "examples", "arches", "compute_in_memory")
)

from _load_spec import get_spec as _get_spec
import accelforge as af


def display_markdown(markdown):
    display(Markdown(markdown))


def get_spec(name: str, add_dummy_main_memory: bool = False) -> af.Spec:
    return _get_spec(name, add_dummy_main_memory=add_dummy_main_memory)


# import difflib
# import re
# import svgutils
# from IPython.display import SVG, display, Markdown
# from .utils import *

# DIAGRAM_DEFAULT_IGNORE = ("system", "macro_in_system", "1bit_x_1bit_mac")


# def grab_from_yaml_file(
#     yaml_file, startfrom=None, same_indent=True, include_lines_before=0
# ):
#     with open(yaml_file, "r") as f:
#         contents = f.readlines()
#     start, end = 0, len(contents)
#     n_whitespace = 0
#     if startfrom is None:
#         return "".join(contents)
#     for i, line in enumerate(contents):
#         if re.findall(r"\b\s*" + startfrom + r"\b", line):
#             start = i
#             n_whitespace = len(re.findall(r"^\s*", line)[0])
#             break
#     else:
#         raise ValueError(f"{startfrom} not found in {yaml_file}")
#     for i, line in enumerate(contents[start + 1 :]):
#         ws = len(re.findall(r"^\s*", line)[0])
#         if ws < n_whitespace or (not same_indent and ws == n_whitespace):
#             end = start + i + 1
#             break
#     return "".join(
#         c[n_whitespace:] for c in contents[start - include_lines_before : end]
#     )


# def scale_svg(svg, scale=0.5):
#     svg = svgutils.transform.fromstring(svg.decode("ascii"))
#     svg = svgutils.compose.Figure(svg.width, svg.height, svg.getroot())
#     svg = svg.scale(scale)
#     svg.width = svg.width * scale
#     svg.height = svg.height * scale
#     return svg


# def display_diagram(diagram, scale=0.5):
#     display(SVG(scale_svg(diagram.create_svg(), scale).tostr()))


# def display_markdown(markdown):
#     display(Markdown(markdown))


# def display_yaml_file(*args, **kwargs):
#     display_yaml_str(grab_from_yaml_file(*args, **kwargs))


# def display_yaml_str(yaml_str):
#     display_markdown(f"```yaml\n{yaml_str}```")


# def get_yaml_file_markdown(yaml_file, *args, **kwargs):
#     return f"```yaml\n{grab_from_yaml_file(yaml_file, *args, **kwargs)}```"


# def get_yaml_str_markdown(yaml_str):
#     return f"```yaml\n{yaml_str}```"


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


# def clean_old_output_files(max_files=50):
#     out_path = os.path.join(THIS_SCRIPT_DIR, "..", "outputs")
#     files = sorted(
#         list(os.path.join(out_path, f) for f in os.listdir(out_path)),
#         key=lambda x: os.path.getmtime(x),
#     )
#     while len(files) > max_files:
#         shutil.rmtree(
#             files.pop(0),
#             ignore_errors=True,
#         )


# def run_test(
#     macro_name: str,
#     test_name: str,
#     show_doc: bool = True,
#     *args,
#     **kwargs,
# ):
#     test_func = get_test(macro_name, test_name)
#     if show_doc:
#         doc = test_func.__doc__
#         doc = "\n".join([line[1:] for line in doc.split("\n")])
#         display_markdown(doc)
#     t = test_func(*args, **kwargs)
#     clean_old_output_files()
#     return t


# def diff_str(a, b):
#     new_a, new_b = [], []
#     a = re.findall(r"[\w\.]+|\s+|.", a)
#     b = re.findall(r"[\w\.]+|\s+|.", b)
#     # print(f'Diffing {a} and {b}')
#     matcher = difflib.SequenceMatcher(None, a, b)
#     for tag, i1, i2, j1, j2 in matcher.get_opcodes():
#         if tag == "equal":
#             new_a.extend(a[i1:i2])
#             new_b.extend(b[j1:j2])
#         elif tag == "replace":
#             new_a.extend([f"\033[31m{l}\033[0m" for l in a[i1:i2]])
#             new_b.extend([f"\033[31m{l}\033[0m" for l in b[j1:j2]])
#         elif tag == "delete":
#             new_a.extend([f"\033[31m{l}\033[0m" for l in a[i1:i2]])
#         elif tag == "insert":
#             new_b.extend([f"\033[31m{l}\033[0m" for l in b[j1:j2]])
#     return "".join(new_a), "".join(new_b)


# def print_side_by_side(a, b):
#     a_lines = a.splitlines()
#     b_lines = b.splitlines()

#     # Use difflib to match up lines
#     matcher = difflib.SequenceMatcher(None, a_lines, b_lines)
#     # Insert blank lines to line up the matches
#     a = []
#     b = []
#     for _, i1, i2, j1, j2 in matcher.get_opcodes():
#         a.extend(a_lines[i1:i2])
#         b.extend(b_lines[j1:j2])
#         a.extend([""] * (len(b) - len(a)))
#         b.extend([""] * (len(a) - len(b)))

#     max_a_len = max(len(line) for line in a)
#     a = [line.ljust(max_a_len) for line in a]

#     for i in range(len(a)):
#         a[i], b[i] = diff_str(a[i], b[i])
#         if a[i] and not b[i]:
#             a[i] = f"\033[31m{a[i]}\033[0m"
#         elif not a[i] and b[i]:
#             b[i] = f"\033[31m{b[i]}\033[0m"

#     for a_line, b_line in zip(a, b):
#         print(f"{a_line}   |   {b_line}")

from math import isclose
import matplotlib.pyplot as plt


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
