import accelforge as af
import os

THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VARIABLES_GLOBAL_PATH = os.path.join(THIS_SCRIPT_DIR, "_include.yaml")


def get_spec(
    arch_name: str,
    compare_with_arch_name: str | None = None,
    add_dummy_main_memory: bool = False,
    n_macros: int = 1,
) -> af.Spec:
    """
    Gets the spec for the given architecture. If `compare_with_arch_name` is given, the
    variables_iso will be grabbed from `compare_with_arch_name` in order to match
    attributes for fair comparison.

    Parameters
    ----------
    arch_name: str
        The name of the architecture to get the spec for.
    compare_with_arch_name: str | None
        The name of the architecture to compare with. If not given, variables will be
        taken from the given `arch_name`.
    n_macros: int
        The number of macros to use in the architecture.
    Returns
    -------
    spec: af.Spec
        The spec for the given architecture.
    """
    if compare_with_arch_name is None:
        compare_with_name = arch_name
    else:
        compare_with_name = compare_with_arch_name

    arch_name_base = arch_name
    arch_name = os.path.join(THIS_SCRIPT_DIR, f"{arch_name}.yaml")
    compare_with_name = os.path.join(THIS_SCRIPT_DIR, f"{compare_with_name}.yaml")
    variables = af.Variables.from_yaml(arch_name, top_key="variables")
    arch = af.Arch.from_yaml(arch_name, top_key="arch")
    workload = af.Workload.from_yaml(arch_name, top_key="workload")
    spec = af.Spec(arch=arch, variables=variables, workload=workload)

    # Helpful to increase spatial utilization. Especially helpful for CiM accelerators,
    # which are often running conv workloads & have weird hardware and/or workload
    # shapes that are difficult to perfectly tile.
    spec.mapper.explore_imperfect_spatial_loops = True

    # It's OK to explore loop orders, but will make the search take longer, and is not
    # very helpful for CiM accelerators. It's not very helpful because loop orders help
    # us capture sliding window reuse, and the CiM macros consume very big tiles at
    # once, giving limited benefit from the small overlap between each tile.
    spec.mapper.explore_loop_orders = False

    spec.mapper.metrics = af.Metrics.ENERGY_DELAY_PRODUCT

    spec.config.expression_custom_functions.append(
        os.path.join(THIS_SCRIPT_DIR, "_include_functions.py")
    )
    # Load architecture-specific helper functions if they exist
    arch_helpers = os.path.join(
        THIS_SCRIPT_DIR, f"{arch_name_base}_helper_functions.py"
    )
    if os.path.exists(arch_helpers):
        spec.config.expression_custom_functions.append(arch_helpers)
    spec.config.component_models.append(
        os.path.join(THIS_SCRIPT_DIR, "components/*.py")
    )
    if n_macros > 1:
        macro = af.arch.Container(
            name="MacroAuto",
            spatial=[{"name": "macro", "fanout": n_macros, "power_gateable": True}],
        )
        spec.arch.nodes.insert(0, macro)
    if add_dummy_main_memory:
        main_memory = af.arch.Memory(
            name="MainMemory",
            component_class="Dummy",
            size=float("inf"),
            tensors={"keep": "~weight"},
        )
        spec.arch.nodes.insert(0, main_memory)
    return spec
