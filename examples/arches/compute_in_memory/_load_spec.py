import accelforge as af
import os

THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VARIABLES_GLOBAL_PATH = os.path.join(THIS_SCRIPT_DIR, "_include.yaml")


def get_spec(
    arch_name: str,
    compare_with_arch_name: str | None = None,
    add_dummy_main_memory: bool = False,
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

    Returns
    -------
    spec: af.Spec
        The spec for the given architecture.
    """
    if compare_with_arch_name is None:
        compare_with_name = arch_name
    else:
        compare_with_name = compare_with_arch_name

    arch_name = os.path.join(THIS_SCRIPT_DIR, f"{arch_name}.yaml")
    compare_with_name = os.path.join(THIS_SCRIPT_DIR, f"{compare_with_name}.yaml")
    variables = af.Variables.from_yaml(arch_name, top_key="variables")
    arch = af.Arch.from_yaml(arch_name, top_key="arch")
    workload = af.Workload.from_yaml(arch_name, top_key="workload")
    spec = af.Spec(arch=arch, variables=variables, workload=workload)

    spec.config.expression_custom_functions.append(
        os.path.join(THIS_SCRIPT_DIR, "_include_functions.py")
    )
    spec.config.component_models.append(
        os.path.join(THIS_SCRIPT_DIR, "components/*.py")
    )
    if add_dummy_main_memory:
        main_memory = af.arch.Memory(
            name="MainMemory",
            component_class="Dummy",
            size=float("inf"),
            tensors={"keep": "~weight"}
        )
        spec.arch.nodes.insert(0, main_memory)
    return spec
