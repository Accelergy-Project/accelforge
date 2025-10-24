from collections import defaultdict

from fastfusion.frontend import arch
from fastfusion.frontend.arch import Leaf, Memory, TensorHolder, Component

from fastfusion.frontend.mapping import Compute, Mapping
from fastfusion.frontend.specification import Specification
from fastfusion.model.looptree.accesses import isl_buffer_accesses_from_buffet_actions
from fastfusion.model.looptree.mapping_utilities import get_leaves
from fastfusion.model.looptree.reuse.isl import IslReuseAnalysisOutput
from fastfusion.model.looptree.reuse.summarized import SummarizedAnalysisOutput

from fastfusion.model.looptree.reuse.summarized.symbolic import Buffet, BuffetStats
from fastfusion.util.parse_expressions import MATH_FUNCS, parse_expression
from fastfusion.util.sympy.broadcast_max import Max, Min

def isl_to_summarized(looptree_results: IslReuseAnalysisOutput, mapping, workload) -> SummarizedAnalysisOutput:
    accesses_stats = isl_buffer_accesses_from_buffet_actions(
        looptree_results,
        mapping,
        workload,
        is_path=False
    )
    buffet_stats = {
        Buffet(level=component, tensor=tensor, einsum=einsum): BuffetStats(    
            max_per_unit_read_actions=accesses.max_per_unit_reads,
            max_per_unit_write_actions=accesses.max_per_unit_writes,
        )
        for (component, tensor, einsum), accesses in accesses_stats.items()
    }
    return SummarizedAnalysisOutput(buffet_stats=buffet_stats)


def component_latency(
    looptree_results: SummarizedAnalysisOutput,
    flattened_arch: list[Leaf],
    mapping: Mapping,
    spec: Specification
):
    component_to_actions: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(lambda: 0))
    name2component: dict[str, Component] = {node.name: node for node in flattened_arch}

    compute_obj = flattened_arch[-1]
    if not isinstance(compute_obj, arch.Compute):
        raise ValueError("Last node in flattened_arch must be a Compute")

    for buffet, buffet_stats in looptree_results.buffet_stats.items():
        component = buffet.level
        actions = component_to_actions[component]
        if component not in name2component:
            raise ValueError(f"Component {component} found in mapping but not arch")

        for action in name2component[component].actions:
            actions[f"{action.name}_actions"] += 0

        if isinstance(name2component[component], TensorHolder):
            actions["read_actions"] += buffet_stats.max_per_unit_read_actions - buffet_stats.min_per_unit_skipped_first_read_actions
            actions["write_actions"] += buffet_stats.max_per_unit_write_actions - buffet_stats.min_per_unit_skipped_first_write_actions
        elif isinstance(name2component[component], arch.Compute):
            pass
        else:
            raise NotImplementedError(f"Component {component} is not a TensorHolder or Compute")

    longest_compute_latency = Max(0, *[s.max_latency for s in looptree_results.compute_stats.values()])
    component_to_actions[compute_obj.name]["compute_actions"] = longest_compute_latency

    component_latency = {}

    symbol_table_base = {
        **dict(spec.variables),
        "variables": spec.variables,
        "max": Max,
        "min": Min,
    }
    
    for component, actions in component_to_actions.items():
        if name2component[component].attributes.latency is None:
            continue
        symbol_table = {
            **symbol_table_base,
            **dict(name2component[component].attributes),
            **actions,
        }
        component_latency[component] = parse_expression(
            name2component[component].attributes.latency,
            symbol_table,
            attr_name="latency",
            location=component
        )

    return component_latency
