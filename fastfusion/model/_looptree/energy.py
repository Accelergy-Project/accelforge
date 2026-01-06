from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from typing import Any
from numbers import Real

from fastfusion.frontend import arch
from fastfusion.frontend.spec import Spec
from fastfusion.model._looptree.reuse.symbolic import SymbolicAnalysisOutput
from fastfusion.frontend.workload import Workload
from fastfusion.frontend.mapping import Mapping


@dataclass
class ActionCount:
    total: Any
    max_per_unit: Any

    @staticmethod
    def default():
        return ActionCount(0, 0)


def gather_actions(
    looptree_results: SymbolicAnalysisOutput, bindings: dict[str, str], use_name=False
):
    actions: dict[tuple[str, str], ActionCount] = {}
    compute_levels = set(c.level for c in looptree_results.compute_stats)

    for buffet, accesses in looptree_results.buffet_stats.items():
        if buffet.level in compute_levels:
            continue

        buf = buffet.level

        if use_name:
            buf = buf
        else:
            buf = bindings[buf]

        key = (buf, "read")

        if key not in actions:
            actions[key] = ActionCount.default()
        actions[key].total += accesses.net_total_read_actions()
        actions[key].max_per_unit += accesses.net_max_per_unit_read_actions()

        key = (buf, "write")
        if key not in actions:
            actions[key] = ActionCount.default()
        actions[key].total += accesses.net_total_write_actions()
        actions[key].max_per_unit += accesses.net_max_per_unit_write_actions()

    for compute, ops in looptree_results.compute_stats.items():
        key = (compute.level, "compute")
        if key not in actions:
            actions[key] = ActionCount.default()
        actions[key].total += ops.total_ops
        actions[key].max_per_unit += ops.max_per_unit_ops

    return actions


def compute_energy_from_actions(
    spec: Spec,
    action_counts: MappingABC[(str, str), Real],
    overall_latency: float,
):
    energy_result = {}
    components = {}
    for (component, action), counts in action_counts.items():
        if counts.total == 0:
            continue
        if component not in components:
            components[component] = spec.arch.find(component)
        component_obj = components[component]
        energy_per_ac = component_obj.actions[action].arguments.energy
        energy_result[(component, action)] = counts.total * energy_per_ac

    for component_obj in spec.arch.get_nodes_of_type(arch.Component):
        energy_result[(component_obj.name, "leak")] = (
            component_obj.attributes.total_leak_power * overall_latency
        )

    return energy_result
