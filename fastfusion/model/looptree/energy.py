from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from typing import Any
from numbers import Real

from fastfusion.frontend import arch
from fastfusion.frontend.component_energy import ComponentEnergy
from fastfusion.model.looptree.reuse.summarized.symbolic import SummarizedAnalysisOutput
from fastfusion.frontend.workload import Workload
from fastfusion.frontend.mapping import Mapping

@dataclass
class ActionCount:
    total: Any
    max_per_unit: Any

    @staticmethod
    def default():
        return ActionCount(0, 0)


def gather_actions(looptree_results: SummarizedAnalysisOutput, bindings: dict[str, str], use_name=False):
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

        key = (buf, 'read')

        if key not in actions:
            actions[key] = ActionCount.default()
        actions[key].total += accesses.net_total_read_actions()
        actions[key].max_per_unit += accesses.net_max_per_unit_read_actions()

        key = (buf, 'write')
        if key not in actions:
            actions[key] = ActionCount.default()
        actions[key].total += accesses.net_total_write_actions()
        actions[key].max_per_unit += accesses.net_max_per_unit_write_actions()

    for compute, ops in looptree_results.compute_stats.items():
        key = (compute.level, 'compute')
        if key not in actions:
            actions[key] = ActionCount.default()
        actions[key].total += ops.total_ops
        actions[key].max_per_unit += ops.max_per_unit_ops

    return actions


def compute_energy_from_actions(action_counts: MappingABC[(str, str), Real],
                                ert: ComponentEnergy,
                                overall_latency: float):
    energy_result = {}
    for (component, action), counts in action_counts.items():
        if counts == 0:
            continue
        action_table = ert.find_action(component, action)
        if action_table is None:
            raise RuntimeError(
                f'Could not find action {action} for component {component}'
            )
        energy_per_ac = action_table.energy
        energy_result[(component, action)] = counts.total * energy_per_ac

    for action_table in ert.entries:
        try:
            leak_energy = overall_latency * action_table.actions["leak"].energy
        except KeyError:
            leak_energy = 0
        energy_result[(action_table.name, 'leak')] = leak_energy

    return energy_result
