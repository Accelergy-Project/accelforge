from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from numbers import Real

from fastfusion.frontend.component_energy import ComponentEnergy
from fastfusion.model.looptree.mapping_utilities import get_einsums_with_complete_mappings
from fastfusion.model.looptree.accesses import buffer_accesses_from_buffet_actions

# from pytimeloop.isl.singular import get_sum_of_pw_qpolynomial


@dataclass
class ActionCount:
    total: Any
    max_per_unit: Any

    @staticmethod
    def default():
        return ActionCount(0, 0)


def gather_actions(looptree_results, mapping, workload, bindings, is_path=False, use_name=False):
    einsums_with_complete_mapping = \
        get_einsums_with_complete_mappings(mapping.nodes, workload, is_path)

    accesses_stats = buffer_accesses_from_buffet_actions(looptree_results,
                                                         mapping,
                                                         workload,
                                                         is_path)
    actions: dict[tuple, ActionCount] = {}
    for (buf, tensor, einsum), accesses in accesses_stats.items():
        if use_name:
            buf = buf
        else:
            buf = bindings[buf]

        key = (buf, 'read')
        if key not in actions:
            actions[key] = ActionCount.default()
        actions[key].total += accesses.total_reads
        actions[key].max_per_unit += accesses.max_per_unit_reads

        key = (buf, 'write')
        if key not in actions:
            actions[key] = ActionCount.default()
        actions[key].total += accesses.total_writes
        actions[key].max_per_unit += accesses.max_per_unit_writes

    # ops = gather_ops(looptree_results.per_einsum_ops, einsums_with_complete_mapping)
    for compute, ops in looptree_results.compute_stats.items():
        key = (compute.level, 'compute')
        if key not in actions:
            actions[key] = ActionCount.default()
        actions[key].total += ops.total_ops
        actions[key].max_per_unit += ops.max_per_unit_ops

    return actions


def compute_energy_from_actions(action_counts: Mapping[(str, str), Real],
                                ert: ComponentEnergy):
    energy_result = {}
    for (component, action), counts in action_counts.items():
        if counts == 0:
            continue
        action_table = ert.entries[component].find_action(action)
        if action_table is None:
            raise RuntimeError(
                f'Could not find action {action} for component {component}'
            )
        energy_per_ac = action_table.energy
        energy_result[(component, action)] = counts.total*energy_per_ac

    return energy_result


def gather_ops(ops, einsums_with_complete_mapping):
    total = 0
    for einsum_id, (tags, v) in ops.items():
        if einsum_id not in einsums_with_complete_mapping:
            continue
        if isinstance(v, isl.PwQPolynomial):
            total += get_sum_of_pw_qpolynomial(v)
        elif isinstance(v, Real):
            total += v
        else:
            total += v
    return total