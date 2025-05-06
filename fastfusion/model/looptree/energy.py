from collections.abc import Mapping
from numbers import Real

from pytimeloop.isl.singular import get_sum_of_pw_qpolynomial
from pytimeloop.timeloopfe.v4.ert import Ert
from pytimeloop.looptree.accesses import *
from pytimeloop.looptree.mapping_utilities import *


def gather_actions(looptree_results, mapping, workload, bindings, is_path=False, use_name=False):
    einsum_name_to_id = workload.einsum_name_to_id()

    einsums_with_complete_mapping = \
        get_einsums_with_complete_mappings(mapping['nodes'], workload, is_path)
    einsums_with_complete_mapping = {
        e if isinstance(e, int) else einsum_name_to_id[e]
        for e in einsums_with_complete_mapping
    }

    accesses_stats = buffer_accesses_from_buffet_actions(looptree_results,
                                                         mapping,
                                                         workload,
                                                         is_path)
    actions = {}
    for (buf, tensor, einsum), accesses in accesses_stats.items():
        if use_name:
            buf = buf
        else:
            buf = bindings[buf]

        key = (buf, 'read')
        if key not in actions:
            actions[key] = 0
        actions[key] += accesses.total_reads

        key = (buf, 'write')
        if key not in actions:
            actions[key] = 0
        actions[key] += accesses.total_writes

    ops = gather_ops(looptree_results.ops, einsums_with_complete_mapping)
    actions[(bindings[max(bindings.keys())], 'compute')] = ops

    return actions


def compute_energy_from_actions(action_counts: Mapping[(str, str), Real],
                   ert: Ert):
    energy_result = {}
    for (component, action), counts in action_counts.items():
        if counts == 0:
            continue
        component_table = ert.find_component(component)
        action_table = component_table.find_action(action)
        if action_table is None:
            raise RuntimeError(
                f'Could not find action {action} for component {component}'
            )
        energy_per_ac = action_table.energy
        energy_result[(component, action)] = counts*energy_per_ac

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