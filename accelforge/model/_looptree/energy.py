from collections.abc import Mapping as MappingABC
import logging
from numbers import Number
from numbers import Real

from accelforge.frontend import arch
from accelforge.frontend.spec import Spec
from accelforge.model._looptree.reuse.symbolic import SymbolicAnalysisOutput
from accelforge.util._base_analysis_types import (
    ActionCount,
    ActionKey,
    VerboseActionKey,
)
from accelforge.model._looptree.types import Network


def gather_actions(
    looptree_results: SymbolicAnalysisOutput,
    bindings: dict[str, str],
    verbose: bool = False,
    use_name=False,
):
    actions: dict[tuple[str, str], ActionCount] = {}
    compute_levels = set(c.level for c in looptree_results.compute_stats)

    buffet_keyer = _get_buffet_keyer(verbose, use_name, bindings)
    compute_keyer = _get_compute_keyer(verbose, use_name, bindings)
    network_keyer = _get_network_keyer(verbose, use_name, bindings)

    for buffet, accesses in looptree_results.buffet_stats.items():
        if buffet.level in compute_levels:
            continue

        level = buffet.level

        if use_name:
            level = level
        else:
            level = bindings[level]

        key = buffet_keyer(buffet, "read")
        if key not in actions:
            actions[key] = ActionCount.default()
        actions[key].total += accesses.net_total_read_actions()
        actions[key].max_per_unit += accesses.net_max_per_unit_read_actions()

        key = buffet_keyer(buffet, "write")
        if key not in actions:
            actions[key] = ActionCount.default()
        actions[key].total += accesses.net_total_write_actions()
        actions[key].max_per_unit += accesses.net_max_per_unit_write_actions()

    for compute, ops in looptree_results.compute_stats.items():
        key = compute_keyer(compute, "compute")
        if key not in actions:
            actions[key] = ActionCount.default()
        actions[key].total += ops.total_ops
        actions[key].max_per_unit += ops.max_per_unit_ops

    for network, stats in looptree_results.network_stats.items():
        key = network_keyer(network, "hops")
        if key not in actions:
            actions[key] = ActionCount.default()
        actions[key].total += stats.total_hops
        actions[key].max_per_unit += stats.max_hops

    return actions


def _get_buffet_keyer(verbose, use_name, bindings):
    if not verbose:

        def get_buffet_key(buffet, action_name) -> ActionKey:
            level = buffet.level
            if use_name:
                level = level
            else:
                level = bindings[level]
            return ActionKey(level, action_name)

    else:

        def get_buffet_key(buffet, action_name) -> VerboseActionKey:
            level = buffet.level
            if use_name:
                level = level
            else:
                level = bindings[level]
            return VerboseActionKey(level, action_name, buffet.tensor, buffet.einsum)

    return get_buffet_key


def _get_compute_keyer(verbose, use_name, bindings):
    if not verbose:

        def compute_keyer(compute, action_name):
            level = compute.level
            if use_name:
                level = level
            else:
                level = bindings[level]
            return ActionKey(level, action_name)

    else:

        def compute_keyer(compute, action_name):
            level = compute.level
            if use_name:
                level = level
            else:
                level = bindings[level]
            return VerboseActionKey(level, action_name, None, compute.einsum)

    return compute_keyer


def _get_network_keyer(verbose, use_name, bindings):
    if not verbose:

        def network_keyer(network: Network, action_name: str):
            component = network.component
            if not use_name:
                component = bindings[component]
            return ActionKey(component, action_name)

    else:

        def network_keyer(network: Network, action_name: str):
            component = network.component
            if not use_name:
                component = bindings[component]
            return VerboseActionKey(
                component, action_name, network.tensor, network.einsum
            )

    return network_keyer


def gather_actions_with_sparse(
    dense_actions: dict[ActionKey | VerboseActionKey, ActionCount],
    sparse_output,
    bindings: dict[str, str] = None,
    verbose: bool = False,
    use_name: bool = False,
) -> dict[ActionKey | VerboseActionKey, ActionCount]:
    """Compose dense action counts with sparse deltas.

    Instead of running gather_actions on mutated reuse, this applies
    precomputed per-buffet/per-compute action deltas from
    SparseAnalysisOutput to the unmodified dense action counts.

    Parameters
    ----------
    dense_actions
        Action counts from gather_actions() on UNMODIFIED reuse.
    sparse_output
        SparseAnalysisOutput with buffet_action_deltas and
        compute_action_deltas populated.
    bindings, verbose, use_name
        Same keying parameters used for the dense_actions call.
    """
    # Deep-copy so we don't mutate the caller's dict
    actions: dict[ActionKey | VerboseActionKey, ActionCount] = {
        k: ActionCount(v.total, v.max_per_unit)
        for k, v in dense_actions.items()
    }

    buffet_keyer = _get_buffet_keyer(verbose, use_name, bindings)
    compute_keyer = _get_compute_keyer(verbose, use_name, bindings)

    # Apply per-buffet deltas (same aggregation as gather_actions)
    for buffet, delta in sparse_output.buffet_action_deltas.items():
        read_key = buffet_keyer(buffet, "read")
        if read_key in actions:
            actions[read_key].total += delta.total_read
            actions[read_key].max_per_unit += delta.max_per_unit_read

        write_key = buffet_keyer(buffet, "write")
        if write_key in actions:
            actions[write_key].total += delta.total_write
            actions[write_key].max_per_unit += delta.max_per_unit_write

    # Apply per-compute deltas
    for compute_key, delta in sparse_output.compute_action_deltas.items():
        key = compute_keyer(compute_key, "compute")
        if key in actions:
            actions[key].total += delta.total_ops
            actions[key].max_per_unit += delta.max_per_unit_ops

    # Merge sparse-specific actions (gated/skipped/metadata)
    if sparse_output.sparse_actions:
        actions.update(sparse_output.sparse_actions)

    return actions


def compute_energy_from_actions(
    spec: Spec,
    action_counts: MappingABC[ActionKey, Real],
    overall_latency: float,
    component_to_non_power_gated_porp: dict[str, int] = None,
) -> dict[ActionKey | VerboseActionKey, Number]:
    if component_to_non_power_gated_porp is None:
        logging.warning(
            "No component_to_non_power_gated_porp provided, will not account for power gating."
        )
        component_to_non_power_gated_porp = {}

    energy_result = {}
    components = {}
    for key, counts in action_counts.items():
        if counts.total == 0:
            continue
        if key.level not in components:
            components[key.level] = spec.arch.find(key.level)
        component_obj = components[key.level]
        try:
            energy_per_ac = component_obj.actions[key.action].energy
        except (KeyError, TypeError):
            energy_per_ac = 0
        energy_result[key] = counts.total * energy_per_ac

    for component_obj in spec.arch.get_nodes_of_type(arch.Component):
        energy_result[ActionKey(component_obj.name, "leak")] = (
            component_obj.total_leak_power
            * overall_latency
            * component_to_non_power_gated_porp.get(component_obj.name, 1)
        )

    return energy_result
