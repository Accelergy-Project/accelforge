from collections.abc import Mapping as MappingABC
import logging
from numbers import Number
from numbers import Real

from accelforge.frontend import arch
from accelforge.frontend.spec import Spec
from accelforge.model._looptree.reuse.symbolic import SymbolicAnalysisOutput
from accelforge.util._frozenset import oset
from accelforge.util._base_analysis_types import (
    ActionCount,
    ActionKey,
    VerboseActionKey,
)
from accelforge.model._looptree.types import Network


def gather_actions(
    looptree_results: SymbolicAnalysisOutput,
    bindings: dict[str, str],
    spec: Spec,
    verbose: bool = False,
    use_name=False,
):
    actions: dict[tuple[str, str], ActionCount] = {}
    compute_levels = oset(c.level for c in looptree_results.compute_stats)

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

    # `ops.total_ops` is a per-op-kind dict ({op_kind: count}). Emit one action
    # key per (level, op_kind), where the action *name* is resolved from the
    # Compute's ComputeAction whose op_kind matches. This is what lets the
    # downstream `compute_energy_from_actions` look up energy via
    # `component.actions[key.action].energy`. With the legacy single-action
    # arch ({op_kind: "mac"}, name: "compute") and the default einsum profile
    # ({"mac": 1}), this collapses to exactly one ("compute") key per level,
    # bit-identical to prior behavior.
    for compute, ops in looptree_results.compute_stats.items():
        for op_kind, total in ops.total_ops.items():
            action_name = _resolve_compute_action_name(spec, compute.level, op_kind)
            key = compute_keyer(compute, action_name)
            if key not in actions:
                actions[key] = ActionCount.default()
            actions[key].total += total
            actions[key].max_per_unit += ops.max_per_unit_ops.get(op_kind, 0)

    for network, stats in looptree_results.network_stats.items():
        key = network_keyer(network, "hops")
        if key not in actions:
            actions[key] = ActionCount.default()
        actions[key].total += stats.total_hops
        actions[key].max_per_unit += stats.max_hops

    _apply_actions_scale(actions, spec)

    return actions

def _apply_actions_scale(actions, spec):
    components = {}
    for key, count in actions.items():
        if key.level not in components:
            components[key.level] = spec.arch.find(key.level)
        scale = getattr(components[key.level], "actions_scale", 1)
        count.total *= scale
        count.max_per_unit *= scale

def _resolve_compute_action_name(spec: Spec, level: str, op_kind: str) -> str:
    """Map (compute level, op_kind) to the matching ComputeAction's name.
    """
    component = spec.arch.find(level)
    return component.action_for_op_kind(op_kind).name

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
        except KeyError as e:
            raise KeyError(
                f"Action {key.action} not found in component {key.level}. Action occurred "
                f"{counts.total} times."
            ) from None
        energy_result[key] = counts.total * energy_per_ac

    for component_obj in spec.arch.get_nodes_of_type(arch.Component):
        energy_result[ActionKey(component_obj.name, "leak")] = (
            component_obj.total_leak_power
            * overall_latency
            * component_to_non_power_gated_porp.get(component_obj.name, 1)
        )

    return energy_result
