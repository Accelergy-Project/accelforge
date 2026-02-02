from dataclasses import dataclass
from pathlib import Path


@dataclass
class LoopTreeStatistics:
    latency: float
    energy: float
    actions: dict
    memory_latency: dict
    capacity_usage: dict


def run_symbolic_model(mapping, workload, architecture):
    from pytimeloop._looptree.reuse import analyze_reuse_and_add_reservations_to_mapping
    from pytimeloop._looptree.energy import gather_actions

    job = Job.make_job(mapping=mapping, workload=workload, architecture=architecture)
    result = analyze_reuse_and_add_reservations_to_mapping(job)
    actions = gather_actions(result, bindings, use_name=True)
    pass


def run_looptree(config_dir, paths, tmp_path, bindings, call_accelergy):
    import islpy as isl
    from bindings.config import Config
    from bindings._looptree import LooptreeModelApp, LooptreeWorkload
    from pytimeloop.file import gather_yaml_configs
    from pytimeloop._looptree.capacity import compute_capacity_usage
    from pytimeloop._looptree.reuse._isl.des import deserialize_looptree_output
    from pytimeloop._looptree.energy import gather_actions, compute_energy_from_actions
    from pytimeloop._looptree.latency import get_latency
    from pytimeloop.timeloopfe.v4fused import Spec
    from pytimeloop.timeloopfe.common.backend_calls import call_accelergy_verbose

    yaml_str = gather_yaml_configs(config_dir, paths)
    config = Config(yaml_str, "yaml")
    model = LooptreeModelApp(config)

    workload = LooptreeWorkload.parse_cfg(config.root["problem"])

    spec = Spec.from_yaml_files([str(config_dir / p) for p in paths])

    if call_accelergy:
        if isinstance(tmp_path, Path):
            tmp_path = str(tmp_path)
        call_accelergy_verbose(spec, tmp_path)
        spec = Spec.from_yaml_files(
            [str(config_dir / p) for p in paths] + [str(Path(tmp_path) / "ERT.yaml")]
        )

    result = deserialize_looptree_output(model.run(), isl.DEFAULT_CONTEXT)

    actions = gather_actions(result, bindings)
    energy = compute_energy_from_actions(actions, spec.ERT)

    latency, comp_latency, mem_latency = get_latency(
        result, spec.mapping, workload, spec.arch, bindings
    )

    capacity_usage = compute_capacity_usage(
        spec.mapping.nodes, result.occupancy, workload
    )
    component_capacity_usage = {}
    for level, component in bindings.items():
        if level in capacity_usage:
            component_capacity_usage[component] = capacity_usage[level]

    return LoopTreeStatistics(
        latency, energy, actions, mem_latency, capacity_usage=component_capacity_usage
    )


def run_looptree_symbolic(config_dir, paths, tmp_path, bindings, call_accelergy):
    from bindings.config import Config
    from bindings._looptree import LooptreeWorkload, LooptreeWorkloadDependencyAnalyzer
    from pytimeloop.file import gather_yaml_configs
    from pytimeloop._looptree.capacity import compute_capacity_usage
    from pytimeloop._looptree.reuse import analyze_reuse_and_add_reservations_to_mapping
    from pytimeloop._looptree.energy import gather_actions, compute_energy_from_actions
    from pytimeloop._looptree.latency import get_latency
    from pytimeloop.timeloopfe.v4fused import Spec
    from pytimeloop.timeloopfe.common.backend_calls import call_accelergy_verbose
    from accelforge.mapper.FFM._make_pmappings.pmapper_job import Job

    yaml_str = gather_yaml_configs(config_dir, paths)

    config = Config(yaml_str, "yaml")
    workload = LooptreeWorkload.parse_cfg(config.root["problem"])
    analyzer = LooptreeWorkloadDependencyAnalyzer(workload)

    spec = Spec.from_yaml_files([str(config_dir / p) for p in paths])

    if call_accelergy:
        if isinstance(tmp_path, Path):
            tmp_path = str(tmp_path)
        call_accelergy_verbose(spec, tmp_path)
        spec = Spec.from_yaml_files(
            [str(config_dir / p) for p in paths] + [str(Path(tmp_path) / "ERT.yaml")]
        )

    job = Job.make_job(mapping=spec.mapping, workload=workload, architecture=spec.arch)
    tile_shapes, result = analyze_reuse_and_add_reservations_to_mapping(job)

    actions = gather_actions(result, bindings, use_name=True)
    energy = compute_energy_from_actions(actions, spec.ERT)

    latency, comp_latency, mem_latency = get_latency(
        result, spec.mapping, workload, spec.arch, bindings
    )

    capacity_usage = compute_capacity_usage(
        spec.mapping.nodes, result.occupancy, workload
    )
    component_capacity_usage = {}
    for level, component in bindings.items():
        if level in capacity_usage:
            component_capacity_usage[component] = capacity_usage[level]

    return LoopTreeStatistics(
        latency, energy, actions, mem_latency, capacity_usage=component_capacity_usage
    )
