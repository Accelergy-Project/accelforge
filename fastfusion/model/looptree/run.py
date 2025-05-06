from dataclasses import dataclass
from pathlib import Path

import islpy as isl

from bindings.config import Config
from bindings.looptree import LooptreeModelApp, LooptreeWorkload, LooptreeWorkloadDependencyAnalyzer

from pytimeloop.file import gather_yaml_configs

from pytimeloop.looptree.capacity import compute_capacity_usage
from pytimeloop.looptree.reuse.isl.des import deserialize_looptree_output
from pytimeloop.looptree.reuse.summarized import analyze_reuse
from pytimeloop.looptree.energy import gather_actions, compute_energy_from_actions
from pytimeloop.looptree.latency import get_latency

from pytimeloop.timeloopfe.v4fused import Specification
from pytimeloop.timeloopfe.common.backend_calls import call_accelergy_verbose


@dataclass
class LoopTreeStatistics:
    latency: float
    energy: float
    actions: dict
    memory_latency: dict
    capacity_usage: dict


def run_looptree(config_dir, paths, tmp_path, bindings, call_accelergy):
    yaml_str = gather_yaml_configs(config_dir, paths)
    config = Config(yaml_str, 'yaml')
    model = LooptreeModelApp(config)

    workload = LooptreeWorkload.parse_cfg(config.root['problem'])

    spec = Specification.from_yaml_files([
        str(config_dir / p) for p in paths
    ])

    if call_accelergy:
        if isinstance(tmp_path, Path):
            tmp_path = str(tmp_path)
        call_accelergy_verbose(spec, tmp_path)
        spec = Specification.from_yaml_files([
            str(config_dir / p) for p in paths
        ] + [str(Path(tmp_path) / 'ERT.yaml')])

    result = deserialize_looptree_output(model.run(), isl.DEFAULT_CONTEXT)

    actions = gather_actions(result, spec.mapping, workload, bindings)
    energy = compute_energy_from_actions(actions, spec.ERT)

    latency, comp_latency, mem_latency = get_latency(result,
                                                     spec.mapping,
                                                     workload,
                                                     spec.architecture,
                                                     bindings)

    capacity_usage = compute_capacity_usage(spec.mapping.nodes,
                                            result.occupancy,
                                            workload)
    component_capacity_usage = {}
    for level, component in bindings.items():
        if level in capacity_usage:
            component_capacity_usage[component] = capacity_usage[level]

    return LoopTreeStatistics(latency,
                              energy,
                              actions,
                              mem_latency,
                              capacity_usage=component_capacity_usage)


def run_looptree_symbolic(config_dir, paths, tmp_path, bindings, call_accelergy):
    yaml_str = gather_yaml_configs(config_dir, paths)

    config = Config(yaml_str, 'yaml')
    workload = LooptreeWorkload.parse_cfg(config.root['problem'])
    analyzer = LooptreeWorkloadDependencyAnalyzer(workload)

    spec = Specification.from_yaml_files([
        str(config_dir / p) for p in paths
    ])

    if call_accelergy:
        if isinstance(tmp_path, Path):
            tmp_path = str(tmp_path)
        call_accelergy_verbose(spec, tmp_path)
        spec = Specification.from_yaml_files([
            str(config_dir / p) for p in paths
        ] + [str(Path(tmp_path) / 'ERT.yaml')])


    tile_shapes, result = analyze_reuse(spec.mapping.nodes, workload, analyzer)

    actions = gather_actions(result, spec.mapping, workload, bindings, use_name=True)
    energy = compute_energy_from_actions(actions, spec.ERT)

    latency, comp_latency, mem_latency = get_latency(result,
                                                     spec.mapping,
                                                     workload,
                                                     spec.architecture,
                                                     bindings)

    capacity_usage = compute_capacity_usage(spec.mapping.nodes,
                                            result.occupancy,
                                            workload)
    component_capacity_usage = {}
    for level, component in bindings.items():
        if level in capacity_usage:
            component_capacity_usage[component] = capacity_usage[level]

    return LoopTreeStatistics(latency,
                              energy,
                              actions,
                              mem_latency,
                              capacity_usage=component_capacity_usage)
