from typing import overload
from sympy import Piecewise

# from fastfusion._model.looptree._isl.singular import get_value_from_singular_qpolynomial
from fastfusion.frontend.arch import Compute
from fastfusion.model._looptree.latency.processors import LATENCY_PROCESSORS
from fastfusion.model._looptree.reuse.isl import IslReuseAnalysisOutput
from fastfusion.model._looptree.reuse import SymbolicAnalysisOutput

from fastfusion.util._sympy.broadcast_max import Max

# from bindings.looptree import SpatialTag


def get_latency(looptree_results, mapping, workload, flattened_arch):
    comp_latency = calculate_compute_latency(looptree_results, mapping, workload)
    mem_latency = memory_latency(looptree_results, flattened_arch, mapping, workload)

    overall_latency = Max(comp_latency, *mem_latency.values())
    return overall_latency, comp_latency, mem_latency


@overload
def calculate_compute_latency(
    reuse_analysis_results: IslReuseAnalysisOutput, mapping, workload
):
    pass


@overload
def calculate_compute_latency(
    reuse_analysis_results: SymbolicAnalysisOutput, mapping, workload
):
    pass


def calculate_compute_latency(reuse_analysis_results, mapping, workload):
    if isinstance(reuse_analysis_results, IslReuseAnalysisOutput):
        return compute_isl_latency(
            reuse_analysis_results.temporal_steps, mapping, workload
        )
    elif isinstance(reuse_analysis_results, SymbolicAnalysisOutput):
        return compute_summarized_latency(
            reuse_analysis_results.compute_stats, mapping, workload
        )


def compute_summarized_latency(compute_stats, mapping, workload):
    # TODO: this is only for single-Einsum!!!
    longest_compute_latency = 0
    for stats in compute_stats.values():
        if longest_compute_latency == 0:
            longest_compute_latency = stats.max_latency
        else:
            longest_compute_latency = Max(longest_compute_latency, stats.max_latency)
    return longest_compute_latency


def compute_isl_latency(temporal_steps, mapping, workload):
    raise NotImplementedError()
    return get_value_from_singular_qpolynomial(
        _compute_latency(mapping.nodes, 0, temporal_steps, workload)[1]
    ).to_python()


def _compute_latency(mapping, top_idx: int, temporal_steps, workload):
    raise NotImplementedError()
    einsum_name_to_id = workload.einsum_name_to_id()

    next_top_idx = top_idx
    for node in mapping:
        next_top_idx += 1

        if node["type"] in LATENCY_PROCESSORS.keys():
            children_latencies = [
                _compute_latency(branch, next_top_idx, temporal_steps, workload)
                for branch in node["branches"]
            ]

            return LATENCY_PROCESSORS[node["type"]](top_idx, children_latencies)
        elif node["type"] == "compute":
            einsum = node["einsum"]
            if "incomplete" in node and node["incomplete"]:
                return ([], 0)
            einsum_id = einsum_name_to_id[einsum]
            return temporal_steps[einsum_id]


def ops_to_latency(dims, map):
    raise NotImplementedError()
    mask = [False] * len(dims)
    new_dims = []
    for i, d in enumerate(dims):
        if d == SpatialTag:
            mask[i] = True
        else:
            new_dims.append(d)
    return map.domain().identity().card()
