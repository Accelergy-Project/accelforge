from typing import overload
from sympy import Max

from pytimeloop.isl.singular import get_value_from_singular_qpolynomial
from pytimeloop.looptree.latency.processors import LATENCY_PROCESSORS
from pytimeloop.looptree.reuse.isl import IslReuseAnalysisOutput
from pytimeloop.looptree.reuse.summarized import SummarizedAnalysisOutput
from pytimeloop.looptree.latency.memory import memory_latency

from bindings.looptree import SpatialTag


def get_latency(looptree_results,
                mapping,
                workload,
                arch,
                bindings):
    comp_latency = calculate_compute_latency(looptree_results,
                                             mapping,
                                             workload)
    mem_latency = memory_latency(looptree_results,
                                 arch,
                                 mapping,
                                 workload,
                                 bindings)
    overall_latency = Max(comp_latency, Max(*mem_latency.values()))
    return overall_latency, comp_latency, mem_latency


@overload
def calculate_compute_latency(reuse_analysis_results: IslReuseAnalysisOutput,
                              mapping,
                              workload):
    pass
@overload
def calculate_compute_latency(reuse_analysis_results: SummarizedAnalysisOutput,
                              mapping,
                              workload):
    pass
def calculate_compute_latency(reuse_analysis_results, mapping, workload):
    if isinstance(reuse_analysis_results, IslReuseAnalysisOutput):
        return compute_isl_latency(reuse_analysis_results.temporal_steps,
                                   mapping,
                                   workload)
    elif isinstance(reuse_analysis_results, SummarizedAnalysisOutput):
        return compute_summarized_latency(
            reuse_analysis_results.temporal_steps,
            mapping,
            workload
        )


def compute_isl_latency(temporal_steps, mapping, workload):
    return get_value_from_singular_qpolynomial(
        _compute_latency(mapping.nodes, 0, temporal_steps, workload)[1]
    ).to_python()


def compute_summarized_latency(temporal_steps, mapping, workload):
    # TODO: this is only for single-Einsum!!!
    return sum(value for key, value in temporal_steps.items())


def _compute_latency(mapping, top_idx: int, temporal_steps, workload):
    einsum_name_to_id = workload.einsum_name_to_id()

    next_top_idx = top_idx
    for node in mapping:
        next_top_idx += 1

        if node['type'] in LATENCY_PROCESSORS.keys():
            children_latencies = [
                _compute_latency(branch, next_top_idx, temporal_steps, workload)
                for branch in node['branches']
            ]

            return LATENCY_PROCESSORS[node['type']](top_idx,
                                                    children_latencies)
        elif node['type'] == 'compute':
            einsum = node['einsum']
            if 'incomplete' in node and node['incomplete']:
                return ([], 0)
            einsum_id = einsum_name_to_id[einsum]
            return temporal_steps[einsum_id]


def ops_to_latency(dims, map):
    mask = [False]*len(dims)
    new_dims = []
    for i, d in enumerate(dims):
        if d == SpatialTag:
            mask[i] = True
        else:
            new_dims.append(d)
    return map.domain().identity().card()