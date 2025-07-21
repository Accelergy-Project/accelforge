from collections import defaultdict

from fastfusion.frontend.architecture import Memory

from fastfusion.frontend.mapping import Compute
from fastfusion.model.looptree.accesses import isl_buffer_accesses_from_buffet_actions
from fastfusion.model.looptree.mapping_utilities import get_leaves
from fastfusion.model.looptree.reuse.isl import IslReuseAnalysisOutput
from fastfusion.model.looptree.reuse.summarized import SummarizedAnalysisOutput

from fastfusion.model.looptree.reuse.summarized.symbolic import Buffet, BuffetStats
from fastfusion.util.sympy.broadcast_max import Max

def isl_to_summarized(looptree_results: IslReuseAnalysisOutput, mapping, workload) -> SummarizedAnalysisOutput:
    accesses_stats = isl_buffer_accesses_from_buffet_actions(
        looptree_results,
        mapping,
        workload,
        is_path=False
    )
    buffet_stats = {
        Buffet(level=component, tensor=tensor, einsum=einsum): BuffetStats(    
            max_per_unit_read_actions=accesses.max_per_unit_reads,
            max_per_unit_write_actions=accesses.max_per_unit_writes,
        )
        for (component, tensor, einsum), accesses in accesses_stats.items()
    }
    return SummarizedAnalysisOutput(buffet_stats=buffet_stats)



def memory_latency(
    looptree_results: SummarizedAnalysisOutput | IslReuseAnalysisOutput,
    flattened_arch,
    mapping,
    workload
):
    if isinstance(looptree_results, IslReuseAnalysisOutput):
        looptree_results = isl_to_summarized(looptree_results, mapping, workload)

    compute_targets = set()
    for compute_node in get_leaves(mapping.nodes, False):
        assert isinstance(compute_node, Compute)
        compute_targets.add(compute_node.compute)

    bandwidths, component_tensor_datawidth = get_bandwidth(flattened_arch)

    component_to_read_writes = defaultdict(lambda: [0, 0])
    for buffet, buffet_stats in looptree_results.buffet_stats.items():
        component = buffet.level
        if component in compute_targets:
            continue
        tensor = buffet.tensor
        if (component, tensor) in component_tensor_datawidth:
            datawidth = component_tensor_datawidth[(component, tensor)]
        elif (component, '*') in component_tensor_datawidth:
            datawidth = component_tensor_datawidth[(component, '*')]
        else:
            raise RuntimeError(f'No datawidth for {component} and {tensor}')
        
        component_to_read_writes[component][0] += buffet_stats.max_per_unit_read_actions*datawidth
        component_to_read_writes[component][1] += buffet_stats.max_per_unit_write_actions*datawidth

    component_latency = {}
    for component, (reads, writes) in component_to_read_writes.items():
        read_bw, write_bw, shared_bw = bandwidths[component]

        # All shared bw for writing
        write_latency = writes / write_bw
        read_latency = reads / read_bw
        shared_latency = (reads + writes) / shared_bw
        component_latency[component] = Max(write_latency, read_latency, shared_latency)

    return component_latency


def get_bandwidth(flattened_arch):
    """Returns a dictionary from memory to bandwidth in bits/cycle"""
    component_bandwidths = {}
    component_tensor_datawidth = {}
    for node in flattened_arch:
        if not isinstance(node, Memory):
            continue
        attributes = node.attributes

        component_bandwidths[node.name] = [
            node.attributes.read_bandwidth,
            node.attributes.write_bandwidth,
            node.attributes.shared_read_write_bandwidth
        ]

        # NOTE: supports per-tensor datawidth in the future. '*' can be tensor name
        component_tensor_datawidth[(node.name, '*')] = attributes.datawidth
    return component_bandwidths, component_tensor_datawidth
