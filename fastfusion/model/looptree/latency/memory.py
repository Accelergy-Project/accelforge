from collections import defaultdict

from fastfusion.frontend.architecture import Memory

from fastfusion.model.looptree.accesses import buffer_accesses_from_buffet_actions
from fastfusion.model.looptree.reuse.isl import IslReuseAnalysisOutput
from fastfusion.model.looptree.reuse.summarized import SummarizedAnalysisOutput

from fastfusion.util.sympy.broadcast_max import Max


def memory_latency(
    looptree_results: IslReuseAnalysisOutput | SummarizedAnalysisOutput,
    flattened_arch,
    mapping,
    workload
):
    accesses_stats = buffer_accesses_from_buffet_actions(
        looptree_results,
        mapping,
        workload,
        is_path=False
    )

    bandwidths, component_tensor_datawidth = get_bandwidth(flattened_arch)

    component_to_read_writes = defaultdict(lambda: [None, None])
    for (component, tensor, _), accesses in accesses_stats.items():
        if (component, tensor) in component_tensor_datawidth:
            datawidth = component_tensor_datawidth[(component, tensor)]
        elif (component, '*') in component_tensor_datawidth:
            datawidth = component_tensor_datawidth[(component, '*')]
        else:
            raise RuntimeError(f'No datawidth for {component} and {tensor}')
        
        if component not in component_to_read_writes:
            component_to_read_writes[component][0] = accesses.max_per_unit_reads*datawidth
            component_to_read_writes[component][1] = accesses.max_per_unit_writes*datawidth
        else:
            component_to_read_writes[component][0] += accesses.max_per_unit_reads*datawidth
            component_to_read_writes[component][1] += accesses.max_per_unit_writes*datawidth

    component_latency = {}
    for component, (reads, writes) in component_to_read_writes.items():
        read_bw, write_bw, shared_bw = bandwidths[component]

        # For numerical stability
        read_bw += 1e-8
        write_bw += 1e-8
        shared_bw += 1e-8

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
