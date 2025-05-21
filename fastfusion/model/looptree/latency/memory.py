from collections import defaultdict
from sympy import Max, Min

from fastfusion.frontend.arch import Memory

from fastfusion.model.looptree.accesses import buffer_accesses_from_buffet_actions
from fastfusion.model.looptree.reuse.isl import IslReuseAnalysisOutput
from fastfusion.model.looptree.reuse.summarized import SummarizedAnalysisOutput


def memory_latency(
    looptree_results: IslReuseAnalysisOutput | SummarizedAnalysisOutput,
    arch,
    mapping,
    workload
):
    accesses_stats = buffer_accesses_from_buffet_actions(
        looptree_results,
        mapping,
        workload,
        is_path=False
    )

    bandwidths, component_tensor_datawidth = get_bandwidth(arch)

    component_to_read_writes = defaultdict(lambda: [None, None])
    for (component, tensor, _), accesses in accesses_stats.items():
        if (component, tensor.name) in component_tensor_datawidth:
            datawidth = component_tensor_datawidth[(component, tensor.name)]
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
        write_latency = writes / (write_bw + shared_bw)
        read_latency = reads / read_bw
        all_shared_for_write_latency = Max(write_latency, read_latency)

        # All shared bw for reading
        write_latency = writes / write_bw
        read_latency = reads / (read_bw + shared_bw)
        all_shared_for_read_latency = Max(write_latency, read_latency)

        # Shared bw shared for reading and writing
        shared_for_read_and_write_latency = (
            (reads + writes)
            / 
            (read_bw + write_bw + shared_bw)
        )

        component_latency[component] = Min(all_shared_for_write_latency,
                                           all_shared_for_read_latency,
                                           shared_for_read_and_write_latency)
    return component_latency


def get_bandwidth(arch):
    """Returns a dictionary from memory to bandwidth in bits/cycle"""
    component_bandwidths = {}
    component_tensor_datawidth = {}
    for node in arch['nodes']:
        if not isinstance(node, Memory):
            continue
        attributes = node.attributes
        n_rd_ports = attributes.get('n_rd_ports', 0)
        n_wr_ports = attributes.get('n_wr_ports', 0)
        n_rdwr_ports = attributes.get('n_rdwr_ports', 0)
        if n_rd_ports + n_wr_ports + n_rdwr_ports < 1:
            n_rdwr_ports = 1

        width = attributes['width']

        component_bandwidths[node['name']] = [
            n_rd_ports*width,
            n_wr_ports*width,
            n_rdwr_ports*width
        ]

        for tensor, datawidth in attributes['datawidth'].items():
            component_tensor_datawidth[(node['name'], tensor)] = datawidth
    return component_bandwidths, component_tensor_datawidth
