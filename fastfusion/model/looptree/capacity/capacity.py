from pytimeloop.isl.singular import get_value_from_singular_qpolynomial

from .aggregators import CAPACITY_AGGREGATORS


def compute_capacity_usage(mapping, occupancy, workload):
    caps = {}
    tensor_name_to_id = workload.data_space_name_to_id()
    einsum_name_to_id = workload.einsum_name_to_id()

    for node in mapping:
        einsums = get_einsums(mapping)
        if node['type'] == 'storage':
            buf = node['target']
            if buf not in caps:
                caps[buf] = 0

            for tensor in node['dspace']:
                tensor_id = tensor_name_to_id[tensor]
                max_cap = 0
                for einsum in einsums:
                    einsum_id = einsum_name_to_id[einsum]
                    key = (buf, tensor_id, einsum_id)
                    if key in occupancy:
                        max_cap = max(
                            max_cap,
                            get_value_from_singular_qpolynomial(occupancy[key][1])
                        )
                caps[buf] += max_cap

        elif node['type'] in ['sequential', 'parallel', 'pipeline']:
            aggregate_capacity = CAPACITY_AGGREGATORS[node['type']]
            child_caps = [
                compute_capacity_usage(b, occupancy, workload)
                for b in node['branches']
            ]
            aggregate_capacity(child_caps, caps)
    return caps


def get_einsums(mapping):
    for node in mapping:
        if node['type'] in ['sequential', 'parallel', 'pipeline']:
            return sum((get_einsums(b) for b in node['branches']), start=[])
        elif node['type'] == 'compute':
            return [node['einsum']]