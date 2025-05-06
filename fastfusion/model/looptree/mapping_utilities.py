from bindings.looptree import LooptreeWorkload


def get_paths(mapping):
    cur_path = []
    for node in mapping:
        cur_path.append(node)
        if node['type'] in ['pipeline', 'sequential']:
            for child in node['branches']:
                for subpath in get_paths(child):
                    yield cur_path + subpath
        elif node['type'] == 'compute':
            yield cur_path.copy()


def get_leaves(mapping, is_path):
    if is_path:
        yield mapping[-1]
        return
    for node in mapping:
        if node['type'] in ['pipeline', 'sequential']:
            for child in node['branches']:
                yield from get_leaves(child, is_path)
        elif node['type'] == 'compute':
            yield node


def get_einsums_with_complete_mappings(mapping, workload, is_path):
    einsums_with_complete_mappings = set()
    for compute_node in get_leaves(mapping, is_path):
        einsum_name = compute_node['einsum']
        if isinstance(einsum_name, int):
            einsum_id = einsum_name
        else:
            einsum_id = workload.einsum_name_to_id()[einsum_name]
        if 'incomplete' not in compute_node:
            einsums_with_complete_mappings.add(einsum_id)
        if 'incomplete' in compute_node and not compute_node['incomplete']:
            einsums_with_complete_mappings.add(einsum_id)
    return einsums_with_complete_mappings


def get_intermediate_tensors(workload: LooptreeWorkload):
    result = set()
    for einsum in workload.einsum_id_to_name():
        written_tensors = workload.tensors_written_by_einsum(einsum)
        for tensor in written_tensors:
            reader_einsums = workload.reader_einsums(tensor)
            for reader in reader_einsums:
                if reader in workload.einsum_id_to_name():
                    result.add(tensor)
                    break

    return result


def get_last_storage_node(mapping, tensor):
    for i, node in enumerate(mapping):
        if node['type'] == 'storage' and tensor in node['dspace']:
            return i
    return None


def get_last_fused_loop_idx(mapping, intermediate_tensors):
    intermediates_remaining = set(intermediate_tensors)
    for i, node in enumerate(mapping):
        if node['type'] == 'storage':
            intermediates_remaining -= set(node['dspace'])
        if not intermediates_remaining:
            return i
    return float('inf')
