from fastfusion.frontend.mapping import Compute, Mapping, Pipeline, Sequential, Storage
from fastfusion.frontend.workload import Workload


def get_paths(mapping: Mapping):
    cur_path = []
    for node in mapping:
        cur_path.append(node)
        if isinstance(node, Pipeline) or isinstance(node, Sequential):
            for child in node.children:
                for subpath in get_paths(child):
                    yield cur_path + subpath
        elif isinstance(node, Compute):
            yield cur_path.copy()


def get_leaves(mapping: Mapping, is_path):
    if is_path:
        yield mapping[-1]
        return
    for node in mapping:
        if isinstance(node, Pipeline) or isinstance(node, Sequential):
            for child in node.children:
                yield from get_leaves(child, is_path)
        elif isinstance(node, Compute):
            yield node


def get_intermediate_tensors(workload: Workload):
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
        if isinstance(node, Storage) and tensor in node.tensor:
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
