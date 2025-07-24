from fastfusion.frontend.mapping import Compute, Mapping, Pipeline, Sequential
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
