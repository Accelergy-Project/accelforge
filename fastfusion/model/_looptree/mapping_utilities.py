from typing import Generator, List, Tuple

from fastfusion.frontend.mapping import (
    Compute,
    Mapping,
    MappingNode,
    Pipeline,
    Sequential,
)
from fastfusion.frontend.workload import Workload


def get_paths(root: Mapping) -> Generator[Tuple[MappingNode, Compute], None, None]:
    """
    Given a MappingNode, get the paths to all all leaves in post-order.

    :param root:    The root of the child exploration.

    :type root:     MappingNode

    :returns:       A generator of all the MappingNodes to a Compute leaf.
    :rtype:         Generator[List[MappingNode]]
    """
    cur_path: List[MappingNode] = []
    for node in root.nodes:
        cur_path.append(node)
        match node:
            # Pipelines or sequentials should have their paths expanded.
            # Mappings naturally get expanded.
            case Mapping() | Pipeline() | Sequential():
                for child in node.nodes:
                    for subpath in get_paths(child):
                        yield tuple(cur_path) + subpath
            # Computes are leaves so should get a yield here.
            case Compute():
                yield tuple(cur_path)
            # Not implemented so continue.
            case _:
                # TODO: Check this is correct
                continue
                raise NotImplementedError(
                    f"{type(node)} does not have type elucidation.\n"
                    f"---\n"
                    f"node={node}"
                )


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
        written_tensors = workload.einsums[einsum].output_tensor_names
        for tensor in written_tensors:
            reader_einsums = workload.reader_einsums(tensor)
            for reader in reader_einsums:
                if reader in workload.einsum_id_to_name():
                    result.add(tensor)
                    break

    return result
