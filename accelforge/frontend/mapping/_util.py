"""
Utilities for interacting with mapping nodes (e.g., iterating over them) that
doesn't make sense to be part of the frontend interface. Placing it here instead
of adding more private methods to the frontend classes, which maybe should focus
on the public interface and private helper classes.
"""
from collections.abc import Iterable

from accelforge.frontend import arch
from accelforge.frontend.mapping.mapping import (
    EvalableList,
    MappingNode,
    Storage,
    Spatial,
)


def iter_binding_blocks(
    nodes: "Iterable[MappingNode]",
) -> list[tuple[Storage, list["Spatial"]]]:
    """
    Iterate over :class:`~.Storage` node of physically-distributed memories
    with the Spatial loops that bind it.

    Spatial loops below a block of Storage nodes bind every distributed Storage
    in the block; a Storage node appearing after spatial loops starts a new
    block. Blocks are returned in the order in which their Storage nodes appear
    top to bottom.
    """
    blocks: list[tuple[Storage, list[Spatial]]] = []
    active: list[tuple[Storage, list[Spatial]]] = []
    spatial_component = None
    spatial_seen = False
    for node in nodes:
        if isinstance(node, Storage):
            component = node.component_object
            assert isinstance(component, arch.Spatialable)
            this_spatial_component = component._physical_spatial_component
            if not component._is_distributed():
                continue
            if spatial_seen or spatial_component != this_spatial_component:
                active = []
                spatial_seen = False
                spatial_component = None
                spatial_component = this_spatial_component
            assert isinstance(component, arch.Memory)
            assert len(node.tensors) == 1, (
                f"Binding is per-tensor, but {node.compact_str()} holds "
                f"{len(node.tensors)} tensors."
            )
            block = (node, [])
            active.append(block)
            blocks.append(block)
        elif isinstance(node, Spatial) and active and node.component == spatial_component:
            spatial_seen = True
            for _, spatials in active:
                spatials.append(node)
    return blocks


def binding_summary(
    nodes: "Iterable[MappingNode]",
) -> dict[tuple[str, str], dict[str, str]]:
    """
    The binding of every bound :class:`~.Storage` of a mapping, as
    ``{(component, tensor): {spatial dimension: bound rank}}``. ``DUPLICATE`` as a rank
    means the tensor is replicated across that dimension rather than partitioned.

    Intended for reporting and for comparing the bindings that different mapper
    configurations choose. A dimension that binds a storage more than once (more than
    one spatial loop over the same dimension) appears once per loop, suffixed with the
    index of the loop.
    """
    summary: dict[tuple[str, str], dict[str, str]] = {}
    for node in nodes:
        if not isinstance(node, Storage) or not node.binding:
            continue
        per_dimension: dict[str, str] = {}
        for i, loop in enumerate(node.binding):
            key = loop.name if loop.name not in per_dimension else f"{loop.name}[{i}]"
            per_dimension[key] = loop.rank_variable
        summary[(node.component, node.tensor)] = per_dimension
    return summary


def drop_binding_loops_for_removed_spatials(
    nodes: "Iterable[MappingNode]", removed: set[int]
) -> None:
    """
    Drop from every distributed :class:`~.Storage`'s binding the loops that pair
    with a spatial loop in `removed`, given as a set of ``id()``.

    A binding is positional: its i-th loop belongs to the i-th spatial loop that
    :func:`iter_binding_blocks` pairs with the Storage. So a caller that removes
    spatial loops from a mapping must call this while those loops are still in
    `nodes` -- afterwards, the pairing that says which binding loops went with
    them is gone.

    Loops are matched by ``id()`` rather than by position for the same reason
    :class:`~accelforge.model._looptree.reuse.symbolic._binding.BindingLookups`
    does: node objects are reused across mappings that reorder and drop nodes.
    """
    nodes = list(nodes)
    if not any(isinstance(n, Storage) and n.binding for n in nodes):
        return
    for storage, spatials in iter_binding_blocks(nodes):
        # A mapping that is still being built may not have a binding yet.
        if len(storage.binding) != len(spatials):
            continue
        storage.binding = EvalableList(
            [
                binding_loop
                for binding_loop, spatial in zip(storage.binding, spatials)
                if id(spatial) not in removed
            ]
        )
