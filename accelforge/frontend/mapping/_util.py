"""
Utilities for interacting with mapping nodes (e.g., iterating over them) that
doesn't make sense to be part of the frontend interface. Placing it here instead
of adding more private methods to the frontend classes, which maybe should focus
on the public interface and private helper classes.
"""
from collections.abc import Iterable, MutableSequence
from dataclasses import dataclass, field

from accelforge.frontend import arch
from accelforge.frontend.renames import TensorName
from accelforge.frontend.mapping.mapping import (
    Binding,
    EvalableList,
    MappingNode,
    Storage,
    Spatial,
)


def find_binding_node(
    nodes: "Iterable[MappingNode]", component: str, tensor: TensorName
) -> Binding | None:
    """
    The :class:`~.Binding` node of `nodes` that binds `tensor` in `component`, or None
    if nothing does.

    A Binding node speaks for the Storage nodes above it, which is where the mapper
    writes them; the search does not depend on that, because a ``(memory, tensor)``
    pair is bound at most once in a mapping of one Einsum -- the binding of a pair is
    what a :class:`~.Binding` node is keyed by, so it could not say two things about
    one pair even if we wanted it to.
    """
    for node in nodes:
        if isinstance(node, Binding) and node.binding_of(component, tensor) is not None:
            return node
    return None


def binding_of(
    nodes: "Iterable[MappingNode]", component: str, tensor: TensorName
) -> "EvalableList[Spatial]":
    """
    The binding of `tensor` in `component` under `nodes`, or an empty list if `nodes`
    binds nothing there.
    """
    node = find_binding_node(nodes, component, tensor)
    if node is None:
        return EvalableList()
    return node.binding_of(component, tensor)


@dataclass
class BindingBlock:
    """
    One tensor held in one physically-distributed memory, the spatial loops that bind
    it to that memory's instances, and the :class:`~.Binding` node -- if there is one
    yet -- that says what each of those loops binds.
    """

    storage: Storage
    """ The Storage node holding the tensor. """

    tensor: TensorName
    """ The tensor being bound. """

    spatials: list[Spatial]
    """ The spatial loops below `storage` that distribute the tensor. """

    nodes: "MutableSequence[MappingNode]" = field(repr=False, default_factory=list)
    """ The mapping this block was found in, which is where its binding lives. """

    _insert_after: MappingNode | None = field(repr=False, default=None)
    """ The last Storage of this block's group: a Binding node created for this block
    goes just below it, so that it sits below every Storage it speaks for. """

    @property
    def component(self) -> str:
        """The memory holding the tensor."""
        return self.storage.component

    @property
    def key(self) -> tuple[str, TensorName]:
        """The ``(memory, tensor)`` pair this block binds."""
        return (self.component, self.tensor)

    @property
    def node(self) -> Binding | None:
        """The Binding node that binds this block, or None if nothing does yet."""
        return find_binding_node(self.nodes, *self.key)

    @property
    def binding(self) -> "EvalableList[Spatial]":
        """
        The loops that bind this block, or an empty list if it is not bound yet. One
        loop per loop of `spatials`, in the same order.
        """
        return binding_of(self.nodes, *self.key)

    def set_binding(self, loops: "Iterable[Spatial]") -> None:
        """
        Bind this block to `loops`, putting them in the Binding node that already binds
        it or in a new one just below the Storage nodes of its group.
        """
        node = self.node
        if node is None:
            node = self._make_binding_node()
        node.set_binding(*self.key, loops)

    def clear_binding(self) -> None:
        """Unbind this block, leaving any Binding node that bound it in place."""
        node = self.node
        if node is not None:
            node.remove_binding(*self.key)

    def _make_binding_node(self) -> Binding:
        """
        The Binding node to write this block's binding into: the one just below the
        Storage nodes of its group, inserted if it is not there yet.

        The position is found by identity rather than remembered as an index, because
        an earlier block of the same group may have inserted a node in the meantime.
        """
        nodes = self.nodes
        after = self._insert_after if self._insert_after is not None else self.storage
        for i, node in enumerate(nodes):
            if node is after:
                break
        else:
            raise ValueError(
                f"{after.compact_str()} is not in the mapping this block came from, "
                f"so there is no telling where its binding goes."
            )
        if i + 1 < len(nodes) and isinstance(nodes[i + 1], Binding):
            return nodes[i + 1]
        node = Binding()
        nodes.insert(i + 1, node)
        return node


def iter_binding_blocks(
    nodes: "Iterable[MappingNode]",
) -> list[BindingBlock]:
    """
    Iterate over the tensors of physically-distributed memories with the Spatial loops
    that bind them, one :class:`BindingBlock` per (Storage node, tensor) pair.

    Spatial loops below a block of Storage nodes bind every distributed Storage in the
    block; a Storage node appearing after spatial loops starts a new block. Blocks are
    returned in the order in which their Storage nodes appear top to bottom, and their
    tensors in the order the Storage node lists them.
    """
    # A list rather than the iterable itself, so that a block can find its own position
    # in it -- but the caller's own sequence when there is one, so that a block whose
    # binding is set can insert a Binding node the caller will see.
    if isinstance(nodes, MutableSequence):
        seq = nodes
    else:
        seq = list(nodes)

    blocks: list[BindingBlock] = []
    active: list[BindingBlock] = []
    spatial_component = None
    spatial_seen = False
    for node in seq:
        if isinstance(node, Storage):
            component = node.component_object
            assert isinstance(component, arch.Spatialable)
            this_spatial_component = component._physical_spatial_component
            if not component._is_distributed():
                continue
            if spatial_seen or spatial_component != this_spatial_component:
                active = []
                spatial_seen = False
                spatial_component = this_spatial_component
            assert isinstance(component, arch.Memory)
            new = [
                BindingBlock(storage=node, tensor=tensor, spatials=[], nodes=seq)
                for tensor in node.tensors
            ]
            # Every block of this group takes its binding node from below the last
            # Storage of the group, so the node speaks for all of them.
            for block in active + new:
                block._insert_after = node
            active.extend(new)
            blocks.extend(new)
        elif (
            isinstance(node, Spatial)
            and active
            and node.component == spatial_component
        ):
            spatial_seen = True
            for block in active:
                block.spatials.append(node)
    return blocks


def binding_summary(
    nodes: "Iterable[MappingNode]",
) -> dict[tuple[str, str], dict[str, str]]:
    """
    The binding of every bound ``(memory, tensor)`` pair of a mapping, as
    ``{(component, tensor): {spatial dimension: bound rank}}``. ``DUPLICATE`` as a rank
    means the tensor is replicated across that dimension rather than partitioned.

    Intended for reporting and for comparing the bindings that different mapper
    configurations choose. A dimension that binds a pair more than once (more than one
    spatial loop over the same dimension) appears once per loop, suffixed with the index
    of the loop.
    """
    summary: dict[tuple[str, str], dict[str, str]] = {}
    for node in nodes:
        if not isinstance(node, Binding):
            continue
        for component, tensor, loops in node.entries():
            per_dimension: dict[str, str] = {}
            for i, loop in enumerate(loops):
                key = (
                    loop.name if loop.name not in per_dimension else f"{loop.name}[{i}]"
                )
                per_dimension[key] = loop.rank_variable
            summary[(component, tensor)] = per_dimension
    return summary


def drop_binding_loops_for_removed_spatials(
    nodes: "Iterable[MappingNode]", removed: set[int]
) -> None:
    """
    Drop from every binding the loops that pair with a spatial loop in `removed`, given
    as a set of ``id()``.

    A binding is positional: its i-th loop belongs to the i-th spatial loop that
    :func:`iter_binding_blocks` pairs with the bound Storage. So a caller that removes
    spatial loops from a mapping must call this while those loops are still in
    `nodes` -- afterwards, the pairing that says which binding loops went with
    them is gone.

    Loops are matched by ``id()`` rather than by position for the same reason
    :class:`~accelforge.model._looptree.reuse.symbolic._binding.BindingLookups`
    does: node objects are reused across mappings that reorder and drop nodes.
    """
    if not isinstance(nodes, MutableSequence):
        nodes = list(nodes)
    if not any(isinstance(n, Binding) for n in nodes):
        return
    for block in iter_binding_blocks(nodes):
        binding = block.binding
        # A mapping that is still being built may not have a binding yet.
        if len(binding) != len(block.spatials):
            continue
        block.set_binding(
            [
                binding_loop
                for binding_loop, spatial in zip(binding, block.spatials)
                if id(spatial) not in removed
            ]
        )
