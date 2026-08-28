from dataclasses import dataclass, field

from accelforge.frontend._workload_isl._symbolic import Irrelevant, Relevant, PartiallyRelevant
from accelforge.frontend.mapping import MappingNode, iter_binding_blocks
from accelforge.frontend.renames import Rank, RANK_DUPLICATE, TensorName


def binding_is_local(
    src_rank: Rank,
    dst_relevancy: Irrelevant | Relevant | PartiallyRelevant
) -> bool:
    """
    Whether data requested by any dst can be found in the src with the greatest
    coordinate lower than or equal to the dst (i.e., nearest src with coord
    <= dst).

    If `src_rank` is None, the data is not distributed,  which we will model as
    locally bound.
    """
    return (
        src_rank is None
        or src_rank == RANK_DUPLICATE
        or (
            isinstance(dst_relevancy, Relevant | PartiallyRelevant)
            and src_rank == dst_relevancy.rank
        )
    )


@dataclass
class BindingLookups:
    """A mapping's bindings, indexed for the two questions the analysis asks of
    them. Both are keyed by the ``id()`` of a spatial loop rather than by its
    position, because the per-tensor mappings of
    :func:`~accelforge.frontend.mapping.Mapping._get_single_tensor_mapping`
    reorder and drop nodes but reuse the very same node objects."""

    rank_by_loop: dict[tuple[int, str, TensorName], Rank] = field(default_factory=dict)
    """The rank bound by each (spatial loop, distributed memory, tensor) triple."""

    replicating_loops: dict[tuple[str, TensorName], list[tuple[int, str]]] = field(
        default_factory=dict
    )
    """``(storage component, tensor)`` to the ``(id(spatial loop), dimension)`` of
    the binding loops that replicate the tensor rather than partition it. A memory
    instance holds a slice of its tensor only along the dimensions the binding
    partitions; along these ones it holds a whole copy, so its occupancy is that
    many times larger than an even split would suggest."""


def index_bindings(nodes: list[MappingNode]) -> BindingLookups:
    """Index the bindings of every physically-distributed Storage of a mapping."""
    lookups = BindingLookups()
    for block in iter_binding_blocks(nodes):
        binding = block.binding
        # A mapping that is still being built may not have a binding yet.
        if len(binding) != len(block.spatials):
            continue
        tensor_key = block.key
        for binding_loop, spatial in zip(binding, block.spatials):
            rank = binding_loop.rank_variable
            lookups.rank_by_loop[(id(spatial), *tensor_key)] = rank
            if rank == RANK_DUPLICATE:
                lookups.replicating_loops.setdefault(tensor_key, []).append(
                    (id(spatial), spatial.name)
                )
    return lookups
