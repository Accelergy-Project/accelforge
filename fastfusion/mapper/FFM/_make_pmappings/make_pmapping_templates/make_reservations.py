from collections.abc import Generator
from typing import Any

import fastfusion.frontend.arch as arch
from fastfusion.frontend.mapping import MappingNode, Reservation, Storage, TensorHolder
from fastfusion.frontend.spec import Spec


def _recursive_iter_fence_positions(
    fence_positions: dict[str, int],
    max_size: int,
) -> Generator[tuple[list[TensorHolder], Any], None, None]:
    if not fence_positions:
        yield {}
    mine = next(iter(fence_positions))
    myval = fence_positions[mine]
    following = {k: v for k, v in fence_positions.items() if k != mine}
    for i in range(myval, max_size):
        following = {k: max(v, i) for k, v in fence_positions.items() if k != mine}
        for following in _recursive_iter_fence_positions(following, max_size):
            yield {mine: i, **following}


def get_reservation_choices(
    mapping: list[TensorHolder],
    arch_flattened: list[arch.Leaf],
) -> Generator[tuple[list[TensorHolder], Any], None, None]:
    # Rules:
    # - In general, reservations go right under their storage node
    # - If a storage node is associated with a fanout, explore putting the reservation
    #   below it, below the next storage node, and so on. Stop once we don't have any
    #   more spatial loops to place. Push down all reservations below this fanout
    #   together.

    # Spatial loops:
    # - Must go below all storage nodes associated with something above the fanout.
    #   -> Memories above fanout must serve all fetches across fanout instances.
    # - Must go above all reservations associated with something below the fanout.
    #   -> Memories below fanout must be reserved for each fanout instance.
    # - If below any storage node associated with the fanout, then must be relevant.
    #   -> No peer-to-peer communication

    # Temporal loops:
    # - If between a storage node and a reservation node, the outermost temporal loop
    #   may be partially relevant. All others must be relevant.

    # Design choices here:
    # - Where to put the 'fence' for each fanout

    fanout_nodes = [n for n in arch_flattened if n.get_fanout() > 1]
    fanout_node_names = set[str](n.name for n in fanout_nodes)
    last_seen_fanout = None
    node2lastfanout = {}

    fence_positions: dict[str, int] = {}
    for i, node in enumerate(mapping):
        if node.component in fanout_node_names:
            fence_positions.setdefault(node.component, i)
            last_seen_fanout = node.component
        node2lastfanout[id(node)] = last_seen_fanout

    def try_add_reservations(
        new_mapping: list[MappingNode],
        reservations_to_add: list[TensorHolder],
        fence_positions: dict[str, int],
    ):
        for res in list(reservations_to_add):
            add = False
            if node2lastfanout[id(res)] is None:
                add = True
            elif i >= fence_positions[node2lastfanout[id(res)]]:
                add = True
            if add:
                new_mapping.append(
                    Reservation(
                        purposes=[res.component],
                        resource=res.component,
                        persistent=res.persistent,
                    )
                )
                reservations_to_add.remove(res)

    # Fence positions are indices of storage nodes below which we'll push all the
    # reservations below that fanout
    for fence_positions in _recursive_iter_fence_positions(
        fence_positions, len(mapping)
    ):
        new_mapping = []
        reservations_to_add = []
        for i, node in enumerate(mapping):
            new_mapping.append(node)
            reservations_to_add.append(node)
            try_add_reservations(new_mapping, reservations_to_add, fence_positions)
        try_add_reservations(new_mapping, reservations_to_add, fence_positions)
        yield new_mapping
