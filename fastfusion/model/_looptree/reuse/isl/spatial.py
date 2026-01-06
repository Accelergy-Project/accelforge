"""
Handles the ISL spatial reuse functions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import islpy as isl

from fastfusion.frontend.mapping import MappingNode
from fastfusion.model._looptree.reuse.isl.isl_functions import (
    insert_equal_dims_map,
    reorder_projector,
)
from fastfusion.model._looptree.reuse.isl.mapping_to_isl.types import (
    TEMPORAL_TAGS,
    Fill,
    Occupancy,
    SpatialTag,
    Tag,
    TaggedMap,
)


class Transfers(TaggedMap):
    """Transfers between regions in spacetime."""


class Reads(TaggedMap):
    """Reads between regions in spacetime."""


@dataclass(frozen=True, slots=True)
class TransferInfo:
    """Data transfer information about a certain [subset] of the chip."""

    # Crucial information to transfer info.
    fulfilled_fill: Transfers
    """Fills done by peer-to-peer transfers."""
    unfulfilled_fill: Fill
    """Fills not performed."""
    parent_reads: Reads
    """Fills done by parent-to-child transfers."""
    hops: isl.PwQPolynomial
    """Peer-to-peer transfer cost metric across spacetime."""

    # Metadata on what is occurring.
    link_transfer: bool


class TransferModel(ABC):
    """
    A peer-to-peer/multicast transfer model for spatial analysis.
    """

    @abstractmethod
    def apply(self, buff: MappingNode, fills: Fill, occs: Occupancy) -> TransferInfo:
        """
        Given a buffer, its fills across time, and its occupancies across time,
        calculate the spatial transfers.

        Parameters
        ----------
        buff:
            The buffer whose spatial analysis is being considered.
        fills:
            The fill of `buffer` across time from parents.
        occs:
            The occupancy of `buffer` across time.

        Returns
        -------
        Fills that were fulfilled, Fills that were unfilled, and parent reads per
        position in spacetime. Then, gets hops per timestep.
        """
        raise NotImplementedError(
            f"{type(self)} has not implemented `apply(self, MappingNode, Fill, Occupancy)`"
        )

    def __repr__(self):
        """Returns what transfer model it is."""
        return f"{type(self)}"


class SimpleLinkTransferModel(TransferModel):
    """
    Basic link transfer model.
    """

    def apply(self, buff: MappingNode, fills: Fill, occs: Occupancy) -> TransferInfo:
        # Sanity check the fill is for the same occupancy. Necessary but insufficient proof.
        assert fills.tags == occs.tags, (
            "Fill and Occupancy mismatch"
            "---------------------------"
            f"Fill: {fills}\n"
            f"Occs: {occs}\n"
        )

        # Gets number of input dimensions, along with spatial and temporal indices.
        n: int = fills.map_.dim(isl.dim_type.in_)
        spatial_dims: list[int] = get_spatial_tags_idxs(fills.tags, buff)
        last_temporal: Optional[int] = get_last_temporal_tag_idx(fills.tags)

        # No temporal or no spatial dims, you're just not moving data across time
        # so no transfers occurring.
        if last_temporal is None or len(spatial_dims) == 0:
            return TransferInfo(
                fulfilled_fill=Transfers(
                    fills.tags, fills.map_.subtract(fills.map_)
                ),  # Empty map
                unfulfilled_fill=fills,  # No fulfilled_fills, so only unfulfilled_fills
                parent_reads=Reads(
                    occs.tags, occs.map_.subtract(occs.map_)
                ),  # Empty map
                hops=isl.PwQPolynomial.from_qpolynomial(
                    isl.QPolynomial.zero_on_domain(fills.map_.domain().get_space())
                ),
                link_transfer=True,
            )
        # Gets the connectivity between points in space.
        connectivity: isl.Map = make_mesh_connectivity(
            len(spatial_dims), occs.map_.get_tuple_name(isl.dim_type.in_)
        )
        padded_connectivity: isl.Map = insert_equal_dims_map(
            connectivity, 0, 0, n - len(spatial_dims) - 1
        )
        permutation: list[int] = make_connectivity_permutation(spatial_dims, n)
        reorder_map: isl.Map = reorder_projector(
            permutation, occs.map_.get_tuple_name(isl.dim_type.in_)
        )
        complete_connectivity: isl.Map = reorder_map.apply_range(
            padded_connectivity
        ).apply_range(reorder_map.reverse())

        # Gets data available from neighbors at each point in space per time.
        available_from_neighbors: isl.Map = complete_connectivity.apply_range(occs.map_)
        # Prunes data that does not need to be fetched from a higher in the mem hierarchy.
        neighbor_filled: isl.Map = fills.map_.intersect(available_from_neighbors)

        return TransferInfo(
            fulfilled_fill=Transfers(fills.tags, neighbor_filled.coalesce()),
            unfulfilled_fill=Fill(
                fills.tags, fills.map_.subtract(neighbor_filled).coalesce()
            ),
            # Empty, since only p2p analyzed.
            parent_reads=Reads(
                fills.tags, neighbor_filled.subtract(neighbor_filled).coalesce()
            ),
            hops=isl.PwQPolynomial.from_qpolynomial(
                isl.QPolynomial.one_on_domain(neighbor_filled.wrap().get_space())
            )
            .intersect_domain(neighbor_filled.wrap())
            .coalesce(),
            link_transfer=True,
        )


def make_mesh_connectivity(n: int, spacetime: str) -> isl.Map:
    """
    Makes a neighbor-to-neighbor mesh connection given a number of spatial dims.

    Parameters
    ----------
    n:
        The number of spatial dimensions.
    spacetime:
        The name of the spacetime the mesh is operating on.

    Returns
    -------
    A direct orthogonal adjacency map on the space `spacetime[t, x_1, x_2, ..., x_n]`
    """
    mesh: isl.Map
    match (n):
        case 2:
            mesh = isl.Map.read_from_str(
                isl.DEFAULT_CONTEXT,
                "{ [t, x, y] -> [t-1, x', y'] : "
                " (y'=y and x'=x-1) or (y'=y and x'=x+1) "
                " or (x'=x and y'=y-1) or (x'=x and y'=y+1) }",
            )
        case 1:
            mesh = isl.Map.read_from_str(
                isl.DEFAULT_CONTEXT,
                "{ [t, x] -> [t-1, x'] : (x'=x-1) or (x'=x+1) }",
            )
        case _:
            raise ValueError(f"Cannot make mesh with {n} spatial dims")

    mesh = mesh.set_tuple_name(isl.dim_type.in_, spacetime).set_tuple_name(
        isl.dim_type.out, spacetime
    )

    return mesh


def make_connectivity_permutation(spatial_idxs: list[int], dims: int) -> list[int]:
    """TODO: Figure out what this is doing."""
    permutation: list[int] = []

    cur_spatial_idx: int = 0
    for i in range(dims):
        if cur_spatial_idx < len(spatial_idxs) and i == spatial_idxs[cur_spatial_idx]:
            cur_spatial_idx += 1
        else:
            permutation.append(i)

    for spatial_idx in spatial_idxs:
        permutation.append(spatial_idx)

    return permutation


def get_spatial_tags_idxs(tags: list[Tag], buffer: MappingNode) -> list[int]:
    """
    Given a list if tags, identify the spatial dimensions belong to a given `buffer`.

    Parameters
    ----------
    tags:
        The `Occupancy` or `Fill` domain dimension tags.
    buffer:
        The `MappingNode` which is the logical-memory we're looking for spatial
        dims over.

    Returns
    -------
    A list of the spatial_dim_idxs in order.
    """
    spatial_dim_idxs: list[int] = [
        i
        for i, tag in enumerate(tags)
        if isinstance(tag, SpatialTag) and tag.buffer == buffer
    ]

    return spatial_dim_idxs


def get_last_temporal_tag_idx(tags: list[Tag]) -> Optional[int]:
    """
    Returns the idx of the deepest temporal tag in the list.

    Parameters
    ----------
    tags:
        A list of `Tags`.

    Returns
    -------
    The index of the last tag that is a `TEMPORAL_TAGS`.
    """
    if len(tags) == 0:
        return None

    for idx, tag in reversed(list(enumerate(tags))):
        if isinstance(tag, TEMPORAL_TAGS):
            return idx

    return None
