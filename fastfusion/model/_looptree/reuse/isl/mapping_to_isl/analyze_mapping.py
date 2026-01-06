"""
Flow of analysis:
-   From mapping, create the iteration space. The iteration space is the
    space of iterators in the mapping.
-   Create the relation from iteration space to operation space.
-   Create the relation from the iteration space to tensor space for each
    (buffer, tensor, einsum) tuple.
-   Run tile shape inference.

Adapted from:
https://github.com/NVlabs/timeloop/blob/4cf6d4cd043bc2a5d2eb02afa9063d7117a4dc11/ \
    src/loop-analysis/mapping-to-isl/fused-mapping-to-isl.cpp
Relevant Name Changes:
-   DataspaceId -> TensorName
-   LogicalBuffer -> Buffet
-   LogicalComputeUnit -> ComputeEinsum
-   Loop.op_dim -> Loop.rank_variable
-   *MappingNode.child -> MappingNode.flatten()[0]
-   Root -> Mapping
-   Compute.kernel -> Compute.einsum
-   Branch -> Split
-   FusedMapping -> Mapping
-   [node]_id -> node (conditional on MappingNode having a valid hashing functor)
"""

from collections import defaultdict, deque
from pprint import pformat
from typing import List, Optional

import islpy as isl

from fastfusion.frontend.mapping import (
    Compute,
    Mapping,
    MappingNode,
    MappingNodeWithChildren,
    Sequential,
    Storage,
)
from fastfusion.frontend.workload import Workload
from fastfusion.frontend._workload_isl._isl import get_projection_map
from fastfusion.frontend.workload import TensorName

from fastfusion.model._looptree.mapping_utilities import get_paths
from fastfusion.model._looptree.types import Buffet
from fastfusion.model._looptree.reuse.isl.isl_functions import project_dim_in_after
from fastfusion.model._looptree.reuse.isl.mapping_to_isl.skews_from_mapping import (
    skews_from_mapping,
)

from . import DUMP_ISL_IR
from .tiling import tiling_from_mapping
from .types import (
    BranchTiling,
    BufferTensorEinsum,
    ComputeEinsum,
    MappingAnalysisResult,
    Occupancy,
    OperationOccupancy,
    SkewsInfo,
)


def buffet_direct_above_sequential(mapping: Mapping) -> defaultdict[Buffet, bool]:
    """
    TODO: Verify this docstring
    For all Buffets (logical objects containing a tensor, its operating einsum,
    and a abstract hardware level), denote whether the buffet is directly above
    a :class:`~.Sequential`, or has an uninterrupted path of other buffets to a
    `~.Sequential`.

    Parameters
    ----------
    mapping:
        The mapping context of the buffets to sequential elements.

    Returns
    -------
    A dictionary of buffets and whether they're directly above a Sequential.
    """
    result: defaultdict[Buffet, bool] = defaultdict(lambda: False)
    # TODO: Figure out if get_paths is just for certain MappingNodesWithChildren
    # or not.
    for path in get_paths(mapping):
        leaf: Compute = path[-1]
        last_bufs: List[Buffet] = []

        node: MappingNode
        for node in path:
            match node:
                # If we have a storage, create a buffet for the current leaf.
                case Storage():
                    # TODO: Verify this port:
                    # https://github.com/NVlabs/timeloop/blob/32370826fdf1aa3c8deb0c93e6b2a2fc7cf053aa/src/loop-analysis/mapping-to-isl/fused-mapping-to-isl.cpp#L518-L520
                    # Note: Buffet seems to have changed a lot?
                    # https://github.com/NVlabs/timeloop/blob/master/include/loop-analysis/isl-ir.hpp#L96
                    last_bufs.extend(
                        Buffet(tensor=tensor, einsum=leaf.einsum, level=node.component)
                        for tensor in node.tensors
                    )
                # TODO: Check that all buffets are unique, because right now
                # it seems it's dependent on the last leaf in traversal?

                # If we encounter a sequential, we know all the last buffet and its
                # parents that are buffets are directly above sequential.
                case Sequential():
                    for buf in last_bufs:
                        result[buf] = result[buf] or True
                    last_bufs.clear()
                # If we encounter no storages or a sequential, we must not be
                # directly above a sequential element, and thus can purge the path.
                case _:
                    for buf in last_bufs:
                        result[buf] = result[buf] or False
                    last_bufs.clear()

    return result


def get_parallelism(mapping: Mapping) -> defaultdict[MappingNode, float]:
    """
    Given a `fastfusion.frontend.mapping.Mapping`, get the parallelism values for
    the Compute leafs.

    Parameters
    ----------
    mapping:
        The mapping to get parallelism for.

    Returns
    -------
    A map relating Compute nodes with their parallelism.
    """
    result: defaultdict[MappingNode, float] = defaultdict()

    # Initiates DFS at the root of the mapping.
    dfs_stack: deque[MappingNode] = deque([mapping])

    while dfs_stack:
        node: MappingNode = dfs_stack.pop()

        match node:
            # Recursively traverse children to find computes for parallelism.
            case MappingNodeWithChildren():
                dfs_stack.extend(node.nodes)
            # If Compute has pre-specified parallelism from internal models, trust
            # that it is right. Otherwise, assume none.
            case Compute():
                if hasattr(node, "parallelism"):
                    result[node] = node.parallelism
                else:
                    result[node] = 1
            case _:
                continue

    return result


def align_dim_names(
    map_: isl.Map,
    reference: isl.Map,
    map_align_dim_type: isl.dim_type = isl.dim_type.in_,
    reference_dim_type: Optional[isl.dim_type] = None,
) -> isl.Map:
    """
    Given an `isl.Map` and a reference `isl.Map`, align as many of the names as
    possible in the first map with the reference map.

    e.g. `map_ = [i] -> [o]` with `reference = [x] -> [y]` becomes `[x] -> [o]`
        with map_

    Parameters
    ----------
    map_:
        The map whose input is being aligned.
    reference:
        The map whose input names are used as reference for aligning `map`.
    map_align_dim_type:
        Dimension tuple in `map_` to align. Defaults to `isl.dim_type.in_`.
    reference_dim_type:
        Dimension tuple in `reference` whose names should be copied. Defaults to
        `map_align_dim_type`.

    Returns
    -------
    A version of `map_` with aligned input names.
    """
    if reference_dim_type is None:
        reference_dim_type = map_align_dim_type

    for dim_idx in range(
        min(map_.dim(map_align_dim_type), reference.dim(reference_dim_type))
    ):
        dim_name: Optional[str] = reference.get_dim_name(reference_dim_type, dim_idx)
        if dim_name is not None:
            map_ = map_.set_dim_name(map_align_dim_type, dim_idx, dim_name)

    return map_


def occupancies_from_mapping(
    mapping: Mapping, workload: Workload
) -> MappingAnalysisResult:
    """
    Given a Mapping and a Workload, extract the data occupancies in memory.

    Parameters
    ----------
    mapping:
        The Mapping of data to hardware.
    workload:
        The Workload occurring on chip.

    Returns
    -------
    The occupancies as an analysis of the Workload on Mapping.
    """
    branch_tiling: BranchTiling = tiling_from_mapping(mapping, workload)
    # tiling: [tile_iteration_space] -> [iteration_space]
    if DUMP_ISL_IR:
        for node, tiling in branch_tiling.items():
            print(f"[Tiling]Node({node}): {tiling}")
            # TODO: Port this line
            # https://github.com/NVlabs/timeloop/blob/32370826fdf1aa3c8deb0c93e6b2a2fc7cf053aa/src/loop-analysis/mapping-to-isl/fused-mapping-to-isl.cpp#L55-L64
            print(f"[Ops]Node({node}): ")

    occupancies: defaultdict[BufferTensorEinsum, Occupancy] = defaultdict()
    skews: SkewsInfo = skews_from_mapping(mapping, workload)
    # skew [Spacetime] -> [tile_iteration_space]
    if DUMP_ISL_IR:
        print(f"skews: {pformat(skews)}")

    ### Somewhere, call the domain space of the returned tilings {einsum}_iterations
    for bte, skew in skews.bte_to_skew.items():
        if DUMP_ISL_IR:
            print(f"{bte} has skew: {skew}")
        tiling = branch_tiling[bte.einsum]

        accesses: Optional[isl.Map] = None
        read_tensors: set[TensorName] = workload.einsums[
            bte.einsum.einsum
        ].input_tensor_names
        write_tensors: set[TensorName] = workload.einsums[
            bte.einsum.einsum
        ].output_tensor_names

        if bte.tensor in read_tensors or bte.tensor in write_tensors:
            accesses = get_projection_map(
                workload.einsums[bte.einsum.einsum], bte.tensor
            )
        else:
            continue

        aligned_skew: isl.Map = align_dim_names(skew.map_, tiling)
        if DUMP_ISL_IR:
            print(f"Skew: {skew.map_}")
            print(f"Aligned Skew: {aligned_skew}")
            print(f"Tiling: {tiling}")
            print(f"{tiling.apply_range(accesses)}")
            print(f"{skew.map_.dim(isl.dim_type.out)}")
        occupancy: isl.Map = aligned_skew.apply_range(
            project_dim_in_after(
                tiling.apply_range(accesses),
                skew.map_.dim(isl.dim_type.out),
                # TODO: fix this unsafe mixing.
            ).set_tuple_name(
                isl.dim_type.in_, aligned_skew.get_tuple_name(isl.dim_type.out)
            )
        )

        occupancies[bte] = Occupancy(skew.tags, occupancy)

    operations_occupancies: defaultdict[ComputeEinsum, OperationOccupancy] = (
        defaultdict()
    )
    for ce, skew in skews.ce_unit_to_skew.items():
        tiling: isl.Map = branch_tiling[ce.branch_leaf_node]
        if DUMP_ISL_IR:
            print(f"skew.map_ {skew.map_}")
        operation_occupancy: isl.Map = skew.map_.apply_range(
            project_dim_in_after(
                tiling, skew.map_.dim(isl.dim_type.out)
            ).set_tuple_name(
                # TODO: Unify the names at some point...
                isl.dim_type.in_,
                skew.map_.get_tuple_name(isl.dim_type.out),
            )
        )
        operations_occupancies[ce] = OperationOccupancy(skew.tags, operation_occupancy)

    return MappingAnalysisResult(
        buffet_to_occupancy=occupancies,
        compute_einsum_to_occupancy=operations_occupancies,
        buffet_direct_above_sequential=buffet_direct_above_sequential(mapping),
        compute_to_assumed_parallelism=get_parallelism(mapping),
        branch_tiling=branch_tiling,
    )
