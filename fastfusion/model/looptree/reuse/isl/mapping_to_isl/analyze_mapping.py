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
-   Loop -> Iteration
-   Loop.op_dim -> Iteration.rank_variable
"""

import os

import logging

from collections import defaultdict, deque
from typing import Tuple

import islpy as isl

from fastfusion.frontend.mapping import (
    # Types
    MappingNode,
    # Mapping objects
    Mapping,
    MappingNodeWithChildren,
    # Physical object types in Mappings.
    Compute,
    Storage,
    # Logical object types in Mappings.
    Iteration
)
from fastfusion.frontend.workload.workload import (
    # Workload class for all of FastFusion.
    Workload,
    # Helpful TypeAlias
    RankVariableName,
)
from fastfusion.frontend.workload.isl import (
    get_einsum_operation_space
)
from fastfusion.frontend.mapping import TensorName
from fastfusion.mapper.FFM.joining.mappinginfo import Loop
from fastfusion.model.looptree.reuse.isl.mapping_to_isl.types import (
    EinsumName,
    Tiling,
    BranchTilings,
    MappingAnalysisResult,
)
from fastfusion.model.looptree.workload import Einsum

DUMP_ISL_IR: bool = os.getenv("FASTFUSION_DUMP_ISL_IR") == "1"
LOG_ISL_IR: bool = os.getenv("FASTFUSION_LOG_ISL_IR") == "1"


def get_mapping_group_einsums(
    mapping: Mapping,
) -> defaultdict[MappingNode, set[EinsumName]]:
    """
    From a mapping, get the group of einsums for a given node.

    :param mapping: The mapping we are getting the grouped einsums for.
    :type mapping:  Mapping

    :return:    A dictionary relating a Node to a set of einsums.
    :rtype:     defaultdict[MappingNode, Set[EinsumName]]
    """
    # Each pair is a (current_node, last_non_branch_node)
    dfs_stack: deque[Tuple[MappingNode, MappingNode]] = deque()
    # Each pair is a (last_non_branch_node, set_of_children_nodes)
    child_stack: deque[Tuple[MappingNode, set[MappingNode]]] = deque()
    result: defaultdict[MappingNode, set[EinsumName]] = defaultdict(set)

    # Start DFS hierarchical search from the root.
    root = mapping.loops[0]
    dfs_stack.append((root, root))

    # Exhaustive DFS search.
    while dfs_stack:
        # Grabs latest node to search.
        node, last_non_branch = dfs_stack.pop()

        # Differentiates behavior by number of child nodes.
        match node:
            case MappingNodeWithChildren():
                match len(node.nodes):
                    # No children, log as a folded result.
                    case 0:
                        # Note:: Check necesary in case Distrobuffers elides
                        # computes into one large unit.
                        if isinstance(node, Compute):
                            result[last_non_branch].add(node.einsum)
                        else:
                            raise TypeError(
                                f"The following node should be of class "
                                f"Compute as it has no children:\n---\n{node}"
                            )
                    # Explore the children further.
                    case 1:
                        dfs_stack.append((node.nodes[0], last_non_branch))
                    # Log all branching children and explore all children.
                    case _:
                        children: set[MappingNode] = set(node.nodes)
                        child_stack.append((last_non_branch, children))
                        dfs_stack.extend((child, child) for child in children)
            # Assumed no children, log as a folded result.
            case Compute():
                result[last_non_branch].add(node.einsum)
            case _:
                raise AttributeError(
                    f"The following node of class {type(node)} has "
                    f"indeterminant number of children:\n---\n"
                    f"{node}"
                )

    # Push up einsums to parents.
    for node, children in reversed(child_stack):
        node_einsum_set: set[EinsumName] = result[node]
        node_einsum_set.update({result[child] for child in children}) # type: ignore

    return result


def get_head_among_einsums(
    einsum_set: set[EinsumName], workload: Workload
) -> set[EinsumName]:
    """
    Gets the provider einsums that only consume data (i.e., sink einsums).

    :param einsum_set:  Set of einsums to consider.
    :param workload:    The workload context the einsums exist in.

    :type einsum_set:   Set[EinsumName]
    :type workload:     Workload

    :return:    The set of all head einsums.
    :rtype:     Set[EinsumName]
    """
    # Returns set of einsums that are not data producers.
    return {
        einsum
        for einsum in einsum_set
        if all(
            not any(
                consumer in einsum_set
                for consumer in workload.einsums_that_read_tensor(output_tensor)
            )
            for output_tensor in workload.tensors_written_by_einsum(einsum)
        )
    }


def add_new_tile_dim(
    old_tiling: Tiling, dim_idx: int, tile_size: int
) -> Tiling:
    """
    Given a tiling, add a new dimension to the tiling.

    :param old_tiling:  The previous tiling the mapper proposed.
    :param dim_idx:     The index of the dimension being tiled.
    :param tile_size:   The size of the tiling on dim_idx.

    :type old_tiling:   Tiling
    :type dim_idx:      int
    :type tile_size:    int

    :return:    The new Tiling with tiled dimension at dim_idx.
    :rtype:     Tiling
    """

    # new_tiling has one extra dimension at the end compared to old_tiling.
    new_tiling = old_tiling.insert_dims(
        isl.dim_type.in_, old_tiling.dim(isl.dim_type.in_), 1
    )

    # Min and max of dim_idx. dimension being tiled as function of tiled dimensions.
    dim_min: isl.PwAff = new_tiling.dim_min(dim_idx)
    dim_max: isl.PwAff = new_tiling.dim_max(dim_idx)

    # Aff from tiled dimensions space to value of newest dim.
    new_dim_id: isl.Aff = isl.Aff.var_on_domain(
        dim_min.get_domain_space().to_local_space(),
        isl.dim_type.set,
        dim_min.dim(isl.dim_type.in_)-1
    )

    # Aff from tiled dimensions space to tile tile size constant.
    tile_size_aff: isl.Aff = isl.Aff.val_on_domain_space(
        dim_min.get_domain_space(), 
        isl.Val.int_from_ui(isl.DEFAULT_CONTEXT, tile_size)
    )

    # PwAff from tiled dimension space to tile_size * newest_dim.
    tile_translate: isl.PwAff = isl.PwAff.from_aff(new_dim_id.mul(tile_size_aff))

    # What dim_min should be given new tiling.
    new_dim_min: isl.PwAff = dim_min.add(tile_translate)

    # What dim_max should be given new tiling.
    new_dim_max: isl.PwAff = new_dim_min.add(
        isl.PwAff.from_aff(tile_size_aff.add_constant_val(-1))
    )

    # TODO: Might be logically equivalent to new_dim_id:
    # https://github.com/NVlabs/timeloop/blob/32370826fdf1aa3c8deb0c93e6b2a2fc7cf053aa/src/loop-analysis/mapping-to-isl/tiling.cpp#L52-L59
    new_iter_id: isl.PwAff = isl.PwAff.from_aff(
        isl.Aff.var_on_domain(
            new_tiling.get_space().domain(),
            isl.dim_type.set,
            old_tiling.dim(isl.dim_type.in_)
        )
    )

    # The set of valid values of the new tiled dimensions.
    iter_set: isl.Set = new_tiling.domain()
    iter_set = iter_set.intersect(
        new_iter_id.le_set(dim_max.div(tile_size_aff).ceil())
    )
    iter_set = iter_set.intersect(
        new_dim_min.ge_set(dim_min)
    )

    # The value of iter dims cannot exceed what was available before tiling.
    new_tiling = new_tiling.intersect_domain(iter_set)

    # The set of operations need to to follow the new tile bounds.
    identity: isl.PwAff = isl.PwAff.from_aff(
        isl.Aff.var_on_domain(
            new_tiling.get_space().range(),
            isl.dim_type.set,
            dim_idx
        )
    )
    new_tiling = new_tiling.intersect(new_dim_min.le_map(identity))
    new_tiling = new_tiling.intersect(new_dim_max.ge_map(identity))

    return new_tiling


def tiling_from_mapping(mapping: Mapping, workload: Workload) -> BranchTilings:
    """
    Given a mapping and a workload generates a tiling.

    :param mapping: A mapping of data to hardware.
    :param workload:The problem being solved.

    :type mapping:  Mapping
    :type workload: Workload

    :return:    BranchTilings associating a node's ID with its tiling.
    :rtype:     BrancTilings
    """
    result: BranchTilings = BranchTilings()
    # Grabs the head einsums.
    mapping_groups: defaultdict[MappingNode, set[EinsumName]] = (
        get_mapping_group_einsums(mapping)
    )
    mapping_group_heads: defaultdict[MappingNode, set[EinsumName]] = defaultdict(
        set, {
            node: get_head_among_einsums(group, workload)
            for node, group in mapping_groups.items()
        }
    )

    tensor_to_reuse_level: defaultdict[TensorName, int] = defaultdict()
    dfs_stack: deque[MappingNode] = deque()

    # Maps last non-branch to tiling of each in the group.
    tiling_info: defaultdict[MappingNode, defaultdict[EinsumName, Tiling]] = (
        defaultdict(defaultdict)
    )

    # Initiates the DFS at the mapping root and appends its info.
    root: MappingNode = mapping.nodes[0]
    dfs_stack.append(root)
    for einsum_name in workload.einsum_names:
        tiling: Tiling = isl.Map.from_range(
           get_einsum_operation_space(workload, einsum_name)
        )
        if DUMP_ISL_IR: print(f"Tiling: {tiling}")
        tiling_info[root][einsum_name] = tiling
    
    while dfs_stack:
        node = dfs_stack.pop()
        heads = mapping_group_heads[node]

        current_node: MappingNode = node
        is_tiling: bool = True

        while is_tiling:
            # Fuses current_node to one of the heads.
            if isinstance(current_node, Iteration):
                if len(heads) != 1:
                    raise ValueError(f"Cannot fuse tiled set with {len(heads)} heads.")
                
                # Grabs rank_var to tile and the head to tile it from.
                rank_var = current_node.rank_variable
                head = next(iter(heads))

                old_tiling: Tiling = tiling_info[node][head]
                # set, not AbstractSet, iteration in python is the same. Downstreeams
                # of "heads" is also constaint.
                isl_rank_idx: int = tuple(
                    workload.einsums[head].rank_variables
                ).index(rank_var)

                # Adds a new tile_dim to the old tiling.
                if isinstance(current_node.tile_shape, int) and current_node.tile_shape != 0:
                    new_tiling: Tiling = add_new_tile_dim(
                        old_tiling, isl_rank_idx, current_node.tile_shape
                    )
                else:
                    raise NotImplementedError(
                        f"Tile size analysis not implemented for type {type(node)} "
                        f"with tile shape {current_node.tile_shape}"
                    )
                
                # Saves the fused tiling.
                tiling_info[node][head] = new_tiling

                iteration_set: isl.Set = new_tiling.domain()
                for einsum in mapping_groups[node]:
                    if einsum == head:
                        continue

                    tiling = tiling_info[node][einsum]
                    tiling = tiling.insert_dims(
                        isl.dim_type.in_, tiling.dim(isl.dim_type.in_), 1
                    )
                    tiling = tiling.intersect_domain(iteration_set)
                
                current_node = current_node.flatten()[0]
            elif isinstance(current_node, Storage):
                # Check if current_node is the highest level of Storage to determine reuse level.
                if current_node.tensor not in tensor_to_reuse_level:
                    random_einsum: EinsumName = next(iter(mapping_groups[node]))
                    tiling: Tiling = tiling_info[node][random_einsum]
                    tensor_to_reuse_level[current_node.tensor] = tiling.dim(isl.dim_type.in_)
                
                current_node = current_node.flatten()[0]


def occupancies_from_mapping(
    mapping: Mapping, workload: Workload
) -> MappingAnalysisResult:
    """
    Given a Mapping and a Workload, extract the data occupancies in memory.

    :param mapping: The Mapping of data to hardware.
    :param workload:The Workload occurring on chip.

    :type mapping:  Mapping
    :type workload: Workload

    :return:    The occupancies as an analysis of the Workload on Mapping.
    :rtype:     MappingAnalysisResult
    """
    result = MappingAnalysisResult()
    result.compute_to_assumed_parallelism = tiling_from_mapping(mapping, workload)
