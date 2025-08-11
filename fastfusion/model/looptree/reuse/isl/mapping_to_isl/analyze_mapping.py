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
from collections.abc import Set
from typing import Tuple

import islpy as isl

from fastfusion.frontend.mapping import (
    # Types
    AnnotatedMappingNode,
    # Mapping objects
    Mapping,
    MappingNodeWithChildren,
    # Physical object types in Mappings.
    Compute,
    Iteration
)
from fastfusion.frontend.workload.workload import Workload
from fastfusion.frontend.workload.isl import
(
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

DUMP_ISL_IR: bool = os.getenv("FASTFUSION_DUMP_ISL_IR") == "1"
LOG_ISL_IR: bool = os.getenv("FASTFUSION_LOG_ISL_IR") == "1"


def get_mapping_group_einsums(
    mapping: Mapping,
) -> defaultdict[AnnotatedMappingNode, Set[EinsumName]]:
    """
    From a mapping, get the group of einsums for a given node.

    :param mapping: The mapping we are getting the grouped einsums for.
    :type mapping:  Mapping

    :return:    A dictionary relating a Node to a set of einsums.
    :rtype:     defaultdict[AnnotatedMappingNode, Set[EinsumName]]
    """
    # Each pair is a (current_node, last_non_branch_node)
    dfs_stack: deque[Tuple[AnnotatedMappingNode, AnnotatedMappingNode]] = deque()
    # Each pair is a (last_non_branch_node, set_of_children_nodes)
    child_stack: deque[Tuple[AnnotatedMappingNode, Set[AnnotatedMappingNode]]] = deque()
    result: defaultdict[AnnotatedMappingNode, Set[EinsumName]] = defaultdict(set)

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
                        children: Set[AnnotatedMappingNode] = set(node.nodes)
                        child_stack.extend((last_non_branch, children))
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
        result[node].update(result[child] for child in children)

    return result


def get_head_among_einsums(
    einsum_set: Set[EinsumName], workload: Workload
) -> Set[EinsumName]:
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
    mapping_groups: defaultdict[AnnotatedMappingNode, Set[EinsumName]] = (
        get_mapping_group_einsums(mapping)
    )
    mapping_group_heads: defaultdict[AnnotatedMappingNode, Set[EinsumName]] = {
        node: get_head_among_einsums(group, workload)
        for node, group in mapping_groups.items()
    }

    tensor_to_reuse_level: defaultdict[TensorName, int]
    dfs_stack: deque[AnnotatedMappingNode] = deque()

    # Maps last non-branch to tiling of each in the group.
    tiling_info: defaultdict[AnnotatedMappingNode, defaultdict[EinsumName, isl.Map]] = (
        {}
    )

    # Initiates the DFS at the mapping root and appends its info.
    root: AnnotatedMappingNode = mapping.nodes[0]
    dfs_stack.append(root)
    for einsum_name in workload.einsum_names:
        tiling: Tiling = isl.Map.from_range(
           get_einsum_operation_space(einsum_name)
        )
        if DUMP_ISL_IR: print(f"Tiling: {tiling}")
        tiling_info[root][einsum_name] = tiling
    
    while dfs_stack:
        node = dfs_stack.pop()
        heads = mapping_group_heads[node]

        current_node: AnnotatedMappingNode = node
        is_tiling: bool = True

        while is_tiling:
            if isinstance(current_node, Iteration):
                if heads.size != 1:
                    raise ValueError(f"Cannot fuse tiled set with {len(heads)} heads.")
                
                rank_var = current_node.rank_variable
                head = next(iter(heads))

                old_tiling: Tiling = tiling_info[current_node][head]
                isl_rank_idx: int = tuple(
                    workload.einsums[einsum_name].rank_variables
                ).index(rank_var)

                


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
