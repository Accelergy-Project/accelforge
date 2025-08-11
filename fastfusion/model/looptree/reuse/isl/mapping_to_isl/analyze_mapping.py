"""
Flow of analysis:
-   From mapping, create the iteration space. The iteration space is the
    space of iterators in the mapping.
-   Create the relation from iteration space to operation space.
-   Create the relation from the iteration space to tensor space for each
    (buffer, tensor, einsum) tuple.
-   Run tile shape inference.

Adapted from: https://github.com/NVlabs/timeloop/blob/4cf6d4cd043bc2a5d2eb02afa9063d7117a4dc11/src/loop-analysis/mapping-to-isl/mapping-to-isl.cpp#L261
-   DataspaceId -> TensorName
-   LogicalBuffer -> Buffet
-   LogicalComputeUnit -> ComputeEinsum
"""

import os

from collections import defaultdict, deque
from collections.abc import Set
from typing import Tuple

import islpy as isl

from fastfusion.frontend.mapping import (
    # Types
    AnnotatedMappingNode,
    # Mapping objects
    Mapping,
    MappingNode,
    MappingNodeWithChildren,
    Nested,
    # Physical object types in Mappings.
    Compute,
)
from fastfusion.frontend.workload.workload import Workload, RankVariableName, TensorName

import fastfusion.model.looptree.reuse.isl.isl_functions as isl_help
from fastfusion.model.looptree.reuse.isl.mapping_to_isl.types import *

DUMP_ISL_IR: bool = os.getenv("FASTFUSION_DUMP_ISL_IR") == "1"
LOG_ISL_IR: bool = os.getenv("FASTFUSION_LOG_ISL_IR") == "1"


def get_mapping_group_einsums(mapping: Mapping) -> defaultdict[NodeID, Set[EinsumID]]:
    """
    From a mapping, get the group of einsums for a given node.

    :param mapping: The mapping we are getting the grouped einsums for.

    :return: A dictionary relating a NodeID to a set of EinsumIDs.
    """
    # Each pair is a (current_node_id, last_non_branch_node_id)
    dfs_stack: deque[Tuple[AnnotatedMappingNode, AnnotatedMappingNode]] = deque()
    # Each pair is a (last_non_branch_node_id, set_of_children_ids)
    child_stack: deque[Tuple[AnnotatedMappingNode, Set[AnnotatedMappingNode]]] = deque()
    result: defaultdict[AnnotatedMappingNode, Set[Einsum]] = defaultdict(set)

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
                                f"The following node should be of class "\
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
                    f"The following node of class {type(node)} has "\
                    f"indeterminant number of children:\n---\n"\
                    f"{node}"
                )

    # Push up einsums to parents.
    for node, children in reversed(child_stack):
        einsum_set: Set[Einsum] = result[node]
        for child in children:
            einsum_set.update(result[child])

    return result


def tiling_from_mapping(mapping: Mapping, workload: Workload) -> BranchTilings:
    """
    Given a mapping and a workload generates a tiling.

    :param mapping: A mapping of data to hardware.
    :param workload: The problem being solved.

    :return:    BranchTilings associating a node's ID with its tiling.
    """
    result: BranchTilings = BranchTilings()


def occupancies_from_mapping(
    mapping: Mapping, workload: Workload
) -> MappingAnalysisResult:
    result = MappingAnalysisResult()
    result.compute_to_assumed_parallelism = tiling_from_mapping(mapping, workload)
