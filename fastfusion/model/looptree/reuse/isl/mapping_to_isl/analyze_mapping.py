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

from fastfusion.frontend.mapper import Mapping
from fastfusion.frontend.mapping import Nested, NodeList
from fastfusion.frontend.workload.workload import Workload, RankVariableName, TensorName

import fastfusion.model.looptree.reuse.isl.isl_functions as isl_help
from fastfusion.model.looptree.reuse.isl.mapping_to_isl.types import *

DUMP_ISL_IR: bool = os.getenv("FASTFUSION_DUMP_ISL_IR") == '1'
LOG_ISL_IR: bool = os.getenv("FASTFUSION_LOG_ISL_IR") == '1'


def get_mapping_group_einsums(
    mapping: Mapping
) -> defaultdict[NodeID, Set[EinsumID]]:
    """
    From a mapping, get the group of einsums for a given node.
    
    :param mapping: The mapping we are getting the grouped einsums for.

    :return: A dictionary relating a NodeID to a set of EinsumIDs.
    """
    # Each pair is a (current_node_id, last_non_branch_node_id)
    dfs_stack: deque[Tuple[NodeID, NodeID]] = deque()
    # Each pair is a (last_non_branch_node_id, set_of_children_ids)
    child_stack: deque[Tuple[NodeID, Set[NodeID]]] = deque()
    result: defaultdict[NodeID, Set[EinsumID]] = {}

    root = mapping.loops[0]
    dfs_stack.append((root, root))

    while dfs_stack:
        node_id, last_non_branch = dfs_stack.pop()

        node 




def tiling_from_mapping(
    mapping: Mapping, workload: Workload
) -> BranchTilings:
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