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
-   *MappingNode.child -> MappingNode.flatten()[0]?
-   Root -> Mapping?
-   Compute.kernel -> Compute.einsum
-   Branch -> Split?
"""

from collections import defaultdict

import islpy as isl

from fastfusion.frontend.mapping import Compute, Mapping
from fastfusion.frontend.workload import Workload
from fastfusion.model.looptree.reuse.isl.mapping_to_isl.skews_from_mapping import skews_from_mapping
from fastfusion.model.looptree.reuse.summarized.symbolic import Buffet

from . import DUMP_ISL_IR, LOG_ISL_IR
from .tiling import tiling_from_mapping
from .types import (
    BranchTiling,
    MappingAnalysisResult,
    Occupancy,
)


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
    branch_tiling: BranchTiling = tiling_from_mapping(mapping, workload)
    if DUMP_ISL_IR:
        for node, tiling in branch_tiling.items():
            print(f"[Tiling]Node({node}): {tiling}")
            # TODO: Port this line 
            # https://github.com/NVlabs/timeloop/blob/32370826fdf1aa3c8deb0c93e6b2a2fc7cf053aa/src/loop-analysis/mapping-to-isl/fused-mapping-to-isl.cpp#L55-L64
            print(f"[Ops]Node({node}): ")
    
    occupancies: defaultdict[Buffet, Occupancy] = defaultdict()
    # TODO: Implement skews_from_mapping
    skews: Skew = skews_from_mapping(mapping, workload)

    for buf, skew in skews.buffet_to_skew:
        if DUMP_ISL_IR:
            print(f"{buf} has skew: {skew}")

    # TODO: Implement both called functions.
    return MappingAnalysisResult(
        branch_tiling=branch_tiling.
        buffet_direct_above_sequential=buffet_direct_above_sequential(mapping),

        compute_to_assumed_parallelism=get_parallelism(mapping)
    ) 