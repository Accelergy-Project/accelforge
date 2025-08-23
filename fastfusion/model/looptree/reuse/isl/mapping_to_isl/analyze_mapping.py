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

from fastfusion.frontend.mapping import Mapping
from fastfusion.frontend.workload import Workload

from .types import MappingAnalysisResult

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
    result.compute_to_assumed_parallelism = get_parallelism(mapping)
    
    branch_tiling: BranchTilings = tiling_from_mapping(mapping, workload)
