"""
Relevant name changes:
- [logical] buffer/lbuf -> buffet
- [logical] comp/lcomp -> compute_einsum
- 
"""

from abc import ABC

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeAlias

import islpy as isl

from fastfusion.frontend.mapping import MappingNode
from fastfusion.model.looptree.reuse import Buffet

class TaggedMap:
    def __init__(self, tags, map):
        self.tags = tags
        self.map = map


class Tag(ABC):
    pass


class TemporalTag(Tag):
    def __init__(self):
        pass


class SpatialTag(Tag):
    def __init__(self, spatial_dim, buffer):
        self.spatial_dim = spatial_dim
        self.buffer = buffer


class PipelineTag(Tag):
    def __init__(self):
        pass


class SequentialTag(Tag):
    def __init__(self):
        pass


TEMPORAL_TAGS = [TemporalTag, SequentialTag]
BRANCH_TAGS = [PipelineTag, SequentialTag]
LOOP_TAGS = [TemporalTag, SpatialTag]


class Occupancy(TaggedMap):
    def __init__(self, tags, map):
        super().__init__(tags, map)

    def __repr__(self):
        return f'Occupancy({self.tags}, {self.map})'


class Skew(TaggedMap):
    def __init__(self, tags, map):
        super().__init__(tags, map)

    def __repr__(self):
        return f'Skew({self.tags}, {self.map})'

@dataclass
class BufferTensorEinsum:
    buffer: str
    tensor: str
    einsum: str


@dataclass(frozen=True)
class ComputeEinsum:
    compute: str
    einsum: str


# Mapper intermediates.
##
#   @brief Iteration -> Operation relation that specifies the tiling.
#
#   The tiling relation allows us to distribute data and operations using the
#   skew and data distribution relations.
#
#   The tiling relation may have unspecified bounds which will be inferred by
#   LoopTree. The tiling relation that goes to the nest analysis is guaranteed
#   to be fully specified.
EinsumName: TypeAlias = str
Tiling: TypeAlias = isl.Map # Tiling of data and operations.
BranchTiling: TypeAlias = defaultdict[MappingNode, Tiling]  # Relation between a node and its tiling.
BuffetTiling: TypeAlias = defaultdict[Buffet, Tiling]   # Relation between a buffet and its tiling.


# Output classes.
@dataclass
class SkewsInfo:
    logical_buffer_to_skew: defaultdict[Buffet, Skew]
    logical_compute_unit_to_skew: defaultdict[ComputeEinsum, Skew]


@dataclass
class MappingAnalysisResult:
    """
    Results of mapping analysis that will become input into reuse
    analysis.

    :param buffet_direct_above_sequential: Whether a buffet is right above 
        a sequential node. This is used when calculating capacity since some data 
        can be dropped earlier than usual when using sequential mapping without tiling.
    :param buffet_to_occupancy: The occupancy of every buffet as defined in 
        the mapping.
    :param compute_einsum_to_occupancy: The occupancy of every compute unit.
    :param node_to_buffets: Buffets found between the current root/branch node and
        the next one.
    :param branch_tiling: Tiling of each branch. The tiling is a relation between tiling
        variables and operations. An uncompletely tiled branch will have multiple-valued
        isl.Map.
    :param compute_to_assumed_parallelism: We can assume an amount of parallelism
        to quickly calculate approx. compute latency by simply dividing number of
        operations with assumed parallelism.
    """
    buffet_direct_above_sequential: defaultdict[ComputeEinsum, Skew]
    buffet_to_occupancy: defaultdict[Buffet, Occupancy]
    compute_einsum_to_occupancy: defaultdict[ComputeEinsum, OperationOccupancy]
    node_to_buffets: defaultdict[MappingNode, Iterable[Buffet]]
    branch_tiling: BranchTiling
    compute_to_assumed_parallelism: defaultdict[MappingNode, float]

