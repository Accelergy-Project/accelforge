"""
Relevant name changes:
- [logical] buffer/lbuf -> buffet
- [logical] comp/lcomp -> compute_einsum
-
"""

from abc import ABC

from collections import defaultdict
from dataclasses import dataclass
from typing import List, TypeAlias

import islpy as isl

from fastfusion.frontend.mapping import Compute, MappingNode
from fastfusion.frontend.workload import TensorName
from fastfusion.model._looptree.types import Buffet


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
"Einsum's identifier."
Tiling: TypeAlias = isl.Map
"Tiling of data and operations."
BranchTiling: TypeAlias = defaultdict[MappingNode, Tiling]
"Relation between a node and its tiling."
BuffetTiling: TypeAlias = defaultdict[Buffet, Tiling]
"Relation between a buffet and its tiling."


@dataclass(frozen=True, slots=True)
class Tag(ABC):  # pylint: disable=too-few-public-methods
    """Associating an element with its type metadata without introspection?"""


class TemporalTag(Tag):  # pylint: disable=too-few-public-methods
    """The associated element is temporally spreading?"""


@dataclass(frozen=True, slots=True)
class SpatialTag(Tag):  # pylint: disable=too-few-public-methods
    """The associated element is spatially spreading?"""

    spatial_dim: int
    "The spatial dim in a given buffer?"
    buffer: MappingNode
    "The buffer the spatial dim is across?"


class PipelineTag(Tag):  # pylint: disable=too-few-public-methods
    """The associated element is pipelined?"""


class SequentialTag(Tag):  # pylint: disable=too-few-public-methods
    """The associated element is serialized?"""


TEMPORAL_TAGS = (TemporalTag, SequentialTag)
BRANCH_TAGS = (PipelineTag, SequentialTag)
LOOP_TAGS = (TemporalTag, SpatialTag)


@dataclass(frozen=True, slots=True)
class TaggedMap:  # pylint: disable=too-few-public-methods
    """A :class:`isl.Map` with its dimensions tagged."""

    tags: List[Tag]
    map_: isl.Map

    def __repr__(self):
        return f"{type(self)}({self.tags}, {self.map_})"


class Occupancy(TaggedMap):  # pylint: disable=too-few-public-methods
    """Location of data in [logical?] hardware elements."""

    def __init__(self, tags: list[Tag], map_: isl.Map):
        assert len(tags) == map_.dim(isl.dim_type.in_), (
            "Occupancy labels input dims with tags\n"
            "-------------------------------------\n"
            f"tags: {tags}\n"
            f"map: {map_}\n"
        )
        super().__init__(tags, map_)


class OperationOccupancy(TaggedMap):  # pylint: disable=too-few-public-methods
    """Location of operations in [logical?] hardware elements."""


class Fill(TaggedMap):
    """Spacetime -> fill of a logical buffer"""

    def __init__(self, tags: list[Tag], map_: isl.Map):
        assert len(tags) == map_.dim(isl.dim_type.in_), (
            "Fill labels input dims with tags\n"
            "--------------------------------\n"
            f"tags: {tags}\n"
            f"map: {map_}\n"
        )
        super().__init__(tags, map_)


class Skew(TaggedMap):  # pylint: disable=too-few-public-methods
    """TODO: Figure out what this is."""


@dataclass(frozen=True, slots=True)
class BufferTensorEinsum:
    """
    A buffet relating a [logical?] hardware element storing data, a tensor it
    contains, and the [logical?] hardware element that is requesting the tensor.

    See Also:
    ---------
    :class:`fastfusion._model.looptree.reuse.Buffet`
    """

    buffer: str
    "The logical name of the buffer supplying the tensor."
    tensor: TensorName
    "The tensor being supplied."
    einsum: Compute
    "The leaf in mapping doing the einsum compute on tensor."


@dataclass(frozen=True, slots=True)
class ComputeEinsum:
    """A logical computation the workload? needs to carry out."""

    compute: str
    """TODO: Figure out what this does."""
    branch_leaf_node: Compute
    """TODO: The compute element at the leaf of a :class:`BranchTiling`"""


# Output classes.
@dataclass(frozen=True, slots=True)
class SkewsInfo:  # pylint: disable=too-few-public-methods
    """TODO: Figure out what this does."""

    bte_to_skew: defaultdict[BufferTensorEinsum, Skew]
    """Relates a :class:`~.BufferTensorEinsum` to a :class:`~.Skew`"""
    ce_unit_to_skew: defaultdict[ComputeEinsum, Skew]
    """Relates a :class:`~.ComputeEinsum` to a :class:`~.Skew`"""


@dataclass(frozen=True, slots=True)
class MappingAnalysisResult:  # pylint: disable=too-few-public-methods
    """
    Results of mapping analysis that will become input into reuse
    analysis.
    """

    buffet_direct_above_sequential: defaultdict[Buffet, bool]
    """
    Whether a buffet is right above a sequential node. This is used when calculating
    capacity since some data can be dropped earlier than usual when using sequential
    mapping without tiling.
    """
    buffet_to_occupancy: defaultdict[BufferTensorEinsum, Occupancy]
    """The occupancy of every buffet as defined in the mapping."""
    compute_einsum_to_occupancy: defaultdict[ComputeEinsum, OperationOccupancy]
    """The occupancy of every compute unit."""
    # TODO: Figure out if this is deprecated:
    # https://github.com/NVlabs/timeloop/blob/32370826fdf1aa3c8deb0c93e6b2a2fc7cf053aa/include/loop-analysis/mapping-to-isl/fused-mapping-to-isl.hpp#L31-L35
    # node_to_buffets
    # Buffets found between the current root/branch node and the next one.
    branch_tiling: BranchTiling
    """
    Tiling of each branch. The tiling is a relation between tiling variables and
    operations. An uncompletely tiled branch will have multiple-valued :class:`isl.Map`.
    """
    compute_to_assumed_parallelism: defaultdict[MappingNode, float]
    """
    We can assume an amount of parallelism to quickly calculate approx. compute
    latency by simply dividing number of operations with assumed parallelism.
    """
