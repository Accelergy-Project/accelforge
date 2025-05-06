from abc import ABC
from dataclasses import dataclass
from typing import TypeAlias


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


@dataclass
class ComputeEinsum:
    compute: str
    einsum: str