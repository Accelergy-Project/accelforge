"""
Flow of analysis:
- From mapping, create the iteration space. The iteration space is the
  space of iterators in the mapping.
- Create the relation from iteration space to operation space.
- Create the relation from the iteration space to tensor space for each
  (buffer, tensor, einsum) tuple. 
- Run tile shape inference.
"""
from dataclasses import dataclass, field

import islpy as isl

from .types import *


@dataclass
class MappingInISL:
    occupancies: dict[BufferTensorEinsum, Occupancy] = field(default_factory=dict)
    buffer_to_skews: dict[BufferTensorEinsum, Skew] = field(default_factory=dict)
    compute_to_skew: dict[ComputeEinsum, Skew] = field(default_factory=dict)


def AnalyzeMapping(mapping, workload) -> dict[BufferTensorEinsum, Occupancy]:
    result: dict[BufferTensorEinsum, Occupancy] = {}

