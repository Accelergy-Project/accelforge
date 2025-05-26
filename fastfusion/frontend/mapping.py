from math import prod

from pydantic import ConfigDict
from fastfusion.frontend import architecture
from fastfusion.frontend.workload.workload import RankVariable, Tensor
from typing import Callable, Iterator, List, Optional, Type, TypeVar, Union, Annotated
from abc import ABC
from fastfusion.util.basetypes import ParsableModel, ParsableList, ParsesTo
from fastfusion.version import assert_version, __version__



T = TypeVar("T")


class MappingNode(ParsableModel, ABC):
    _constraint_lambdas: List[Callable[[], bool]] = []
    _must_be_here: bool = False  # Can the mapper move this node?
    _required: bool = False  # Must the mapper keep this node?

class Iteration(ParsableModel):
    rank_variable: Union[set[RankVariable], RankVariable]
    loop_bound: ParsesTo[Union[int, None]] = None
    tile_shape: ParsesTo[Union[int, None]] = None
    stride: ParsesTo[Union[int, None]] = None

class Temporal(Iteration):
    def compact_string(self) -> str:
        return f"{self.rank_variable}-{self.loop_bound}"


class Spatial(Iteration):
    dimension: str
    across: Optional[architecture.Leaf] = None

    def compact_string(self) -> str:
        return f"S{self.dimension}-{self.rank_variable}-{self.loop_bound}"


class Storage(MappingNode):
    tensor: Tensor
    memory: architecture.Memory
    _must_exist: bool = False  # Must the mapper keep this node?
    _backing: bool = False  # Is this node a backing storage?

    def compact_string(self) -> str:
        return f"[{self.memory.name} {self.tensor.name}]"

    @property
    def tensor_name(self) -> str:
        return self.tensor.name

    @property
    def memory_name(self) -> str:
        return self.memory.name


class Split(MappingNode):
    children: ParsableList[MappingNode]


class Compute(MappingNode):
    einsum: str
    compute: str

    def compact_string(self) -> str:
        return f"C{self.einsum}"


class Mapping(ParsableModel):
    version: Annotated[str, assert_version] = __version__
    nodes: ParsableList[MappingNode] = ParsableList()
