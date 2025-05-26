from pydantic import BaseModel, model_validator
from fastfusion.frontend import architecture
from typing import Callable, List, Union, Annotated, Literal, TypeVar
from abc import ABC
from fastfusion.util.basetypes import ParsableModel, ParsableList, ParsesTo
from fastfusion.version import assert_version, __version__


# =============================================================================
# LoopTree Mapping Nodes
# =============================================================================

class MappingNode(ParsableModel, ABC):
    _constraint_lambdas: List[Callable[[], bool]] = []
    _must_be_here: bool = False  # Can the mapper move this node?
    _required: bool = False  # Must the mapper keep this node?


class Pattern(BaseModel):
    stride: ParsesTo[Literal['symbol'] | int]
    initial_tile_shape: ParsesTo[Literal['symbol'] | int | None] = None
    tile_shape: ParsesTo[Literal['symbol'] | int | None] = None


class Iteration(MappingNode):
    rank_variable: str
    loop_bound: ParsesTo[Union[Literal['symbol'], int, None]] = None
    tile_shape: ParsesTo[Union[Literal['symbol'], int, None]] = None
    tile_pattern: ParsesTo[Union[Pattern, None]] = None

    @model_validator(mode='after')
    def check_at_least_one_tiling_info(self):
        if self.loop_bound is None and self.tile_shape is None and self.tile_pattern is None:
            raise ValueError('Must give at least one of loop_bound, tile_shape, or tile_pattern')
        return self


class Temporal(Iteration):
    def compact_string(self) -> str:
        return f"{self.rank_variable}-{self.loop_bound}"


class Spatial(Iteration):
    dimension: int
    across: str

    def compact_string(self) -> str:
        return f"S{self.dimension}-{self.rank_variable}-{self.loop_bound}"


class Storage(MappingNode):
    tensor: list[str]
    memory: str
    _must_exist: bool = False  # Must the mapper keep this node?
    _backing: bool = False  # Is this node a backing storage?
    _memory: architecture.Memory = None # Reference to memory node for convenience

    def compact_string(self) -> str:
        return f"[{self.memory.name} {self.tensor.name}]"


class Split(MappingNode):
    children: ParsableList[MappingNode]


class Pipeline(Split):
    pass


class Sequential(Split):
    pass


class Compute(MappingNode):
    einsum: str
    compute: str

    def compact_string(self) -> str:
        return f"C{self.einsum}"


# =============================================================================
# Nodes That May Only be Inserted by the Model
# =============================================================================

class Reservation(MappingNode):
    tensor: str
    memory: str


class Fill(MappingNode):
    tensor: str
    memory: str


# =============================================================================
# Top-level Mapping
# =============================================================================

type MappingNodeTypes = Union[
    Temporal,
    Spatial,
    Storage,
    Pipeline,
    Sequential,
    Compute,
    Reservation,
    Fill
]


class Mapping(ParsableModel):
    version: Annotated[str, assert_version] = __version__
    nodes: ParsableList[MappingNodeTypes] = ParsableList()

