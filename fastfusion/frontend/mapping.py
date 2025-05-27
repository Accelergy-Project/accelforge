import copy
from pydantic import BaseModel, model_validator
from fastfusion.frontend import architecture
from typing import Callable, List, Union, Annotated, Literal, TypeVar, TypeAlias
from abc import ABC
from fastfusion.util.basetypes import ParsableModel, ParsableList, ParsesTo
from fastfusion.version import assert_version, __version__

from fastfusion.frontend import architecture
from fastfusion.frontend.workload.workload import RankVariableName, TensorName
from typing import Callable, Iterator, List, Optional, Type, TypeVar, Union, Annotated
from abc import ABC
from fastfusion.util.basetypes import ParsableModel, ParsableList, ParsesTo
from fastfusion.version import assert_version, __version__

T = TypeVar("T")


# =============================================================================
# LoopTree Mapping Nodes
# =============================================================================

class MappingNode(ParsableModel, ABC):
    _constraint_lambdas: List[Callable[[], bool]] = []
    _must_be_here: bool = False  # Can the mapper move this node?
    _required: bool = False  # Must the mapper keep this node?
    # children: ParsableList["MappingNode"] = ParsableList()
    
    def mermaid_graph(self, _is_root: bool = True) -> str:
        import mermaid as mermaid_import
        from mermaid.graph import Graph
        lines = []
        if _is_root:
            lines = [
                "graph TD",
                "linkStyle default interpolate basis"
            ]
        else:
            lines.append(f"{self._mermaid_graph_node_name()}[\"{str(self)}\"]")
        
        for node in self.children:
            node_name = node._mermaid_graph_node_name()
            lines.append(f"{self._mermaid_graph_node_name()} --> {node_name}")
            lines.append(node.mermaid_graph(_is_root=False))
        return "\n".join(lines)

    def _mermaid_graph_node_name(self) -> str:
        return f"{self.__class__.__name__}_{id(self)}"

    def _mermaid_graph_node(self) -> str:
        return f"{self._mermaid_graph_node_name()}[\"{str(self)}\"]"

class Pattern(ParsableModel):
    stride: ParsesTo[Literal['symbol'] | int]
    initial_tile_shape: ParsesTo[Literal['symbol'] | int | None] = None
    tile_shape: ParsesTo[Literal['symbol'] | int | None] = None


class Iteration(MappingNode):
    rank_variable: Union[set[RankVariableName], RankVariableName]
    loop_bound: ParsesTo[Union[Literal['symbol'], int, None]] = None
    tile_shape: ParsesTo[Union[Literal['symbol'], int, None]] = None
    tile_pattern: ParsesTo[Union[Pattern, None]] = None
    assume_perfect_factor: bool = True

    # @model_validator(mode='after')
    # def check_at_least_one_tiling_info(self):
    #     n_non_none = sum([
    #         self.loop_bound is not None,
    #         self.tile_shape is not None,
    #         self.tile_pattern is not None
    #     ])
    #     if n_non_none != 1:
    #         raise ValueError('Must give exactly one of loop_bound, tile_shape, or tile_pattern')
    #     return self

class Temporal(Iteration):
    def compact_string(self) -> str:
        return f"{self.rank_variable}-{self.loop_bound}"


class Spatial(Iteration):
    dimension: Union[int, str]
    across: str
    across_object: Optional[architecture.Leaf] = None

    def compact_string(self) -> str:
        return f"S{self.dimension}-{self.rank_variable}-{self.loop_bound}"


class Storage(MappingNode):
    tensors: ParsableList[TensorName]
    memory: str
    memory_object: Optional[architecture.Memory] = None # Reference to memory node for convenience
    _must_exist: bool = False  # Must the mapper keep this node?
    _backing: bool = False  # Is this node a backing storage?

    def compact_string(self) -> str:
        tname = ",".join(self.tensors)
        return f"[{self.memory} {tname}]"
    
    def __str__(self) -> str:
        tname = ", ".join(self.tensors)
        return f"{tname} in {self.memory}"
    
    @property
    def tensor(self) -> TensorName:
        if len(self.tensors) != 1:
            raise ValueError(
                f"Storage node {repr(self)} has {len(self.tensors)} tensors. "
                f"Access the tensors property instead."
            )
        return self.tensors[0]
    
    @property
    def _mermaid_graph_node(self) -> str:
        return f"[({str(self)})]"

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

MappingNodeTypes: TypeAlias = Union[
    Temporal,
    Spatial,
    Storage,
    Pipeline,
    Sequential,
    Compute,
    Reservation,
    Fill
]


class Mapping(ParsableModel): # TODO: Make this a partial mapping
    version: Annotated[str, assert_version] = __version__
    nodes: ParsableList[MappingNodeTypes] = ParsableList()

    @property
    def n_fused_loops(self) -> int:
        seen_tensors = set()
        n_fused_loops = 0
        n_seen_loops = 0
        for node in self.nodes:
            if isinstance(node, Iteration):
                n_seen_loops += 1
            elif isinstance(node, Storage):
                if set(node.tensors) - seen_tensors:
                    n_fused_loops = n_seen_loops
                seen_tensors.update(node.tensors)
        return n_fused_loops
    
    def make_pmapping_compatibility(self, tile_shapes: list[int]) -> "Mapping":
        compatibility = Mapping(nodes=[])
        i = 0
        for node in self.nodes:
            while i < len(tile_shapes) and tile_shapes[i] is None:
                i += 1
            if i >= len(tile_shapes):
                break
            new_node = copy.deepcopy(node)
            if isinstance(node, Iteration):
                new_node.tile_shape = tile_shapes[i]
                i += 1
            compatibility.nodes.append(new_node)
        return compatibility
    
    @property
    def loops(self) -> list[Iteration]:
        return [node for node in self.nodes if isinstance(node, Iteration)]
        
        
        
class MappingTree(MappingNode): # TODO: Make this a full mapping
    version: Annotated[str, assert_version] = __version__

