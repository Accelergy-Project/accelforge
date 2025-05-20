from math import prod
from fastfusion.frontend import arch
from fastfusion.frontend.workload.workload_spec import RankVariable, Tensor
from fastfusion.yamlparse.nodes import DictNode, ListNode
from .version import assert_version
from typing import Callable, Iterator, List, Optional, Type, TypeVar, Union
from abc import ABC

class Mapping(DictNode):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("version", default="0.5", callfunc=assert_version)
        super().add_attr("nodes", MappingNodeList)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.version: str = self["version"]
        self.nodes: MappingNodeList = self["nodes"]

T = TypeVar("T")

class MappingNodeList(ListNode):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", MappingNode)

    def __getitem__(self, key: Union[str, int]) -> "MappingNode":
        return super().__getitem__(key)
    
    def compact_string(self) -> str:
        return " ".join([n.compact_string() for n in self])
    
    def enumerate_type(self, required_type: Union[Type[T], tuple[Type[T], ...]]) -> Iterator[tuple[int, T]]:
        for i, child in enumerate(self):
            if isinstance(child, required_type):
                yield i, child

class MappingNode(DictNode, ABC):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._constraint_lambdas: List[Callable[[], bool]] = []
        self._must_be_here: bool = False # Can the mapper move this node?
        self._required: bool = False # Must the mapper keep this node?

class Iteration(DictNode):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("rank_variable", (set, RankVariable))
        super().add_attr("loop_bound", (int, None), default=None)
        super().add_attr("tile_shape", (int, None), default=None)
        super().add_attr("stride", (int, None), default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank_variable: set[RankVariable] | RankVariable = self["rank_variable"]
        self.loop_bound: Optional[int] = self["loop_bound"]
        self.stride: Optional[int] = self["stride"]
        self.tile_shape: Optional[int] = self["tile_shape"]

    def _matching_iterations(self, children: List[MappingNode]) -> List["Iteration"]:
        return [c for c in children if isinstance(c, Iteration) and c.rank_variable == self.rank_variable]
    
class Temporal(Iteration):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        
    def compact_string(self) -> str:
        return f"{self.rank_variable}-{self.loop_bound}"

class Spatial(Iteration):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("dimension", str)
        super().add_attr("across", arch.Leaf, default=None)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dimension: str = self["dimension"]
        self.across: Optional[arch.Leaf] = self["across"]

    def compact_string(self) -> str:
        return f"S{self.dimension}-{self.rank_variable}-{self.loop_bound}"

class TileShape(DictNode):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", DictNode, part_name_match=True, no_change_key=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tile_shape: DictNode[str, int] = self

class Storage(MappingNode):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("tensor", Tensor)
        super().add_attr("memory", arch.Memory)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tensor: Tensor = self["tensor"]
        self.memory: arch.Memory = self["memory"]
        self._must_exist: bool = False # Must the mapper keep this node?
        self._backing: bool = False # Is this node a backing storage?

    def compact_string(self) -> str:
        return f"[{self.memory.name} {self.tensor.name}]"

    @property
    def tile_shape(self) -> dict[RankVariable, int]:
        raise NotImplementedError("tile_shape is not implemented for Storage")

    @property
    def tile_size(self) -> int:
        raise NotImplementedError("tile_size is not implemented for Storage")

    @property
    def tensor_name(self) -> str:
        return self.tensor.name

    @property
    def memory_name(self) -> str:
        return self.memory.name

class Split(MappingNode):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("children", MappingNodeList, [])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.children: MappingNodeList = self["children"]
