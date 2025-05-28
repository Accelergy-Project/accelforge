from abc import ABC
from logging import Logger
from numbers import Number
from typing import Any, Dict, List, Optional, Tuple, Union, Annotated, TypeVar, TypeAlias
from pydantic import ConfigDict, RootModel, BaseModel

from fastfusion.util.basetypes import ParsableModel, ParsableList, ParsesTo, PostCall, parse_field, T
from fastfusion.util.parse_expressions import ParseError

from .component_classes import ComponentAttributes, SubcomponentAction
from . import constraints
from fastfusion.version import assert_version, __version__

from fastfusion.frontend.constraints import ConstraintGroup

ARCHNODE_TYPE: TypeAlias = Union["Memory", "Compute", "Hierarchical"]

class ArchNode(ParsableModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Make sure all leaf names are unique
        leaves = {}
        for l in self.get_instances_of_type(Leaf):
            n = l.name
            leaves.setdefault(n, l)
            assert l is leaves[n], f"Duplicate name {n} found in architecture"

    def name2leaf(self, name: str) -> "Leaf":
        if isinstance(self, Leaf) and getattr(self, "name", None) == name:
            return self
        for element in self if isinstance(self, list) else self.values():
            try:
                return element.name2leaf(name)
            except (AttributeError, ValueError):
                pass
        raise ValueError(f"Leaf {name} not found in {self}")

    def find(self, *args, **kwargs) -> "Leaf":
        return self.name2leaf(*args, **kwargs)

class ArchNodes(ParsableList):
    def combine(self, other: "ArchNodes") -> "ArchNodes":
        return ArchNodes(self + other)

    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()})"

    def parse_expressions(self, *args, **kwargs):
        class PostCallArchNode(PostCall):
            def __call__(self, field, value, parsed, symbol_table):
                if isinstance(parsed, Container):
                    symbol_table.update(parsed.attributes)
                return parsed
        return super().parse_expressions(*args, **kwargs, post_calls=(PostCallArchNode(),))

class Spatial(ParsableModel):
    fanout_X: ParsesTo[Union[int, float]] = 1
    fanout_Y: ParsesTo[Union[int, float]] = 1

    def validate_fanout(self):
        for target in ["fanout_X", "fanout_Y"]:
            v = getattr(self, target)
            assert int(v) == v, f"{target} must be an integer, but is {v}"
            assert v > 0, f"{target} must be positive, but is {v}"

    def get_fanout(self):
        return int(self.fanout_X * self.fanout_Y)

    def to_fanout_string(self):
        return f"[1..{self.get_fanout()}]"


class Leaf(ArchNode, ABC):
    name: str
    attributes: ComponentAttributes
    spatial: Spatial = Spatial()
    constraints: ConstraintGroup = ConstraintGroup()

    def parse_expressions(self, *args, **kwargs):
        class PostCallLeaf(PostCall):
            def __call__(self, field, value, parsed, symbol_table):
                if field == "attributes":
                    symbol_table.update(parsed.model_dump())
                return parsed
        return super().parse_expressions(*args, **kwargs, post_calls=(PostCallLeaf(),), order=("attributes",))

    def get_fanout(self):
        return self.spatial.get_fanout()
    
    def _parse_constraints(self, outer_scope: dict[str, Any]):
        self.constraints.name = self.name
        return self.constraints._parse(outer_scope)


class Component(Leaf, ABC):
    component_class: str
    enabled: ParsesTo[bool] = True
    power_gated_at: ParsesTo[Optional[str]] = None
    actions: ParsableList[SubcomponentAction]
    
    def _update_actions(self, new_actions: ParsableList[SubcomponentAction]):
        has_actions = set(x.name for x in self.actions)
        for action in new_actions:
            if action.name not in has_actions:
                self.actions.append(action)


class Actions(ParsableList[SubcomponentAction]):
    pass


class Container(Leaf, ABC):
    pass

MEMORY_ACTIONS = ParsableList(
    [SubcomponentAction(name="read"),
    SubcomponentAction(name="write"),
    SubcomponentAction(name="leak"),
    ]
)

COMPUTE_ACTIONS = ParsableList(
    [SubcomponentAction(name="compute"),
    SubcomponentAction(name="leak"),
    ]
)

class Memory(Component):
    attributes: 'MemoryAttributes'
    actions: ParsableList[SubcomponentAction] = MEMORY_ACTIONS
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._update_actions(MEMORY_ACTIONS)


class Compute(Component):
    actions: ParsableList[SubcomponentAction] = COMPUTE_ACTIONS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._update_actions(COMPUTE_ACTIONS)


class Attributes(ComponentAttributes):
    pass


class MemoryAttributes(Attributes):
    datawidth: ParsesTo[Union[int, float]]
    size: ParsesTo[Union[int, float]]
    multiple_buffering: ParsesTo[Union[int, float]] = 1
    shared_read_write_bandwidth: ParsesTo[Union[int, float]] = float('inf')
    read_bandwidth: ParsesTo[Union[int, float]] = float('inf')
    write_bandwidth: ParsesTo[Union[int, float]] = float('inf')


class Branch(ArchNode, ABC):
    nodes: ArchNodes[ARCHNODE_TYPE] = ArchNodes()
    
    def __init__(self, **kwargs):
        if "nodes" in kwargs:
            kwargs["nodes"] = ArchNodes([make_arch_node(node) for node in kwargs["nodes"]])
        super().__init__(**kwargs)

def make_arch_node(node: Union[Memory, Compute, "Hierarchical", Any]):
    tag_to_class = {
        "!Memory": Memory,
        "!Compute": Compute,
        "!Hierarchical": Hierarchical,
    }
    
    if isinstance(node, ArchNode):
        for k, v in tag_to_class.items():
            if isinstance(node, v):
                node._yaml_tag = k
                return node
        raise ValueError(f"Unknown node type {type(node)}. Should be one of {sorted(tag_to_class.values())}")
    
    tag = getattr(node, "_yaml_tag", None)
    if tag is None:
        raise ValueError(f"Unknown node type. Set _yaml_tag in {node}")
    tag = str(tag)
    if tag not in tag_to_class:
        raise ValueError(f"Unknown node type {tag}. Set _yaml_tag "
                         f" in {node} to one of {sorted(tag_to_class.keys())}")
    
    node = tag_to_class[tag](**node)
    node._yaml_tag = tag
    return node

def arch_nodes_factory(nodes: ArchNodes[ARCHNODE_TYPE]):
    return ArchNodes([make_arch_node(node) for node in nodes])

class Hierarchical(Branch):
    def _flatten(self, attributes: dict, fanout: int = 1, return_fanout: bool = False):
        nodes = []
        for i, node in enumerate(self.nodes):
            try:
                if isinstance(node, Hierarchical):
                    new_nodes, new_fanout = node._flatten(
                        attributes, fanout, return_fanout=True
                    )
                    nodes.extend(new_nodes)
                    fanout *= new_fanout
                elif isinstance(node, Leaf) and not isinstance(node, Container):
                    fanout *= node.get_fanout()
                    node2 = node.model_copy()
                    node2.attributes = type(node.attributes)(
                        **{**attributes.model_dump(), **node.attributes.model_dump()}
                    )
                    node2.attributes.n_instances *= fanout
                    nodes.append(node2)
                elif isinstance(node, Container):
                    fanout *= node.get_fanout()
                else:
                    raise TypeError(f"Can't flatten {node}")
            except ParseError as e:
                e.add_field(node)
                raise e
        if return_fanout:
            return nodes, fanout
        return nodes

class Architecture(Hierarchical):
    version: Annotated[str, assert_version] = __version__
