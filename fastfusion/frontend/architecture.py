from abc import ABC
from logging import Logger
import math
from numbers import Number
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Annotated,
    TypeVar,
    TypeAlias,
)
from pydantic import ConfigDict, RootModel, BaseModel, Tag

from fastfusion.util.basetypes import (
    InferFromTag,
    ParsableDict,
    ParsableModel,
    ParsableList,
    ParsesTo,
    PostCall,
    get_tag,
)
from fastfusion.util.parse_expressions import ParseError, parse_expression

from .component_classes import ComponentAttributes, SubcomponentAction
from . import constraints
from fastfusion.version import assert_version, __version__
from pydantic import Discriminator

from fastfusion.frontend.constraints import ConstraintGroup


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

        return super().parse_expressions(
            *args, **kwargs, post_calls=(PostCallArchNode(),)
        )


class Spatial(ParsableModel):
    fanout: ParsableDict[str, ParsesTo[int]] = ParsableDict()

    def get_fanout(self):
        return int(math.prod(self.fanout.values()))

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

        return super().parse_expressions(
            *args, **kwargs, post_calls=(PostCallLeaf(),), order=("attributes",)
        )

    def get_fanout(self):
        return self.spatial.get_fanout()

    def _parse_constraints(self, outer_scope: dict[str, Any]):
        self.constraints.name = self.name
        return self.constraints._parse(outer_scope, location=f"{self.name} constraints")


class Component(Leaf, ABC):
    component_class: Optional[str] = None
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


class ArchMemoryActionArguments(ComponentAttributes):
    bits_per_action: ParsesTo[Union[int, float]] = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ArchMemoryAction(SubcomponentAction):
    arguments: ArchMemoryActionArguments


MEMORY_ACTIONS = ParsableList(
    [
        ArchMemoryAction(name="read", arguments={"bits_per_action": 1}),
        ArchMemoryAction(name="write", arguments={"bits_per_action": 1}),
        SubcomponentAction(name="leak"),
    ]
)

COMPUTE_ACTIONS = ParsableList(
    [
        SubcomponentAction(name="compute"),
        SubcomponentAction(name="leak"),
    ]
)


def _parse_tensor2bits(
    to_parse: dict[str, Any], location: str, symbol_table: dict[str, Any]
) -> dict[str, Any]:
    result = {}
    for key, value in to_parse.items():
        if isinstance(value, Number):
            result[key] = value
            continue
        result[key] = parse_expression(
            expression=value,
            symbol_table=symbol_table,
            attr_name=key,
            location=location,
        )
    return result


class Attributes(ComponentAttributes):
    pass


class TensorHolderAttributes(Attributes):
    datawidth: ParsesTo[Union[dict, int, float]] = 1
    bandwidth_reads_plus_writes_per_cycle: ParsesTo[Union[int, float]] = float("inf")
    bandwidth_reads_per_cycle: ParsesTo[Union[int, float]] = float("inf")
    bandwidth_writes_per_cycle: ParsesTo[Union[int, float]] = float("inf")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.datawidth, dict):
            self.datawidth = {"All()": self.datawidth}

    def parse_expressions(self, *args, **kwargs):
        class MyPostCall(PostCall):
            def __call__(self, field, value, parsed, symbol_table):
                if field == "datawidth":
                    parsed = _parse_tensor2bits(
                        parsed, location="datawidth", symbol_table=symbol_table
                    )
                return parsed

        return super().parse_expressions(*args, **kwargs, post_calls=(MyPostCall(),))


class MemoryAttributes(TensorHolderAttributes):
    size: ParsesTo[Union[int, float]]
    multiple_buffering: ParsesTo[Union[int, float]] = 1


class TensorHolder(Component):
    actions: ParsableList[ArchMemoryAction] = MEMORY_ACTIONS
    attributes: TensorHolderAttributes

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._update_actions(MEMORY_ACTIONS)


class Memory(TensorHolder):
    attributes: "MemoryAttributes"


class ProcessingStage(TensorHolder):
    pass


class Compute(Component):
    actions: ParsableList[SubcomponentAction] = COMPUTE_ACTIONS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._update_actions(COMPUTE_ACTIONS)


class Branch(ArchNode, ABC):
    # nodes: ArchNodes[InferFromTag[Compute, Memory, "Hierarchical"]] = ArchNodes()
    nodes: ArchNodes[
        Annotated[
            Union[
                Annotated[Compute, Tag("Compute")],
                Annotated[Memory, Tag("Memory")],
                Annotated[ProcessingStage, Tag("ProcessingStage")],
            ],
            Discriminator(get_tag),
        ]
    ] = ArchNodes()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


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


# We had to reference Hierarchical before it was defined
Branch.model_rebuild()
