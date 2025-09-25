from abc import ABC
import math
from numbers import Number
from typing import (
    Any,
    Optional,
    Union,
    Annotated,
)
from pydantic import Tag
import pydantic

from fastfusion.util.basetypes import (
    ParsableModel,
    ParsableList,
    ParsesTo,
    PostCall,
    get_tag,
)
from fastfusion.util.parse_expressions import ParseError, parse_expression

from .components import ComponentAttributes, SubcomponentAction
from . import constraints
from fastfusion.version import assert_version, __version__
from pydantic import Discriminator

from fastfusion.frontend.constraints import ConstraintGroup, MiscOnlyConstraints


class ArchNode(ParsableModel):
    def model_post_init(self, __context__=None) -> None:
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
    name: str
    fanout: ParsesTo[int]


class Leaf(ArchNode, ABC):
    name: str
    attributes: ComponentAttributes = ComponentAttributes()
    spatial: ParsableList[Spatial] = ParsableList()
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
        return int(math.prod(x.fanout for x in self.spatial))

    def _parse_constraints(self, outer_scope: dict[str, Any]):
        self.constraints.name = self.name
        return self.constraints._parse(outer_scope, location=f"{self.name} constraints")


class Component(Leaf, ABC):
    component_class: Optional[str] = None
    actions: ParsableList[SubcomponentAction]

    def _update_actions(self, new_actions: ParsableList[SubcomponentAction]):
        has_actions = set(x.name for x in self.actions)
        for action in new_actions:
            if action.name not in has_actions:
                self.actions.append(action)

    def get_component_class(self) -> str:
        if self.component_class is None:
            raise ParseError(
                f"component_class must be set to a valid string. "
                f"Got {self.component_class}.",
                source_field=f"{self.name}.component_class",
            )
        return self.component_class


class Actions(ParsableList[SubcomponentAction]):
    pass


class Container(Leaf, ABC):
    pass


class ArchMemoryActionArguments(ComponentAttributes):
    bits_per_action: ParsesTo[Union[int, float]] = 1


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

    def model_post_init(self, __context__=None) -> None:
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


class TensorHolder(Component):
    actions: ParsableList[ArchMemoryAction] = MEMORY_ACTIONS
    attributes: TensorHolderAttributes = pydantic.Field(
        default_factory=TensorHolderAttributes
    )

    def model_post_init(self, __context__=None) -> None:
        self._update_actions(MEMORY_ACTIONS)


class Memory(TensorHolder):
    attributes: "MemoryAttributes" = pydantic.Field(default_factory=MemoryAttributes)


class ProcessingStage(TensorHolder):
    pass


class ComputeAttributes(ComponentAttributes):
    computes_per_cycle: ParsesTo[int] = 1


class Compute(Component):
    actions: ParsableList[SubcomponentAction] = COMPUTE_ACTIONS
    attributes: ComputeAttributes = pydantic.Field(default_factory=ComputeAttributes)
    constraints: MiscOnlyConstraints = MiscOnlyConstraints()

    def model_post_init(self, __context__=None) -> None:
        self._update_actions(COMPUTE_ACTIONS)


class Branch(ArchNode, ABC):
    # nodes: ArchNodes[InferFromTag[Compute, Memory, "Hierarchical"]] = ArchNodes()
    nodes: ArchNodes[
        Annotated[
            Union[
                Annotated[Compute, Tag("Compute")],
                Annotated[Memory, Tag("Memory")],
                Annotated[ProcessingStage, Tag("ProcessingStage")],
                Annotated["Parallel", Tag("Parallel")],
                Annotated["Hierarchical", Tag("Hierarchical")],
            ],
            Discriminator(get_tag),
        ]
    ] = ArchNodes()


class Parallel(Branch):
    def _flatten(
        self,
        attributes: dict,
        compute_node: str,
        fanout: int = 1,
        return_fanout: bool = False,
    ):
        nodes = []

        def _parse_node(node: Leaf, fanout: int):
            fanout *= node.get_fanout()
            node2 = node.model_copy()
            node2.attributes = type(node.attributes)(
                **{**attributes.model_dump(), **node.attributes.model_dump()}
            )
            node2.attributes.n_instances *= fanout
            nodes.append(node2)
            return fanout

        for node in self.nodes:
            if isinstance(node, Compute) and node.name == compute_node:
                fanout = _parse_node(node, fanout)
                break
            if isinstance(node, Branch):
                computes = node.get_instances_of_type(Compute)
                if compute_node in [c.name for c in computes]:
                    new_nodes, new_fanout = node._flatten(
                        attributes, compute_node, fanout, return_fanout=True
                    )
                    nodes.extend(new_nodes)
                    fanout *= new_fanout
                    break
        else:
            raise ParseError(f"Compute node {compute_node} not found in parallel node")

        return nodes, fanout if return_fanout else nodes


class Hierarchical(Branch):
    def _flatten(
        self,
        attributes: dict,
        compute_node: str,
        fanout: int = 1,
        return_fanout: bool = False,
    ):
        nodes = []

        def _parse_node(node: Leaf, fanout: int):
            fanout *= node.get_fanout()
            node2 = node.model_copy()
            node2.attributes = type(node.attributes)(
                **{**attributes.model_dump(), **node.attributes.model_dump()}
            )
            node2.attributes.n_instances *= fanout
            nodes.append(node2)
            return fanout

        for i, node in enumerate(self.nodes):
            try:
                if isinstance(node, (Hierarchical, Parallel)):
                    if isinstance(node, Parallel) and i < len(self.nodes) - 1:
                        raise ParseError(
                            f"Parallel node {node.name} must be the last node in a hierarchical node"
                        )
                    new_nodes, new_fanout = node._flatten(
                        attributes, compute_node, fanout, return_fanout=True
                    )
                    nodes.extend(new_nodes)
                    fanout *= new_fanout
                    if any(
                        isinstance(n, Compute) and n.name == compute_node
                        for n in new_nodes
                    ):
                        break
                elif isinstance(node, Compute):
                    if node.name == compute_node:
                        fanout = _parse_node(node, fanout)
                        break
                elif isinstance(node, Leaf) and not isinstance(node, Container):
                    fanout = _parse_node(node, fanout)
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


class Arch(Hierarchical):
    version: Annotated[str, assert_version] = __version__
    global_cycle_period: ParsesTo[Union[int, float]] = (
        '"Set me with Specification().arch.global_cycle_period = [value]"'
    )

    def parse_expressions(self, symbol_table: dict[str, Any], *args, **kwargs):
        # Parse global_cycle_period
        global_cycle_period = parse_expression(
            expression=self.global_cycle_period,
            symbol_table=symbol_table,
            attr_name="global_cycle_period",
            location="global_cycle_period",
        )
        symbol_table["global_cycle_period"] = global_cycle_period
        new, symbol_table = super().parse_expressions(symbol_table, *args, **kwargs)
        new.global_cycle_period = global_cycle_period
        return new, symbol_table


# We had to reference Hierarchical before it was defined
Branch.model_rebuild()
