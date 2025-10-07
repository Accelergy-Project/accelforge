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
from fastfusion.util.setexpressions import InvertibleSet, eval_set_expression
from fastfusion.frontend.renames import TensorName

from .components import ComponentAttributes, SubcomponentAction
from . import constraints
from fastfusion.version import assert_version, __version__
from pydantic import Discriminator

from fastfusion.frontend.constraints import ConstraintGroup, MiscOnlyConstraints


class ArchNode(ParsableModel):
    """ A node in the architecture. """
    def model_post_init(self, __context__=None) -> None:
        # Make sure all leaf names are unique
        leaves = {}
        for l in self.get_instances_of_type(Leaf):
            n = l.name
            leaves.setdefault(n, l)
            assert l is leaves[n], f"Duplicate name {n} found in architecture"

    def name2leaf(self, name: str) -> "Leaf":
        """
        Finds a `Leaf` node with the given name. 
        :raises ValueError: If the `Leaf` node with the given name is not found.
        """
        if isinstance(self, Leaf) and getattr(self, "name", None) == name:
            return self
        for element in self if isinstance(self, list) else self.values():
            try:
                return element.name2leaf(name)
            except (AttributeError, ValueError):
                pass
        raise ValueError(f"Leaf {name} not found in {self}")

    def find(self, *args, **kwargs) -> "Leaf":
        """
        Finds a `Leaf` node with the given name. 
        :raises ValueError: If the `Leaf` node with the given name is not found.
        """
        return self.name2leaf(*args, **kwargs)


class ArchNodes(ParsableList):
    """ A list of `ArchNode`s. """

    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()})"

    def _parse_expressions(self, *args, **kwargs):
        class PostCallArchNode(PostCall):
            def __call__(self, field, value, parsed, symbol_table):
                if isinstance(parsed, Container):
                    symbol_table.update(parsed.attributes)
                return parsed

        return super()._parse_expressions(
            *args, **kwargs, post_calls=(PostCallArchNode(),)
        )


class Spatial(ParsableModel):
    """ A one-dimensional spatial fanout in the architecture. """
    
    name: str
    """ 
    The name of the dimension over which this spatial fanout is occurring (e.g., X or Y).
    """

    fanout: ParsesTo[int]
    """ The size of this fanout. """

    reuse: Union[str, InvertibleSet[TensorName], set[TensorName]] = "All()"
    """ The tensors that are reused in this fanout. This expression will be parsed for
    each pmapping template. """

    def _parse(self, symbol_table: dict[str, Any], location: str):
        return type(self)(
            name=self.name,
            fanout=self.fanout,
            reuse=set(eval_set_expression(
                self.reuse,
                symbol_table,
                expected_space_name="tensors",
                location=location + ".reuse",
            ))
        )


class Leaf(ArchNode, ABC):
    """ A leaf node in the architecture. This is an abstract class that represents any
    node that is not a `Branch`. """

    name: str
    """ The name of this `Leaf`. """

    attributes: ComponentAttributes = ComponentAttributes()
    """ The attributes of this `Leaf`. """

    spatial: ParsableList[Spatial] = ParsableList()
    """ 
    The spatial fanouts of this `Leaf`. 
    
    Spatial fanouts describe the spatial organization of components in the architecture.
    A spatial fanout of size N for this node means that there are N instances of this
    node. Multiple spatial fanouts lead to a multi-dimensional fanout. Spatial
    constraints apply to the data exchange across these instances. Spatial fanouts
    specified at this level also apply to lower-level `Leaf` nodes in the architecture.
    """

    constraints: ConstraintGroup = ConstraintGroup()
    """ Mapping constraints applied to this `Leaf`. """

    def _parse_expressions(self, *args, **kwargs):
        class PostCallLeaf(PostCall):
            def __call__(self, field, value, parsed, symbol_table):
                if field == "attributes":
                    symbol_table.update(parsed.model_dump())
                return parsed

        return super()._parse_expressions(
            *args, **kwargs, post_calls=(PostCallLeaf(),), order=("attributes",)
        )

    def get_fanout(self) -> int:
        """ The spatial fanout of this node. """
        return int(math.prod(x.fanout for x in self.spatial))

    def _parse_constraints(self, outer_scope: dict[str, Any]) -> ConstraintGroup:
        self.constraints.name = self.name
        return self.constraints._parse(outer_scope, location=f"{self.name} constraints")


class Component(Leaf, ABC):
    """ A component object in the architecture. This is overridden by different
    component types, such as `Memory` and `Compute`. """

    component_class: Optional[str] = None
    """ The class of this `Component`. Used if an energy or area model needs to be
    called for this `Component`. """

    actions: ParsableList[SubcomponentAction]
    """ The actions that this `Component` can perform. """

    def _update_actions(self, new_actions: ParsableList[SubcomponentAction]):
        has_actions = set(x.name for x in self.actions)
        for action in new_actions:
            if action.name not in has_actions:
                self.actions.append(action)

    def get_component_class(self) -> str:
        """ Returns the class of this `Component`. 
        
        :raises ParseError: If the `component_class` is not set.
        """
        if self.component_class is None:
            raise ParseError(
                f"component_class must be set to a valid string. "
                f"Got {self.component_class}.",
                source_field=f"{self.name}.component_class",
            )
        return self.component_class


class Actions(ParsableList[SubcomponentAction]):
    """ A list of actions that a `Component` can perform. """
    pass


class Container(Leaf, ABC):
    """ A `Container` is an abstract node in the architecture that contains other nodes. 
    For example, a P` may be a `Container` that contains `Memory`s and `Compute` units.
    """

    pass


class ArchMemoryActionArguments(ComponentAttributes):
    """ Arguments for any `Memory` action. """

    bits_per_action: ParsesTo[Union[int, float]] = 1
    """ The number of bits accessed in this action. For example, bits_per_action to 16
    that each call to this action yields 16 bits. """


class ArchMemoryAction(SubcomponentAction):
    """ An action that a `Memory` can perform. """

    arguments: ArchMemoryActionArguments
    """ The arguments for this `ArchMemoryAction`. """


MEMORY_ACTIONS = ParsableList(
    [
        ArchMemoryAction(name="read", arguments={"bits_per_action": 1}),
        ArchMemoryAction(name="write", arguments={"bits_per_action": 1}),
        SubcomponentAction(name="leak"),
    ]
)

COMPUTE_ACTIONS = ParsableList(
    [
        SubcomponentAction(name="compute"), SubcomponentAction(name="leak"),
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
            expression=value, symbol_table=symbol_table, attr_name=key,
            location=location,
        )
    return result


class Attributes(ComponentAttributes):
    """ Attributes for any `Component`. """
    pass


class TensorHolderAttributes(Attributes):
    """ Attributes for a `TensorHolder`. `TensorHolder`s are components that hold tensors
    (usually `Memory`s). When specifying these attributes, it is recommended to
    underscore-prefix attribute names. See `UNDERSCORE_PREFIX_DISCUSSION`.
    
    Number of accesses calculation is as follows. These bullets are written for "read",
    and the same applies for "write".
    
    - (# Values Read) comes from the mapping
    - (# Bits Read) = (datawidth) * (# Values Read). If datawidth is a dictionary, this
      calculation is done for each tensor.
    - (# Reads) = (# Bits Read) / bits_per_action of the read action. By default,
      bits_per_action is 1.
    - (Read Energy) = (# Reads) * (Energy specified in the read action)
    - (Read Latency) = (# Reads) / bandwidth_reads_per_cycle
    - ...
    - (Read+Write Latency) = ((# Reads) + (# Writes)) /
      (bandwidth_reads_plus_writes_per_cycle)
    - (Total Latency) = max((Read Latency), (Write Latency), (Read+Write Latency))

    """

    datawidth: ParsesTo[Union[dict, int, float]] = 1
    """ Number of bits per value stored in this `TensorHolder`. If this is a dictionary,
    keys in the dictionary are parsed as expressions and may reference one or more
    `Tensor`s. """

    bandwidth_reads_plus_writes_per_cycle: ParsesTo[Union[int, float]] = float("inf")
    """ Total bandwidth in number of reads + number of writes per cycle. """

    bandwidth_reads_per_cycle: ParsesTo[Union[int, float]] = float("inf")
    """ Total bandwidth in number of reads per cycle. """

    bandwidth_writes_per_cycle: ParsesTo[Union[int, float]] = float("inf")
    """ Total bandwidth in number of writes per cycle. """

    def model_post_init(self, __context__=None) -> None:
        if not isinstance(self.datawidth, dict):
            self.datawidth = {"All()": self.datawidth}

    def _parse_expressions(self, *args, **kwargs):
        class MyPostCall(PostCall):
            def __call__(self, field, value, parsed, symbol_table):
                if field == "datawidth":
                    parsed = _parse_tensor2bits(
                        parsed, location="datawidth", symbol_table=symbol_table
                    )
                return parsed

        return super()._parse_expressions(*args, **kwargs, post_calls=(MyPostCall(),))


class MemoryAttributes(TensorHolderAttributes):
    """ Attributes for a `Memory`. """

    size: ParsesTo[Union[int, float]]
    """ The size of this `Memory` in bits. """


class TensorHolder(Component):
    """ A `TensorHolder` is a component that holds tensors. These are usually `Memory`s,
    but can also be `ProcessingStage`s. """

    actions: ParsableList[ArchMemoryAction] = MEMORY_ACTIONS 
    """ The actions that this `TensorHolder` can perform. """

    attributes: TensorHolderAttributes = pydantic.Field(
        default_factory=TensorHolderAttributes
    )
    """ The `TensorHolderAttributes` that describe this `TensorHolder`. """

    def model_post_init(self, __context__=None) -> None:
        self._update_actions(MEMORY_ACTIONS)


class Memory(TensorHolder):
    """ A `Memory` is a `TensorHolder` that stores data over time, allowing for temporal
    reuse. """

    attributes: "MemoryAttributes" = pydantic.Field(default_factory=MemoryAttributes)
    """ The attributes of this `Memory`. """


class ProcessingStage(TensorHolder):
    """ A `ProcessingStage` is a `TensorHolder` that does not store data over time, and
    therefore does not allow for temporal reuse. Use this as a toll that charges reads
    and writes every time a piece of data moves through it. 
    
    The access counts of `Memory` and `ProcessingStage` classes are identical; they vary
    in how they affect the access counts of other components. Every write to a
    `ProcessingStage` is written to the next `Memory` (which may be above or below
    depending on where the write came from), and same for reads."""


class ComputeAttributes(ComponentAttributes):
    computes_per_cycle: ParsesTo[int] = 1


class Compute(Component):
    actions: ParsableList[SubcomponentAction] = COMPUTE_ACTIONS 
    """ The actions that this `Compute` can perform. """
    
    attributes: ComputeAttributes = pydantic.Field(default_factory=ComputeAttributes)
    """ The attributes of this `Compute`. """
    
    constraints: MiscOnlyConstraints = MiscOnlyConstraints()
    """ Mapping constraints applied to this `Compute`. """

    def model_post_init(self, __context__=None) -> None:
        self._update_actions(COMPUTE_ACTIONS)


class Branch(ArchNode, ABC):
    # nodes: ArchNodes[InferFromTag[Compute, Memory, "Hierarchical"]] = ArchNodes()
    nodes: ArchNodes[
        Annotated[
            Union[
                Annotated[Compute, Tag("Compute")], Annotated[Memory, Tag("Memory")],
                Annotated[ProcessingStage, Tag("ProcessingStage")],
                Annotated["Parallel", Tag("Parallel")], Annotated["Hierarchical",
                Tag("Hierarchical")],
            ], Discriminator(get_tag),
        ]
    ] = ArchNodes()


class Parallel(Branch):
    def _flatten(
        self, attributes: dict, compute_node: str, fanout: int = 1, return_fanout: bool
        = False,
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
        self, attributes: dict, compute_node: str, fanout: int = 1, return_fanout: bool
        = False,
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
                            f"Parallel node {node.name} must be the last node in a "
                            "hierarchical node"
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
    """ The version of the architecture specification. """

    global_cycle_period: ParsesTo[Union[int, float]] = (
        '"Set me with Specification().arch.global_cycle_period = [value]"'
    )
    """ 
    The global clock frequency of the system. All cycle counts are multiplied by this
    value to get the actual time.
    """

    def _parse_expressions(self, symbol_table: dict[str, Any], *args, **kwargs):
        # Parse global_cycle_period 
        global_cycle_period = parse_expression(
            expression=self.global_cycle_period, symbol_table=symbol_table,
            attr_name="global_cycle_period", location="global_cycle_period",
        ) 
        symbol_table["global_cycle_period"] = global_cycle_period 
        new, symbol_table = super()._parse_expressions(symbol_table, *args, **kwargs)
        new.global_cycle_period = global_cycle_period 
        return new, symbol_table


# We had to reference Hierarchical before it was defined Branch.model_rebuild()
