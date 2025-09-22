"""
A module containing the visualization and types needed to run mapspace exploratioon
in FastFusion.
"""

from abc import ABC
import copy
from dataclasses import dataclass, replace
import pydot

from typing import (
    # Collections
    Any,
    Iterator,
    List,
    # Object definitions
    Annotated,
    Callable,
    Literal,
    # Type constructions
    Type,
    TypeVar,
    TypeAlias,
    # Variable meta-mandates
    Optional,
    Union,
)
from collections.abc import Set
from pydantic import ConfigDict, Discriminator, Tag
import sympy

from fastfusion.util.basetypes import (
    # Parsing helpers for the input files.
    ParsableModel,
    ParsableList,
    ParsesTo,
    # Retrieves information from YAML tags.
    get_tag,
    InferFromTag,
)
from fastfusion.frontend.workload.workload import RankVariableName, TensorName
from fastfusion.version import assert_version, __version__
from fastfusion.frontend import arch

T = TypeVar("T", bound="MappingNode")
"""TypeVar T: Restricts the allowable types to types of MappingNodes."""

NodeList: TypeAlias = ParsableList[
    Annotated[
        Union[
            Annotated["Split", Tag("Split")],
            Annotated["Compute", Tag("Compute")],
            Annotated["Storage", Tag("Storage")],
            Annotated["Temporal", Tag("Temporal")],
            Annotated["Spatial", Tag("Spatial")],
            Annotated["Sequential", Tag("Sequential")],
            Annotated["Pipeline", Tag("Pipeline")],
            Annotated["Nested", Tag("Nested")],
            Annotated["Reservation", Tag("Reservation")],
            # NOFILL: Annotated["Fill", Tag("Fill")],
            Annotated["Mapping", Tag("Mapping")],
            Annotated["ProcessingStage", Tag("ProcessingStage")],
        ],
        Discriminator(get_tag),
    ]
]
"""
TypeAlias NodeList: ParsableList that can contain and discriminate between
MappingNodes of different types.
"""


def comma_separated_list(items: list[str]) -> list[str]:
    return items
    result = []
    for item in items[:-1]:
        result.append(item)
        result.append(",")
    result.extend(items[-1:])
    return result


# =============================================================================
# Color Map for Visualization
# =============================================================================

# digraph G {

# "A" -> "B" [label=<This is my label <br/> It has line breaks. I <FONT COLOR="Red">love</FONT>  background<br/>Colors.>]

# C [label=<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
#   <TR><TD COLSPAN="3">This is my label </TD></TR>
#   <TR><TD>It has line breaks. I </TD><TD BGCOLOR="red">love</TD><TD>background</TD></TR>
#   <TR><TD COLSPAN="3">Colors</TD></TR>
#   </TABLE>>]
# }


class ColorMap:
    """
    A ColorMap used to visualize the Mapping objects.
    """

    def __init__(self, keys: list[str]):
        self.keys = keys
        self.color_list = self._make_color_map(len(keys))
        self.color_map = {key: self.color_list[i] for i, key in enumerate(keys)}

    # b'Error: Syntax error: non-space string used after </TABLE> in line 1 \n...  reused via MainMemory</HTML> ...\nin label of node Storage_139993997346128
    # Label: <<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0"><TR><TD BGCOLOR="#90EE90">W1</TD></TR></TABLE> reused via MainMemory>

    def format_list(self, items: list[str]) -> str:
        result = ['<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0"><TR>']
        for i, item in enumerate(items):
            start = '<TD ALIGN="CENTER">'  # if i < len(items) - 1 else f'</TR><TR><TD ALIGN="CENTER" COLSPAN="100">'
            if item in self.color_map:
                start = f'<TD ALIGN="CENTER" BORDER="5" COLOR="{self.color_map[item]}">'
            end = "</TD>"
            result.append(f"{start}{item}{end}")
        result.append("</TR></TABLE>>")
        return "".join(result)

        # This makes a colored bar under the text
        # result = ['<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">']
        # # First row: text
        # result.append('<TR>')
        # for item in items:
        #     result.append(f'<TD ALIGN="CENTER" STYLE="margin:0;padding:0;">{item}</TD>')
        # result.append('</TR>')
        # # Second row: color bar (height 20, width 40, minimal spacing)
        # result.append('<TR>')
        # for item in items:
        #     if item in self.color_map:
        #         result.append(f'<TD BGCOLOR="{self.color_map[item]}" HEIGHT="10" WIDTH="15" FIXEDSIZE="TRUE" STYLE="margin:0;padding:0;"></TD>')
        #     else:
        #         result.append('<TD HEIGHT="20" WIDTH="40" FIXEDSIZE="TRUE" STYLE="margin:0;padding:0;"></TD>')
        # result.append('</TR>')
        # result.append('</TABLE>>')
        # return ''.join(result)

    def _make_color_map(self, n_colors: int) -> list[str]:
        if n_colors <= 0:
            return []

        # High contrast, distinguishable colors for borders
        base_colors = [
            "#FF0000",  # Red
            "#00FF00",  # Green
            "#0000FF",  # Blue
            "#FFFF00",  # Yellow
            "#FF00FF",  # Magenta
            "#00FFFF",  # Cyan
            "#FF8000",  # Orange
            "#8000FF",  # Purple
            "#008000",  # Dark Green
            "#800000",  # Dark Red
            "#000080",  # Dark Blue
            "#808000",  # Olive
        ]

        if n_colors <= len(base_colors):
            return base_colors[:n_colors]

        # For more colors, generate additional colors with maximum distinction
        colors = base_colors.copy()

        # Use evenly spaced hues for maximum distinction
        for i in range(len(base_colors), n_colors):
            # Evenly space hues around the color wheel
            hue = i / n_colors

            # Use high saturation and value for maximum contrast
            saturation = 1.0  # Full saturation
            value = 1.0  # Full value

            # Convert HSV to RGB
            h = hue * 6
            c = value * saturation
            x = c * (1 - abs(h % 2 - 1))
            m = value - c

            if h < 1:
                r, g, b = c, x, 0
            elif h < 2:
                r, g, b = x, c, 0
            elif h < 3:
                r, g, b = 0, c, x
            elif h < 4:
                r, g, b = 0, x, c
            elif h < 5:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x

            r = int((r + m) * 255)
            g = int((g + m) * 255)
            b = int((b + m) * 255)

            colors.append(f"#{r:02x}{g:02x}{b:02x}")

        return colors


# =============================================================================
# LoopTree Mapping Nodes
# =============================================================================


class MappingNode(ParsableModel, ABC):
    """
    Represents a Node in the Mapping, which can be a loop, a storage node, a compute
    node, etc.
    """

    _constraint_lambdas: List[Callable[[], bool]] = []
    """ Constraints that apply to this node. """
    
    _must_be_here: bool = False
    """ Can the mapper move this node? """
    
    _required: bool = False
    """ Must the mapper keep this node? """

    def _render_node_name(self) -> str:
        return f"{self.__class__.__name__}_{id(self)}"

    def _render_node_label(self) -> str:
        return self.__str__()
        # return f"[\"{str(self)}\"]"
        # return self.__class__.__name__
        # return f"[\"{self.__class__.__name__}\"]"

    def _render_node_shape(self) -> str:
        return "box"

    def _render_node(self) -> str:
        return pydot.Node(
            self._render_node_name(),
            label=self._render_node_label(),
            shape=self._render_node_shape(),
            style="filled",
            fillcolor=self._render_node_color(),
            margin=0,
        )

    def _parent2next(self) -> "MappingNode":
        return self

    def _parent2child(
        self, parent: "MappingNode"
    ) -> list[tuple["MappingNode", "MappingNode"]]:
        return []

    def _render_make_children(self) -> list[str]:
        return []

    def get_nodes_of_type(self, *types: Type[T]) -> List[T]:
        nodes: List[T] = []
        if isinstance(self, types):
            nodes.append(self)
        if isinstance(self, MappingNodeWithChildren):
            for node in self.nodes:
                if isinstance(node, types):
                    nodes.append(node)
                if isinstance(node, MappingNodeWithChildren):
                    nodes.extend(node.get_nodes_of_type(*types))
        return nodes

    def flatten(self) -> list["MappingNode"]:
        if isinstance(self, MappingNodeWithChildren):
            result = [self]
            for node in self.nodes:
                result.extend(node.flatten())
            return result
        return [self]

    def _render_node_color(self) -> str:
        return "white"



class TilePattern(ParsableModel):
    stride: ParsesTo[Literal["symbol"] | sympy.Symbol | int | str] = "symbol"
    """ The stride of the pattern. """

    initial_tile_shape: ParsesTo[Literal["symbol"] | sympy.Symbol | int | None | str] = "symbol"
    """ The initial tile shape. """

    calculated_n_iterations: Literal["symbol"] | sympy.Symbol | int | None | str = None
    """ The number of iterations in the pattern. Do not set this! Used internally by the
    mapper! """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    def _symbol_attrs(self) -> tuple[str, ...]:
        return ("stride", "initial_tile_shape", "calculated_n_iterations")

    def __str__(self) -> str:
        s = []
        if self.calculated_n_iterations is not None:
            s.append(f"in [0..{self.calculated_n_iterations})")
        if self.initial_tile_shape is not None:
            s.append(f"initial={self.initial_tile_shape}")
        if self.stride is not None:
            s.append(f"stride={self.stride}")
        return " ".join(s)


    def update(self, **kwargs) -> "TilePattern":
        return type(self)(**{**self.model_dump(), **kwargs})


    def symbol2str(self) -> "TilePattern":
        def _symbol2str(x: sympy.Symbol | int | None) -> str | int | None:
            return x.name if isinstance(x, sympy.Symbol) else x
        return type(self)(**{
            x: _symbol2str(getattr(self, x))
            for x in self._symbol_attrs()
        })

    def symbols_as_strings(self) -> set[str]:
        symbols: set[str] = set()
        for x in self._symbol_attrs():
            x = getattr(self, x)
            if isinstance(x, sympy.Symbol):
                symbols.add(x.name)
            elif isinstance(x, str):
                symbols.add(x)
        return symbols

    def prepend_symbols(self, prepend: str) -> "TilePattern":
        def _prepend(x: sympy.Symbol | int | None) -> str | int | None:
            if isinstance(x, sympy.Symbol):
                x = x.name
            return prepend + x if isinstance(x, str) else x

        return self.update({
            x: _prepend(getattr(self, x))
            for x in self._symbol_attrs()
        })

    def __eq__(self, other: "TilePattern") -> bool:
        return all(getattr(self, x) == getattr(other, x) for x in self._symbol_attrs())

    def __hash__(self) -> int:
        return hash((self.initial_tile_shape, self.stride))

class Iteration(MappingNode):
    """
    A bounded loop over a rank with a given shape and/or pattern.
    """

    rank_variable: Union[Set[RankVariableName], RankVariableName]
    """ The set of rank variables that are iterated over in this loop. """

    tile_pattern: ParsesTo[TilePattern] = TilePattern()
    """ The tile pattern. """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    assume_perfect_factor: bool = True
    """ Whether the Mapper assumes perfect factorization is necessary to perform an
    operation. """

    _fused: bool = False
    """ Whether this Iteration is fused with another. """

    def __str__(self) -> str:
        x = []
        return f"for {self.rank_variable} {self.tile_pattern}"

    def __eq__(self, other: "Iteration") -> bool:
        return (
            isinstance(other, Iteration)
            and self.rank_variable == other.rank_variable
            and self.tile_pattern == other.tile_pattern
        )

    def _render_node_shape(self) -> str:
        return "box"

    def _render_node_color(self) -> str:
        return "#FCC2FC"

    def compact_str(self) -> str:
        rv = self.rank_variable
        if isinstance(rv, (set, frozenset)):
            rv = ",".join(sorted(rv))
        return f"{rv} {self.tile_pattern}"
        
    def merge(self, other: "Iteration", **kwargs) -> "Iteration":
        if not isinstance(other, Iteration):
            raise ValueError(f"Expected Iteration, got {type(other)}")
        if self.tile_pattern != other.tile_pattern:
            raise ValueError(f"Tile patterns do not match: {self.tile_pattern} != {other.tile_pattern}")

        my_rv, other_rv = self.rank_variable, other.rank_variable
        my_rv = my_rv if isinstance(my_rv, (set, frozenset)) else set((my_rv,))
        other_rv = other_rv if isinstance(other_rv, (set, frozenset)) else set((other_rv,))
        return type(self)(
            rank_variable=my_rv | other_rv,
            tile_pattern=self.tile_pattern,
            assume_perfect_factor=self.assume_perfect_factor,
            **kwargs
        )

    @property
    def initial_tile_shape(self) -> Union[int, sympy.Symbol]:
        return self.tile_pattern.initial_tile_shape

    @property
    def stride(self) -> Union[int, sympy.Symbol]:
        return self.tile_pattern.stride

    @property
    def calculated_n_iterations(self) -> int:
        return self.tile_pattern.calculated_n_iterations

    @initial_tile_shape.setter
    def initial_tile_shape(self, value: Union[int, sympy.Symbol]) -> None:
        self.tile_pattern = self.tile_pattern.update(initial_tile_shape=value)

    @stride.setter
    def stride(self, value: Union[int, sympy.Symbol]) -> None:
        self.tile_pattern = self.tile_pattern.update(stride=value)

    @calculated_n_iterations.setter
    def calculated_n_iterations(self, value: int) -> None:
        self.tile_pattern = self.tile_pattern.update(calculated_n_iterations=value)


class Temporal(Iteration):
    """
    A Temporal :class:`~.Iteration`.
    """

    def compact_str(self) -> str:
        return f"T-{super().compact_str()}"

    def __eq__(self, other: "Temporal") -> bool:
        return isinstance(other, Temporal) and super().__eq__(other)
    
    def merge(self, other: "Temporal") -> "Temporal":
        if not isinstance(other, Temporal):
            raise ValueError(f"Expected Temporal, got {type(other)}")
        return super().merge(other)


class Spatial(Iteration):
    """ A spatial loop. """

    name: Union[int, str]
    """ The dimension the spatial is occuring over. """
    
    component: str
    """ The hardware feature name hosting the iteration. """
    
    component_object: Optional[arch.Leaf] = None
    """ The hardware feature hosting the Iteration. """
    
    component_object: Optional[arch.Leaf] = None

    def compact_str(self) -> str:
        return f"S-{self.name}-{super().compact_str()}"

    def __str__(self) -> str:
        return f"S-{self.name} " + super().__str__()

    def __eq__(self, other: "Spatial") -> bool:
        return (
            isinstance(other, Spatial)
            and super().__eq__(other)
            and self.name == other.name
            and self.component == other.component
            and self.component_object == other.component_object
        )

    def merge(self, other: "Spatial") -> "Spatial":
        if not isinstance(other, Spatial):
            raise ValueError(f"Expected Spatial, got {type(other)}")
        if self.name != other.name:
            raise ValueError(f"Names do not match: {self.name} != {other.name}")
        if self.component != other.component:
            raise ValueError(f"Components do not match: {self.component} != {other.component}")
        return super().merge(
            other, 
            name=self.name, 
            component=self.component, 
            component_object=self.component_object
        )


class TensorHolder(MappingNode):
    """ A node that represents a hardware Component holding a set of tensors. """

    tensors: ParsableList[TensorName]
    """ The names of the tensors being held in this node. """
    
    component: str
    """ The name of the component holding the tensors. """
    
    component_object: Optional[arch.Component] = None
    """ The component object hosting the tensors. """
    
    _must_keep_tensors: ParsableList[TensorName] = ParsableList()
    """ Which tensor(s) the Mapper must keep here. """
    
    _backing: Set[TensorName] = set()
    """ Which tensor(s) are backed by this node. """
    
    _lower: bool = True
    """ Whether the tensor names are compressed to lowercase. """

    def compact_str(self) -> str:
        tname = ",".join(self.tensors)
        return f"[{tname} in {self.component}]"

    def __str__(self, color_map: ColorMap = None) -> str:
        tensors = self.tensors
        if color_map is not None:
            # format_list = comma_separated_list(tensors) + [f" reused via {self.component}"]
            format_list = [f"{self.component} reuses"] + comma_separated_list(tensors)
            return color_map.format_list(format_list)
        return f"{self.component} reuses {', '.join(tensors)}"

    @property
    def tensor(self) -> TensorName:
        if len(self.tensors) != 1:
            raise ValueError(
                f"TensorHolder node {repr(self)} has {len(self.tensors)} tensors. "
                f"Access the tensors property instead."
            )
        return self.tensors[0]

    def _render_node_shape(self) -> str:
        return "cylinder"

    def _render_node_color(self) -> str:
        return "#D7FCD7"


    def merge(self, other: "TensorHolder") -> "TensorHolder":
        if not isinstance(other, TensorHolder):
            raise ValueError(f"Expected TensorHolder, got {type(other)}")

        if self.component != other.component:
            raise ValueError(f"Components do not match: {self.component} != {other.component}")

        new = type(self)(
            tensors=self.tensors + other.tensors,
            component=self.component,
            component_object=self.component_object,
        )
        new._must_keep_tensors = self._must_keep_tensors + other._must_keep_tensors
        new._backing = self._backing | other._backing
        new._lower = self._lower
        return new

class Storage(TensorHolder):
    """
    A Storage component that acts as a :class:`~.TensorHolder`.
    """

    def merge(self, other: "Storage") -> "Storage":
        if not isinstance(other, Storage):
            raise ValueError(f"Expected Storage, got {type(other)}")
        return super().merge(other)


class ProcessingStage(TensorHolder):
    """
    A ProcessingStage that acts as a :class:`~.TensorHolder`.
    """

    pass


class Compute(MappingNode):
    """ A node that represents a compute operation. These nodes are the leaves of the
    LoopTree."""

    einsum: str
    """ The Einsum being computed. """
    
    compute: str
    """ The type of computation being performed. """
    
    component_object: Optional[arch.Compute] = None
    """ The compute object performing the computation. """

    def compact_str(self) -> str:
        return f"{self.compute} computes {self.einsum}"

    def __str__(self) -> str:
        return f"{self.compute} computes {self.einsum}"

    def _render_node_shape(self) -> str:
        return "ellipse"

    def _render_node_color(self) -> str:
        return "#E0EEFF"


class MappingNodeWithChildren(MappingNode):
    """
    A :class:`~.MappingNode` that also contains children.
    """

    nodes: NodeList = ParsableList()
    """ The child nodes. """

    def _parent2child(
        self, parent: MappingNode
    ) -> list[tuple[MappingNode, MappingNode]]:
        mine = [(self, node) for node in self.nodes]
        for child in self.nodes:
            mine.extend(child._parent2child(self))
        return mine

    def _parent2next(self) -> MappingNode:
        return None

    def _render_make_children(self) -> list[str]:
        lines = []
        for child in self.nodes:
            lines.append(child._render_node())
            lines.extend(child._render_make_children())
        return lines

    def get_backers(self) -> list[TensorHolder]:
        backing = []
        for child in self.nodes:
            if isinstance(child, TensorHolder) and child._backing:
                backing.append(child)
            elif isinstance(child, MappingNodeWithChildren):
                backing.extend(child.get_backers())
        return backing

    def clear_nodes_of_type(self, *types: type) -> None:
        new_nodes = []
        for node in self.nodes:
            if isinstance(node, types):
                continue
            if isinstance(node, MappingNodeWithChildren):
                node.clear_nodes_of_type(*types)
            new_nodes.append(node)
        self.nodes = ParsableList(new_nodes)

    def clear_nodes(self, *nodes: MappingNode) -> None:
        new_nodes: list[MappingNode] = []
        for node in self.nodes:
            if node in nodes:
                continue
            if isinstance(node, MappingNodeWithChildren):
                node.clear_nodes(*nodes)
            new_nodes.append(node)
        self.nodes = ParsableList(new_nodes)

    def _consolidate_tensor_holders(self) -> None:
        new_nodes = []
        for node in self.nodes:
            if isinstance(node, TensorHolder):
                found = False
                for n in new_nodes[::-1]:
                    if isinstance(n, TensorHolder) and n.component == node.component:
                        n.tensors.extend(
                            n2 for n2 in node.tensors if n2 not in n.tensors
                        )
                        found = True
                        break
                    if isinstance(n, Iteration):
                        break
                if not found:
                    new_nodes.append(node)
            else:
                new_nodes.append(node)
            if isinstance(node, MappingNodeWithChildren):
                node._consolidate_tensor_holders()
        assert new_nodes, "BUG"
        self.nodes = ParsableList(new_nodes)

    def _consolidate_reservations(self) -> None:
        new_nodes = []
        for node in self.nodes:
            if isinstance(node, Reservation):
                found = False
                for n in new_nodes[::-1]:
                    if isinstance(n, Reservation) and n.resource == node.resource:
                        n.purposes.extend(node.purposes)
                        found = True
                        break
                    if isinstance(n, Iteration):
                        break
                if not found:
                    new_nodes.append(node)
            else:
                new_nodes.append(node)
            if isinstance(node, MappingNodeWithChildren):
                node._consolidate_reservations()
        assert new_nodes, "BUG"
        self.nodes = ParsableList(new_nodes)

    def _elevate_tensor_holders_above_splits(self) -> None:
        new_nodes: list[MappingNode] = []
        for node in self.nodes:
            if isinstance(node, Split):
                shared_tensor_holders = node._get_shared_tensor_holders()
                new_nodes.extend(shared_tensor_holders)
                node.clear_nodes(*shared_tensor_holders)
            if isinstance(node, MappingNodeWithChildren):
                node._elevate_tensor_holders_above_splits()
            new_nodes.append(node)
        self.nodes = ParsableList(new_nodes)

    def _propagate_reservations_between_splits(self) -> None:
        for node in self.nodes:
            if isinstance(node, MappingNodeWithChildren):
                node._propagate_reservations_between_splits()

        if not isinstance(self, Split):
            return

        for i, node1 in enumerate(self.nodes):
            for j in range(i + 2, len(self.nodes)):
                node2 = self.nodes[j]
                reservations1 = node1.get_nodes_of_type(Reservation)
                reservations2 = node2.get_nodes_of_type(Reservation)

                shared_reservations = []
                for reservation1 in reservations1:
                    for reservation2 in reservations2:
                        if reservation1 == reservation2:
                            shared_reservations.append(reservation1)
                            break

                for s in shared_reservations:
                    for k in range(i + 1, j):
                        node3 = self.nodes[k]
                        if not isinstance(node3, Nested):
                            raise ValueError(f"Expected Nested node, got {type(node3)}")
                        reservations3 = node3.get_nodes_of_type(Reservation)
                        if s not in reservations3:
                            node3.nodes.insert(0, copy.deepcopy(s))

    def _move_tensor_holders_above_reservations(self) -> None:
        groups = []
        cur_group = []
        for node in self.nodes:
            if isinstance(node, MappingNodeWithChildren):
                node._move_tensor_holders_above_reservations()
            if not isinstance(node, (TensorHolder, Reservation)):
                groups.append(cur_group)
                cur_group = []
            cur_group.append(node)
        groups.append(cur_group)
        groups = [g for g in groups if g]

        groups = [
            [x for x in g if not isinstance(x, (TensorHolder, Reservation))]
            + [x for x in g if isinstance(x, (TensorHolder))]
            + [x for x in g if isinstance(x, (Reservation))]
            for g in groups
        ]
        self.nodes = ParsableList([x for g in groups for x in g])


class Split(MappingNodeWithChildren):
    """
    A :class:`~.MappingNodeWithChildren` that determines a Tensor split between
    the child nodes.
    """

    pass

    def __str__(self) -> str:
        return "Split"

    def _render_node_shape(self) -> str:
        return "hexagon"

    def _get_shared_tensor_holders(self) -> list[TensorHolder]:
        tensor_holders = [n.get_nodes_of_type(TensorHolder) for n in self.nodes]
        shared_tensor_holders = []
        for i in range(len(tensor_holders)):
            for j in range(i + 1, len(tensor_holders)):
                for a in tensor_holders[i]:
                    for b in tensor_holders[j]:
                        if a._backing & b._backing and a not in shared_tensor_holders:
                            assert len(a.tensors) == 1 and len(b.tensors) == 1, "BUG"
                            shared_tensor_holders.append(a)
                            break
        return shared_tensor_holders

    def _render_node_color(self) -> str:
        return "#FFFFE0"


LoopGroup: TypeAlias = list[Iteration]
NonLoopGroup: TypeAlias = list[MappingNode]


class Nested(MappingNodeWithChildren):
    """
    A :class:`~.MappingNodeWithChildren` where the last Node may, but is not
    obligated to be, a :class:`~.MappingNodeWithChildren` and where all other
    nodes are guaranteed to to be not :class:`~.MappingNodeWithChildren`.
    """

    def model_post_init(self, __context__=None) -> None:
        for node in list(self.nodes)[:-1]:
            assert not isinstance(
                node, MappingNodeWithChildren
            ), f"Nested node has a child with children. Only the last child can have children."

    def _parent2child(
        self, parent: MappingNode
    ) -> list[tuple[MappingNode, MappingNode]]:
        parent2child = []
        for node in self.nodes:
            parent2child.append((parent, node))
            parent2child.extend(node._parent2child(parent))
            parent = node._parent2next()
        return parent2child

    def _parent2next(self) -> MappingNode:
        if not self.nodes:
            raise ValueError("Nested node has no children")
        return self.nodes[-1]._parent2next()

    # def _render_connect_children(self, names_lines: list[tuple[str, str]], parent_name: str=None) -> list[str]:
    #     return super()._render_connect_children(names_lines)

    def _render_node_label(self) -> str:
        if not self.nodes:
            raise ValueError("Nested node has no children")
        return self.nodes[0]._render_node_label()

    def _render_node_name(self) -> str:
        if not self.nodes:
            raise ValueError("Nested node has no children")
        return self.nodes[0]._render_node_name()

    def get_n_shared_loops(self, other: "Nested") -> int:
        my_backing = set(
            (t, s.component) for s in self.get_backers() for t in s._backing
        )
        other_backing = set(
            (t, s.component) for s in other.get_backers() for t in s._backing
        )
        shared_backing = my_backing & other_backing

        if not shared_backing:
            return 0

        n_shared_loops = 0
        for i, node in enumerate(self.nodes):
            if isinstance(node, Iteration):
                n_shared_loops += 1
            if (
                isinstance(node, Reservation)
                and (node.purpose, node.resource) in shared_backing
            ):
                return n_shared_loops
            if isinstance(node, Split):
                for child in node.nodes:
                    max_child_n_shared_loops = 0
                    try:
                        max_child_n_shared_loops = max(
                            max_child_n_shared_loops, child.get_n_shared_loops(other)
                        )
                    except ValueError:
                        pass
                    return max_child_n_shared_loops

        raise ValueError("BUG")

    def _break_into_reorderable_groups(
        self, stop_at_n_loops: int
    ) -> list[list[MappingNode]]:
        # We can reorder loops relative to each other
        groups = []
        cur_group = None

        seen_loops = 0

        if stop_at_n_loops == 0 and not any(
            isinstance(node, Iteration) for node in self.nodes
        ):
            return [list(self.nodes)]

        i = 0
        for i, node in enumerate(self.nodes):
            if seen_loops >= stop_at_n_loops:
                break
            is_iteration = isinstance(node, Iteration)
            if cur_group is None:
                cur_group = []
            elif (
                is_iteration and not all(isinstance(x, Iteration) for x in cur_group)
            ) or (
                not is_iteration and any(isinstance(x, Iteration) for x in cur_group)
            ):
                groups.append(cur_group)
                cur_group = []
            cur_group.append(node)
            assert not isinstance(node, Sequential) or i == len(self.nodes) - 1, "BUG"
            if isinstance(node, Iteration):
                seen_loops += 1

        if cur_group:
            groups.append(cur_group)

        final_group = self.nodes[i:]
        groups.append(final_group)

        if seen_loops < stop_at_n_loops:
            raise ValueError(
                f"Expected {stop_at_n_loops} loops, but only found {seen_loops}"
            )

        # Lower reservations. If reservations are in the second-to-last group
        # # non-iteration group, lower them to the last group.
        # if len(groups) > 3:
        #     assert not any(isinstance(x, Iteration) for x in groups[-1]), "BUG"
        #     assert not any(isinstance(x, Iteration) for x in groups[-3]), "BUG"
        #     reservations = [x for x in groups[-2] if isinstance(x, Reservation)]
        #     groups[-1].extend(reservations)
        #     groups[-3] = [x for x in groups[-3] if x not in reservations]

        return groups

    def merge(self, other: "Nested", n_shared_loops: int) -> "Nested":

        # Break up the nodes above the indices. We need to have them in the format of
        # [(loop, other stuff...), (loop, other stuff...), ...]
        my_groups = self._break_into_reorderable_groups(stop_at_n_loops=n_shared_loops)
        my_remaining = my_groups.pop(-1)
        other_groups = other._break_into_reorderable_groups(
            stop_at_n_loops=n_shared_loops
        )
        other_remaining = other_groups.pop(-1)

        # Reorder so that the loops are in the same order. We can't reorder groups that
        # have other stuff in them because that'll change the behavior of the mapping.
        zipped_groups = []

        def _pop_loop_group(groups: list[list[MappingNode]]) -> list[MappingNode]:
            while groups and not any(isinstance(x, Iteration) for x in groups[0]):
                zipped_groups.append(groups.pop(0))
            return groups.pop(0) if groups else []

        my_loop_group = _pop_loop_group(my_groups)
        other_loop_group = _pop_loop_group(other_groups)
        while (my_groups or my_loop_group) and (other_groups or other_loop_group):
            if not my_loop_group:
                my_loop_group = _pop_loop_group(my_groups)
                continue
            if not other_loop_group:
                other_loop_group = _pop_loop_group(other_groups)
                continue

            # Add matching loops from the two groups. If we can't find a match, raise an
            # error.
            to_add = None
            for i, a in enumerate(my_loop_group):
                for j, b in enumerate(other_loop_group):
                    if a == b:
                        to_add = [a]
                        my_loop_group.pop(i)
                        other_loop_group.pop(j)
                        break

            if to_add is None:
                raise ValueError(
                    f"No matching loop found for {my_loop_group} and {other_loop_group}"
                )

            zipped_groups.append(to_add)

        assert not my_loop_group and not other_loop_group, "BUG"

        zipped_groups.extend(my_groups)
        zipped_groups.extend(other_groups)

        flattened = list(x for group in zipped_groups for x in group)
        new_nodes = [x for x in flattened if not isinstance(x, Sequential)]
        new_nodes.extend([x for x in flattened if isinstance(x, Sequential)])

        if isinstance(my_remaining[0], Sequential) and isinstance(
            other_remaining[0], Sequential
        ):
            my_remaining[0].nodes.extend(other_remaining[0].nodes)
            assert len(my_remaining) == 1 and len(other_remaining) == 1, "BUG"
            new_nodes.append(my_remaining[0])
        elif isinstance(my_remaining[0], Sequential):
            my_remaining[0].nodes.append(Nested(nodes=other_remaining))
            assert len(my_remaining) == 1, "BUG"
            new_nodes.append(my_remaining[0])
        elif isinstance(other_remaining[0], Sequential):
            other_remaining[0].nodes.append(Nested(nodes=my_remaining))
            assert len(other_remaining) == 1, "BUG"
            new_nodes.append(other_remaining[0])
        else:
            new_nodes.append(
                Sequential(
                    nodes=[Nested(nodes=my_remaining), Nested(nodes=other_remaining)]
                )
            )

        return Nested(nodes=new_nodes)

    def beautify_loops(
        self, rank_variable_bounds: Optional[dict[str, dict[str, int]]] = None
    ):
        to_remove = []
        rank_variable_bounds = rank_variable_bounds or {}

        for i, node in enumerate(self.nodes):
            if not isinstance(node, Iteration):
                continue
            prev_tile_shape = None
            for j in range(i - 1, -1, -1):
                node2 = self.nodes[j]
                if not isinstance(node2, Iteration):
                    continue
                if node2.stride is None:
                    continue
                if node2.rank_variable != node.rank_variable:
                    continue
                prev_tile_shape = node2.stride
                break
            if prev_tile_shape is None:
                prev_tile_shape = rank_variable_bounds.get(node.rank_variable, None)
            if prev_tile_shape is not None:
                if node.stride == prev_tile_shape:
                    to_remove.append(i)
                    continue
                elif node.stride is not None and prev_tile_shape is not None:
                    node.tile_pattern = node.tile_pattern.update(
                        calculated_n_iterations=prev_tile_shape / node.stride,
                    )

        def safe_int_cast(x: int | float | None) -> int | float | None:
            try:
                int_x = int(x)
                return int_x if int_x == x else x
            except:
                pass    
            return x

        for i, node in enumerate(self.nodes):
            if not isinstance(node, Iteration):
                continue
            node.tile_pattern = node.tile_pattern.update(
                initial_tile_shape=safe_int_cast(node.tile_pattern.initial_tile_shape),
                stride=safe_int_cast(node.tile_pattern.stride),
            )

        self.nodes = [node for i, node in enumerate(self.nodes) if i not in to_remove]

    def compact_str(self) -> str:
        result = []
        prev = None
        for node in self.nodes:
            try:
                prev = prev.merge(node)
            except:
                if prev is not None:
                    result.append(prev)
                prev = node
        if prev is not None:
            result.append(prev)
        
        return " ".join(node.compact_str() for node in result)

class Pipeline(Split):
    """
    A :class:`~.Split` where the tensors are stored and processed
    in parallel.
    """

    pass


class Sequential(Split):
    """
    A :class:`~.Split` where the tensors are stored and processed
    in series.
    """

    pass


# =============================================================================
# Nodes That May Only be Inserted by the Model
# =============================================================================


class Reservation(MappingNode):
    """ A node that reserves a hardware resource for a specific task. """

    purposes: ParsableList[str]
    """ The reasons for reserving the resource. """
    
    resource: str
    """ The resource being reserved. """

    def compact_str(self) -> str:
        return f'{",".join(self.purposes)} reserves {self.resource}'

    def __str__(self, color_map: ColorMap = None) -> str:
        purposes = self.purposes
        if color_map is not None:
            # format_list = comma_separated_list(purposes) + [f" reserves {self.resource}"]
            format_list = [f"{self.resource} reserved for"] + comma_separated_list(
                purposes
            )
            return color_map.format_list(format_list)
        return f"{self.resource} reserved for {",".join(purposes)}"

    def _render_node_shape(self) -> str:
        return "component"

    @property
    def purpose(self) -> str:
        if len(self.purposes) == 1:
            return self.purposes[0]
        raise ValueError(f"Reservation has multiple purposes: {self.purposes}")

    def __eq__(self, other: "Reservation") -> bool:
        return isinstance(other, Reservation) and self.purposes == other.purposes and self.resource == other.resource

    def _render_node_color(self) -> str:
        return "#E8E8E8"  # Light gray


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
    # Fill,
    TensorHolder,
]
"""TypeAlias MappingNodeTypes: The types of MappingNodes possible."""


class Mapping(Nested):
    """ A Mapping of a workload onto a hardware architecture. """

    version: Annotated[str, assert_version] = __version__

    def get_fused_slice(self, intermediate_tensors: set[TensorName]) -> "Mapping":
        """
        Return a mapping with:
        - All backing reservation nodes for intermediate tensors
        - Loop nodes above any backing reservation nodes
        """
        # All intermediate tensors that can be found in this mapping
        # Note: `intermediate_tensors` may be for **whole workload**.
        relevant_intermediate_tensors = set()
        for node in self.nodes:
            if isinstance(node, Reservation):
                if node.purpose in intermediate_tensors:
                    relevant_intermediate_tensors.add(node.purpose)

        fused_slice = Mapping(nodes=[])
        to_add = []
        for node in self.nodes:
            node = copy.copy(node)
            if isinstance(node, Reservation):
                if node.purpose not in relevant_intermediate_tensors:
                    continue
                fused_slice.nodes.extend(to_add + [node])
                to_add = []
                relevant_intermediate_tensors.remove(node.purpose)
                if len(relevant_intermediate_tensors) == 0:
                    break
            elif isinstance(node, Iteration):
                to_add.append(node)
        return fused_slice

    @property
    def loops(self) -> list[Iteration]:
        return [node for node in self.nodes if isinstance(node, Iteration)]

    def _render_node_label(self) -> str:
        return f"Root"

    def render(self) -> str:
        graph = pydot.Dot(graph_type="digraph", rankdir="TD", ranksep=0.2)
        graph.set_node_defaults(shape="box", fontname="Arial", fontsize="12")
        graph.set_edge_defaults(fontname="Arial", fontsize="10")
        # Enable HTML-like labels for color support
        graph.set_node_defaults(label="")
        for node in self._render_make_children():
            graph.add_node(node)

        color_keys = set()
        all_nodes = self.flatten()
        for node in all_nodes:
            if isinstance(node, TensorHolder):
                color_keys.update(node.tensors)
            if isinstance(node, Reservation):
                color_keys.update(node.purposes)

        color_map = ColorMap(sorted(color_keys))

        for node in all_nodes:
            if isinstance(node, (TensorHolder, Reservation)):
                graph_nodes = graph.get_node(node._render_node_name())
                for graph_node in graph_nodes:
                    # Set HTML-like label for color support
                    new_label = node.__str__(color_map)
                    graph_node.set_label(new_label)
                    # graph_node.set_fillcolor(color_map[node._color_key()])
                    # graph_node.set_style('filled')

        added_edges = set()
        for parent, child in self._parent2child(None):
            if parent is not None:
                parent_name = parent._render_node_name()
                child_name = child._render_node_name()
                if (parent_name, child_name) not in added_edges:
                    graph.add_edge(pydot.Edge(parent_name, child_name))
                    added_edges.add((parent_name, child_name))
        return graph.create_svg(prog="dot")

    @classmethod
    def from_pmappings(
        cls,
        pmappings: list[Nested],
        rank_variable_bounds: Optional[dict[str, dict[str, int]]] = None,
    ) -> "Mapping":
        pmappings = list(copy.deepcopy(pmappings))
        for pmapping in pmappings:
            pmapping.beautify_loops(rank_variable_bounds)

        while len(pmappings) > 1:
            highest_n_shared_loops = 0
            highest_shared_pmapping_index = 0
            for i, pmapping in enumerate(pmappings):
                shared_index = 0
                for j in range(i + 1, len(pmappings)):
                    shared_index = max(
                        pmapping.get_n_shared_loops(pmappings[j]), shared_index
                    )
                if shared_index > highest_n_shared_loops:
                    highest_n_shared_loops = shared_index
                    highest_shared_pmapping_index = i

            def einsum_names(pmapping: Nested) -> str:
                return ",".join(n.einsum for n in pmapping.get_nodes_of_type(Compute))

            names_a = einsum_names(pmappings[highest_shared_pmapping_index])
            names_b = einsum_names(pmappings[highest_shared_pmapping_index + 1])
            print(
                f"Merging with shared loops {highest_n_shared_loops}: {names_a} <--> {names_b}."
            )
            pmappings[highest_shared_pmapping_index] = pmappings[
                highest_shared_pmapping_index
            ].merge(
                pmappings.pop(highest_shared_pmapping_index + 1),
                highest_n_shared_loops,
            )

        mapping: Mapping = cls(nodes=pmappings)
        mapping._elevate_tensor_holders_above_splits()
        mapping._propagate_reservations_between_splits()
        mapping._consolidate_tensor_holders()
        mapping._consolidate_reservations()
        mapping._move_tensor_holders_above_reservations()
        return mapping

        # import mermaid as md
        # from mermaid.graph import Graph
        # lines = []
        # lines = [
        #     "graph TD",
        #     "%%{init: {'flowchart': {'nodeSpacing': 30, 'rankSpacing': 30, 'padding': 2}, 'themeVariables': {'fontFamily': 'Arial, sans-serif'}}}%%"
        # ]
        # lines.extend(self._render_make_children())
        # for parent, child in self._parent2child(None):
        #     if parent is not None:
        #         lines.append(f"{parent._render_node_name()} --> {child._render_node_name()}")
        #     # if _is_root:
        # #     lines.extend([
        # #         "",
        # #         "classDef default fill:#fff,stroke:#000,stroke-width:1px,color:#000,font-family:Arial,font-size:12px,padding:2px;",
        # #         "classDef compact fill:#fff,stroke:#000,stroke-width:1px,color:#000,font-family:Arial,font-size:12px,padding:2px;"
        # #     ])

        # # Create the graph with the flowchart script
        # flowchart_script = "\n".join(lines)
        # graph = Graph('Flowchart', flowchart_script)

        # # Set the configuration for compact layout
        # config = md.Config()
        # config.theme = 'base'
        # # config.theme_variables = {
        # #     'primaryColor': '#ffffff',
        # #     'primaryTextColor': '#000000',
        # #     'primaryBorderColor': '#000000',
        # #     'lineColor': '#000000',
        # #     'fontSize': '12px'
        # # }
        # # config.flowchart = {
        # #     'nodeSpacing': 20,
        # #     'rankSpacing': 10,
        # #     'curve': 'linear'
        # # }
        # graph.config = config

        # return md.Mermaid(graph)


class MappingTree(MappingNode):  # TODO: Make this a full mapping
    version: Annotated[str, assert_version] = __version__


Split.model_rebuild()
Nested.model_rebuild()
Mapping.model_rebuild()
