import math
from typing import (
    Any,
    Iterator,
    TypeVar,
    Annotated,
    Type,
    Union,
)
from pydantic import Tag
import pydot

from accelforge.util._basetypes import (
    EvalableModel,
    EvalableList,
    _PostCall,
    _get_tag,
)

from accelforge.util.exceptions import EvaluationError

from pydantic import Discriminator
from accelforge.util._basetypes import _uninstantiable
from accelforge.util.parallel import _SVGJupyterRender
from accelforge.util._visualization import _pydot_graph


class ArchNode(EvalableModel):
    """A node in the architecture."""

    def find(self, name: str) -> "Leaf":
        """Finds a `Leaf` node with the given name.

        Raises
        ------
        ValueError
            If the `Leaf` node with the given name is not found.
        """
        if isinstance(self, Leaf) and getattr(self, "name", None) == name:
            return self

        if isinstance(self, Branch):
            for element in self.nodes:
                try:
                    return element.find(name)
                except (AttributeError, ValueError):
                    pass
        raise ValueError(f"Leaf {name} not found in {self}")

    def _render_node_name(self) -> str:
        """The name for a Pydot node."""
        return f"{self.__class__.__name__}_{id(self)}"

    def _render_node_label(self) -> str:
        """The label for a Pydot node."""
        return self.__str__()

    def _render_node_shape(self) -> str:
        """The shape for a Pydot node."""
        return "box"

    def _render_node_color(self) -> str:
        """The color for a Pydot node."""
        return "white"

    def _render_node(self) -> pydot.Node:
        """Render this node using Pydot."""
        return pydot.Node(
            self._render_node_name(),
            label=self._render_node_label(),
            shape=self._render_node_shape(),
            style="filled",
            fillcolor=self._render_node_color(),
            margin=0.05,
        )

    def _render_make_children(self) -> list[pydot.Node]:
        """Renders this node and its children, returning them as a list."""
        return [self._render_node()]


class ArchNodes(EvalableList):
    """A list of ArchNodes."""

    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()})"

    def _eval_expressions(self, symbol_table: dict[str, Any], *args, **kwargs):
        class PostCallArchNode(_PostCall):
            def __call__(self, field, value, evaluated, symbol_table):
                if isinstance(evaluated, Leaf):
                    symbol_table[evaluated.name] = evaluated
                return evaluated

        for i, node in enumerate(self):
            symbol_table[i] = node

        return super()._eval_expressions(
            symbol_table, *args, **kwargs, post_calls=(PostCallArchNode(),)
        )


@_uninstantiable
class Leaf(ArchNode):
    """A leaf node in the architecture. This is an abstract class that represents any
    node that is not a `Branch`."""

    name: str
    """ The name of this `Leaf`. """

    def __str__(self) -> str:
        """String representation of the Leaf node."""
        result = self.name
        if self.spatial:
            spatial_str = ", ".join(f"{s.fanout}× {s.name}" for s in self.spatial)
            result = f"{result} [{spatial_str}]"
        return result

    def _spatial_str(self, include_newline=True) -> str:
        if not self.spatial:
            return ""
        result = ", ".join(f"{s.fanout}× {s.name}" for s in self.spatial)
        return f"\n[{result}]" if include_newline else result

    def _render_node_label(self) -> str:
        return f"{self.name}" + self._spatial_str()

    def _render_make_children(self) -> list[pydot.Node]:
        return [self._render_node()]


T = TypeVar("T")


@_uninstantiable
class Branch(ArchNode):
    nodes: ArchNodes[
        Annotated[
            Union[
                Annotated["Compute", Tag("Compute")],
                Annotated["Memory", Tag("Memory")],
                Annotated["Toll", Tag("Toll")],
                Annotated["Container", Tag("Container")],
                Annotated["Hierarchical", Tag("Hierarchical")],
                Annotated["Fork", Tag("Fork")],
            ],
            Discriminator(_get_tag),
        ]
    ] = ArchNodes()

    def get_nodes_of_type(self, types: Type[T] | tuple[Type[T], ...]) -> Iterator[T]:
        for node in self.nodes:
            if isinstance(node, types):
                yield node
            elif isinstance(node, Branch):
                yield from node.get_nodes_of_type(types)

    def _render_make_children(self) -> list[pydot.Node]:
        """Renders this node and all children, returning them as a list."""
        result = []
        for node in self.nodes:
            children = node._render_make_children()
            result.extend(child for child in children if child is not None)
        return result


class Hierarchical(Branch):
    def _flatten(
        self,
        compute_node: str,
        fanout: int = 1,
        return_fanout: bool = False,
    ):
        from accelforge.frontend.arch.components import Compute
        nodes = []

        for i, node in enumerate(self.nodes):
            try:
                if isinstance(node, Hierarchical):
                    if isinstance(node, Fork):
                        # If it's a compute node and our node is not in the fork, skip
                        # it
                        try:
                            node.find(compute_node)
                        except ValueError:
                            continue
                    new_nodes, new_fanout = node._flatten(
                        compute_node, fanout, return_fanout=True
                    )
                    nodes.extend(new_nodes)
                    fanout *= new_fanout
                    if any(
                        isinstance(n, Compute) and n.name == compute_node
                        for n in new_nodes
                    ):
                        break
                    assert not isinstance(node, Fork)
                elif isinstance(node, Compute):
                    if node.name == compute_node:
                        fanout *= node.get_fanout()
                        nodes.append(node)
                        break
                elif isinstance(node, Leaf):
                    fanout *= node.get_fanout()
                    nodes.append(node)
                else:
                    raise TypeError(f"Can't flatten {node}")
            except EvaluationError as e:
                e.add_field(node)
                raise e

        if return_fanout:
            return nodes, fanout
        return nodes

    def _parent2child_names(
        self, parent_name: str = None
    ) -> tuple[list[tuple[str, str]], str]:
        from accelforge.frontend.arch.components import Compute
        edges = []
        current_parent_name = parent_name

        for node in self.nodes:
            if isinstance(node, (Hierarchical, Fork)):
                child_edges, last_child_name = node._parent2child_names(
                    current_parent_name
                )
                edges.extend(child_edges)
                if last_child_name is not None:
                    current_parent_name = last_child_name
            elif isinstance(node, Compute):
                # Compute nodes branch off to the side like a Fork
                if current_parent_name is not None:
                    edges.append((current_parent_name, node._render_node_name()))
            else:
                if current_parent_name is not None:
                    edges.append((current_parent_name, node._render_node_name()))

                # Update parent for next iteration
                current_parent_name = node._render_node_name()

        return edges, current_parent_name

    def _render_node(self) -> pydot.Node:
        """Hierarchical nodes should not be rendered."""
        return None

    def _render_make_children(self) -> list[pydot.Node]:
        """Renders only children, not the Hierarchical node itself."""
        result = []
        for node in self.nodes:
            children = node._render_make_children()
            result.extend(child for child in children if child is not None)
        return result

    def render(self) -> str:
        """Renders the architecture as a Pydot graph."""
        graph = _pydot_graph()

        # Render all nodes (Hierarchical nodes return None and are filtered out)
        for node in self._render_make_children():
            if node is not None:
                graph.add_node(node)

        # Add edges
        edges, _ = self._parent2child_names()
        for parent_name, child_name in edges:
            graph.add_edge(pydot.Edge(parent_name, child_name))

        return _SVGJupyterRender(graph.create_svg(prog="dot").decode("utf-8"))

    def _repr_svg_(self) -> str:
        return self.render()


class Fork(Hierarchical):
    """
    A Fork is a Hierarchical that branches off from the main path. The nodes
    inside the Fork are a separate branch, while the main path continues to the next
    sibling after the Fork.
    """

    def _parent2child_names(
        self, parent_name: str = None
    ) -> tuple[list[tuple[str, str]], str]:
        edges = []

        # Process children as a hierarchical stack within the fork
        current_parent_name = parent_name

        for node in self.nodes:
            # If this node is a Hierarchical, or Fork, it's transparent
            if isinstance(node, (Hierarchical, Fork)):
                # Get edges from the child and its last node
                child_edges, last_child_name = node._parent2child_names(
                    current_parent_name
                )
                edges.extend(child_edges)
                # The last child of this branch becomes the new parent for subsequent nodes
                if last_child_name is not None:
                    current_parent_name = last_child_name
            else:
                if current_parent_name is not None:
                    edges.append((current_parent_name, node._render_node_name()))

                # Update parent for next iteration within the fork
                current_parent_name = node._render_node_name()

        # Return the original parent as the exit node so next sibling continues from parent
        return edges, parent_name
