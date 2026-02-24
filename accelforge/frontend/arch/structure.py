from copy import deepcopy
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

from accelforge.frontend.arch.spatialable import Spatialable

from pydantic import Discriminator
from accelforge.util._basetypes import _uninstantiable
from accelforge.util.parallel import _SVGJupyterRender
from accelforge.util._visualization import _pydot_graph

_FIND_SENTINEL = object()

D = TypeVar("D")
T = TypeVar("T")


class ArchNode(EvalableModel):
    """A node in the architecture."""

    def find(self, name: str, default: Any = _FIND_SENTINEL) -> Union["Leaf", Any]:
        """

        Finds a `Leaf` node with the given name.

        Parameters
        ----------
        name: str
            The name of the `Leaf` node to find.
        default: Any
            The value to return if the `Leaf` node with the given name is not found.
            Otherwise, raises a ValueError.

        Raises
        ------
        ValueError
            If the `Leaf` node with the given name is not found.

        Returns
        -------
        Leaf
            The `Leaf` node with the given name.
        default
            The value to return if the `Leaf` node with the given name is not found.
        """
        if isinstance(self, Leaf) and getattr(self, "name", None) == name:
            return self

        if isinstance(self, Branch):
            for element in self.nodes:
                try:
                    return element.find(name)
                except (AttributeError, ValueError):
                    pass
        if default is not _FIND_SENTINEL:
            return default
        raise ValueError(f"Leaf {name} not found in {self}")

    def is_above(self, node_a: str, node_b: str) -> bool:
        """Returns whether node_a is above node_b in a hierarchy."""
        self.find(node_a)
        self.find(node_b)
        for node, parents in self.iterate_hierarchically():
            if node.name != node_b:
                continue
            return any(p.name == node_a for p in parents)

    def find_first_of_type_above(
        self,
        node_type: T,
        name: str,
        default: D = _FIND_SENTINEL
    ) -> T | D:
        """
        Returns the first node with type `node_type` above `name`.

        If `name` does not exist, raises an error.

        If no node of `node_type` is found, either `default` is
        returned (if provided) or raises an error.
        """
        # Check if name exists
        # This *should* raise even if default != _FIND_SENTINEL
        self.find(name)

        for node, parents in self.iterate_hierarchically():
            if node.name != name:
                continue
            for p in reversed(parents):
                if isinstance(p, node_type):
                    return p
            if default is not _FIND_SENTINEL:
                return default
            raise ValueError(f"Parent of {name} with type {node_type} not found")
        raise RuntimeError(
            "BUG: find() finds node but iterate_hierarchically() does not"
        )

    def iterate_hierarchically(self, _parents=None):
        """
        Iterates over all Leaf nodes while also yielding the list of all Leaf
        nodes that are hierarchical parents over the current node.
        """
        if _parents is None:
            _parents = []

        if isinstance(self, Leaf):
            yield self, _parents
            _parents.append(self)
            return

        assert isinstance(self, Branch)
        if isinstance(self, (Fork, Array)):
            for node in self.nodes:
                yield from node.iterate_hierarchically(list(_parents))
        elif isinstance(self, Hierarchical):
            for node in self.nodes:
                yield from node.iterate_hierarchically(_parents)
        else:
            raise RuntimeError("unhandled structure type")

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

    def _render_node_label(self) -> str:
        return f"{self.name}" + self._spatial_str()

    def _render_make_children(self) -> list[pydot.Node]:
        return [self._render_node()]


@_uninstantiable
class Branch(ArchNode):
    nodes: ArchNodes[
        Annotated[
            Union[
                Annotated["Compute", Tag("Compute")],
                Annotated["Memory", Tag("Memory")],
                Annotated["Toll", Tag("Toll")],
                Annotated["Container", Tag("Container")],
                Annotated["Network", Tag("Network")],
                Annotated["Hierarchical", Tag("Hierarchical")],
                Annotated["Array", Tag("Array")],
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

    def _power_gating(
        self, compute_name, used_fanout
    ) -> tuple[dict[str, float], float]:
        from accelforge.frontend.arch.structure import Fork
        from accelforge.frontend.arch.components import Compute

        result = {}
        non_power_gated_porp = 1
        found_compute = False
        i_have_compute = self.find(compute_name, default=None) is not None
        for i, node in enumerate(self.nodes):
            if isinstance(node, Fork):
                compute_in_fork = node.find(compute_name, default=None) is not None
                found_compute |= compute_in_fork

                r, _ = node._power_gating(compute_name, used_fanout)
                r = {k: v * non_power_gated_porp for k, v in r.items()}

                if node.non_forked_power_gateable and compute_in_fork:
                    non_power_gated_porp = 0
                if node.forked_power_gateable and not compute_in_fork:
                    r = {k: 0 for k in r}
                result.update(r)

            elif isinstance(node, Hierarchical):
                found_compute |= node.find(compute_name, default=None) is not None
                r, new_porp = node._power_gating(compute_name, used_fanout)
                r = {k: v * non_power_gated_porp for k, v in r.items()}
                result.update(r)
                non_power_gated_porp *= new_porp

            elif isinstance(node, Leaf):
                for s in node.spatial:
                    if found_compute or not i_have_compute:
                        assert (node.name, s.name) not in used_fanout, "BUG"
                        porp = 0
                    else:
                        porp = used_fanout.get((node.name, s.name), 1) / s.fanout
                    if s.power_gateable:
                        non_power_gated_porp *= porp
                result[node.name] = non_power_gated_porp

                if isinstance(node, Compute) and node.name == compute_name:
                    found_compute = True

            else:
                raise TypeError(f"Unknown node type: {type(node)}")

        return result, non_power_gated_porp


class Array(Branch, Spatialable):
    def model_post_init(self, __context__=None) -> None:
        for node in self.nodes:
            if isinstance(node, Fork):
                raise EvaluationError("cannot have fork inside array")

    def _flatten(
        self,
        compute_node: str,
        fanout: int = 1,
        return_fanout: bool = False,
    ):
        from accelforge.frontend.arch.components import Compute
        nodes = []

        for node in self.nodes:
            try:
                if isinstance(node, Branch):
                    raise RuntimeError("do not put branches inside array")
                elif isinstance(node, Leaf):
                    fanout *= node.get_fanout()
                    node = deepcopy(node)
                    node.spatial = EvalableList()
                    nodes.append(node)
                else:
                    raise RuntimeError(f"unhandled structure type {node}")
            except EvaluationError as e:
                e.add_field(node)
                raise e

        if return_fanout:
            return nodes, fanout
        return nodes

    def _render_node_label(self) -> str:
        return f"Array {self._spatial_str()}"

    def _render_node_color(self) -> str:
        return "#FCC2FC"

    def _parent2child_names(
        self, parent_name: str = None
    ) -> tuple[list[tuple[str, str, str]], str]:
        from accelforge.frontend.arch.components import Compute

        edges = []
        current_parent_name = parent_name

        for node in self.nodes:
            if isinstance(node, Branch):
                raise EvaluationError("do not put branch in array")
            elif isinstance(node, Compute):
                # Compute nodes branch off to the side like a Fork
                if current_parent_name is not None:
                    edges.append((current_parent_name, node._render_node_name(), "dashed"))
            else:
                if current_parent_name is not None:
                    edges.append((current_parent_name, node._render_node_name(), "dashed"))
        return edges, self._render_node_name()

    def _render_make_children(self) -> list[pydot.Node]:
        """Renders only children, not the Hierarchical node itself."""
        result = [self._render_node()]
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
                        if node.find(compute_node, default=None) is None:
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
                elif isinstance(node, Array):
                    new_nodes = node._flatten(
                        compute_node, fanout, return_fanout=False
                    )
                    nodes.extend(new_nodes)
                    nodes.append(node)
                    fanout *= node.get_fanout()
                    if any(
                        isinstance(n, Compute) and n.name == compute_node
                        for n in new_nodes
                    ):
                        break
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
            elif isinstance(node, Array):
                child_edges, last_child_name = node._parent2child_names(
                    node._render_node_name()
                )
                edges.extend(child_edges)
                edges.append((current_parent_name, last_child_name, "solid"))
                if last_child_name is not None:
                    current_parent_name = last_child_name
            elif isinstance(node, Compute):
                # Compute nodes branch off to the side like a Fork
                if current_parent_name is not None:
                    edges.append((current_parent_name, node._render_node_name(), "solid"))
            else:
                if current_parent_name is not None:
                    edges.append((current_parent_name, node._render_node_name(), "solid"))

                # Update parent for next iteration
                current_parent_name = node._render_node_name()

        return edges, current_parent_name

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
        for parent_name, child_name, style in edges:
            graph.add_edge(pydot.Edge(parent_name, child_name, style=style))

        return _SVGJupyterRender(graph.create_svg(prog="dot").decode("utf-8"))

    def _repr_svg_(self) -> str:
        return self.render()


class Fork(Hierarchical):
    """
    A Fork is a Hierarchical that branches off from the main path. The nodes
    inside the Fork are a separate branch, while the main path continues to the next
    sibling after the Fork.
    """

    forked_power_gateable: bool = False
    """
    Whether the child branch (the nodes inside this Fork) can be power gated when
    the main branch is active. If True, these nodes will not leak when the main
    branch is being used.
    """

    non_forked_power_gateable: bool = False
    """
    Whether the main branch (the siblings after this Fork in the parent) can be power
    gated when the child branch is active. If True, those nodes will not leak when
    this Fork's child branch is being used.
    """

    def _parent2child_names(
        self, parent_name: str = None
    ) -> tuple[list[tuple[str, str]], str]:
        edges, _ = super()._parent2child_names(parent_name)
        return edges, parent_name
