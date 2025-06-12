import copy
from pydantic import BaseModel, Discriminator, Tag, model_validator
from fastfusion.frontend import architecture
from typing import Callable, List, Union, Annotated, Literal, TypeVar, TypeAlias
from abc import ABC
from fastfusion.util.basetypes import ParsableModel, ParsableList, ParsesTo, get_tag
from fastfusion.version import assert_version, __version__

from fastfusion.frontend import architecture
from fastfusion.frontend.workload.workload import RankVariableName, TensorName
from typing import Callable, Iterator, List, Optional, Type, TypeVar, Union, Annotated
from abc import ABC
from fastfusion.util.basetypes import ParsableModel, ParsableList, ParsesTo, InferFromTag
from fastfusion.version import assert_version, __version__
import pydot

T = TypeVar("T")

node_list: TypeAlias = ParsableList[Annotated[
        Union[
            Annotated["Split", Tag("Split")],
            Annotated["Compute", Tag("Compute")],
            Annotated["Storage", Tag("Storage")],
            Annotated["Temporal", Tag("Temporal")],
            Annotated["Spatial", Tag("Spatial")],
            Annotated["Sequential", Tag("Sequential")],
            Annotated["Pipeline", Tag("Pipeline")],
            Annotated["Nested", Tag("Nested")],
        ], 
    Discriminator(get_tag)
]]

# =============================================================================
# LoopTree Mapping Nodes
# =============================================================================

class MappingNode(ParsableModel, ABC):
    _constraint_lambdas: List[Callable[[], bool]] = []
    _must_be_here: bool = False  # Can the mapper move this node?
    _required: bool = False  # Must the mapper keep this node?
    # children: ParsableList["MappingNode"] = ParsableList()

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
        return pydot.Node(self._render_node_name(), label=self._render_node_label(), shape=self._render_node_shape())
        return f"{self._render_node_name()}{self._render_node_label()}"
    
    def _parent2next(self) -> "MappingNode":
        return self
    
    def _parent2child(self, parent: "MappingNode") -> list[tuple["MappingNode", "MappingNode"]]:
        return []
    
    def _render_make_children(self) -> list[str]:
        return []

class Pattern(ParsableModel):
    stride: ParsesTo[Literal['symbol'] | int]
    initial_tile_shape: ParsesTo[Literal['symbol'] | int | None] = None
    tile_shape: ParsesTo[Literal['symbol'] | int | None] = None

class Iteration(MappingNode):
    rank_variable: Union[set[RankVariableName], RankVariableName]
    loop_bound: ParsesTo[Union[Literal['symbol'], int, None]] = None
    tile_shape: ParsesTo[Union[Literal['symbol'], int, None]] = None
    tile_pattern: ParsesTo[Union[Literal['symbol'], Pattern, None]] = None
    assume_perfect_factor: bool = True
    _fused: bool = False

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
    
    def __str__(self) -> str:
        x = []
        if self.tile_shape is not None:
            x.append(f"shape {self.tile_shape}")
        if self.tile_pattern is not None:
            x.append(f"pattern {self.tile_pattern}")
        if self.loop_bound is not None:
            x.append(f"in [0..{self.loop_bound})")
        return f"for {self.rank_variable} {' '.join(x)}"

class Temporal(Iteration):
    def compact_string(self) -> str:
        if self.loop_bound is not None:
            return f"{self.rank_variable} shape {self.tile_shape}"
        elif self.tile_pattern is not None:
            return f"{self.rank_variable} patrn {self.tile_pattern}"
        elif self.loop_bound is not None:
            return f"{self.rank_variable} bound {self.loop_bound}"
        else:
            return f"{self.rank_variable} None"

class Spatial(Iteration):
    dimension: Union[int, str]
    across: str
    across_object: Optional[architecture.Leaf] = None

    def compact_string(self) -> str:
        return f"S{self.dimension}-{self.rank_variable}-{self.loop_bound}"

    def __str__(self) -> str:
        return f"S{self.dimension}" + super().__str__()

class Storage(MappingNode):
    tensors: ParsableList[TensorName]
    memory: str
    memory_object: Optional[architecture.Memory] = None # Reference to memory node for convenience
    _must_keep_tensors: ParsableList[TensorName] = ParsableList() # Must the mapper keep these tensors here?
    _backing: bool = False  # Is this node a backing storage?
    _even_with_below: bool = False
    _lower: bool = True

    def compact_string(self) -> str:
        tname = ",".join(self.tensors)
        return f"[{self.memory} {tname}]"
    
    def __str__(self) -> str:
        tname = ", ".join(self.tensors)
        return f"{tname} in {self.memory}"
    
        # return f"[(\"{tensors} in {self.memory}\")]"
    
    @property
    def tensor(self) -> TensorName:
        if len(self.tensors) != 1:
            raise ValueError(
                f"Storage node {repr(self)} has {len(self.tensors)} tensors. "
                f"Access the tensors property instead."
            )
        return self.tensors[0]
    
    def _render_node_shape(self) -> str:
        return "cylinder"



class Compute(MappingNode):
    einsum: str
    compute: str = "MAC"

    def compact_string(self) -> str:
        return f"Einsum {self.einsum}"
    
    def __str__(self) -> str:
        return f"Einsum {self.einsum}"
    
    def _render_node_shape(self) -> str:
        return "ellipse"

class MappingNodeWithChildren(MappingNode):
    nodes: node_list = ParsableList()

    def _parent2child(self, parent: MappingNode) -> list[tuple[MappingNode, MappingNode]]:
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
    
    def get_backing_storage_nodes(self) -> list[Storage]:
        backing = []
        for child in self.nodes:
            if isinstance(child, Storage) and child._backing:
                backing.append(child)
            elif isinstance(child, MappingNodeWithChildren):
                backing.extend(child.get_backing_storage_nodes())
        return backing


class Split(MappingNodeWithChildren):
    pass

    def __str__(self) -> str:
        return "Split"
    
    def _render_node_shape(self) -> str:
        return "hexagon"

    def merge_branches(self) -> "Split":
        branch2backing_storage = [node.get_backing_storage_nodes() for node in self.nodes]
        i = 0
        nodes = self.nodes
        while i < len(nodes) - 1:
            this_backing_storage = branch2backing_storage[i]
            next_backing_storage = branch2backing_storage[i + 1]
            if not set(str(b) for b in this_backing_storage) & set(str(b) for b in next_backing_storage):
                i += 1
                continue
            if not isinstance(nodes[i], Nested) or not isinstance(nodes[i + 1], Nested):
                raise ValueError(
                    f"Can only merge branches if each is a Nested node. Is this mapping a sequential "
                    f"of partial mappings?"
                )
            nodes[i] = nodes[i].merge(nodes.pop(i + 1))
            branch2backing_storage.pop(i + 1)
        return type(self)(nodes=nodes)

class Nested(MappingNodeWithChildren):
    def _parent2child(self, parent: MappingNode) -> list[tuple[MappingNode, MappingNode]]:
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
    
    def merge_branches(self) -> "Nested":
        return type(self)(nodes=[
            n.merge_branches()
            for n in self.nodes
        ])
        
    def merge(self, other: "Nested") -> "Nested":
        # 1. Get the split location. This will be right below the last shared backing storage node.
        my_backing_storage = [n.get_backing_storage_nodes() for n in self.nodes]
        other_backing_storage = [n.get_backing_storage_nodes() for n in other.nodes]
        my_backing_set = set(str(b) for b in my_backing_storage)
        other_backing_set = set(str(b) for b in other_backing_storage)
        shared_backing = my_backing_set & other_backing_set
        if not shared_backing:
            raise ValueError(
                f"No shared backing storage nodes between {self} and {other}"
            )

        insert_above_my = None
        for i, b in enumerate(my_backing_storage):
            if b in shared_backing:
                insert_above_my = i
        insert_above_other = None
        for i, b in enumerate(other_backing_storage):
            if b in shared_backing:
                insert_above_other = i
        if insert_above_my is None or insert_above_other is None:
            raise ValueError(
                f"Could not find shared backing storage node between {self} and {other}"
            )

    def beautify_loops(self, rank_variable_bounds: Optional[dict[str, dict[str, int]]] = None):
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
                if node2.tile_shape is None:
                    continue
                if node2.rank_variable != node.rank_variable:
                    continue
                prev_tile_shape = node2.tile_shape
                break
            if prev_tile_shape is None:
                prev_tile_shape = rank_variable_bounds.get(node.rank_variable, None)
            if prev_tile_shape is not None:
                if node.tile_shape == prev_tile_shape:
                    to_remove.append(i)
                    continue
                elif node.tile_shape is not None and prev_tile_shape is not None:
                    node.loop_bound = prev_tile_shape / node.tile_shape

        self.nodes = [node for i, node in enumerate(self.nodes) if i not in to_remove]
                        

class Pipeline(Split):
    pass


class Sequential(Split):
    pass

# =============================================================================
# Nodes That May Only be Inserted by the Model
# =============================================================================

class ModelOnlyNode:
    pass

class Reservation(MappingNode, ModelOnlyNode):
    tensor: str
    memory: str

    def compact_string(self) -> str:
        return f'Reservation {self.tensor} in {self.memory}'


class Fill(MappingNode, ModelOnlyNode):
    tensor: str
    memory: str

    def compact_string(self) -> str:
        return f'Fill {self.tensor} in {self.memory}'

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


class Mapping(Nested):
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
            if isinstance(node, Storage):
                for tensor in node.tensors:
                    if tensor in intermediate_tensors:
                        relevant_intermediate_tensors.add(tensor)

        fused_slice = Mapping(nodes=[])
        to_add = []
        for node in self.nodes:
            new_node = copy.deepcopy(node)
            if isinstance(new_node, Reservation):
                tensor = new_node.tensor
                if tensor not in relevant_intermediate_tensors:
                    continue
                fused_slice.nodes.extend(to_add + [new_node])
                to_add = []
                relevant_intermediate_tensors.remove(tensor)
                if len(relevant_intermediate_tensors) == 0:
                    break
            else:
                to_add.append(new_node)
        return fused_slice

    @property
    def loops(self) -> list[Iteration]:
        return [node for node in self.nodes if isinstance(node, Iteration)]
    
    def _render_node_label(self) -> str:
        return f"Root"

    def render(self, merge_branches: bool = False) -> str:
        if merge_branches:
            self = self.merge_branches()

        graph = pydot.Dot(graph_type='digraph', rankdir='TD')
        graph.set_node_defaults(shape="box", fontname="Arial", fontsize="12")
        graph.set_edge_defaults(fontname="Arial", fontsize="10")
        # graph.add_nodes_from(self._render_make_children())
        for node in self._render_make_children():
            graph.add_node(node)

        backing_storage_nodes = self.get_backing_storage_nodes()
        for a in backing_storage_nodes:
            for b in backing_storage_nodes:
                if str(a) == str(b) and id(a) != id(b):
                    edge = pydot.Edge(a._render_node_name(), b._render_node_name())
                    edge.set_constraint('false')
                    graph.add_edge(edge)
                    for node in [graph.get_node(a._render_node_name()), graph.get_node(b._render_node_name())]:
                        for n in node:
                            n.set_fillcolor('cyan')
                            n.set_style('filled')
            
        added_edges = set()
        for parent, child in self._parent2child(None):
            if parent is not None:
                parent_name = parent._render_node_name()
                child_name = child._render_node_name()
                if (parent_name, child_name) not in added_edges:
                    graph.add_edge(pydot.Edge(parent_name, child_name))
                    added_edges.add((parent_name, child_name))
        return graph.create_svg(prog='dot')
        
        
        
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

class MappingTree(MappingNode): # TODO: Make this a full mapping
    version: Annotated[str, assert_version] = __version__

Split.model_rebuild()
Nested.model_rebuild()
Mapping.model_rebuild()