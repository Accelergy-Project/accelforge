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
            Annotated["Reservation", Tag("Reservation")],
            Annotated["Fill", Tag("Fill")],
            Annotated["Mapping", Tag("Mapping")],
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

    def __eq__(self, other: "Iteration") -> bool:
        return isinstance(other, Iteration) and \
               self.rank_variable == other.rank_variable and \
               self.loop_bound == other.loop_bound and \
               self.tile_shape == other.tile_shape and \
               self.tile_pattern == other.tile_pattern

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
        
    def __eq__(self, other: "Temporal") -> bool:
        return isinstance(other, Temporal) and \
               super().__eq__(other)

class Spatial(Iteration):
    dimension: Union[int, str]
    across: str
    across_object: Optional[architecture.Leaf] = None

    def compact_string(self) -> str:
        return f"S{self.dimension}-{self.rank_variable}-{self.loop_bound}"

    def __str__(self) -> str:
        return f"S{self.dimension}" + super().__str__()
    
    def __eq__(self, other: "Spatial") -> bool:
        return isinstance(other, Spatial) and \
               super().__eq__(other) and \
               self.dimension == other.dimension and \
               self.across == other.across and \
               self.across_object == other.across_object

class Storage(MappingNode):
    tensors: ParsableList[TensorName]
    memory: str
    memory_object: Optional[architecture.Memory] = None # Reference to memory node for convenience
    _must_keep_tensors: ParsableList[TensorName] = ParsableList() # Must the mapper keep these tensors here?
    _backing: set[TensorName] = set()  # Is this node a backing storage?
    _even_with_below: bool = False
    _lower: bool = True

    def compact_string(self) -> str:
        tname = ",".join(self.tensors)
        return f"[{self.memory} {tname} {self._lower}]"
    
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


    def clear_nodes_of_type(self, *types: type) -> "MappingNodeWithChildren":
        new_nodes = []
        for node in self.nodes:
            if isinstance(node, types):
                continue
            if isinstance(node, MappingNodeWithChildren):
                node = node.clear_nodes_of_type(*types)
            new_nodes.append(node)
        return type(self)(nodes=new_nodes)
    
    def consolidate_storage(self) -> "MappingNodeWithChildren":
        new_nodes = []
        for node in self.nodes:
            if isinstance(node, Storage):
                found = False
                for n in new_nodes[::-1]:
                    if isinstance(n, Storage) and n.memory == node.memory:
                        n.tensors.extend(n2 for n2 in node.tensors if n2 not in n.tensors)
                        found = True
                        break
                    if isinstance(n, Iteration):
                        break
                if not found:
                    new_nodes.append(copy.deepcopy(node))
            elif isinstance(node, MappingNodeWithChildren):
                new_nodes.append(node.consolidate_storage())
            else:
                new_nodes.append(node)
        assert new_nodes, "BUG"
        return type(self)(nodes=new_nodes)

    def get_nodes_of_type(self, *types: type) -> list[MappingNode]:
        mine = []
        for node in self.nodes:
            if isinstance(node, types):
                mine.append(node)
            if isinstance(node, MappingNodeWithChildren):
                mine.extend(node.get_nodes_of_type(*types))
        return mine

class Split(MappingNodeWithChildren):
    pass

    def __str__(self) -> str:
        return "Split"
    
    def _render_node_shape(self) -> str:
        return "hexagon"

LoopGroup: TypeAlias = list[Iteration]
NonLoopGroup: TypeAlias = list[MappingNode]

class Nested(MappingNodeWithChildren):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for node in list(self.nodes)[:-1]:
            assert not isinstance(node, MappingNodeWithChildren), (
                f"Nested node has a child with children. Only the last child can have children."
            )
    
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
    
    def get_n_shared_loops(self, other: "Nested") -> int:
        my_backing_storage = set(
            (t, s.memory)
            for s in self.get_backing_storage_nodes() for t in s._backing
        )
        other_backing_storage = set(
            (t, s.memory)
            for s in other.get_backing_storage_nodes() for t in s._backing
        )
        shared_storage = my_backing_storage & other_backing_storage
        
        if not shared_storage:
            return 0
        
        n_shared_loops = 0
        for i, node in enumerate(self.nodes):
            if isinstance(node, Iteration):
                n_shared_loops += 1
            if isinstance(node, Reservation) and (node.tensor, node.memory) in shared_storage:
                return n_shared_loops
            if isinstance(node, Split):
                raise ValueError("Can't check for n_shared_loops beneath a split")
            
        raise ValueError("BUG")
    
    def _break_into_reorderable_groups(self, stop_at_n_loops: int) -> list[list[MappingNode]]:
        # We can reorder loops relative to each other
        groups = []
        cur_group = None
        
        seen_loops = 0
        
        if stop_at_n_loops == 0 and not any(isinstance(node, Iteration) for node in self.nodes):
            return []
        
        for i, node in enumerate(self.nodes):
            if seen_loops >= stop_at_n_loops and isinstance(node, Iteration):
                break
            is_iteration = isinstance(node, Iteration)
            if cur_group is None:
                cur_group = []
            elif (is_iteration and not all(isinstance(x, Iteration) for x in cur_group)) or \
                 (not is_iteration and any(isinstance(x, Iteration) for x in cur_group)):
                groups.append(cur_group)
                cur_group = []
            cur_group.append(node)
            assert not isinstance(node, Sequential) or i == len(self.nodes) - 1, "BUG"
            if isinstance(node, Iteration):
                seen_loops += 1
            
        if cur_group:
            groups.append(cur_group)
            
        if seen_loops < stop_at_n_loops:
            raise ValueError(f"Expected {stop_at_n_loops} loops, but only found {seen_loops}")
            
        return groups
    
    def merge(self,
              other: "Nested",
              n_shared_loops: int
              ) -> "Nested":
        

        # Break up the nodes above the indices. We need to have them in the format of
        # [(loop, other stuff...), (loop, other stuff...), ...]
        my_groups = self._break_into_reorderable_groups(stop_at_n_loops=n_shared_loops)
        other_groups = other._break_into_reorderable_groups(stop_at_n_loops=n_shared_loops)
        
        # Reorder so that the loops are in the same order. We can't reorder groups that
        # have other stuff in them because that'll change the behavior of the mapping.
        zipped_groups = []
        def _pop_loop_group(groups: list[list[MappingNode]]) -> list[MappingNode]:
            while groups and not any(isinstance(x, Iteration) for x in groups[0]):
                zipped_groups.append(groups.pop(0))
            return groups.pop(0) if groups else []
        
        my_loop_group = _pop_loop_group(my_groups)
        other_loop_group = _pop_loop_group(other_groups)
        while my_groups and other_groups:
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
                raise ValueError(f"No matching loop found for {my_loop_group} and {other_loop_group}")

            zipped_groups.append(to_add)

        zipped_groups.extend(my_groups)
        zipped_groups.extend(other_groups)

        flattened = list(x for group in zipped_groups for x in group)
        new_nodes = [x for x in flattened if not isinstance(x, Sequential)]
        new_nodes.extend([x for x in flattened if isinstance(x, Sequential)])
        
        loops_left = n_shared_loops
        my_remaining = list(self.nodes)
        for i, node in enumerate(my_remaining):
            if isinstance(node, Iteration):
                loops_left -= 1
            if loops_left <= -1:
                break
        my_remaining = my_remaining[i:] if my_remaining[i:] else my_remaining
        
        loops_left = n_shared_loops
        other_remaining = list(other.nodes)
        for i, node in enumerate(other_remaining):
            if isinstance(node, Iteration):
                loops_left -= 1
            if loops_left <= -1:
                break
        other_remaining = other_remaining[i:] if other_remaining[i:] else other_remaining
        
        if isinstance(my_remaining[0], Sequential) and isinstance(other_remaining[0], Sequential):
            my_remaining[0].nodes.extend(other_remaining[0].nodes)
        elif isinstance(my_remaining[0], Sequential):
            my_remaining[0].nodes.append(Nested(nodes=other_remaining))
        elif isinstance(other_remaining[0], Sequential):
            other_remaining[0].nodes.append(Nested(nodes=my_remaining))
        else:
            new_nodes.append(Sequential(nodes=[Nested(nodes=my_remaining), Nested(nodes=other_remaining)]))

        return Nested(nodes=new_nodes)


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
                    
        def safe_int_cast(x: int | float | None) -> int | float | None:
            try:
                int_x = int(x)
            except:
                return x
            return int_x if int_x == x else x
                    
        for i, node in enumerate(self.nodes):
            if not isinstance(node, Iteration):
                continue
            node.loop_bound = safe_int_cast(node.loop_bound)
            node.tile_shape = safe_int_cast(node.tile_shape)

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
        return f'R {self.tensor} in {self.memory}'


class Fill(MappingNode, ModelOnlyNode):
    tensor: str
    memory: str

    def compact_string(self) -> str:
        return f'F {self.tensor} in {self.memory}'

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
            if isinstance(node, Reservation):
                if node.tensor in intermediate_tensors:
                    relevant_intermediate_tensors.add(node.tensor)

        fused_slice = Mapping(nodes=[])
        to_add = []
        for node in self.nodes:
            node = copy.deepcopy(node)
            if isinstance(node, Reservation):
                if node.tensor not in relevant_intermediate_tensors:
                    continue
                fused_slice.nodes.extend(to_add + [node])
                to_add = []
                relevant_intermediate_tensors.remove(node.tensor)
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
    
    
    
    @classmethod
    def from_pmappings(cls, pmappings: list[Nested], rank_variable_bounds: Optional[dict[str, dict[str, int]]] = None) -> "Mapping":
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
                        pmapping.get_n_shared_loops(pmappings[j]),
                        shared_index
                    )
                if shared_index > highest_n_shared_loops:
                    highest_n_shared_loops = shared_index
                    highest_shared_pmapping_index = i

            # def einsum_names(pmapping: Nested) -> str:
            #     return ",".join(n.einsum for n in pmapping.get_nodes_of_type(Compute))
            # names_a = einsum_names(pmappings[highest_shared_pmapping_index])
            # names_b = einsum_names(pmappings[highest_shared_pmapping_index + 1])
            # print(f'Merging with shared loops {highest_n_shared_loops}: {names_a} <--> {names_b}.')
            pmappings[highest_shared_pmapping_index] = pmappings[highest_shared_pmapping_index].merge(
                pmappings.pop(highest_shared_pmapping_index + 1),
                highest_n_shared_loops,
            )

        return cls(nodes=pmappings)
        
        
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
