from abc import ABC
from logging import Logger
from numbers import Number
from typing import Any, Dict, List, Optional, Tuple, Union

from .components import ComponentEnergyAreaDictNode, SubcomponentAction
from fastfusion.yamlparse.nodes import DictNode, ListNode, Node
from . import constraints
from .version import assert_version

from ruamel.yaml.scalarstring import DoubleQuotedScalarString

BUFFER_CLASSES = ("DRAM", "SRAM", "regfile", "smartbuffer", "storage")
COMPUTE_CLASSES = ("mac", "intmac", "fpmac", "compute")


class ArchNode(Node):
    """
    A node in the architecture hierarchy.

    Methods:
        name2leaf: Finds a leaf node with the given name.
        find: Alias for name2leaf method.
        name2constraints: Retrieves the constraints of a leaf node with the given name.

    Raises:
        ValueError: If the leaf node with the given name is not found.

    Returns:
        None
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Make sure all leaf names are unique
        leaves = {}
        for l in self.get_nodes_of_type(Leaf):
            n = l.name
            leaves.setdefault(n, l)
            assert l is leaves[n], f"Duplicate name {n} found in architecture"

    def name2leaf(self, name: str) -> "Leaf":
        """
        Finds a leaf node with the given name.

        Args:
            name (str): The name of the leaf node to find.

        Returns:
            Leaf: The leaf node with the given name.

        Raises:
            ValueError: If the leaf node with the given name is not found.
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
        Alias for name2leaf function.
        """
        return self.name2leaf(*args, **kwargs)

    def name2constraints(self, name: str) -> "constraints.ConstraintGroup":
        """
        Retrieves the constraints of a leaf node with the given name.

        Args:
            name (str): The name of the leaf node.

        Returns:
            constraints.ConstraintGroup: The constraints of the leaf node.
        """
        return self.name2leaf(name).constraints


class ArchNodes(ArchNode, ListNode):
    """
    A collection of architectural nodes.

    This class inherits from `ArchNode` and `ListNode` classes.

    Attributes:
        None

    Methods:
        declare_attrs: Declares attributes for the architectural nodes.
        __init__: Initializes an instance of the `ArchNodes` class.
        combine: Combines two `ArchNodes` instances.
        __repr__: Returns a string representation of the `ArchNodes` instance.
        parse_expressions: Parses expressions in the `ArchNodes` instance.

    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("!Storage", Storage)
        super().add_attr("!Compute", Compute)
        super().add_attr("!Container", Container)
        super().add_attr("!Hierarchical", Hierarchical)
        super().add_attr(
            "!Component",
            (Storage, Compute, Component),
            callfunc=component_factory,
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def combine(self, other: "ArchNodes") -> "ArchNodes":
        """
        Combines two `ArchNodes` instances.

        Args:
            other: Another `ArchNodes` instance to combine with.

        Returns:
            A new `ArchNodes` instance that is the combination of self and other.

        """
        return ArchNodes(self + other)

    def __repr__(self):
        """
        Returns a string representation of the `ArchNodes` instance.

        Returns:
            A string representation of the `ArchNodes` instance.

        """
        return f"{self.__class__.__name__}({super().__repr__()})"

    def parse_expressions(
        self,
        symbol_table: Optional[Dict[str, Any]] = None,
        parsed_ids: Optional[set] = None,
    ):
        """
        Parses expressions in the `ArchNodes` instance.

        Args:
            symbol_table: A dictionary representing the symbol table.
            parsed_ids: A set of parsed IDs.

        Returns:
            The parsed `ArchNodes` instance.

        """
        n_symbol_table = {} if symbol_table is None else symbol_table.copy()
        for l in self.get_nodes_of_type(Leaf):
            n_symbol_table[l.name] = l

        def callfunc(x, sym_table):
            if isinstance(x, Container):
                sym_table.setdefault("_parent_container_attributes", {})
                sym_table.update(x.attributes)
            return x

        return super().parse_expressions(n_symbol_table, parsed_ids, callfunc)

    def __getitem__(self, key: Union[str, int]) -> ArchNode:
        return super().__getitem__(key)


class Branch(ArchNode, DictNode, ABC):
    """
    A branch in the architecture.

    Attributes:
        nodes (ArchNodes): List of child nodes in the branch.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("nodes", ArchNodes, [])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nodes: ArchNodes = self["nodes"]

    def parse_expressions(
        self,
        symbol_table: Optional[Dict[str, Any]] = None,
        parsed_ids: Optional[set] = None,
    ):
        return super().parse_expressions(symbol_table, parsed_ids)


class Hierarchical(Branch):
    """
    A hierarchical branch in the architecture.
    """

    def _flatten(self, attributes: dict, fanout: int = 1, return_fanout: bool = False):
        def replace(src, dst):
            for k, v in src.items():
                found = dst.get(k, "<SPECIFY ME>")
                if isinstance(found, str) and "<SPECIFY ME>" in found:
                    dst[k] = v

        nodes = []
        for node in self.nodes:
            if isinstance(node, Hierarchical):
                new_nodes, new_fanout = node._flatten(
                    attributes, fanout, return_fanout=True
                )
                nodes.extend(new_nodes)
                fanout *= new_fanout
            elif isinstance(node, Leaf) and not isinstance(node, Container):
                fanout *= node.get_fanout()
                node2 = type(node)(**node)
                node2.attributes = type(node.attributes)(
                    {**attributes, **node.attributes}
                )
                node2.attributes["n_instances"] *= fanout
                replace(attributes, node2.attributes)
                nodes.append(node2)
            elif isinstance(node, Container):
                fanout *= node.get_fanout()
                replace(node.attributes, attributes)
            else:
                raise TypeError(f"Can't flatten {node}")
        if return_fanout:
            return nodes, fanout
        return nodes


class Architecture(Hierarchical):
    """
    An architecture.

    Attributes:
        version (Union[str, Number]): The version of the architecture.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("version", default="0.5", callfunc=assert_version)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.version: Union[str, Number] = self["version"]

    def get_storage_nodes(self):
        return self.get_nodes_of_type(Storage)

    def get_compute_node(self):
        return self.get_nodes_of_type(Compute)[0]


class Leaf(ArchNode, DictNode, ABC):
    """
    A leaf node in the architecture hierarchy.

    Attributes:
        name (str): The name of the leaf node.
        attributes (Attributes): The attributes associated with the leaf node.
        spatial (Spatial): The spatial attributes of the leaf node.
        constraints (ConstraintGroup): The constraint group associated with the leaf node.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("name", str)
        # Class named _class to avoid clashing with class keyword
        super().add_attr("attributes", Attributes, {})
        super().add_attr("spatial", Spatial, {})
        super().add_attr(
            "constraints", constraints.ConstraintGroup, {"name": "parentname"}
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name: str = self["name"]
        self.attributes: Attributes = self["attributes"]
        self.spatial: Spatial = self["spatial"]
        self.constraints: constraints.ConstraintGroup = self["constraints"]
        self.constraints.name = self.name

    def parse_expressions(
        self,
        symbol_table: Optional[Dict[str, Any]] = None,
        parsed_ids: Optional[set] = None,
    ):
        """
        Parse the expressions in the leaf node.

        Args:
            symbol_table (Optional[Dict[str, Any]]): The symbol table for parsing expressions.
            parsed_ids (Optional[set]): The set of parsed IDs.

        Returns:
            Attributes: The parsed attributes.
        """
        n_symbol_table = {} if symbol_table is None else symbol_table.copy()

        def callfunc(x, sym_table):
            # Fill the attributes with the parent attributes
            sym_table["attributes"] = {
                **sym_table.get("_parent_container_attributes", {}),
                **sym_table.get("attributes", {}),
            }
            return x

        callfunc(None, n_symbol_table)
        super().parse_expressions(n_symbol_table, parsed_ids)
        return self.attributes

    def get_fanout(self):
        return self.spatial.get_fanout()
    
    def _parse_constraints(self, outer_scope: dict[str, Any]):
        self.constraints.name = self.name
        # self.constraints.validate_spatial(self.spatial.fanout_X, self.spatial.fanout_Y)
        return self.constraints._parse(outer_scope)


class Component(Leaf, ABC):
    """
    A component in the architecture.

    Attributes:
        _class (str): The class of the component.
        required_actions (List[str]): The list of required actions for the component.
        enabled (bool): Indicates whether the component is enabled or not.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("class", str)
        super().add_attr("enabled", bool, True)
        super().add_attr("power_gated_at", str, None)
        super().add_attr("actions", Actions, [])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._class: str = self["class"]
        self.enabled: bool = self["enabled"]
        self.power_gated_at: Optional[str] = self["power_gated_at"]
        self.actions: Actions = self["actions"]

    def _check_unrecognized(self, *args, **kwargs):
        return super()._check_unrecognized(*args, **kwargs)


class Actions(ListNode):
    """
    A list of actions.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", SubcomponentAction)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, key: Union[str, int]) -> "SubcomponentAction":
        return super().__getitem__(key)


class Container(Leaf, ABC):
    """
    A container in the architecture.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Storage(Component):
    """
    A storage component.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("attributes", StorageAttributes, {})
        super().add_attr("actions", Actions, [{"name": "read"}, {"name": "write"}])

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.attributes: StorageAttributes = self["attributes"]


class Compute(Component):
    """
    A compute component.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("actions", Actions, [{"name": "compute"}])


class Spatial(DictNode):
    """
    A spatial configuration in a system architecture.

    Attributes:
        fanout_X (int): The number of elements in the X dimension.
        fanout_Y (int): The number of elements in the Y dimension.
        get_fanout (Callable): A function that returns the fanout of the spatial configuration.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("fanout_X", (int), 1)
        super().add_attr("fanout_Y", (int), 1)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fanout_X: int = self["fanout_X"]
        self.fanout_Y: int = self["fanout_Y"]

    def validate_fanout(self):
        for target in ["fanout_X", "fanout_Y"]:
            v = self[target]
            assert int(v) == v, f"{target} must be an integer, but is {v}"
            assert v > 0, f"{target} must be positive, but is {v}"

    def get_fanout(self):
        return self.fanout_X * self.fanout_Y

    def to_fanout_string(self):
        return f"[1..{self.get_fanout()}]"


class Attributes(ComponentEnergyAreaDictNode):
    """
    A class representing attributes for a node in the architecture.

    Attributes:
        has_power_gating (bool): Indicates whether the node has power gating.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", part_name_match=True, no_change_key=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._default_parse = True


class StorageAttributes(Attributes):
    """Represents the attributes of a storage element.

    This class provides methods to declare and initialize various attributes
    related to a storage element, such as datawidth, technology, size,
    multiple buffering, and bandwidth.

    Attributes:
        datawidth (Union[str, int]): The datawidth of the storage element.
        technology (Union[str, int]): The technology used for the storage element.
        size (Union[str, int]): The size of the storage element.
        multiple_buffering (Union[str, Number]): The level of multiple buffering.
        shared_bandwidth (Union[str, Number]): The shared bandwidth of the storage element.
        read_bandwidth (Union[str, Number]): The read bandwidth of the storage element.
        write_bandwidth (Union[str, Number]): The write bandwidth of the storage element.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("size", (int, str))
        super().add_attr("datawidth", Datawidth, callfunc=datawidth_factory)
        super().add_attr("multiple_buffering", Number, 1)

        # Bandwidth and latency
        super().add_attr("shared_bandwidth", Number, None)
        super().add_attr("read_bandwidth", Number, None)
        super().add_attr("write_bandwidth", Number, None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datawidth: Datawidth = self["datawidth"]
        self.size: Optional[int] = self["size"]
        self.multiple_buffering: Number = self["multiple_buffering"]
        self.shared_bandwidth: Optional[Number] = self["shared_bandwidth"]
        self.read_bandwidth: Optional[Number] = self["read_bandwidth"]
        self.write_bandwidth: Optional[Number] = self["write_bandwidth"]

    def parse_expressions(self, *args, **kwargs):
        super().parse_expressions(*args, **kwargs)


def component_factory(*args, **kwargs) -> "Component":
    """
    Factory function for creating components based on the provided arguments.

    Args:
        *args: Variable length arguments. Either a single dictionary or keyword arguments.
        **kwargs: Keyword arguments. Either a single dictionary or keyword arguments.

    Returns:
        Component: The created component.

    Raises:
        TypeError: If both a dictionary and keyword arguments are provided, or if no dictionary is provided.
        TypeError: If the provided argument is not a dictionary.
        AssertionError: If the 'class' attribute is missing in the provided dictionary.
        AssertionError: If the 'class' attribute is not a string.
        ValueError: If the element class is unknown.

    """
    all_args = list(args) + ([kwargs] if kwargs else [])
    f = "Pass either a dictionary or keyword arguments, but not both."
    if len(all_args) > 1:
        raise TypeError(f"Too many arguments given to component_factory(). {f}")
    if len(all_args) == 0:
        raise TypeError(f"No dictionary given to component_factory(). {f}")
    if not isinstance(all_args[0], dict):
        raise TypeError(f"No dictionary given to component_factory(). {f}")

    kwargs = all_args[0]
    assert "class" in kwargs, f"Component missing 'class' attribute."
    assert isinstance(
        kwargs.get("class", None), str
    ), f'Component "class" attribute must be a string. Got {kwargs["class"]}'
    element_class = kwargs["class"]
    class2class = {BUFFER_CLASSES: Storage, COMPUTE_CLASSES: Compute}

    for c, target in class2class.items():
        if any([e in element_class for e in c]):
            return target(**kwargs)

    raise ValueError(
        f"Unknown element class {element_class}. " f"Accepted classes: {class2class}"
    )


class Datawidth(DictNode):
    """
    A class representing the datawidth of a storage element. It is a dictionary
    with keys for each tensor and values being the datawidth of the tensor.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", int, part_name_match=True, no_change_key=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def datawidth_factory(*args, **kwargs) -> "Datawidth":
    if isinstance(args[0], dict):
        return Datawidth(**args[0])
    if isinstance(args[0], int):
        return Datawidth({"*": args[0]})
    raise ValueError(
        f"Invalid datawidth: {args[0]}. Expected a single integer or a dictionary "
        f"with tensor names as keys and datawidths as values."
    )
