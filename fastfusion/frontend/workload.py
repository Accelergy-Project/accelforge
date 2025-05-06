from numbers import Number
import re
from fastfusion.yamlparse.nodes import ListNode, DictNode
from typing import List, Set, Union
from .version import assert_version

CLIST_OPERATORS = [
    "EQ",
    "NE",
    "LT",
    "GT",
    "LE",
    "GE",
    "NG",
    "NL",
    "AND",
    "OR",
]

ISL_REGEX = re.compile(
    r"\b(?!(?:" + "|".join(CLIST_OPERATORS) + r")\b)[a-zA-Z#$@][a-zA-Z0-9_]*\b"
)


class Tensor:
    def __init__(self, name: str):
        self.name = name


class Workload(DictNode):
    """
    The top-level workload object in Timeloop.

    Attributes:
        version (str): The version of the workload.
        instance (Instance): The instance object for the workload.
        shape (Shape): The shape object for the workload.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("version", default="0.5", callfunc=assert_version)
        super().add_attr("einsums", EinsumList, [])
        super().add_attr("shape", ShapeDict, {})

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.version: str = self["version"]
        self.shape: Shape = self["shape"]
        self.einsums: EinsumList = self["einsums"]
        self._validate()

        self._tensors: set[Tensor] = set()
        self._einsums_that_read_tensor: dict[Tensor, list[Einsum]] = {}
        self._einsums_that_write_tensor: dict[Tensor, list[Einsum]] = {}
        self._tensors_read_by_einsum: dict[str, set[Tensor]] = {}
        self._tensors_written_by_einsum: dict[str, set[Tensor]] = {}
        self._gather_tensors()

    def _validate(self):
        tensor2ranks = {}
        einsum_names = set()
        for einsum in self.einsums:
            if einsum.name in einsum_names:
                raise ValueError(f"Einsum name {einsum.name} is not unique")
            einsum_names.add(einsum.name)
            for tensor in einsum.tensor_accesses:
                tensor2ranks.setdefault(tensor.name, tensor.ranks)
                if tensor2ranks[tensor.name] != tensor.ranks:
                    raise ValueError(
                        f"Tensor {tensor.name} has inconsistent ranks. Found "
                        f"{tensor2ranks[tensor.name]} and {tensor.ranks}. "
                        f"Tensor is in Einsums "
                        f"{', '.join(e.name for e in self.einsums_with_tensor(tensor))}"
                    )

    def _gather_tensors(self):
        for einsum in self.einsums:
            for tensor_access in einsum.tensor_accesses:
                tensor_name = tensor_access.name
                tensor = Tensor(tensor_name)
                self._tensors.add(tensor)
                if tensor_access.output:
                    if tensor in self._einsums_that_write_tensor:
                        self._einsums_that_write_tensor[tensor].append(einsum)
                    else:
                        self._einsums_that_write_tensor[tensor] = [einsum]

                    if einsum.name in self._tensors_written_by_einsum:
                        self._tensors_written_by_einsum[einsum.name].add(tensor)
                    else:
                        self._tensors_written_by_einsum[einsum.name] = {tensor}
                else:
                    if tensor in self._einsums_that_read_tensor:
                        self._einsums_that_read_tensor[tensor].append(einsum)
                    else:
                        self._einsums_that_read_tensor[tensor] = [einsum]

                    if einsum.name in self._tensors_read_by_einsum:
                        self._tensors_read_by_einsum[einsum.name].add(tensor)
                    else:
                        self._tensors_read_by_einsum[einsum.name] = {tensor}


    def einsums_with_tensor(self, tensor: Tensor) -> set["Einsum"]:
        return (
            self._einsums_that_read_tensor[tensor]
            +
            self._einsums_that_write_tensor[tensor]
        )

    def tensors_read_by_einsum(self, einsum_name: str) -> set[Tensor]:
        return self._tensors_read_by_einsum[einsum_name]

    def tensors_written_by_einsum(self, einsum_name: str) -> set[Tensor]:
        return self._tensors_written_by_einsum[einsum_name]

    def einsums_that_read_tensor(self, tensor: Tensor) -> set["Einsum"]:
        return self._einsums_that_read_tensor[tensor]

    def einsums_that_write_tensor(self, tensor: Tensor) -> set["Einsum"]:
        return self._einsums_that_write_tensor[tensor]

    def get_shape_isl_string(self, einsum_name: str) -> str:
        einsum = self.einsums[einsum_name]
        einsum_shape = einsum.shape
        global_shape = [self.shape[r] for r in einsum.rank_variables if r in self.shape]
        return " and ".join(einsum_shape + global_shape)

    @property
    def tensors(self) -> set["TensorAccess"]:
        return self._tensors

    def mermaid_graph(self) -> str:
        """
        Get the mermaid graph of the workload.
        Returns a Mermaid graph string showing relationships between Einsums and Tensors.
        Einsums are shown as rectangles and Tensors as circles.
        """
        import mermaid as md
        from mermaid.graph import Graph
        lines = [
            "graph LR",
            "linkStyle default interpolate basis"
        ]
        
        # Add all tensors as nodes (circles)
        tensors = []
        seen_tensor_names = set()
        for einsum in self.einsums:
            lines.append(f"\tEinsum_{einsum.name}[\"<b>{einsum.name}</b>\n<small>{einsum.to_formatted_string(compress=True)}</small>\"]")
            for tensor in einsum.tensor_accesses:
                if tensor.name not in seen_tensor_names:
                    tensors.append(tensor)
                    seen_tensor_names.add(tensor.name)
                    lines.append(f"\tTensor_{tensor.name}{{{{\"<b>{tensor.name}</b>\n\"}}}}")
        
        # Add all einsums as nodes (rectangles)
        for einsum in self.einsums:
            # Add edges from tensors to einsums
            for tensor in einsum.tensor_accesses:
                if tensor.output:
                    # Output tensor: einsum -> tensor
                    lines.append(f"\tEinsum_{einsum.name} --> Tensor_{tensor.name}")
                else:
                    # Input tensor: tensor -> einsum
                    lines.append(f"\tTensor_{tensor.name} --> Einsum_{einsum.name}")
        
        # Create the graph with the flowchart script
        flowchart_script = "\n".join(lines)
        graph = Graph('Flowchart', flowchart_script)
        
        # Set the configuration to ignore node order
        config = md.Config()
        graph.config = config

        return md.Mermaid(graph)


class EinsumList(ListNode):
    """
    A list of einsums in the workload.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", Einsum)


class Einsum(DictNode):
    """
    An einsum object in the workload.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("name", str)
        super().add_attr("tensor_accesses", TensorAccessList)
        super().add_attr("shape", Shape, [], callfunc=shape_factory)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name: str = self["name"]
        self.tensor_accesses: TensorAccessList = self["tensor_accesses"]
        self.shape: Shape = self["shape"]
        
    @property
    def rank_variables(self) -> set[str]:
        if not self.tensor_accesses:
            return set()
        return set.union(*[t.rank_variables for t in self.tensor_accesses])
    
    @property
    def tensor_names(self) -> set[str]:
        return set([t.name for t in self.tensor_accesses])
    
    def to_formatted_string(self, compress: bool = False) -> str:
        lhs_join = ",\n" if compress else " , "
        rhs_join = "#215;\n" if compress else " #215; "
        lhs = lhs_join.join([t.to_formatted_string() for t in self.tensor_accesses if t.output])
        rhs = rhs_join.join([t.to_formatted_string() for t in self.tensor_accesses if not t.output])
        return f"{lhs}=\n{rhs}" if compress else f"{lhs} = {rhs}"


class TensorAccessList(ListNode):
    """
    A list of tensors in the workload.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", TensorAccess)

class TensorAccess(DictNode):
    """
    A tensor object.

    Attributes:
        name (str): The name of the data space.
        projection (list): The projection of the data space.
        read_write (str, bool, int): The read-write attribute of the data space.
        factors (list): The factors derived from the projection.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("name", str)
        super().add_attr("projection", dict, callfunc=projection_factory)
        super().add_attr("output", (str, bool, int), False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.name: str = self["name"]
        self.projection: list = self["projection"]
        self.factors: list = []

        projection = [x for x in self.projection]
        while projection:
            factor = projection.pop(0)
            if isinstance(factor, list):
                projection += factor
            else:
                self.factors.append(factor)
                
    def to_formatted_string(self) -> str:
        subscript = ",".join(self.projection.values())
        if isinstance(self.projection, ImpliedProjection):
            return f"{self.name}<sub>{subscript}</sub>"
        
        string = [self.name]
        for k, v in self.projection.items():
            if len(string) < len(self.projection):
                string.append(f"<sup>{k},</sup><sub>{v},</sub>")
            else:
                string.append(f"<sup>{k}</sup><sub>{v}</sub>")
        return "".join(string)

    @property
    def ranks(self):
        return list(self.projection.keys())

    @property
    def rank_variables(self):
        # Projection values may be expressions, so we need to grab all identifiers
        return set(re.findall(ISL_REGEX, " ".join(self.projection.values())))

    # def __eq__(self, other):
    #     return self.name == other.name

    # def __hash__(self):
    #     return hash(self.name)


class ImpliedProjection(dict):
    pass

def projection_factory(projection: dict | list):
    if isinstance(projection, list):
        for i, x in enumerate(projection):
            if not isinstance(x, str):
                raise TypeError(f"Element at index {i} must be a string, got {type(x)}")
            if not ISL_REGEX.match(x):
                raise ValueError(
                    f"Element '{x}' at index {i} is not a valid ISL identifier"
                    f"In a projection list, all elements must be valid ISL identifiers."
                    f"For expressions, use a dictionary projection."
                )
        projection = ImpliedProjection({x.upper(): x for x in projection})
    elif not isinstance(projection, dict):
        raise TypeError(
            f"Invalid projection: {projection}. Must be a list of "
            f"rank variables or a dictionary of rank variable to projection."
        )
    for key in projection:
        if not isinstance(key, str):
            raise TypeError(f"Invalid projection key: {key}. Must be a string.")
        if not key.isidentifier():
            raise ValueError(
                f"Invalid projection key: {key}. Must be a valid identifier."
                f"Check with the Python isidentifier() function."
            )
    return projection

class ShapeDict(DictNode):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", Shape, part_name_match=True, no_change_key=True, callfunc=shape_factory)


def shape_factory(shape: list | str):
    if isinstance(shape, str):
        shape = [shape]
    return Shape(shape)

class Shape(ListNode):
    """
    A shape object in the workload.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", str)

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], str):
            args = [args[0]]
        super().__init__(*args, **kwargs)

    @property
    def rank_variables(self) -> set[str]:
        if not self:
            return set()
        return set.union(*[set(re.findall(ISL_REGEX, x)) for x in self])
