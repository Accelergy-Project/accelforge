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

ISL_REGEX = re.compile(r'\b(?!(?:' + '|'.join(CLIST_OPERATORS) + r')\b)[a-zA-Z#$@][a-zA-Z0-9_]*\b')

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
        super().add_attr("shape", Shape, [])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.version: str = self["version"]
        self.shape: Shape = self["shape"]
        self.einsums: EinsumList = self["einsums"]

    def tensors_read_by_einsum(self, einsum_name: str) -> set["Tensor"]:
        """
        Get the tensors read by the given einsum.
        """
        return {t for t in self.einsums[einsum_name].tensors if not t.output}

    def tensors_written_by_einsum(self, einsum_name: str) -> set["Tensor"]:
        """
        Get the tensors written by the given einsum.
        """
        return {t for t in self.einsums[einsum_name].tensors if t.output}

    def einsums_that_read_tensor(self, tensor: "Tensor") -> set["Einsum"]:
        """
        Get the einsums that read the given tensor.
        """
        # Avoid nested loop by checking each einsum's tensors directly
        return {e for e in self.einsums if tensor in e.tensors and not tensor.output}

    def einsums_that_write_tensor(self, tensor: "Tensor") -> set["Einsum"]:
        """
        Get the einsums that write to the given tensor.
        """
        return {e for e in self.einsums if tensor in e.tensors and tensor.output}
    
    def get_shape_isl_string(self, einsum_name: str) -> str:
        """
        Get the shape of the given einsum as an ISL string.
        """
        einsum = self.einsums[einsum_name]
        einsum_shape = einsum.shape
        global_shape = [self.shape[r] for r in einsum.rank_variables if r in self.shape]
        return " and ".join(einsum_shape + global_shape)
    
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
        super().add_attr("rank_variables", list, [])
        super().add_attr("tensors", TensorList)
        super().add_attr("shape", Shape, [])
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name: str = self["name"]
        self.rank_variables: list = self["rank_variables"]
        self.tensors: TensorList = self["tensors"]
        self.shape: Shape = self["shape"]
        
class TensorList(ListNode):
    """
    A list of tensors in the workload.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", Tensor)
        
class Tensor(DictNode):
    """
    A tensor object in the workload.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("name", str)
        super().add_attr("shape", Shape)
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", Tensor)


class Tensor(DictNode):
    """
    A data space object.

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
        super().add_attr("read_write", (str, bool, int), False)

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

    @property
    def ranks(self):
        return list(self.projection.keys())
    
    @property
    def rank_variables(self):
        # Projection values may be expressions, so we need to grab all identifiers
        return set(re.findall(ISL_REGEX, " ".join(self.projection.values())))
    
    
    def __eq__(self, other):
        return self.name == other.name
    
    def __hash__(self):
        return hash(self.name)

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
        projection = {x: x.upper() for x in projection}
    elif not isinstance(projection, dict):
        raise TypeError(
            f"Invalid projection: {projection}. Must be a list of "
            f"rank variables or a dictionary of rank variable to projection."
        )
    for key in projection:
        if not isinstance(key, str):
            raise TypeError(
                f"Invalid projection key: {key}. Must be a string."
            )
        if not key.isidentifier():
            raise ValueError(
                f"Invalid projection key: {key}. Must be a valid identifier."
                f"Check with the Python isidentifier() function."
            )
    return projection

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