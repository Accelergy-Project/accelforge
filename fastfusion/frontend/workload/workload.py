import re

from pydantic_core import CoreSchema
from fastfusion.util.basetypes import ParsableDict, ParsableList, ParsableModel
from fastfusion.util.setexpressions import InvertibleSet, eval_set_expression
from fastfusion.frontend.renames import Renames
from fastfusion.version import assert_version, __version__
from typing import Annotated, Any, Callable
from pydantic_core.core_schema import CoreSchema, chain_schema, no_info_plain_validator_function


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


class CompareByName(ParsableModel):
    name: str

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and "name" not in kwargs:
            kwargs["name"] = args[0]
            super().__init__(**kwargs)
        else:
            super().__init__(*args, **kwargs)
        
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return f"{self.name}"
    
    def __eq__(self, other):
        return self.name == getattr(other, "name", other)
    
    def __hash__(self):
        return hash(self.name)
    
    def __lt__(self, other):
        return self.name < other.name
    
    def __le__(self, other):
        return self.name <= other.name
    
    def __ge__(self, other):
        return self.name >= other.name
    
    def __gt__(self, other):
        return self.name > other.name
    
    def __ne__(self, other):
        return self.name != other.name

    # @classmethod
    # def __get_pydantic_core_schema__(cls, source_type: Any, handler: Callable) -> CoreSchema:
    #     # Create a schema that validates strings and converts them to CompareByName instances
    #     # or accepts existing CompareByName instances
    #     return no_info_plain_validator_function(
    #         lambda x: x if isinstance(x, cls) else cls(str(x))
    #     )

class Tensor(CompareByName):
    pass

class RankVariable(CompareByName):
    pass
    
class Rank(CompareByName):
    pass

class TensorAccess(ParsableModel):
    name: str
    projection: dict[str, str]
    output: bool = False
    factors: list = []

    def __init__(self, **data):
        if "projection" in data:
            data["projection"] = projection_factory(data["projection"])
        super().__init__(**data)
        
        projection = [x for x in self.projection.values()]
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
    def ranks(self) -> tuple[Rank, ...]:
        return tuple(Rank(x) for x in self.projection.keys())

    @property
    def rank_variables(self) -> set[RankVariable]:
        # Projection values may be expressions, so we need to grab all identifiers
        return set(RankVariable(x) for x in re.findall(ISL_REGEX, " ".join(self.projection.values())))

    @property
    def relevant_rank_variables(self) -> set[RankVariable]:
        return self.rank_variables

    @property
    def fully_relevant_rank_variables(self) -> set[RankVariable]:
        return set(RankVariable(x) for x in self.projection.values() if ISL_REGEX.match(x))

    @property
    def partially_relevant_rank_variables(self) -> set[RankVariable]:
        return self.rank_variables - self.fully_relevant_rank_variables

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

def shape_factory(shape: list | str):
    if isinstance(shape, str):
        shape = [shape]
    return Shape(shape)

class Shape(ParsableList):
    @property
    def rank_variables(self) -> set[str]:
        if not self:
            return set()
        return set.union(*[set(re.findall(ISL_REGEX, x)) for x in self])

class Einsum(ParsableModel):
    name: str
    tensor_accesses: ParsableList[TensorAccess]
    shape: Shape[str] = Shape()

    @property
    def rank_variables(self) -> set[RankVariable]:
        if not self.tensor_accesses:
            return set()
        return set.union(*[t.rank_variables for t in self.tensor_accesses])
    
    @property
    def tensor_names(self) -> set[str]:
        return set([t.name for t in self.tensor_accesses])
    
    @property
    def tensors(self) -> set[Tensor]:
        return set([Tensor(t.name) for t in self.tensor_accesses])
    
    @property
    def tensor2rank_variables(self) -> dict[str, set[RankVariable]]:
        return {t.name: t.rank_variables for t in self.tensor_accesses}
    
    def to_formatted_string(self, compress: bool = False) -> str:
        lhs_join = ",\n" if compress else " , "
        rhs_join = "#215;\n" if compress else " #215; "
        lhs = lhs_join.join([t.to_formatted_string() for t in self.tensor_accesses if t.output])
        rhs = rhs_join.join([t.to_formatted_string() for t in self.tensor_accesses if not t.output])
        return f"{lhs}=\n{rhs}" if compress else f"{lhs} = {rhs}"
    
    def input_tensors(self) -> set[Tensor]:
        return {Tensor(t.name) for t in self.tensor_accesses if not t.output}
    
    def output_tensors(self) -> set[Tensor]:
        return {Tensor(t.name) for t in self.tensor_accesses if t.output}

class Workload(ParsableModel):
    version: Annotated[str, assert_version] = __version__
    einsums: ParsableList[Einsum] = []
    shape: ParsableDict[str, str] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._validate()

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

    def einsums_with_tensor(self, tensor: Tensor) -> list["Einsum"]:
        return [e for e in self.einsums if tensor.name in e.tensor_names]

    def tensors_read_by_einsum(self, einsum_name: str) -> set[Tensor]:
        return self.einsums[einsum_name].input_tensors()

    def tensors_written_by_einsum(self, einsum_name: str) -> set[Tensor]:
        return self.einsums[einsum_name].output_tensors()

    def einsums_that_read_tensor(self, tensor: Tensor) -> list["Einsum"]:
        return [e for e in self.einsums if tensor.name in e.input_tensors()]

    def einsums_that_write_tensor(self, tensor: Tensor) -> list["Einsum"]:
        return [e for e in self.einsums if tensor.name in e.output_tensors()]

    def get_shape_isl_string(self, einsum_name: str) -> str:
        einsum = self.einsums[einsum_name]
        einsum_shape = einsum.shape
        global_shape = [self.shape[r] for r in einsum.rank_variables if r in self.shape]
        return " and ".join(term for term in einsum_shape + global_shape)

    @property
    def tensors(self) -> set[Tensor]:
        return {Tensor(t.name) for e in self.einsums for t in e.tensor_accesses}

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

    def get_constraint_symbol_table(
            self, 
            einsum_name: str,
            renames: Renames | None = None,
        ) -> dict[str, InvertibleSet]:
        einsum = self.einsums[einsum_name]
        inputs = einsum.input_tensors()
        outputs = einsum.output_tensors()
        all_ = inputs | outputs
        intermediates = {t for t in all_ if self.einsums_that_read_tensor(t) and self.einsums_that_write_tensor(t)}
        shared = {
            t for t in all_ if len(set(e.name for e in self.einsums_that_read_tensor(t)) | set(e.name for e in self.einsums_that_write_tensor(t))) > 1
        }

        element_to_child_space = {}
        all_rank_variables = einsum.rank_variables
        for tensor in all_:
            rank_variables = einsum.tensor_accesses[tensor.name].rank_variables
            element_to_child_space[tensor.name] = InvertibleSet(
                instance=rank_variables,
                full_space=all_rank_variables,
                space_name=f"rank_variables",
            )
                
        kwargs = dict(
            full_space=all_,
            space_name=f"tensors",
            child_access_name="rank_variables",
            element_to_child_space=element_to_child_space,
        )
        symbol_table = {
            "All": InvertibleSet(instance=all_, **kwargs),
            "Inputs": InvertibleSet(instance=inputs, **kwargs),
            "Outputs": InvertibleSet(instance=outputs, **kwargs),
            "Intermediates": InvertibleSet(instance=intermediates, **kwargs),
            "Shared": InvertibleSet(instance=shared, **kwargs),
            **{t.name: InvertibleSet(instance=(t,), **kwargs) for t in all_},
        }
        
        if renames is not None:
            rename = renames.get_renames_for_einsum(einsum_name)
            for rename_tensor in rename.tensor_accesses:
                einsum.output_tensors()
                name = rename_tensor.name
                source = rename_tensor.source
                injective = rename_tensor.injective
                symbol_table[name] = eval_set_expression(source, symbol_table, "tensors", injective=injective)
                
        for rank_variable in einsum.rank_variables:
            symbol_table[rank_variable] = InvertibleSet(instance=(rank_variable,), space_name="rank_variables", full_space=einsum.rank_variables)
                
        return symbol_table