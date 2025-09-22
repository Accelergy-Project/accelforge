import re
from itertools import product

from fastfusion.util.basetypes import ParsableDict, ParsableList, ParsableModel
from fastfusion.util.parse_expressions import ParseError
from fastfusion.util.setexpressions import InvertibleSet, eval_set_expression
from fastfusion.version import assert_version, __version__
from typing import Annotated, TypeAlias, Union
from fastfusion.frontend.renames import EinsumName, RankVariableName, Rename, RenameList, Renames, TensorName, RankName, rename_list_factory


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


SymbolTable: TypeAlias = dict[str, InvertibleSet]


class TensorAccess(ParsableModel):
    name: TensorName
    projection: dict[str, str] | list[str]
    output: bool = False
    factors: list = []

    def model_post_init(self, __context__=None) -> None:
        self.projection = projection_factory(self.projection)

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
    def rank2rank_variables(self) -> dict[RankName, set[RankVariableName]]:
        return {
            RankName(rank): set(RankVariableName(rank_var)
                                for rank_var in re.findall(ISL_REGEX, projection))
            for rank, projection in self.projection.items()
        }

    @property
    def rank_variable2ranks(self) -> dict[RankVariableName, set[RankName]]:
        result = {}
        for rank, projection in self.projection.items():
            for rank_var in re.findall(ISL_REGEX, projection):
                rank_set: set = result.setdefault(rank_var, set())
                rank_set.add(rank)
        return result

    @property
    def ranks(self) -> tuple[RankName, ...]:
        return tuple(RankName(x) for x in self.projection.keys())

    @property
    def rank_variables(self) -> set[RankVariableName]:
        # Projection values may be expressions, so we need to grab all identifiers
        return set(RankVariableName(x) for x in re.findall(ISL_REGEX, " ".join(self.projection.values())))

    @property
    def relevant_rank_variables(self) -> set[RankVariableName]:
        return self.rank_variables

    @property
    def fully_relevant_rank_variables(self) -> set[RankVariableName]:
        return set(RankVariableName(x) for x in self.projection.values() if ISL_REGEX.match(x))

    @property
    def partially_relevant_rank_variables(self) -> set[RankVariableName]:
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
    name: EinsumName
    tensor_accesses: ParsableList[TensorAccess]
    shape: Shape[str] = Shape()
    is_copy_operation: bool = False
    renames: RenameList[Rename] = RenameList()

    def __init__(self, *args, **kwargs):
        if "renames" in kwargs:
            kwargs["renames"] = rename_list_factory(kwargs["renames"])
        super().__init__(*args, **kwargs)

    @property
    def rank_variables(self) -> set[RankVariableName]:
        if not self.tensor_accesses:
            return set()
        return set.union(*[t.rank_variables for t in self.tensor_accesses])
    
    @property
    def tensor_names(self) -> set[TensorName]:
        return set([TensorName(t.name) for t in self.tensor_accesses])
    
    @property
    def tensors(self) -> set[TensorName]:
        return set([TensorName(t.name) for t in self.tensor_accesses])
    
    @property
    def tensor2rank_variables(self) -> dict[TensorName, set[RankVariableName]]:
        return {TensorName(t.name): t.rank_variables for t in self.tensor_accesses}

    @property
    def tensor2fully_relevant_rank_variables(self) -> dict[TensorName,
                                                           set[RankVariableName]]:
        return {TensorName(t.name): t.fully_relevant_rank_variables
                for t in self.tensor_accesses}

    @property
    def tensor2partially_relevant_rank_variables(self) -> dict[TensorName,
                                                               set[RankVariableName]]:
        fully_relevant_rank_vars = self.tensor2fully_relevant_rank_variables
        return {
            TensorName(t.name): t.rank_variables-fully_relevant_rank_vars[t.name]
            for t in self.tensor_accesses
        }

    @property
    def tensor2irrelevant_rank_variables(self) -> dict[TensorName,
                                                        set[RankVariableName]]:
        partially_relevant = self.tensor2partially_relevant_rank_variables
        fully_relevant = self.tensor2fully_relevant_rank_variables
        rank_variables = self.rank_variables
        return {TensorName(t.name): rank_variables - fully_relevant[t.name] - partially_relevant[t.name]
                for t in self.tensor_accesses}

    def to_formatted_string(self, compress: bool = False) -> str:
        lhs_join = ",\n" if compress else " , "
        rhs_join = "#215;\n" if compress else " #215; "
        lhs = lhs_join.join([t.to_formatted_string() for t in self.tensor_accesses if t.output])
        rhs = rhs_join.join([t.to_formatted_string() for t in self.tensor_accesses if not t.output])
        return f"{lhs}=\n{rhs}" if compress else f"{lhs} = {rhs}"

    def input_tensors(self) -> set[TensorName]:
        return {TensorName(t.name) for t in self.tensor_accesses if not t.output}

    def output_tensors(self) -> set[TensorName]:
        return {TensorName(t.name) for t in self.tensor_accesses if t.output}

    def copy_source_tensor(self) -> TensorName | None:
        if not self.is_copy_operation:
            return None
        input_tensors = self.input_tensors()
        if len(input_tensors) != 1:
            raise ValueError(f"Copy Einsum {self.name} has {len(input_tensors)} input tensors, expected 1")
        return input_tensors.pop()

    @property
    def rank_variable2ranks(self) -> dict[RankVariableName, set[RankName]]:
        result: dict[RankVariableName, set[RankName]] = {}
        for tensor_access in self.tensor_accesses:
            new = tensor_access.rank_variable2ranks
            for rank_var, ranks in new.items():
                result.setdefault(rank_var, set()).update(ranks)
        return result

class Workload(ParsableModel):
    version: Annotated[str, assert_version] = __version__
    einsums: ParsableList[Einsum] = []
    shape: ParsableDict[RankVariableName, str] = {}

    def model_post_init(self, __context__=None) -> None:
        self._validate()

    def _validate(self):
        tensor2ranks = {}
        einsum_names = set()
        for einsum in self.einsums:
            if einsum.name in einsum_names:
                raise ValueError(f"Einsum name {einsum.name} is not unique")
            einsum_names.add(einsum.name)
            for tensor_accesses in einsum.tensor_accesses:
                tensor2ranks.setdefault(tensor_accesses.name, tensor_accesses.ranks)
                if tensor2ranks[tensor_accesses.name] != tensor_accesses.ranks:
                    raise ValueError(
                        f"TensorName {tensor_accesses.name} has inconsistent ranks. Found "
                        f"{tensor2ranks[tensor_accesses.name]} and {tensor_accesses.ranks}. "
                        f"TensorName is in Einsums "
                        f"{', '.join(e.name for e in self.einsums_with_tensor(tensor_accesses.name))}"
                    )

    @property
    def einsum_names(self) -> list[EinsumName]:
        return [EinsumName(e.name) for e in self.einsums]

    def einsums_with_tensor(self, tensor: TensorName) -> list["Einsum"]:
        return [e for e in self.einsums if tensor in e.tensor_names]

    def tensors_read_by_einsum(self, einsum_name: str) -> set[TensorName]:
        return self.einsums[einsum_name].input_tensors()

    def tensors_written_by_einsum(self, einsum_name: str) -> set[TensorName]:
        return self.einsums[einsum_name].output_tensors()

    def einsums_that_read_tensor(self, tensor: TensorName) -> list["Einsum"]:
        return [e for e in self.einsums if tensor in e.input_tensors()]

    def einsums_that_write_tensor(self, tensor: TensorName) -> list["Einsum"]:
        return [e for e in self.einsums if tensor in e.output_tensors()]
    
    def accesses_for_tensor(self, tensor: TensorName) -> list[TensorAccess]:
        return [t for e in self.einsums for t in e.tensor_accesses if t.name == tensor]

    def get_shape_isl_string(self, einsum_name: str) -> str:
        einsum = self.einsums[einsum_name]
        einsum_shape = einsum.shape
        global_shape = [self.shape[r] for r in einsum.rank_variables if r in self.shape]
        return " and ".join(term for term in einsum_shape + global_shape)
    
    @property
    def intermediate_tensor_names(self) -> set[TensorName]:
        return {t for t in self.tensor_names if self.einsums_that_read_tensor(t) and self.einsums_that_write_tensor(t)}

    @property
    def tensor_names(self) -> set[TensorName]:
        return {TensorName(t.name) for e in self.einsums for t in e.tensor_accesses}

    # def render(self) -> str:
    #     import mermaid as md
    #     from mermaid.graph import Graph
    #     lines = [
    #         "graph LR",
    #         "linkStyle default interpolate basis"
    #     ]
        
    #     # Add all tensors as nodes (circles)
    #     tensors = []
    #     seen_tensor_names = set()
    #     for einsum in self.einsums:
    #         lines.append(f"\tEinsum_{einsum.name}[\"<b>{einsum.name}</b>\n<small>{einsum.to_formatted_string(compress=True)}</small>\"]")
    #         for tensor_access in einsum.tensor_accesses:
    #             if tensor_access.name not in seen_tensor_names:
    #                 tensors.append(tensor)
    #                 seen_tensor_names.add(tensor_access.name)
    #                 lines.append(f"\tTensor_{tensor_access.name}{{{{\"<b>{tensor_access.name}</b>\n\"}}}}")
        
    #     # Add all einsums as nodes (rectangles)
    #     for einsum in self.einsums:
    #         # Add edges from tensors to einsums
    #         for tensor_access in einsum.tensor_accesses:
    #             if tensor_access.output:
    #                 # Output tensor: einsum -> tensor
    #                 lines.append(f"\tEinsum_{einsum.name} --> Tensor_{tensor_access.name}")
    #             else:
    #                 # Input tensor: tensor -> einsum
    #                 lines.append(f"\tTensor_{tensor_access.name} --> Einsum_{einsum.name}")
        
    #     # Create the graph with the flowchart script
    #     flowchart_script = "\n".join(lines)
    #     graph = Graph('Flowchart', flowchart_script)
        
    #     # Set the configuration to ignore node order
    #     config = md.Config()
    #     graph.config = config

    #     return md.Mermaid(graph)

    def get_constraint_symbol_table(
        self,
        einsum_name: EinsumName,
        renames: Union[Renames, None] = None,
    ) -> SymbolTable:
        """
        Return a table that maps symbols (e.g., Nothing, All, Inputs) to
        tensors or rank variables.
        """
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
        for tensor in self.tensor_names:
            if tensor in all_:
                rank_variables = einsum.tensor_accesses[tensor].rank_variables
            else:
                rank_variables = set()
            element_to_child_space[tensor] = InvertibleSet(
                instance=rank_variables,
                full_space=all_rank_variables,
                space_name=f"rank_variables",
            )
                
        kwargs_tensors = dict(
            full_space=all_,
            space_name=f"tensors",
            child_access_name="rank_variables",
            element_to_child_space=element_to_child_space,
        )
        kwargs_rank_variables = dict(
            full_space=all_rank_variables,
            space_name=f"rank_variables",
        )

        symbol_table = {
            "Nothing": InvertibleSet(instance=(), **kwargs_tensors),
            "Tensors": InvertibleSet(instance=all_, **kwargs_tensors),
            "All": InvertibleSet(instance=all_, **kwargs_tensors),
            "Inputs": InvertibleSet(instance=inputs, **kwargs_tensors),
            "Outputs": InvertibleSet(instance=outputs, **kwargs_tensors),
            "Intermediates": InvertibleSet(instance=intermediates, **kwargs_tensors),
            "Shared": InvertibleSet(instance=shared, **kwargs_tensors),
            **{t: InvertibleSet(instance=(t,), **kwargs_tensors) for t in all_},
            **{r: InvertibleSet(instance=(r,), **kwargs_rank_variables) for r in all_rank_variables},
            "Einsum": einsum_name,
            "Einsum_Object": einsum,
        }
        
        taken_renames = set()
        for rename in self.einsums[einsum_name].renames:
            symbol_table[rename.name] = eval_set_expression(
                rename.source,
                symbol_table,
                None,
                f"Einsum {einsum_name} renames",
                rename.expected_count,
            )
            taken_renames.add(rename.name)

        if renames is not None:
            rename = renames.get_renames_for_einsum(einsum_name)
            for rename_tensor in rename.tensor_accesses:
                if (name := rename_tensor.name) in taken_renames:
                    continue
                source = rename_tensor.source
                expected_count = rename_tensor.expected_count
                try:
                    symbol_table[name] = eval_set_expression(source, symbol_table, "tensors", f"{name} tensor renames", expected_count=expected_count)
                except ParseError as e:
                    e.add_field(einsum_name)
                    raise
            for rename_rank_variable in rename.rank_variables:
                if (name := rename_rank_variable.name) in taken_renames:
                    continue
                source = rename_rank_variable.source
                expected_count = rename_rank_variable.expected_count
                try:
                    symbol_table[name] = eval_set_expression(source, symbol_table, "rank_variables", f"{name} rank variable renames", expected_count=expected_count)
                except ParseError as e:
                    e.add_field(einsum_name)
                    raise

        for rank_variable in einsum.rank_variables:
            symbol_table[rank_variable] = InvertibleSet(instance=(rank_variable,), space_name="rank_variables", full_space=einsum.rank_variables)

        for t in self.tensor_names:
            if t not in symbol_table:
                symbol_table[t] = InvertibleSet(
                    instance=(),
                    space_name="tensors",
                    full_space=all_,
                    child_access_name="rank_variables",
                    element_to_child_space=element_to_child_space,
                )

        return symbol_table

    def get_mixable_ranks(self) -> dict[RankName, set[RankName]]:
        rank2rankvars = {}
        for tensor in self.tensor_names:
            for acc in self.accesses_for_tensor(tensor):
                for rank, rank_vars in acc.rank2rank_variables.items():
                    rank2rankvars.setdefault(rank, set()).update(rank_vars)

        rank_var_to_ranks = {}
        for rank, rank_vars in rank2rankvars.items():
            for rank_var in rank_vars:
                rank_var_to_ranks.setdefault(rank_var, set()).add(rank)

        rank_to_ranks = {r: set((r,)) for r in rank2rankvars.keys()}
        update_with = list(rank_var_to_ranks.values())
        changed = True
        while changed:
            changed = False
            for ranks in rank_to_ranks.values():
                for u in update_with:
                    if u & ranks:
                        changed = changed or (u - ranks)
                        ranks.update(u)

        return rank_to_ranks

    def get_tensor_copies(self) -> dict[TensorName, set[TensorName]]:
        tensor_copies = {}
        for einsum in self.einsums:
            if not einsum.is_copy_operation:
                continue
            input_tensor = einsum.copy_source_tensor()
            for output_tensor in einsum.output_tensors():
                tensor_copies.setdefault(input_tensor, set()).add(output_tensor)
                tensor_copies.setdefault(output_tensor, set()).add(input_tensor)
        return tensor_copies