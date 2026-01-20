"""
All the objects used for a Workload description in FastFusion.
"""

from itertools import product
import itertools
import logging
import re
from typing import TypeAlias

import pydot

from fastfusion.util.parallel import _SVGJupyterRender

from fastfusion.util._basetypes import ParsableDict, ParsableList, ParsableModel
from fastfusion.util._visualization import _pydot_graph
from fastfusion.frontend.renames import (
    EinsumName,
    RankVariable,
    Rename,
    RenameList,
    Renames,
    TensorName,
    Rank,
    rename_list_factory,
)
from fastfusion.util._parse_expressions import ParseError, parse_expression
from fastfusion.util._setexpressions import InvertibleSet, eval_set_expression
from fastfusion._version import __version__

from fastfusion.frontend.renames import (
    EinsumName,
    RankVariable,
    Rename,
    RenameList,
    Renames,
    TensorName,
    Rank,
    rename_list_factory,
)


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

_ISL_REGEX = re.compile(
    r"\b(?!(?:" + "|".join(CLIST_OPERATORS) + r")\b)[a-zA-Z#$@][a-zA-Z0-9_]*\b"
)
"""
Pattern[AnyStr@compile] _ISL_REGEX: A compiled regex pattern that matches
words that are not exactly in CLIST_OPERATORS (case-sensitive), start with a
letter, `#`, `$`, or `@`, and are followed by zero or more letters, digits,
or underscores.
"""


def isl_expression_has_variable(expression: str, variable: RankVariable) -> bool:
    """
    Returns True if the given ISL expression has the given rank variable.

    Parameters
    ----------
    expression : str
        The ISL expression to check.
    variable : RankVariable
        The rank variable to check for.

    Returns
    -------
    bool
        True if the given ISL expression has the given rank variable.
    """
    return variable in re.findall(_ISL_REGEX, expression)


SymbolTable: TypeAlias = dict[str, InvertibleSet]


class TensorAccess(ParsableModel):
    """Information about how an Einsum accesses a tensor."""

    name: TensorName
    """ The name of the tensor. """

    projection: dict[str, str] | list[str]
    """
    How the rank variables of the Einsum project into the tensor. If this is a list,
    then it is assumed that each of the elements of the list is a single rank variable
    and they index into the tensor in ranks that equal the uppercase of the rank
    variable. For example:

    name: X, projection: [a, b, c] means X[A=a, B=b, C=c]

    If this is a dictionary, it is a mapping from rank names to rank variable
    expressions. This can be used to either project into a non-matching rank name or to
    project into a tensor using an expression. For example:

    name: X, projection: {A: a, B2: b, C: a+b} means X[A=a, B2=b, C=a+b]
    """

    output: bool = False
    """ Whether the tensor is an output. False means the tensor is an input. """

    persistent: bool = False
    """ If True, then a copy of this tensor must remain in backing storage for the full
    duration of the workload's execution. """

    backing_storage_size_scale: float = 1.0
    """ If != 1, then the backing storage size will be scaled by this factor. """

    bits_per_value: int | str | None = None
    """ Bits per value for this tensor. """

    def model_post_init(self, __context__=None) -> None:
        self.projection: ImpliedProjection = _projection_factory(self.projection)

    def _to_formatted_string(self) -> str:
        """Returns a string representation of the tensor access for Pydot nodes."""
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
    def rank2rank_variables(self) -> dict[Rank, set[RankVariable]]:
        """
        Returns a dictionary of rank names to the rank variables that project into that
        rank.
        """
        return {
            Rank(rank): set(
                RankVariable(rank_var)
                for rank_var in re.findall(_ISL_REGEX, projection)
            )
            for rank, projection in self.projection.items()
        }

    @property
    def rank_variable2ranks(self) -> dict[RankVariable, set[Rank]]:
        """
        Returns a dictionary of rank variables to the ranks into which that rank
        variable projects.
        """
        result = {}
        for rank, projection in self.projection.items():
            for rank_var in re.findall(_ISL_REGEX, projection):
                rank_set: set = result.setdefault(rank_var, set())
                rank_set.add(rank)
        return result

    @property
    def ranks(self) -> tuple[Rank, ...]:
        """Returns the ranks of this access's tensor."""
        return tuple(Rank(x) for x in self.projection.keys())

    @property
    def rank_variables(self) -> set[RankVariable]:
        """Returns all rank variables used in this access."""
        # Projection values may be expressions, so we need to grab all identifiers
        return set(
            RankVariable(x)
            for x in re.findall(_ISL_REGEX, " ".join(self.projection.values()))
        )

    @property
    def directly_indexing_rank_variables(self) -> set[RankVariable]:
        """
        Returns the rank variables that directly index into this tensor without any
        expression (e.g., "M=m", NOT "M=m+n").
        """
        return set(
            RankVariable(x) for x in self.projection.values() if _ISL_REGEX.match(x)
        )

    @property
    def expression_indexing_rank_variables(self) -> set[RankVariable]:
        """
        Returns the rank variables that indirectly index into this tensor through an
        expression (e.g., "M=m+n") instead of a direct index (e.g., "M=m").
        """
        return self.rank_variables - self.directly_indexing_rank_variables


class ImpliedProjection(dict):
    """
    Holds a projection that has been implied by a list of rank variables. The implied
    rank names are uppercased versions of the rank variables; for example, [a, b, c] ->
    {A: a, B: b, C: c}.
    """


def _projection_factory(projection: dict | list):
    if isinstance(projection, list):
        for i, x in enumerate(projection):
            if not isinstance(x, str):
                raise TypeError(f"Element at index {i} must be a string, got {type(x)}")
            if not _ISL_REGEX.match(x):
                raise ValueError(
                    f"Element '{x}' at index {i} is not a valid ISL identifier"
                    f"In a projection list, all elements must be valid ISL identifiers."
                    f"For expressions, use a dictionary projection."
                )
        projection = ImpliedProjection({x.upper(): x for x in projection})
    elif not isinstance(projection, dict):
        raise TypeError(
            f"Invalid projection: {projection}. Must be a list of rank variables or a "
            f"dictionary of rank variable to projection."
        )
    for key in projection:
        if not isinstance(key, str):
            raise TypeError(f"Invalid projection key: {key}. Must be a string.")
        if not key.isidentifier():
            raise ValueError(
                f"Invalid projection key: {key}. Must be a valid identifier. Check with "
                f"the Python isidentifier() function."
            )
    return projection


class Shape(ParsableList):
    """
    Specifies valid values for the rank variables. This is a list of strings, each one
    an ISL expression. The total space is considered to be the logal AND of all the
    expressions in the list.
    """

    @property
    def rank_variables(self) -> set[str]:
        """Returns all rank variables used in this shape."""
        if not self:
            return set()
        return set.union(*[set(re.findall(_ISL_REGEX, x)) for x in self])


class Einsum(ParsableModel):
    """
    Represents an Einsum, which is a single computation step in the workload. The Einsum
    includes a set of rank variables, which are used to index into tensors. Rank
    variables iterate through an iteration space.

    For example, if the Einsum is A[m, n] += B[k, n] * C[k, n] and we define the
    iteration space as "0 <= m < 10, 0 <= n < 10, 0 <= k < 10", then the Einsum will
    iterate through all possible values of (m, n, k) in the iteration space, indexing
    into tensors for each and updating A[m, n] with B[k, n] * C[k, n].
    """

    name: EinsumName
    """ The name of the Einsum. """
    tensor_accesses: ParsableList[TensorAccess]
    """ The tensors accessed by this Einsum, and how they are accessed. """
    iteration_space_shape: Shape[str] = Shape()
    """
    Bounds of valid rank variable values. This is a list of expressions, each one an ISL
    expression. Additionally, global iteration_space_shape expressions are appended to
    the list if their rank variables are present in the Einsum's rank_variables. For
    example, if the global scope has "m: 0 <= m < 10" and the Einsum has "m" in its
    rank_variables, then "0 <= m < 10" will be appended to the iteration_space_shape.
    """
    rank_sizes: ParsableDict[Rank, int] = ParsableDict()
    """
    Sizes of ranks. This is a dictionary of rank names to sizes. Sizes are integers, and
    the rank's bounds are 0 <= rank < size. Accesses outside of these bounds are
    skipped.
    """
    is_copy_operation: bool = False
    """ Whether the Einsum is a copy operation. Copy operations take the input tensor
    and directly place them at the location of the output tensor(s) without any
    computation. If the destination tensor is at the same location, then this is a
    no-op."""
    renames: RenameList[Rename] = RenameList()
    """ Renames of the Einsum. Renames here can be used to rename rank variables or
    tensors. When this Einsum is executed on an architecture, the architecture can use
    renamed tensors and rank variables to access the tensors and rank variables. """
    n_instances: int = 1
    """
    Number of times to repeat the Einsum. Multiplied by `Workload.n_instances` to get
    the total number of Einsum instances. Energy, latency, and other summable metrics
    are multiplied by this value. Persistent reservations are also multiplied by this
    value, but non-persistent reservations are not, as they are assumed to be freed
    between each instance.
    """

    def model_post_init(self, __context__=None) -> None:
        if self.name == "Total":
            raise ValueError(
                f'Einsum name "Total" is reserved for totaling across Einsums.'
                f"Use a different name for the Einsum."
            )

    def __init__(self, *args, **kwargs):
        if "renames" in kwargs:
            kwargs["renames"] = rename_list_factory(kwargs["renames"])
        super().__init__(*args, **kwargs)

    @property
    def rank_variables(self) -> set[RankVariable]:
        """Returns all rank variables used in this Einsum."""
        if not self.tensor_accesses:
            return set()
        return set.union(*[t.rank_variables for t in self.tensor_accesses])

    @property
    def ranks(self) -> set[Rank]:
        """Returns all ranks used in this Einsum."""
        if not self.tensor_accesses:
            return set()
        return set.union(*[set(t.ranks) for t in self.tensor_accesses])

    @property
    def input_tensor_names(self) -> set[TensorName]:
        """Returns the names of the input tensors of this Einsum."""
        return set([TensorName(t.name) for t in self.tensor_accesses if not t.output])

    @property
    def output_tensor_names(self) -> set[TensorName]:
        """Returns the names of the output tensors of this Einsum."""
        return set([TensorName(t.name) for t in self.tensor_accesses if t.output])

    @property
    def tensor_names(self) -> set[TensorName]:
        """Returns the names of all tensors of this Einsum."""
        return set([TensorName(t.name) for t in self.tensor_accesses])

    @property
    def tensor2rank_variables(self) -> dict[TensorName, set[RankVariable]]:
        """Returns a dictionary of tensor names to the rank variables that project into
        that tensor."""
        return {TensorName(t.name): t.rank_variables for t in self.tensor_accesses}

    @property
    def tensor2directly_indexing_rank_variables(
        self,
    ) -> dict[TensorName, set[RankVariable]]:
        """
        Returns a dictionary of tensor names to the rank variables that directly index
        into that tensor. Direct indexing means that the rank variable is used as a
        direct index into the tensor, without any expression (e.g., "M=m", NOT "M=m+n").
        """
        return {
            TensorName(t.name): t.directly_indexing_rank_variables
            for t in self.tensor_accesses
        }

    @property
    def tensor2expression_indexing_rank_variables(
        self,
    ) -> dict[TensorName, set[RankVariable]]:
        """
        Returns a dictionary of tensor names to the rank variables that indirectly index
        into that tensor through an expression (e.g., "M=m+n") instead of a direct index
        (e.g., "M=m").
        """
        fully_relevant_rank_vars = self.tensor2directly_indexing_rank_variables
        return {
            TensorName(t.name): t.rank_variables - fully_relevant_rank_vars[t.name]
            for t in self.tensor_accesses
        }

    @property
    def tensor2irrelevant_rank_variables(
        self,
    ) -> dict[TensorName, set[RankVariable]]:
        """
        Returns a dictionary of tensor names to the rank variables that are irrelevant
        to that tensor. Irrelevant rank variables are rank variables that are not used
        to index into the tensor.
        """
        partially_relevant = self.tensor2expression_indexing_rank_variables
        fully_relevant = self.tensor2directly_indexing_rank_variables
        rank_variables = self.rank_variables
        return {
            TensorName(t.name): rank_variables
            - fully_relevant[t.name]
            - partially_relevant[t.name]
            for t in self.tensor_accesses
        }

    def _to_formatted_string(self, compress: bool = False) -> str:
        """
        Returns a string representation of this Einsum for use in a Pydot graph.

        Parameters
        ----------
        compress : bool, optional
            If True, the string will be compressed to a single line.

        Returns
        -------
        str
            A string representation of this Einsum for use in a Pydot graph.
        """
        lhs_join = ",\n" if compress else " , "
        rhs_join = " \n " if compress else "  "
        lhs = lhs_join.join(
            [t._to_formatted_string() for t in self.tensor_accesses if t.output]
        )
        rhs = rhs_join.join(
            [t._to_formatted_string() for t in self.tensor_accesses if not t.output]
        )
        return f"{lhs}=\n{rhs}" if compress else f"{lhs} = {rhs}"

    def copy_source_tensor(self) -> TensorName | None:
        """
        If this Einsum is a copy operation, returns the name of the tensor that is the
        source of the copy. Otherwise, returns None.
        """
        if not self.is_copy_operation:
            return None
        input_tensors = self.input_tensor_names
        if len(input_tensors) != 1:
            raise ValueError(
                f"Copy Einsum {self.name} has {len(input_tensors)} input tensors, expected 1"
            )
        return input_tensors.pop()

    @property
    def rank_variable2ranks(self) -> dict[RankVariable, set[Rank]]:
        """
        Returns a dictionary of rank variables to the ranks that are indexed into by
        that rank variable.
        """
        result: dict[RankVariable, set[Rank]] = {}
        for tensor_access in self.tensor_accesses:
            new = tensor_access.rank_variable2ranks
            for rank_var, ranks in new.items():
                result.setdefault(rank_var, set()).update(ranks)
        return result

    @property
    def indexing_expressions(self) -> set[str]:
        """
        Returns a list of all the expressions that index into the tensors of this
        Einsum.
        """
        result = set()
        for tensor_access in self.tensor_accesses:
            for _, projection in tensor_access.projection.items():
                result.add(projection)
        return result


class Workload(ParsableModel):
    """
    The workload specification as a cascade of Einsums, with each Einsum being a
    computation step in the workload.
    """

    # version: Annotated[str, assert_version] = __version__
    # """ The version of the workload specification. """

    einsums: ParsableList[Einsum] = ParsableList()
    """ The Einsums in the workload. """

    iteration_space_shape: ParsableDict[RankVariable, str] = ParsableDict()
    """
    Bounds of valid rank variable values. This is a dictionary of rank variable
    names to bounds of valid rank variable values. The bounds are specified as a string
    in the ISL format. For example, "0 <= a < 10" means that the rank variable `a` must
    be between 0 and 10, including 0 but not 10. Bounds are included for all Einsums
    that include that rank variable.
    """

    rank_sizes: ParsableDict[Rank, int] = ParsableDict()
    """
    Rank sizes. This is a dictionary of rank names to sizes. Sizes are integers, and the
    rank's bounds are 0 <= rank < size. Accesses outside of these bounds are skipped.
    """

    n_instances: int = 1
    """
    Number of times to repeat the workload. Multiplied by `Einsum.n_instances` to get
    the total number of Einsum instances. Energy, latency, and other summable metrics
    are multiplied by this value. Persistent reservations are also multiplied by this
    value, but non-persistent reservations are not, as they are assumed to be freed
    between each instance.
    """

    bits_per_value: ParsableDict[str, int | str] = ParsableDict()
    """
    Bits per value for each tensor. The workload-level bits_per_value is overridden if
    bits_per_action is specified for any given tensor access. This is a dictionary of
    set expressions to bits per value for the tensors given by those expressions. For
    example, we may write "Inputs: 8" to set the bits per value to 8 for all input
    tensors, unless overridden.
    """

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
                        "TensorName is in Einsums "
                        f"{', '.join(
                            e.name for e in self.einsums_with_tensor(tensor_accesses.name)
                        )}"
                    )

    @property
    def einsum_names(self) -> list[EinsumName]:
        """Returns the names of the Einsums in the workload."""
        return [EinsumName(e.name) for e in self.einsums]

    def einsums_with_tensor(self, tensor: TensorName) -> list["Einsum"]:
        """
        Returns the Einsums in the workload that access the given tensor.

        Parameters
        ----------
        tensor : TensorName
            The tensor to check.

        Returns
        -------
        list[Einsum]
            The Einsums in the workload that access the given tensor. Order is the same
            as the order in this workload's Einsums list.
        """
        return [e for e in self.einsums if tensor in e.tensor_names]

    def einsums_with_tensor_as_input(self, tensor: TensorName) -> list["Einsum"]:
        """
        Returns the Einsums in the workload that use the given tensor as an input.

        Parameters
        ----------
        tensor : TensorName
            The tensor to check.

        Returns
        -------
        list[Einsum]
            The Einsums in the workload that use the given tensor as an input. Order is
            the same as the order in this workload's Einsums list.
        """
        return [e for e in self.einsums if tensor in e.input_tensor_names]

    def einsums_with_tensor_as_output(self, tensor: TensorName) -> list["Einsum"]:
        """
        Returns the Einsums in the workload that have the given tensor as an output.

        Parameters
        ----------
        tensor : TensorName
            The tensor to check.

        Returns
        -------
        list[Einsum]
            The Einsums in the workload that have the given tensor as an output. Order
            is the same as the order in this workload's Einsums list.
        """
        return [e for e in self.einsums if tensor in e.output_tensor_names]

    def accesses_for_tensor(self, tensor: TensorName) -> list[TensorAccess]:
        """
        Returns all TensorAccess objects that access the given tensor across all
        Einsums.

        Parameters
        ----------
        tensor : TensorName
            The tensor to check.

        Returns
        -------
        list[TensorAccess]
            The TensorAccess objects that access the given tensor across all Einsums.
            Order is the same as the order in this workload's Einsums list.
        """
        return [t for e in self.einsums for t in e.tensor_accesses if t.name == tensor]

    def get_iteration_space_shape_isl_string(self, einsum_name: str) -> str:
        """
        Returns the ISL string representing the iteration space of the given Einsum.

        Parameters
        ----------
        einsum_name : str
            The name of the Einsum for which to get the iteration space shape.

        Returns
        -------
        str
            The ISL string representing the iteration space shape of the given Einsum.
        """
        einsum = self.einsums[einsum_name]
        einsum_shape = einsum.iteration_space_shape
        my_ispace = self.iteration_space_shape
        global_shape = [my_ispace[r] for r in einsum.rank_variables if r in my_ispace]
        rank_sizes = einsum.rank_sizes
        global_rank_sizes = {
            r: self.rank_sizes[r] for r in einsum.ranks if r in self.rank_sizes
        }

        exprs = einsum_shape + global_shape
        for tensor in einsum.tensor_accesses:
            for rank, projection in tensor.projection.items():
                if rank in rank_sizes:
                    exprs.append(f"0 <= {projection} < {rank_sizes[rank]}")
                elif rank in global_rank_sizes:
                    exprs.append(f"0 <= {projection} < {global_rank_sizes[rank]}")

        return " and ".join(exprs)

    def _check_consistent_persistent(self):
        for tensor in self.tensor_names:
            persistents = {
                e.tensor_accesses[tensor].persistent
                for e in self.einsums_with_tensor(tensor)
            }
            if len(persistents) > 1:
                raise ValueError(
                    f"Tensor {tensor} is used in multiple Einsums with different "
                    f"persistent values. Persistent values must be consistent across "
                    f"all Einsums that use the tensor."
                )

    @property
    def tensor_names_used_in_multiple_einsums(self) -> set[TensorName]:
        """Returns the names of the tensors that are used in multiple Einsums."""
        self._check_consistent_persistent()
        return {t for t in self.tensor_names if len(self.einsums_with_tensor(t)) > 1}

    @property
    def tensor_names(self) -> set[TensorName]:
        """Returns the names of all tensors in the workload."""
        return {TensorName(t.name) for e in self.einsums for t in e.tensor_accesses}

    def _repr_svg_(self) -> str:
        return self.render()

    def render(self) -> str:
        """Renders the workload as a Pydot graph. Returns an SVG string."""
        graph = _pydot_graph()

        # Add all tensors as nodes (circles)
        tensors = []
        seen_tensor_names = set()
        for einsum in self.einsums:
            node = pydot.Node(
                f"Einsum_{einsum.name}",
                shape="box",
                label=f"<{einsum._to_formatted_string(compress=True)}>",
            )
            graph.add_node(node)
            for tensor_access in einsum.tensor_accesses:
                if tensor_access.name not in seen_tensor_names:
                    tensors.append(tensor_access.name)
                    seen_tensor_names.add(tensor_access.name)
                    node = pydot.Node(
                        f"Tensor_{tensor_access.name}",
                        shape="oval",
                        label=f"<{tensor_access._to_formatted_string()}>",
                    )
                    graph.add_node(node)

        # Add all einsums as nodes (rectangles)
        for einsum in self.einsums:
            # Add edges from tensors to einsums
            for tensor_access in einsum.tensor_accesses:
                if tensor_access.output:
                    # Output tensor: einsum -> tensor
                    edge = pydot.Edge(
                        f"Einsum_{einsum.name}", f"Tensor_{tensor_access.name}"
                    )
                    graph.add_edge(edge)
                else:
                    # Input tensor: tensor -> einsum
                    edge = pydot.Edge(
                        f"Tensor_{tensor_access.name}", f"Einsum_{einsum.name}"
                    )
                    graph.add_edge(edge)
        return _SVGJupyterRender(graph.create_svg(prog="dot").decode("utf-8"))

    def get_constraint_symbol_table(
        self,
        einsum_name: EinsumName | list[EinsumName],
        renames: Renames | None = None,
    ) -> SymbolTable | dict[EinsumName, SymbolTable]:
        """
        Return a table that maps symbols (e.g., Nothing, All, Inputs, X) to tensors or
        rank variables.

        Parameters
        ----------
        einsum_name : EinsumName | list[EinsumName]
            The name of the Einsum for which to get the constraint symbol table. If a
            list, a dictionary of all Einsum names in the list to symbol tables is
            returned.
        renames : Renames, optional
            The renames to apply to the symbol table. If None, no renames are applied.

        Returns
        -------
        SymbolTable | dict[EinsumName, SymbolTable]
            A table that maps symbols (e.g., Nothing, All, Inputs, X) to tensors or rank
            variables. If einsum_name is a list, a dictionary of all Einsum names in the
            list to symbol tables is returned.

        Raises
        ------
        ParseError
            If the renames are invalid or any parsing fails.
        """
        if isinstance(einsum_name, list):
            return {
                e: self.get_constraint_symbol_table(e, renames) for e in einsum_name
            }

        einsum = self.einsums[einsum_name]
        inputs = einsum.input_tensor_names
        outputs = einsum.output_tensor_names
        all_ = inputs | outputs
        intermediates = {
            t
            for t in all_
            if self.einsums_with_tensor_as_input(t)
            and self.einsums_with_tensor_as_output(t)
        }
        shared = {
            t
            for t in all_
            if len(
                set(e.name for e in self.einsums_with_tensor_as_input(t))
                | set(e.name for e in self.einsums_with_tensor_as_output(t))
            )
            > 1
        }
        persistent = {
            t for t in all_ if self.einsums[einsum_name].tensor_accesses[t].persistent
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
                space_name="rank_variables",
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
            "All": InvertibleSet(instance=all_, **kwargs_tensors),
            "Tensors": InvertibleSet(instance=all_, **kwargs_tensors),
            "Nothing": InvertibleSet(instance=(), **kwargs_tensors),
            "Inputs": InvertibleSet(instance=inputs, **kwargs_tensors),
            "Outputs": InvertibleSet(instance=outputs, **kwargs_tensors),
            "Intermediates": InvertibleSet(instance=intermediates, **kwargs_tensors),
            "Shared": InvertibleSet(instance=shared, **kwargs_tensors),
            "Persistent": InvertibleSet(instance=persistent, **kwargs_tensors),
            **{t: InvertibleSet(instance=(t,), **kwargs_tensors) for t in all_},
            **{
                r: InvertibleSet(instance=(r,), **kwargs_rank_variables)
                for r in all_rank_variables
            },
            "Einsum": einsum_name,
            "Above": InvertibleSet(instance=(), **kwargs_tensors),
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
                    symbol_table[name] = eval_set_expression(
                        source,
                        symbol_table,
                        "tensors",
                        f"{name} tensor renames",
                        expected_count=expected_count,
                    )
                except ParseError as e:
                    e.add_field(einsum_name)
                    raise
            for rename_rank_variable in rename.rank_variables:
                if (name := rename_rank_variable.name) in taken_renames:
                    continue
                source = rename_rank_variable.source
                expected_count = rename_rank_variable.expected_count
                try:
                    symbol_table[name] = eval_set_expression(
                        source,
                        symbol_table,
                        "rank_variables",
                        f"{name} rank variable renames",
                        expected_count=expected_count,
                    )
                except ParseError as e:
                    e.add_field(einsum_name)
                    raise

        for rank_variable in einsum.rank_variables:
            symbol_table[rank_variable] = InvertibleSet(
                instance=(rank_variable,),
                space_name="rank_variables",
                full_space=einsum.rank_variables,
            )

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

    def _get_ranks_that_share_indexing_rank_variables(self) -> dict[Rank, set[Rank]]:
        """
        Returns a dictionary of ranks to the ranks with which they share indexing rank
        variables. For example, if one einsum indexes into rank A with rank variable a
        and another einsum indexes into rank B with rank variable a, then A and B share
        the indexing rank variable a. Then we'd have in our return value both A: {A, B}
        and B: {A, B}. This is transitive and reflexive.

        Returns
        -------
        dict[Rank, set[Rank]]
            A dictionary of ranks to the ranks with which they share indexing rank
            variables. The ranks are the keys, and the values are sets of ranks that
            share indexing rank variables with the key.
        """
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
        """
        Returns a dictionary specifying which tensors are copies of which other tensors.
        For example, if einsum A copies tensor X into tensors Y and Z, then we'd have in
        the return value X: {Y, Z}, Y: {X, Z}, and Z: {X, Y}. This is transitive.

        Returns
        -------
        dict[TensorName, set[TensorName]]
            A dictionary specifying which tensors are copies of which other tensors. The
            keys are the tensors that are copies, and the values are sets of tensors
            that are copies of the key.
        """
        tensor_copies = {}
        for einsum in self.einsums:
            if not einsum.is_copy_operation:
                continue
            input_tensor = einsum.copy_source_tensor()
            for output_tensor in einsum.output_tensor_names:
                tensor_copies.setdefault(input_tensor, set()).add(output_tensor)
                tensor_copies.setdefault(output_tensor, set()).add(input_tensor)
        return tensor_copies

    def _get_bits_per_value(
        self,
        einsum2symbol_table: dict[EinsumName, SymbolTable],
    ) -> dict[TensorName, int]:
        """
        Returns a dictionary of tensor names to bits per value for the given Einsum.

        Parameters
        ----------
        einsum2symbol_table : dict[EinsumName, SymbolTable]
            A dictionary of Einsum names to symbol tables.

        Returns
        -------
        dict[TensorName, int]
            A dictionary of tensor names to bits per value for the given Einsum.
        """
        bits_per_value_per_einsum = {}
        bits_per_value = {}

        for einsum, symbol_table in einsum2symbol_table.items():
            cur_bpv = {}
            seen_tensors = set()

            # Parse the global bits per value expressions
            for set_expression, n_bits in self.bits_per_value.items():
                tensors = eval_set_expression(
                    set_expression,
                    symbol_table,
                    "tensors",
                    f"workload.bits_per_value for Einsum {einsum}",
                )
                n_bits = parse_expression(
                    n_bits,
                    symbol_table,
                    "bits_per_value",
                    location=f"workload.bits_per_value[{set_expression}] for Einsum {einsum}",
                )
                if tensors & seen_tensors:
                    raise ValueError(
                        f"Tensors {sorted(tensors & seen_tensors)} are used in multiple "
                        f"entries in the workload.bits_per_value dictionary. This "
                        f"occured while parsing the bits per value for Einsum {einsum}."
                    )
                cur_bpv.update({t: n_bits for t in tensors})

            # Override with the local bits per value expressions
            for tensor_access in self.einsums[einsum].tensor_accesses:
                if tensor_access.bits_per_value is not None:
                    cur_bpv[tensor_access.name] = parse_expression(
                        tensor_access.bits_per_value,
                        symbol_table,
                        "bits_per_value",
                        str(tensor_access.bits_per_value),
                        location=f"workload.einsums[{einsum}].tensor_accesses[{tensor_access.name}].bits_per_value",
                    )
                if tensor_access.name not in cur_bpv:
                    if tensor_access.name in bits_per_value:
                        logging.warning(
                            f"Einsum {einsum} does not specify the bits per value "
                            f"for tensor {tensor_access.name}, but it is already "
                            f"specified in another Einsum. Taking the other value."
                        )
                    else:
                        raise ValueError(
                            f"Tensor {tensor_access.name} in Einsum {einsum} does "
                            f"not have a bits per value specified. Ensure that the "
                            f"tensor is either covered by the set expressions in the "
                            f"workload.bits_per_value dictionary or bits_per_value is "
                            f"specified for the tensor access."
                        )

            # Check for consistency across Einsums
            for prev_einsum, prev_bpv in bits_per_value_per_einsum.items():
                for t0, t1 in itertools.product(cur_bpv.keys(), prev_bpv.keys()):
                    b0 = cur_bpv[t0]
                    b1 = prev_bpv[t1]
                    if b0 != b1:
                        raise ValueError(
                            f"Tensor {t0} has bits per value {b0} in Einsum "
                            f"{einsum} and {b1} in Einsum {prev_einsum}. Bits per "
                            "value must be consistent across all Einsums that access "
                            "the same tensor."
                        )

            bits_per_value_per_einsum[einsum] = cur_bpv
            bits_per_value.update(cur_bpv)

        return bits_per_value
