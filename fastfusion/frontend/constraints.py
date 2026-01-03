import copy
import logging
import re
from abc import ABC
from typing import Annotated, Any, Callable, List, Optional

from fastfusion._accelerated_imports import np
from fastfusion.util._basetypes import ParsableList, ParsableModel, ParsesTo
from fastfusion.util._parse_expressions import parse_expression
from fastfusion.util._setexpressions import InvertibleSet, eval_set_expression
from fastfusion.frontend.workload.workload import RankVariable, TensorName
from fastfusion._version import assert_version, __version__


# class LoopOrder(ParsableList[RankVariable]):
#     """
#     A loop_order of ranks.
#     """

#     def _parse(self, symbol_table: dict[str, Any], location: str):
#         # return [x._parse(symbol_table) for x in self]
#         return type(self)(
#             [
#                 eval_set_expression(x, symbol_table, "rank_variables", location)
#                 for x in self
#             ],
#         )


class Comparison(ParsableModel):
    """
    A comparison between a rank variable's bound and a value. A comparison is performed
    for each rank variable.

    The LHS of each comparison is the loop bound of a loop that affects this rank
    variable. The RHS is the given value.

    For example, if the expression resolves to [a, b], the operator is "<=", and the
    value is 10, and we have loops "for a0 in [0..A0)" and "for b0 in [0..B0)", then a
    mapping is only valid if A0 <= 10 and B0 <= 10.
    """

    expression: str | InvertibleSet[RankVariable] | set[RankVariable]
    """ The expression to compare. This expression should resolve to a set of rank
    variables. A comparison is performed for each rank variable independently, and the
    result passes if and only if all comparisons pass. The LHS of each comparison is the
    loop bound of a loop that affects this rank variable. The RHS is the given value.
    """

    operator: str
    """ The operator to use for the comparison. Supported operators are:
    - == (equal to)
    - <= (less than or equal to)
    - >= (greater than or equal to)
    - < (less than)
    - > (greater than)
    - product== (product of all loop bounds is equal to)
    - product<= (product of all loop bounds is less than or equal to)
    - product>= (product of all loop bounds is greater than or equal to)
    - product< (product of all loop bounds is less than)
    - product> (product of all loop bounds is greater than)
    """

    value: ParsesTo[int]
    """ The value to compare against. """

    def _parse(self, symbol_table: dict[str, Any], location: str):
        # if len(self) != 3:
        #     raise ValueError(f"Comparison can only have 3 elements. got {len(self)}")
        new = type(self)(
            expression=eval_set_expression(
                self.expression, symbol_table, "rank_variables", location
            ),
            operator=self.operator,
            value=self.value,
        )
        if len(new.expression) == 1 and "product" in new.operator:
            new.operator = new.operator.replace("product", "")
        return new

    def _constrained_to_one(self) -> bool:
        return self.value == 1 and self.operator in [
            "==",
            "<=",
            "product==",
            "product<=",
        ]

    def _split_expression(self) -> List[set[RankVariable]]:
        if "product" in self.operator:
            return [self.expression]
        return sorted(set((x,)) for x in self.expression)

    def _to_constraint_lambda(
        self,
        increasing_sizes: bool,
    ) -> Callable[[bool, np.ndarray], bool | np.ndarray]:
        # Equal operators can only evaluate when all sizes are known
        eq_op = lambda final: (
            np.equal
            if final
            else (np.less_equal if increasing_sizes else np.greater_equal)
        )

        # If we're increasing, we can evaluate leq immediately. If we're
        # decreasing, we can evaluate geq immediately. The other must wait
        # until all sizes are known.
        le_wrapper = lambda op: lambda final, sizes: (
            op(sizes) if final or increasing_sizes else True
        )
        ge_wrapper = lambda op: lambda final, sizes: (
            op(sizes) if final or not increasing_sizes else True
        )

        _all = lambda sizes: np.all(sizes, axis=1)
        _prod = lambda sizes: np.prod(sizes, axis=1)

        # fmt: off
        operator_to_wrapper = {
            "==":        lambda final, sizes: _all(eq_op(final)(sizes, self.value)),
            "product==": lambda final, sizes: eq_op(final)(_prod(sizes), self.value),
            "<=":        le_wrapper(lambda sizes: _all(sizes)  <= self.value),
            ">=":        ge_wrapper(lambda sizes: _all(sizes)  >= self.value),
            "<":         le_wrapper(lambda sizes: _all(sizes)  <  self.value),
            ">":         ge_wrapper(lambda sizes: _all(sizes)  >  self.value),
            "product<=": le_wrapper(lambda sizes: _prod(sizes) <= self.value),
            "product>=": ge_wrapper(lambda sizes: _prod(sizes) >= self.value),
            "product<":  le_wrapper(lambda sizes: _prod(sizes) <  self.value),
            "product>":  ge_wrapper(lambda sizes: _prod(sizes) >  self.value),
        }
        # fmt: on

        if self.operator in operator_to_wrapper:
            return operator_to_wrapper[self.operator]
        raise KeyError(
            f"Unknown operator: {self.operator}. Known operators: {list(operator_to_wrapper.keys())}"
        )


class Tensors(ParsableModel):
    keep: str | InvertibleSet[TensorName] | set[TensorName] = "<Defaults to Nothing>"
    """
    A set expression describing which tensors must be kept in this
    :class:`fastfusion.frontend.arch.TensorHolder`. If this is not defined, then all
    tensors must be kept.
    """

    may_keep: str | InvertibleSet[TensorName] | set[TensorName] = (
        "<Nothing if keep is defined, else All>"
    )
    """
    A set expression describing which tensors may optionally be kept in this
    :class:`fastfusion.frontend.arch.TensorHolder`. The mapper will explore both keeping
    and not keeping each of these tensors. If this is not defined, then all tensors may
    be kept.
    """

    tile_shape: ParsableList[Comparison] = []
    """
    The tile shape for each rank variable. This is given as a list of
    :class:`~.Comparison` objects, where each comparison must evaluate to True for a
    valid mapping.
    """

    no_refetch_from_above: str | InvertibleSet[TensorName] | set[TensorName] = "~All"
    """
    The tensors that are not allowed to be refetched from above. This is given as a set
    of :class:`~.TensorName` objects or a set expression that resolves to them. These
    tensors must be fetched at most one time from above memories, and may not be
    refetched across any temporal or spatial loop iterations. Tensors may be fetched in
    pieces (if they do not cause re-fetches of any piece).
    """

    tensor_order_options: ParsableList[
        ParsableList[str | InvertibleSet[TensorName] | set[TensorName]]
    ] = ParsableList()
    """
    Options for the order of tensor storage nodes in the mapping. This is given as a
    list-of-lists-of-sets. Each list-of-sets is a valid order of tensor storage nodes.
    Order is given from highest in the mapping to lowest.

    For example, an option could be [input | output, weight], which means that there is
    no relative ordering required between input and output, but weight must be below
    both.
    """

    def _parse_tensor_order_options(
        self, symbol_table: dict[str, Any], location: str
    ) -> "Tensors":
        result = type(self)(
            tensor_order_options=[
                [
                    eval_set_expression(x, symbol_table, "tensors", location)
                    for x in order_choice
                ]
                for order_choice in self.tensor_order_options
            ],
        )
        # Assert that there are no intersecting sets
        for order in result.tensor_order_options:
            for i, s0 in enumerate(order):
                for j, s1 in enumerate(order):
                    if i == j:
                        continue
                    if s0 & s1:
                        raise ValueError(
                            f"Intersecting entries in dataflow constraint: {s0} and {s1}"
                        )
        return result

    def _parse_keep(self, symbol_table: dict[str, Any], location: str) -> "Tensors":
        keep, may_keep = self.keep, self.may_keep
        if may_keep == "<Nothing if keep is defined, else All>":
            may_keep = "All" if keep == "<Defaults to Nothing>" else "~All"
        if keep == "<Defaults to Nothing>":
            keep = "Nothing"

        may_keep_first = isinstance(keep, str) and re.findall(r"\bmay_keep\b", keep)
        keep_first = isinstance(may_keep, str) and re.findall(r"\bkeep\b", may_keep)
        if keep_first and may_keep_first:
            raise ValueError(
                f"Keep and may_keep constraints reference each other: "
                f"{keep} and {may_keep}"
            )

        if may_keep_first:
            may_keep = eval_set_expression(may_keep, symbol_table, "tensors", location)
            symbol_table = copy.copy(symbol_table)
            symbol_table["may_keep"] = may_keep
            keep = eval_set_expression(keep, symbol_table, "tensors", location)
            return type(self)(keep=keep, may_keep=may_keep)
        else:
            keep = eval_set_expression(keep, symbol_table, "tensors", location)
            symbol_table = copy.copy(symbol_table)
            symbol_table["keep"] = keep
            may_keep = eval_set_expression(may_keep, symbol_table, "tensors", location)
            return type(self)(keep=keep, may_keep=may_keep)

    def _parse_non_keep(self, symbol_table: dict[str, Any], location: str) -> "Tensors":
        return type(self)(
            tile_shape=[x._parse(symbol_table, location) for x in self.tile_shape],
            no_refetch_from_above=eval_set_expression(
                self.no_refetch_from_above, symbol_table, "tensors", location
            ),
        )


class Loop(ParsableModel, ABC):
    """Constraints that apply to loops. Do not use this directly; use :class:`~.Spatial`
    or :class:`~.Temporal` instead.
    """

    loop_bounds: ParsableList[Comparison] = ParsableList()
    """ Bounds for this loop. This is a list of :class:`~.Comparison` objects, all of
    which must be satisfied by the loops to which this constraint applies. """

    def _parse(self, symbol_table: dict[str, Any], location: str):
        return type(self)(
            loop_bounds=[x._parse(symbol_table, location) for x in self.loop_bounds],
        )


class Spatial(Loop):
    """
    A :class:`~.Loop` constraints that apply to spatial loops.
    """

    name: str
    """ The dimension name across which different spatial iterations occur. """

    min_utilization: int | float | str = 0.0
    """ The minimum utilization of spatial instances, as a value from 0 to 1. A mapping
    is invalid if less than this porportion of this dimension's fanout is utilized.
    Mappers that support it (e.g., FFM) may, if no mappings satisfy this constraint,
    return the highest-utilization mappings.
    """

    must_reuse: str | InvertibleSet[TensorName] | set[TensorName] = "Nothing"
    """ A set of tensors or a set expression representing tensors that must be reused
    across spatial iterations. Spatial loops may only be placed that reuse ALL tensors
    given here.
    """

    def _parse(self, symbol_table: dict[str, Any], location: str):
        return type(self)(
            name=self.name,
            loop_bounds=[x._parse(symbol_table, location) for x in self.loop_bounds],
            min_utilization=parse_expression(
                self.min_utilization, symbol_table, "min_utilization", location
            ),
            must_reuse=eval_set_expression(
                self.must_reuse, symbol_table, "tensors", location
            ),
        )


class Temporal(Loop):
    """
    A :class:`~.Loop` constraints that apply to temporal loops.
    """

    rmw_first_update: str | InvertibleSet[TensorName] | set[TensorName] = "Nothing"
    """ A set of tensors or a set expression representing tensors that incur a
    read-modify-write the first time they are updated in a memory. For tensors outputted
    by an Einsum, the first update of a value only incurs a read, because the previous
    value is null. If a tensor is given here, then the first update of that tensor will
    incur a read and write.
    """

    def _parse(self, symbol_table: dict[str, Any], location: str):
        new_temporal = super()._parse(symbol_table, location)
        new_temporal.rmw_first_update = eval_set_expression(
            self.rmw_first_update, symbol_table, "tensors", location
        )
        return new_temporal


class Misc(ParsableModel):
    """
    Miscellaneous constraints that do not fit into the other categories.
    """

    enabled: str | bool = True
    """ Whether this component is enabled. If the expression resolves to False, then
    the component is disabled. """


class MiscOnlyConstraints(ParsableModel):
    """
    Miscellaneous constraints that do not fit into the other categories.
    """

    misc: Misc = Misc()
    """ Miscellaneous constraints that do not fit into the other categories. """


class ConstraintGroup(MiscOnlyConstraints):
    """A group of constraints that apply to a component."""

    spatial: ParsableList[Spatial] = ParsableList()
    """ Constraints that apply to spatial loops across spatial instances of this
    component. """

    # temporal: Temporal = Temporal()

    tensors: Tensors = Tensors()
    """ Constraints that apply to tensors stored in this component. """


class _ConstraintLambda:
    def __init__(
        self,
        constraint: Comparison,
        target_mapping_nodes: list[Spatial],
        rank_variables: set[str],
    ):
        self.constraint = constraint
        self.constraint_lambda = (
            None if constraint is None else constraint._to_constraint_lambda(True)
        )
        self.target_mapping_nodes = target_mapping_nodes
        self.rank_variables = rank_variables
        self._target_node_indices = None
        self._target_loop_indices = None

    def __call__(self, rank_variables: set[RankVariable], sizes: np.ndarray) -> bool:
        final = self.rank_variables.issubset(rank_variables)
        return self.constraint_lambda(final, sizes)

    def _constrained_node_str(self) -> str:
        return f"constrains {self._target_node_indices}"


class _TileShapeConstraintLambda(_ConstraintLambda):
    def pretty_str(self) -> str:
        return f"Tile shape {self.constraint.operator} {self.constraint.value} {self._constrained_node_str()}"


class _LoopBoundsConstraintLambda(_ConstraintLambda):
    def pretty_str(self) -> str:
        return f"Loop bounds {self.constraint.operator} {self.constraint.value} {self._constrained_node_str()}"


class _MinUtilizationConstraintLambda(_ConstraintLambda):
    def __init__(
        self,
        target_mapping_nodes: list[Spatial],
        rank_variables: set[str],
        min_utilization: float,
    ):
        super().__init__(None, target_mapping_nodes, rank_variables)
        self.min_utilization = min_utilization

    def __call__(self, complete_indices: list[int], utilizations: np.ndarray) -> bool:
        # final = self.rank_variables.issubset(rank_variables)
        final = set(self._target_loop_indices).issubset(set(complete_indices))
        if not final:
            return np.ones(utilizations.shape[0], dtype=np.bool)

        # Some utilizations are already above the minimum. Return those.
        result = utilizations >= self.min_utilization
        if np.sum(result) > 0:
            return result

        # Nobody is amove the minimum. Return the best we can do.
        max_utilization = np.max(utilizations, axis=0)
        return utilizations == max_utilization

    def pretty_str(self) -> str:
        return f"Min utilization {self.min_utilization} {self._constrained_node_str()}"


# class Constraints(ParsableModel):
#     # version: Annotated[str, assert_version] = __version__
#     constraints: ParsableList[ConstraintGroup] = ParsableList()
