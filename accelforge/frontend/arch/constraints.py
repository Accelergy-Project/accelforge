from typing import (
    Callable,
    List,
)

from accelforge.util._basetypes import (
    EvalableModel,
    EvalsTo,
    TryEvalTo,
)
import numpy as np

from accelforge.util._setexpressions import InvertibleSet
from accelforge.frontend.renames import RankVariable


class Comparison(EvalableModel):
    """
    A comparison between a rank variable's bound and a value. A comparison is performed
    for each rank variable.

    The LHS of each comparison is the loop bound of a loop that affects this rank
    variable. The RHS is the given value.

    For example, if the expression resolves to [a, b], the operator is "<=", and the
    value is 10, and we have loops "for a0 in [0..A0)" and "for b0 in [0..B0)", then a
    mapping is only valid if A0 <= 10 and B0 <= 10.
    """

    expression: TryEvalTo[InvertibleSet[RankVariable]]
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

    value: EvalsTo[int]
    """ The value to compare against. """

    _str_repr: str = None
    """ A string to print for this comparison when __str__ is called. If None, a default
    string will be used. """

    def _eval_expressions(self, *args, **kwargs):
        result, symbol_table = super()._eval_expressions(*args, **kwargs)
        if len(result.expression) == 1 and "product" in result.operator:
            result.operator = result.operator.replace("product", "")
        return result, symbol_table

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

    def __str__(self) -> str:
        if self._str_repr is not None:
            return self._str_repr
        return f"({sorted(self.expression)}) {self.operator} ({self.value})"


class _ConstraintLambda:
    def __init__(
        self,
        constraint: Comparison,
        target_mapping_nodes: list["Spatial"],
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

    def __repr__(self):
        return f"_ConstraintLambda({self.constraint}, {self.target_mapping_nodes}, {self.rank_variables})"

    def __call__(self, rank_variables: set[RankVariable], sizes: np.ndarray) -> bool:
        final = self.rank_variables.issubset(rank_variables)
        return self.constraint_lambda(final, sizes)

    def _constrained_node_str(self) -> str:
        return f"constrains {self._target_node_indices}"

    def __bool__(self) -> bool:
        return bool(self.target_mapping_nodes)

    def __str__(self) -> str:
        return self.constraint.__str__()


class _TileShapeConstraintLambda(_ConstraintLambda):
    def __str__(self) -> str:
        if self.constraint._str_repr is not None:
            return self.constraint._str_repr
        return "tile_shape " + super().__str__()

    def pretty_str(self) -> str:
        return f"Tile shape {self.constraint.operator} {self.constraint.value} {self._constrained_node_str()}"


class _LoopBoundsConstraintLambda(_ConstraintLambda):
    def __str__(self) -> str:
        if self.constraint._str_repr is not None:
            return self.constraint._str_repr
        return "loop_bounds " + super().__str__()

    def pretty_str(self) -> str:
        return f"Loop bounds {self.constraint.operator} {self.constraint.value} {self._constrained_node_str()}"


class _MinUsageConstraintLambda(_ConstraintLambda):
    def __init__(
        self,
        target_mapping_nodes: list["Spatial"],
        rank_variables: set[str],
        min_usage: float,
    ):
        super().__init__(None, target_mapping_nodes, rank_variables)
        self.min_usage = min_usage

    def __call__(self, complete_indices: list[int], usages: np.ndarray) -> bool:
        # final = self.rank_variables.issubset(rank_variables)
        final = set(self._target_loop_indices).issubset(set(complete_indices))
        if not final:
            return np.ones(usages.shape[0], dtype=np.bool)

        # Some usages are already above the minimum. Return those.
        result = usages >= self.min_usage
        if np.sum(result) > 0:
            return result

        # Nobody is amove the minimum. Return the best we can do.
        max_usage = np.max(usages, axis=0)
        return usages == max_usage

    def pretty_str(self) -> str:
        return f"Min usage {self.min_usage} {self._constrained_node_str()}"
