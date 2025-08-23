import copy
import logging
from typing import Annotated, Any, Callable, List, Optional, Tuple, Union

from fastfusion.accelerated_imports import np
from fastfusion.util.basetypes import ParsableList, ParsableModel, ParsesTo
from fastfusion.util.parse_expressions import parse_expression
from fastfusion.util.setexpressions import InvertibleSet, eval_set_expression
from fastfusion.frontend.workload.workload import RankVariableName, TensorName
from fastfusion.version import assert_version, __version__



class LoopOrder(ParsableList[RankVariableName]):
    """
    A loop_order of ranks.
    """

    def _parse(self, symbol_table: dict[str, Any], location: str):
        # return [x._parse(symbol_table) for x in self]
        return type(self)(
            [eval_set_expression(x, symbol_table, "rank_variables", location) for x in self],
        )

class Comparison(ParsableModel):
    expression: Union[str, InvertibleSet[RankVariableName], set[RankVariableName]]
    operator: str
    value: ParsesTo[int]
    
    def _parse(self, symbol_table: dict[str, Any], location: str):
        # if len(self) != 3:
        #     raise ValueError(f"Comparison can only have 3 elements. got {len(self)}")
        new = type(self)(
            expression=eval_set_expression(self.expression, symbol_table, "rank_variables", location),
            operator=self.operator,
            value=self.value,
        )
        if len(new.expression) == 1 and 'product' in new.operator:
            new.operator = new.operator.replace('product', '')
        return new
        
    def constrained_to_one(self) -> bool:
        return self.value == 1 and self.operator in ["==", "<=", "product==", "product<="]
    
    def split_expression(self) -> List[set[RankVariableName]]:
        if "product" in self.operator:
            return [self.expression]
        return sorted(set((x, )) for x in self.expression)

    def to_constraint_lambda(
            self,
            increasing_sizes: bool,
        ) -> Callable[[bool, np.ndarray], Union[bool, np.ndarray]]:
        # Equal operators can only evaluate when all sizes are known
        eq_op = lambda final: (np.equal if final else (np.less_equal if increasing_sizes else np.greater_equal))

        # If we're increasing, we can evaluate leq immediately. If we're
        # decreasing, we can evaluate geq immediately. The other must wait
        # until all sizes are known.
        le_wrapper = lambda op: lambda final, sizes: op(sizes) if final or increasing_sizes else True
        ge_wrapper = lambda op: lambda final, sizes: op(sizes) if final or not increasing_sizes else True
        
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
        raise KeyError(f"Unknown operator: {self.operator}. Known operators: {list(operator_to_wrapper.keys())}")

    def force_to_one(self):
        return self.value == 1 and self.operator in ["==", "<=", "product==", "product<="]


class Tensors(ParsableModel):
    bypass: Union[str, InvertibleSet[TensorName], set[TensorName]] = "~All"
    keep: Union[str, InvertibleSet[TensorName], set[TensorName]] = "~All"
    tile_shape: ParsableList[Comparison] = []
    no_refetch_from_above: Union[str, InvertibleSet[TensorName], set[TensorName]] = "~All"

    def _parse_keep_bypass(self, symbol_table: dict[str, Any], location: str):
        if "bypass" in self.keep and "keep" in self.bypass:
            raise ValueError(
                f"Bypass and keep constraints reference each other: {self.bypass} and {self.keep}"
            )

        if isinstance(self.bypass, str) and "keep" in self.bypass:
            keep = eval_set_expression(self.keep, symbol_table, "tensors", location)
            symbol_table = copy.copy(symbol_table)
            symbol_table["keep"] = keep
            bypass = eval_set_expression(self.bypass, symbol_table, "tensors", location)
            return type(self)(bypass=bypass, keep=keep)
        else:
            bypass = eval_set_expression(self.bypass, symbol_table, "tensors", location)
            symbol_table = copy.copy(symbol_table)
            symbol_table["bypass"] = bypass
            keep = eval_set_expression(self.keep, symbol_table, "tensors", location)
            return type(self)(bypass=bypass, keep=keep)

    def _parse_non_keep_bypass(self, symbol_table: dict[str, Any], location: str):
        return type(self)(
            tile_shape=[x._parse(symbol_table, location) for x in self.tile_shape],
            no_refetch_from_above=eval_set_expression(self.no_refetch_from_above, symbol_table, "tensors", location),
        )

class Iteration(ParsableModel):
    version: Annotated[str, assert_version] = __version__
    reuse: Union[str, InvertibleSet[TensorName], set[TensorName]] = "All"
    loop_bounds: ParsableList[Comparison] = ParsableList()
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _parse(self, symbol_table: dict[str, Any], location: str):
        return type(self)(
            loop_bounds=[x._parse(symbol_table, location) for x in self.loop_bounds],
            reuse=eval_set_expression(self.reuse, symbol_table, "tensors", location),
        )
        
class Spatial(Iteration):
    name: str
    min_utilization: Union[float, str] = 0.0
    def combine(self, other: "Spatial"):
        if self.reuse != other.reuse:
            raise ValueError(f"Cannot combine iterations with different reuse constraints. Got {self.reuse} and {other.reuse}.")
        return type(self)(
            loop_bounds=self.loop_bounds + other.loop_bounds,
            reuse=self.reuse,
        )
        
    @property
    def name(self):
        return self.name
    
    def _parse(self, symbol_table: dict[str, Any], location: str):
        return type(self)(
            name=self.name,
            loop_bounds=[x._parse(symbol_table, location) for x in self.loop_bounds],
            reuse=eval_set_expression(self.reuse, symbol_table, "tensors", location),
            min_utilization=parse_expression(self.min_utilization, symbol_table, "min_utilization", location),
        )

class Temporal(Iteration):
    rmw_first_update: List[str] = []

    def _parse(self, symbol_table: dict[str, Any], location: str):
        new_temporal = super()._parse(symbol_table, location)
        new_temporal.rmw_first_update = eval_set_expression(self.rmw_first_update, symbol_table, "tensors", location)
        return new_temporal

    def combine(self, other: "Temporal"):
        if self.rmw_first_update != other.rmw_first_update:
            raise ValueError(f"Cannot combine iterations with different rmw_first_update constraints. Got {self.rmw_first_update} and {other.rmw_first_update}.")
        return type(self)(
            loop_bounds=self.loop_bounds + other.loop_bounds,
            rmw_first_update=self.rmw_first_update + other.rmw_first_update,
            reuse=self.reuse,
        )
        
class Dataflow(ParsableModel):
    tensor_order_options: ParsableList[ParsableList[Union[str, InvertibleSet[TensorName], set[TensorName]]]] = ParsableList()

    def _parse(self, symbol_table: dict[str, Any], location: str):
        result = type(self)(
            tensor_order_options=[
                [eval_set_expression(x, symbol_table, "tensors", location) for x in order_choice]
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
                        raise ValueError(f"Intersecting entries in dataflow constraint: {s0} and {s1}")
        return result

class Misc(ParsableModel):
    enabled: Union[str, bool] = True


class MiscOnlyConstraints(ParsableModel):
    name: Optional[str] = None
    misc: Misc = Misc()


class ConstraintGroup(MiscOnlyConstraints):
    spatial: ParsableList[Spatial] = ParsableList()
    temporal: Temporal = Temporal()
    tensors: Tensors = Tensors()
    dataflow: Dataflow = Dataflow()

class ConstraintLambda:
    def __init__(self, constraint: Comparison, target_mapping_nodes: list[Spatial], rank_variables: set[str]):
        self.constraint = constraint
        self.constraint_lambda = None if constraint is None else constraint.to_constraint_lambda(True)
        self.target_mapping_nodes = target_mapping_nodes
        self.rank_variables = rank_variables
        self._target_node_indices = None
        self._target_loop_indices = None

    def __call__(self, rank_variables: set[RankVariableName], sizes: np.ndarray) -> bool:
        final = self.rank_variables.issubset(rank_variables)
        return self.constraint_lambda(final, sizes)
    
    def _constrained_node_str(self) -> str:
        return f"constrains {self._target_node_indices}"

class TileShapeConstraintLambda(ConstraintLambda):
    def pretty_str(self) -> str:
        return f"Tile shape {self.constraint.operator} {self.constraint.value} {self._constrained_node_str()}"


class LoopBoundsConstraintLambda(ConstraintLambda):
    def pretty_str(self) -> str:
        return f"Loop bounds {self.constraint.operator} {self.constraint.value} {self._constrained_node_str()}"


class MinUtilizationConstraintLambda(ConstraintLambda):
    def __init__(self, target_mapping_nodes: list[Spatial], rank_variables: set[str], min_utilization: float):
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


class Constraints(ParsableModel):
    version: Annotated[str, assert_version] = __version__
    constraints: ParsableList[ConstraintGroup] = ParsableList()
