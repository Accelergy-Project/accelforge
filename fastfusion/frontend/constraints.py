import copy
import logging
from typing import Annotated, Any, Callable, List, Optional, Tuple, Union

import numpy as np
from fastfusion.util.basetypes import ParsableList, ParsableModel, ParsesTo
from fastfusion.util.setexpressions import InvertibleSet, eval_set_expression
from fastfusion.frontend.workload.workload import RankVariable, Tensor
from fastfusion.version import assert_version, __version__



class LoopOrder(ParsableList[RankVariable]):
    """
    A loop_order of ranks.
    """

    def _parse(self, symbol_table: dict[str, Any]):
        # return [x._parse(symbol_table) for x in self]
        return type(self)(
            [eval_set_expression(x, symbol_table, "rank_variables") for x in self],
        )

class Comparison(ParsableModel):
    expression: Union[str, InvertibleSet[RankVariable], set[RankVariable]]
    operator: str
    value: ParsesTo[int]
    
    def _parse(self, symbol_table: dict[str, Any]):
        # if len(self) != 3:
        #     raise ValueError(f"Comparison can only have 3 elements. got {len(self)}")
        return type(self)(
            expression=eval_set_expression(self.expression, symbol_table, "rank_variables"),
            operator=self.operator,
            value=self.value,
        )
        
    def constrained_to_one(self) -> bool:
        return self.value == 1 and self.operator in ["==", "<=", "product==", "product<="]

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
            "==":        lambda final, sizes: _all(eq_op(final)(sizes), self.value),
            "product==": lambda final, sizes: eq_op(final)(_prod((sizes), self.value)),
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
        
# class ComparisonList(ParsableList):
#     def __init__(self, *args, **kwargs):
#         new_args = []
#         for arg in args:
#             if isinstance(arg, Comparison):
#                 new_args.extend(arg)
#             elif isinstance(arg, list):
#                 if len(arg) != 3:
#                     raise ValueError(
#                         "To make a comparison from a list, it must have "
#                         "three elements: expression, operator, value. Got "
#                         f"{arg}"
#                     )
#                 new_args.append(Comparison(arg))
#             else:
#                 new_args.append(arg)
#         super().__init__(*new_args, **kwargs)
        
# def parse_comparison_list(x: ParsableList[RankVariable]) -> ComparisonList:
#     if not isinstance(x, list):
#         raise ValueError(f"Must be a list. Got {type(x)}")
#     return ComparisonList(*x)

class Storage(ParsableModel):
    bypass: Union[str, InvertibleSet[Tensor], set[Tensor]] = "~All"
    keep: Union[str, InvertibleSet[Tensor], set[Tensor]] = "~All"
    coalesce: Union[str, InvertibleSet[Tensor], set[Tensor]] = "All"
    tile_shape: ParsableList[Comparison] = []

    def _parse_keep_bypass(self, symbol_table: dict[str, Any]):
        return type(self)(
            bypass=eval_set_expression(self.bypass, symbol_table, "tensors"),
            keep=eval_set_expression(self.keep, symbol_table, "tensors"),
        )

    def _parse_non_keep_bypass(self, symbol_table: dict[str, Any]):
        return type(self)(
            coalesce=eval_set_expression(self.coalesce, symbol_table, "tensors"),
            tile_shape=[x._parse(symbol_table) for x in self.tile_shape],
        )

class Iteration(ParsableModel):
    version: Annotated[str, assert_version] = __version__
    reuse: Union[str, InvertibleSet[Tensor], set[Tensor]] = "All"
    loop_bounds: ParsableList[Comparison] = ParsableList()
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _parse(self, symbol_table: dict[str, Any]):
        return type(self)(
            loop_bounds=[x._parse(symbol_table) for x in self.loop_bounds],
            reuse=eval_set_expression(self.reuse, symbol_table, "tensors"),
        )
        
class Spatial(Iteration):
    def combine(self, other: "Spatial"):
        if self.reuse != other.reuse:
            raise ValueError(f"Cannot combine iterations with different reuse constraints. Got {self.reuse} and {other.reuse}.")
        return type(self)(
            loop_bounds=self.loop_bounds + other.loop_bounds,
            reuse=self.reuse,
        )

class Temporal(Iteration):
    rmw_first_update: List[str] = []

    def _parse(self, symbol_table: dict[str, Any]):
        new_temporal = super()._parse(symbol_table)
        new_temporal.rmw_first_update = eval_set_expression(self.rmw_first_update, symbol_table, "tensors")
        return new_temporal

    def combine(self, other: "Temporal"):
        if self.rmw_first_update != other.rmw_first_update:
            raise ValueError(f"Cannot combine iterations with different rmw_first_update constraints. Got {self.rmw_first_update} and {other.rmw_first_update}.")
        return type(self)(
            loop_bounds=self.loop_bounds + other.loop_bounds,
            rmw_first_update=self.rmw_first_update + other.rmw_first_update,
            reuse=self.reuse,
        )

class ConstraintGroup(ParsableModel):
    name: Optional[str] = None
    spatial: Spatial = Spatial()
    spatial_X: Spatial = Spatial()
    spatial_Y: Spatial = Spatial()
    temporal: Temporal = Temporal()
    storage: Storage = Storage()
    
    # def validate_spatial(self, fanout_X: int, fanout_Y: int):
    #     has_X = self.spatial_X.notempty_recursive()
    #     has_Y = self.spatial_Y.notempty_recursive()
    #     has_spatial = self.spatial.notempty_recursive()
    #     if has_X and fanout_X == 1:
    #         logging.warning(
    #             f"Spatial_X constraint is set for {self.name}, but fanout_X is 1. "
    #             f"The constraint will be ignored."
    #         )
    #     if has_Y and fanout_Y == 1:
    #         logging.warning(
    #             f"Spatial_Y constraint is set for {self.name}, but fanout_Y is 1. "
    #             f"The constraint will be ignored."
    #         )
    #     if has_spatial and (has_X or has_Y):
    #         raise ValueError(
    #             f"{self.name} has a \"spatial\" constraint, but has fanout in both "
    #             "X and Y dimensions. Please specify spatial_X and spatial_Y constraints "
    #             "instead."
    #         )
    #     if has_spatial and (has_X or has_Y):
    #         raise ValueError(
    #             f"{self.name} has a \"spatial\" constraint, and a \"spatial_X\" or "
    #             "\"spatial_Y\" constraint. Please specify either one \"spatial\" constraint "
    #             "or both \"spatial_X\" and \"spatial_Y\" constraints."
    #         )
            
    def get_spatial_constraint(self, for_X: bool=False, for_Y: bool=False) -> "Spatial":
        base = copy.deepcopy(self.spatial)
        
        
        if for_X:
            base.combine(self.spatial_X)
        elif for_Y:
            base.combine(self.spatial_Y)
        return base
        
        # if not for_X and not for_Y:
        #     raise ValueError(
        #         f"{self.name} has no spatial constraints. Please specify either "
        #         "spatial_X or spatial_Y constraints."
        #     )
        # if for_X and self.spatial_X.notempty_recursive():
        #     return self.spatial_X
        # if for_Y and self.spatial_Y.notempty_recursive():
        #     return self.spatial_Y
        # return self.spatial
            
    def _parse_storage(self, symbol_table: dict[str, Any]):
        return self.storage._parse(symbol_table)

    def _parse(
        self, 
        symbol_table: dict[str, Any],
        fanout_X: int,
        fanout_Y: int,
    ):
        self.validate_spatial(fanout_X, fanout_Y)
        return type(self)(
            self.spatial._parse(symbol_table),
            self.spatial_X._parse(symbol_table),
            self.spatial_Y._parse(symbol_table),
            self.temporal._parse(symbol_table),
            self.storage._parse(symbol_table),
        )


class Constraints(ParsableModel):
    version: Annotated[str, assert_version] = __version__
    constraints: ParsableList[ConstraintGroup] = ParsableList()
