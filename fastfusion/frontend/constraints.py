import copy
import logging
from typing import Any, Callable, List, Tuple, Union

import numpy as np
from fastfusion.frontend._set_parsing import eval_set_expression
from fastfusion.yamlparse.nodes import DictNode, ListNode, CombinableListNode
from .version import assert_version

def constraint_factory(constraint: dict):
    # Support the old "type" field
    if "type" in constraint:
        ctype = constraint["type"]
        type2class = {
            "spatial": Spatial,
            "spatial_X": Spatial,
            "spatial_Y": Spatial,
            "temporal": Temporal,
            "storage": Storage,
        }
        constraint = {k: v for k, v in constraint.items() if k != "type"}
        return ConstraintGroup(ctype=type2class[ctype](**constraint))
    return ConstraintGroup(**constraint)


class Constraints(DictNode):
    """
    Class representing constraints.

    Attributes:
        version (str): The version of the constraints.
        targets (ConstraintsList): The list of targets for the constraints.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("version", default="0.5", callfunc=assert_version)
        super().add_attr("constraints", ConstraintsList, [])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.version: str = self["version"]
        self.constraints: ConstraintsList = self["constraints"]


class ConstraintsList(CombinableListNode):
    """
    A class representing a list of constraints.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", ConstraintGroup, callfunc=constraint_factory)


class ConstraintGroup(DictNode):
    """
    A group of constraints.

    Attributes:
        spatial (Spatial): The spatial constraint.
        temporal (Temporal): The temporal constraint.
        tensors (Storage): The tensors constraint.
        max_overbooked_proportion (MaxOverbookedProportion): The maximum overbooked proportion constraint.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("name", str, "SET ME.")
        super().add_attr("spatial", Spatial, {})
        super().add_attr("spatial_X", Spatial, {})
        super().add_attr("spatial_Y", Spatial, {})
        super().add_attr("temporal", Temporal, {})
        super().add_attr("storage", Storage, {})

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name: str = self["name"]
        self.spatial: Spatial = self["spatial"]
        self.spatial_X: Spatial = self["spatial_X"]
        self.spatial_Y: Spatial = self["spatial_Y"]
        self.temporal: Temporal = self["temporal"]
        self.storage: Storage = self["storage"]
        

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
        return ConstraintGroup(
            self.spatial._parse(symbol_table),
            self.spatial_X._parse(symbol_table),
            self.spatial_Y._parse(symbol_table),
            self.temporal._parse(symbol_table),
            self.storage._parse(symbol_table),
            __node_skip_parse=True,
        )
        

class Iteration(DictNode):
    """
    An iteration (spatial or temporal) constraint.

    Attributes:
        factors (LoopBounds): The factors associated with the iteration.
        loop_order (LoopOrder): The loop_order associated with the iteration.
        default_max_factor (int): The default maximum factor value.
        default_min_factor (int): The default minimum factor value.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("reuse", str, "All")
        super().add_attr("loop_bounds", ShapeList, [])
        super().add_attr("loop_order", LoopOrder, [], LoopOrder)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reuse: List[str] = self["reuse"]
        self.loop_bounds: LoopBounds = self["loop_bounds"]
        self.loop_order: LoopOrder = self["loop_order"]
        
    def _parse(self, symbol_table: dict[str, Any]):
        return type(self)(
            loop_bounds=self.loop_bounds._parse(symbol_table),
            loop_order=self.loop_order._parse(symbol_table),
            reuse=self.reuse,
            __node_skip_parse=True,
        )
            
class ShapeList(ListNode):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", LoopBounds)
        
    def __getitem__(self, key: Union[str, int]) -> "LoopBounds":
        return super().__getitem__(key)

    def _parse(self, symbol_table: dict[str, Any]):
        # return [x._parse(symbol_table) for x in self]
        return ShapeList(
            [x._parse(symbol_table) for x in self],
            __node_skip_parse=True,
        )
        
    def get_constrained_one_rank_variables(self) -> set[str]:
        return set.union(*[set(x.get_constrained_one_rank_variables()) for x in self], set())
    
class Spatial(Iteration):
    """
    A spatial iteration constraint.
    """

    def __init__(self, *args, **kwargs):
        if "split" in kwargs:
            raise KeyError(
                "The split attribute is not supported. If you have fanout in "
                "both the X and Y dimensions, specify spatial_X and spatial_Y "
                "constraints instead."
            )
        super().__init__(*args, **kwargs)


class Temporal(Iteration):
    """
    A temporal iteration constraint.

    Attributes:
        rmw_first_update (List[str]): A list of workload tensorss that should
        have read-modify-write for the first update (rather than a write only).
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("rmw_first_update", str, "~All")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rmw_first_update: List[str] = self["rmw_first_update"]
        
    def _parse(self, symbol_table: dict[str, Any]):
        new_temporal = super()._parse(symbol_table)
        new_temporal.rmw_first_update = eval_set_expression(self.rmw_first_update, symbol_table, "tensors")
        return new_temporal

class Storage(DictNode):
    """
    A constraint class for specifying tensors properties.

    Attributes:
        bypass (List[str]): List of bypass tensors names.
        keep (List[str]): List of keep tensors names.
        no_coalesce (List[str]): List of no_coalesce tensors names.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("bypass", str, "~All")
        super().add_attr("keep", str, "~All")
        super().add_attr("coalesce", str, "All")
        super().add_attr("uneven", bool, False)
        super().add_attr("tile_shape", ShapeList, [])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bypass: str = self["bypass"]
        self.keep: str = self["keep"]
        self.coalesce: str = self["coalesce"]
        self.uneven: bool = self["uneven"]
        self.tile_shape: ShapeList = self["tile_shape"]

    def _parse_keep_bypass(self, symbol_table: dict[str, Any]):
        return type(self)(
            bypass=eval_set_expression(self.bypass, symbol_table, "tensors"),
            keep=eval_set_expression(self.keep, symbol_table, "tensors"),
            __node_skip_parse=True,
        )

    def _parse_non_keep_bypass(self, symbol_table: dict[str, Any]):
        # new_storage = Storage()
        # new_storage.bypass = eval_set_expression(self.bypass, symbol_table, "tensors")
        # new_storage.keep = eval_set_expression(self.keep, symbol_table, "tensors")
        # new_storage.coalesce = eval_set_expression(self.coalesce, symbol_table, "tensors")
        # return new_storage
        return type(self)(
            coalesce=eval_set_expression(self.coalesce, symbol_table, "tensors"),
            tile_shape=self.tile_shape._parse(symbol_table),
            __node_skip_parse=True,
        )

class LoopOrder(ListNode):
    """
    A loop_order of ranks.
    """

    def _parse(self, symbol_table: dict[str, Any]):
        # return [x._parse(symbol_table) for x in self]
        return LoopOrder(
            [eval_set_expression(x, symbol_table, "rank_variables") for x in self],
            __node_skip_parse=True,
        )


class LoopBounds(ListNode):
    """
    A list of factors used to describe loop bounds
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr(0, str)
        super().add_attr(1, (
            "==",
            "<=",
            ">=",
            "<",
            ">",
            "product==",
            "product<=",
            "product>=",
            "product<",
            "product>",
        ))
        super().add_attr(2, int)
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(self) != 3:
            raise ValueError(f"LoopBounds can only have 3 elements. got {len(self)}")
        self.expression = self[0]
        self.operator = self[1]
        self.value = self[2]
    
    def _parse(self, symbol_table: dict[str, Any]):
        if len(self) != 3:
            raise ValueError(f"LoopBounds can only have 3 elements. got {len(self)}")
        return type(self)(
            [eval_set_expression(self.expression, symbol_table, "rank_variables"),
            self.operator,
            self.value],
            __node_skip_parse=True,
        )
        
    def get_constrained_one_rank_variables(self) -> set[str]:
        if self.value != 1:
            return set()
        if self.operator not in ["==", "<=", "product==", "product<="]:
            return set()
        return self.expression

    def to_constraint_lambda(
            self, 
            increasing_sizes: bool,
        ) -> Callable[Tuple[bool, int, ...], bool]:
        
        # def relevant_sizes(sizes: np.ndarray, rank_variable_to_index: dict[str, int]) -> np.ndarray:
        #     return sizes[:, [rank_variable_to_index[x] for x in self.expression]]

        # Equal operators can only evaluate when all sizes are known
        if self.operator == "==":
            return lambda final, sizes: not final or np.all(sizes == self.value, axis=1)
        if self.operator == "product==":
            return lambda final, sizes: not final or np.all(np.prod(sizes, axis=1) == self.value)

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
            return operator_to_wrapper[self.operator](lambda sizes: np.all(sizes, axis=1) <= self.value)
        raise KeyError(f"Unknown operator: {self.operator}. Known operators: {list(operator_to_wrapper.keys())}")
