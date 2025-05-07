import copy
import logging
from typing import Any, List
from fastfusion.frontend._set_parse import eval_set_expression
from fastfusion.yamlparse.nodes import DictNode, ListNode, CombinableListNode
from .version import assert_version

class InvertibleSet(set):
    def __init__(self, *args, full_space: set, **kwargs):
        super().__init__(*args, **kwargs)
        self.full_space = full_space

    def __invert__(self):
        return InvertibleSet(self.full_space - self, full_space=self.full_space)

class ConstraintSetResolver:
    def __init__(
        self,
        tensor2rank_variables: dict[str, list[str]],
        intermediate_tensors: set[str],
    ):
        self._all_tensors = set(self.tensor2rank_variables.keys())
        self._all_rank_variables = set.union(*self.tensor2rank_variables.values())

        all_tensors = set(self.tensor2rank_variables.keys())
        self._tensor_set = {
            name: self.make_tensor_set([name]) for name in tensor2rank_variables
        }
        self._tensor_set["All"] = self.make_tensor_set(all_tensors)
        self._tensor_set["Intermediates"] = self.make_tensor_set(intermediate_tensors)

        all_rank_variables = set.union(*tensor2rank_variables.values())
        self._rank_variable_sets = {
            name: self.make_variable_set(tensor2rank_variables[name])
            for name in tensor2rank_variables
        }
        self._rank_variable_sets["All"] = self.make_variable_set(all_rank_variables)
        self._rank_variable_sets["Intermediates"] = self.make_variable_set(
            tensor_names=intermediate_tensors
        )

    def make_tensor_set(self, tensor_names: list[str]) -> InvertibleSet:
        return InvertibleSet(set(tensor_names), full_space=self._all_tensors)

    def make_variable_set(
        self, variable_names: list[str] = (), tensor_names: list[str] = ()
    ) -> InvertibleSet:
        variable_names = set.union(
            variable_names, *(self._rank_variable_sets[t] for t in tensor_names)
        )
        return InvertibleSet(variable_names, full_space=self._all_rank_variables)

    def resolve_tensor_set(
        self, expression: str, extra_tensor_set: dict[str, set] = {}
    ) -> list[str]:
        extra_sets = {k: self.make_tensor_set(v) for k, v in extra_tensor_set.items()}
        return self._resolve(expression, {**self._tensor_set, **extra_sets})

    def resolve_rank_variable_set(
        self, expression: str, extra_tensor_set: dict[str, set] = {}
    ) -> list[str]:
        extra_sets = {
            k: self.make_variable_set(tensor_names=v)
            for k, v in extra_tensor_set.items()
        }
        return self._resolve(expression, {**self._rank_variable_sets, **extra_sets})

    def _resolve(self, expression: str, sets: dict[str, InvertibleSet]) -> list[str]:
        if not expression:
            return set()
        try:
            return eval(expression, {"__builtins__": {}}, sets)
        except Exception as e:
            raise ValueError(
                f"Invalid set expression: {expression}. Error: {str(e)}. "
                f"Available sets: {', '.join(sets.keys())}"
            )

class ResolvesToTensorSet:
    def resolve(self, resolver: ConstraintSetResolver) -> list[str]:
        try:
            return set.union(*(resolver.resolve_tensor_set(e) for e in self))
        except Exception as e:
            raise ValueError(f"Error in {self.get_name()}: {e}")


class EntriesResolveToRankVariableSet:
    def resolve(self, resolver: ConstraintSetResolver) -> list[str]:
        try:
            return [resolver.resolve_rank_variable_set(e) for e in self]
        except Exception as e:
            raise ValueError(f"Error in {self.get_name()}: {e}")


class FirstEntryResolvesToRankVariableSet:
    def resolve(self, resolver: ConstraintSetResolver) -> list[str]:
        try:
            return [resolver.resolve_rank_variable_set(self[0])] + self[1:]
        except Exception as e:
            raise ValueError(f"Error in {self.get_name()}: {e}")


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
        fanoutX: int,
        fanoutY: int,
    ):
        self.validate_spatial(fanoutX, fanoutY)
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
        super().add_attr("loop_bounds", LoopBoundsList, [])
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
    
class LoopBoundsList(ListNode):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", LoopBounds)
        
    def _parse(self, symbol_table: dict[str, Any]):
        # return [x._parse(symbol_table) for x in self]
        return LoopBoundsList(
            *[x._parse(symbol_table) for x in self],
            __node_skip_parse=True,
        )
    
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bypass: str = self["bypass"]
        self.keep: str = self["keep"]
        self.coalesce: str = self["coalesce"]
        
    def _parse(self, symbol_table: dict[str, Any]):
        # new_storage = Storage()
        # new_storage.bypass = eval_set_expression(self.bypass, symbol_table, "tensors")
        # new_storage.keep = eval_set_expression(self.keep, symbol_table, "tensors")
        # new_storage.coalesce = eval_set_expression(self.coalesce, symbol_table, "tensors")
        # return new_storage
        return type(self)(
            bypass=eval_set_expression(self.bypass, symbol_table, "tensors"),
            keep=eval_set_expression(self.keep, symbol_table, "tensors"),
            coalesce=eval_set_expression(self.coalesce, symbol_table, "tensors"),
            __node_skip_parse=True,
        )

class LoopOrder(ListNode, EntriesResolveToRankVariableSet):
    """
    A loop_order of ranks.
    """

    def _parse(self, symbol_table: dict[str, Any]):
        # return [x._parse(symbol_table) for x in self]
        return LoopOrder(
            [eval_set_expression(x, symbol_table, "rank_variables") for x in self],
            __node_skip_parse=True,
        )


class LoopBounds(ListNode, FirstEntryResolvesToRankVariableSet):
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

    # def get_rank_variables(self):
    #     pass
    
    def _parse(self, symbol_table: dict[str, Any]):
        if len(self) != 3:
            raise ValueError(f"LoopBounds can only have 3 elements. got {len(self)}")
        return type(self)(
            [eval_set_expression(self.expression, symbol_table, "rank_variables"),
            self.operator,
            self.value],
            __node_skip_parse=True,
        )
        
        # new_loop_bounds = LoopBounds(*self)
        # new_loop_bounds.expression = eval_set_expression(self.expression, symbol_table, "rank_variables")
        # new_loop_bounds.operator = self.operator
        # new_loop_bounds.value = self.value
        # new_loop_bounds[0] = new_loop_bounds.expression
        # new_loop_bounds[1] = new_loop_bounds.operator
        # new_loop_bounds[2] = new_loop_bounds.value
        # return new_loop_bounds


"""
### Constraints Specification

## Keywords

Keywords specify a set of tensors. The following keywords are supported:

- All -> All tensors
- Intermediates -> All intermediate tensors
- (Any tensor name) -> Specific tensor
- (Any memory name) -> All tensors stored in a memory

## Constraint Types

### Storage

Keep, bypass, and coalesce constraints are lists. Entries in the list are joined
with a union. Keywords are replaced by the set of tensors they represent.

### Spatial and Temporal

Reuse acts like keep, bypass, and coalesce.

- loop_bound and tile_size: These are lists of constraints. A mapping is valid
  iff all constraints in the list are valid. Keywords are replaced by the set of
  ranks in the tensors that they represent. The .shape attribute returns the
  shape of either the loop bounds or the tile size, depending on the constraint.
  The .product attribute returns the product of the shape.

"""
