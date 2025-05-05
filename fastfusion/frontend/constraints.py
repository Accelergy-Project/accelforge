import logging
from typing import List, Union, Set, Dict, Type, Any
from fastfusion.yamlparse.nodes import DictNode, ListNode, isempty, CombinableListNode
from .version import assert_version

# Registry for classes that need to declare attrs
_classes_to_declare: Set[Type] = set()


def declare_attrs_after_imports():
    """Call declare_attrs on all registered classes after imports are complete."""
    for cls in _classes_to_declare:
        cls.declare_attrs()


class DeclarableNode:
    """Mixin class that delays declare_attrs until after imports."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "declare_attrs"):
            _classes_to_declare.add(cls)


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
        super().add_attr("name", str)
        super().add_attr("spatial", Spatial, {})
        super().add_attr("spatial_X", Spatial, {})
        super().add_attr("spatial_Y", Spatial, {})
        super().add_attr("temporal", Temporal, {})
        super().add_attr("tensors", Storage, {})

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name: str = self["name"]
        self.spatial: Spatial = self["spatial"]
        self.spatial_X: Spatial = self["spatial_X"]
        self.spatial_Y: Spatial = self["spatial_Y"]
        self.temporal: Temporal = self["temporal"]
        self.tensors: Storage = self["tensors"]


class Iteration(DictNode):
    """
    An iteration (spatial or temporal) constraint.

    Attributes:
        factors (LoopBounds): The factors associated with the iteration.
        permutation (LoopOrder): The permutation associated with the iteration.
        default_max_factor (int): The default maximum factor value.
        default_min_factor (int): The default minimum factor value.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("reuse", TensorList, ["All"])
        super().add_attr("factors", LoopBounds, [], LoopBounds)
        super().add_attr("permutation", LoopOrder, [], LoopOrder)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reuse: List[str] = self["reuse"]
        self.factors: LoopBounds = self["factors"]
        self.permutation: LoopOrder = self["permutation"]


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
        super().add_attr("rmw_first_update", TensorList, [])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rmw_first_update: List[str] = self["rmw_first_update"]


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
        super().add_attr("bypass", TensorList, [])
        super().add_attr("keep", TensorList, [])
        super().add_attr("coalesce", TensorList, ["All"])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bypass: List[str] = self["bypass"]
        self.keep: List[str] = self["keep"]
        self.coalesce: List[str] = self["coalesce"]


class LoopOrder(ListNode, EntriesResolveToRankVariableSet):
    """
    A permutation of ranks.
    """

    pass


class LoopBounds(ListNode, FirstEntryResolvesToRankVariableSet):
    """
    A list of factors used to describe loop bounds
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", str)

    def get_rank_variables(self):
        pass


class TensorList(ListNode, ResolvesToTensorSet):
    """
    A list of workload tensorss.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Override the in operator
    def __contains__(self, item):
        return super().__contains__(item) or super().__contains__("*")


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
