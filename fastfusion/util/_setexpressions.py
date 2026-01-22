import functools
from fastfusion.util._parse_expressions import ParseError
from pydantic import BaseModel, ConfigDict
from typing import Iterator, Optional, TypeVar, Generic, Any, Union
from fastfusion.util._parse_expressions import MATH_FUNCS

T = TypeVar("T")


def _reconstruct_invertible_set(state):
    """Helper function to reconstruct InvertibleSet during unpickling."""
    obj = object.__new__(InvertibleSet)
    obj.__dict__.update(state)
    return obj


class InvertibleSet(BaseModel, Generic[T]):
    instance: frozenset[T]
    full_space: frozenset[T]
    space_type: type[T]
    # child_access_name: Optional[str] = None
    element_to_child_space: Optional[dict[str, Any]] = None
    _bits_per_value: Optional[int] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def bits_per_value(self) -> int:
        if self._bits_per_value is None:
            raise ValueError(f"Bits per value is not set for set {self.instance}.")
        return self._bits_per_value

    @bits_per_value.setter
    def bits_per_value(self, value: int):
        self._bits_per_value = value

    def __reduce__(self):
        return (_reconstruct_invertible_set, (self.__dict__,))

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __deepcopy__(self, memo):
        """Custom deepcopy implementation to avoid pydantic deepcopy issues."""
        import copy
        cls = type(self)
        # Create a new instance without calling __init__
        new_obj = cls.__new__(cls)
        # Mark it in the memo to handle circular references
        memo[id(self)] = new_obj
        # Deep copy the __dict__ directly to avoid triggering setattr
        new_obj.__dict__.update(copy.deepcopy(self.__dict__, memo))
        # Initialize pydantic's internal attributes if they don't exist
        if not hasattr(new_obj, '__pydantic_fields_set__'):
            object.__setattr__(new_obj, '__pydantic_fields_set__', set())
        if not hasattr(new_obj, '__pydantic_extra__'):
            object.__setattr__(new_obj, '__pydantic_extra__', {})
        if not hasattr(new_obj, '__pydantic_private__'):
            object.__setattr__(new_obj, '__pydantic_private__', {})
        return new_obj

    def __repr__(self):
        return f"InvertibleSet({self.instance})"

    def __str__(self):
        return self.__repr__()

    def __invert__(self):
        return self.to_my_space(self.full_space - self.instance)

    def check_match_space_name(self, other):
        if self.space_type != other.space_type:
            raise ValueError(
                f"Can not perform set operations between different spaces "
                f"{self.space_type} and {other.space_type}."
            )

    def to_my_space(self, other) -> Union[set, "InvertibleSet"]:
        return InvertibleSet(
            instance=other.instance if isinstance(other, InvertibleSet) else other,
            full_space=self.full_space,
            space_type=self.space_type,
            # child_access_name=self.child_access_name,
            element_to_child_space=self.element_to_child_space,
        )

    @staticmethod
    def _make_set(x) -> set:
        return x.instance if isinstance(x, InvertibleSet) else x

    def __and__(self, other: "InvertibleSet[T]") -> "InvertibleSet[T]":
        a, b = self._make_set(self), self._make_set(other)
        return self.to_my_space(a & b)

    def __or__(self, other: "InvertibleSet[T]") -> "InvertibleSet[T]":
        a, b = self._make_set(self), self._make_set(other)
        return self.to_my_space(a | b)

    def __sub__(self, other: "InvertibleSet[T]") -> "InvertibleSet[T]":
        a, b = self._make_set(self), self._make_set(other)
        return self.to_my_space(a - b)

    def __xor__(self, other: "InvertibleSet[T]") -> "InvertibleSet[T]":
        a, b = self._make_set(self), self._make_set(other)
        return self.to_my_space(a ^ b)

    def __call__(self):
        return self

    def _cast_to_child_space(self, *args, **kwargs):
        if not self.full_space:
            raise ValueError(f"Full space is empty for set {self.space_type}.")
        for item in self:
            if item not in self.element_to_child_space:
                raise ValueError(
                    f"Item {item} is not in the element_to_child_space "
                    f"for set {self.space_type}."
                )

        if not self.element_to_child_space:
            raise ValueError(
                f"Element to child space is not set for set {self.space_type}."
            )

        first_child_space_item: InvertibleSet = next(
            iter(self.element_to_child_space.values())
        )
        return first_child_space_item.to_my_space(
            set.union(*(set(self.element_to_child_space[item]) for item in self), set())
        )

    def __bool__(self):
        return bool(self.instance)

    def __len__(self):
        return len(self.instance)

    def __contains__(self, item):
        return item in self.instance

    def __iter__(self):
        return iter(self.instance)

    def __getitem__(self, item):
        return self.instance[item]

    def iter_one_element_sets(self) -> Iterator["InvertibleSet[T]"]:
        for item in self.instance:
            yield InvertibleSet(
                instance=set((item,)),
                full_space=self.full_space,
                space_type=self.space_type,
                # child_access_name=self.child_access_name,
                element_to_child_space=self.element_to_child_space,
            )

    @property
    def rank_variables(self) -> set["RankVariable"]:
        from fastfusion.frontend.workload import RankVariable
        from fastfusion.frontend.renames import TensorName
        if self.space_type == TensorName:
            return self._cast_to_child_space()
        raise ValueError(
            f"Can not get rank variables for a set with space type "
            f"{self.space_type.__name__}."
        )

    @property
    def tensors(self) -> set["TensorName"]:
        from fastfusion.frontend.renames import TensorName
        if self.space_type == TensorName:
            return self
        raise ValueError(
            f"Can not get tensors for a set with space type "
            f"{self.space_type.__name__}."
        )


def set_expression_type_check(
    result: InvertibleSet[T],
    expected_space: type[T],
    expected_count: int | None = None,
    location: str | None = None,
) -> None:
    if not isinstance(result, InvertibleSet):
        raise TypeError(f"Expected a InvertibleSet, got {type(result)}: {result}")
    if expected_space is not None and result.space_type != expected_space:
        raise ValueError(f"Expected a set with space type '{expected_space.__name__}', got {result.space_type.__name__}")
    if expected_count is not None and len(result) != expected_count:
        raise ValueError(f"Expected {expected_count=} elements, got {len(result)}: {result.instance}")

def eval_set_expression(
    expression: str | InvertibleSet,
    symbol_table: dict[str, InvertibleSet],
    expected_space: type[T],
    location: str,
    expected_count: int | None = None,
) -> InvertibleSet[T]:
    try:
        err = None
        if not isinstance(expression, (InvertibleSet, str)):
            raise TypeError(f"Expected a string, got {type(expression)}: {expression}")

        prev_result = "NOT_FOUND"
        if isinstance(expression, str):
            result = prev_result
            if expression in symbol_table:
                result = symbol_table[expression]
            elif expression[-2:] == "()" and expression[:-2] in symbol_table:
                try:
                    result = symbol_table[expression[:-2]]()
                except:
                    pass
        else:
            result = expression

        if id(result) == id(prev_result):
            result = eval(expression, {"__builtins__": MATH_FUNCS}, symbol_table)

        if not isinstance(result, InvertibleSet):
            raise TypeError(
                f"Returned a non-InvertibleSet with type {type(result)}: {result}"
            )
        set_expression_type_check(result, expected_space, expected_count, location)

    except Exception as e:
        def strformat(v):
            v = str(v)
            return v if len(v) <= 100 else v[:100] + "..."
        err = ParseError(
            f'{e}. Set expression: "{expression}". Symbol table:\n\t'
            + "\n\t".join(f"{k}: {strformat(v)}" for k, v in symbol_table.items())
        )
        if location is not None:
            err.add_field(location)
    if err:
        raise err
    return result
