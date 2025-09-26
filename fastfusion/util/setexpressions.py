from fastfusion.util.parse_expressions import ParseError
from fastfusion.util.util import fzs
from pydantic import BaseModel, ConfigDict
from typing import Iterator, Optional, TypeVar, Generic, Any, Union
from fastfusion.util.parse_expressions import MATH_FUNCS

T = TypeVar("T")


class InvertibleSet(BaseModel, Generic[T]):
    model_config = ConfigDict(extra="allow")  # Allow extra fields to be added
    instance: frozenset[T]
    full_space: frozenset[T]
    space_name: str
    child_access_name: Optional[str] = None
    element_to_child_space: Optional[dict[str, Any]] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.child_access_name:
            setattr(
                self,
                self.child_access_name,
                lambda *args, **kwargs: self._cast_to_child_space(*args, **kwargs),
            )

        # Make sure there's no extra fields
        extra = getattr(self, "__pydantic_extra__", {})
        extra = {k: v for k, v in extra.items() if k != self.child_access_name}
        if extra:
            raise ValueError(f"Extra fields are not allowed: {extra}")

    def __repr__(self):
        return f"InvertibleSet({self.instance})"

    def __str__(self):
        return self.__repr__()

    # def __new__(
    #         cls,
    #         *args,
    #         full_space: set,
    #         space_name: str,
    #         child_access_name: str=None,
    #         element_to_child_space: dict[str, str]=None,
    #         **kwargs):
    #     # Create the frozenset instance
    #     instance = super().__new__(cls, *args, **kwargs)
    #     # Set our custom attributes
    #     instance.full_space = full_space
    #     instance.space_name = space_name
    #     instance.child_access_name = child_access_name
    #     instance.element_to_child_space = element_to_child_space
    #     if child_access_name:
    #         setattr(instance, child_access_name, instance._cast_to_child_space)
    #     return instance

    def __invert__(self):
        return self.to_my_space(self.full_space - self.instance)

    def check_match_space_name(self, other):
        if self.space_name != other.space_name:
            raise ValueError(
                f"Can not perform set operations between different spaces "
                f"{self.space_name} and {other.space_name}."
            )

    def to_my_space(self, other) -> Union["InvertibleSet", set]:
        return InvertibleSet(
            instance=other.instance if isinstance(other, InvertibleSet) else other,
            full_space=self.full_space,
            space_name=self.space_name,
            child_access_name=self.child_access_name,
            element_to_child_space=self.element_to_child_space,
        )

    def __and__(self, other: "InvertibleSet[T]") -> "InvertibleSet[T]":
        return self.to_my_space(self.instance & other.instance)

    def __or__(self, other: "InvertibleSet[T]") -> "InvertibleSet[T]":
        return self.to_my_space(self.instance | other.instance)

    def __sub__(self, other: "InvertibleSet[T]") -> "InvertibleSet[T]":
        return self.to_my_space(self.instance - other.instance)

    def __xor__(self, other: "InvertibleSet[T]") -> "InvertibleSet[T]":
        return self.to_my_space(self.instance ^ other.instance)

    def __call__(self):
        return self

    def _cast_to_child_space(self, *args, **kwargs):
        if not self.full_space:
            raise ValueError(f"Full space is empty for set {self.space_name}.")
        for item in self:
            if item not in self.element_to_child_space:
                raise ValueError(
                    f"Item {item} is not in the element_to_child_space "
                    f"for set {self.space_name}."
                )

        if not self.element_to_child_space:
            raise ValueError(
                f"Element to child space is not set for set {self.space_name}."
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
                space_name=self.space_name,
                child_access_name=self.child_access_name,
                element_to_child_space=self.element_to_child_space,
            )


def eval_set_expression(
    expression: str,
    symbol_table: dict[str, InvertibleSet],
    expected_space_name: str,
    location: str,
    expected_count: int | None = None,
) -> InvertibleSet:
    try:
        if not isinstance(expression, str):
            raise TypeError(f"Expected a string, got {type(expression)}: {expression}")
        prev_result = "NOT_FOUND"
        result = prev_result
        if expression in symbol_table:
            result = symbol_table[expression]
        elif expression[-2:] == "()" and expression[:-2] in symbol_table:
            try:
                result = symbol_table[expression[:-2]]()
            except:
                pass

        if id(result) == id(prev_result):
            result = eval(expression, {"__builtins__": MATH_FUNCS}, symbol_table)

        if not isinstance(result, InvertibleSet):
            raise TypeError(
                f"Returned a non-InvertibleSet with type {type(result)}: {result}"
            )
        if expected_count is not None and len(result) != expected_count:
            raise ValueError(
                f"Expected {expected_count=} elements, got {len(result)}: {result.instance}"
            )
        if expected_space_name is not None and result.space_name != expected_space_name:
            raise ValueError(
                f'Returned a set with space name "{result.space_name}", '
                f'expected "{expected_space_name}"'
            )
    except Exception as e:
        err = ParseError(
            f'{e}. Set expression: "{expression}". Symbol table:\n\t'
            + "\n\t".join(f"{k}: {v}" for k, v in symbol_table.items())
        )
        if location is not None:
            err.add_field(location)
        raise err from e
    return result
