from fastfusion.util.util import fzs
from fastfusion.yamlparse.nodes import ListNode

class InvertibleSet(fzs):
    def __new__(
            cls,
            *args,
            full_space: set,
            space_name: str,
            child_access_name: str=None,
            element_to_child_space: dict[str, str]=None,
            **kwargs):
        # Create the frozenset instance
        instance = super().__new__(cls, *args, **kwargs)
        # Set our custom attributes
        instance.full_space = full_space
        instance.space_name = space_name
        instance.child_access_name = child_access_name
        instance.element_to_child_space = element_to_child_space
        if child_access_name:
            setattr(instance, child_access_name, instance._cast_to_child_space)
        return instance

    def __invert__(self):
        return self.to_my_space(self.full_space - self)
    
    def check_match_space_name(self, other):
        if self.space_name != other.space_name:
            raise ValueError(
                f"Can not perform set operations between different spaces "
                f"{self.space_name} and {other.space_name}."
            )
            
    def to_my_space(self, other) -> "InvertibleSet":
        return InvertibleSet(
            other,
            full_space=self.full_space,
            space_name=self.space_name,
            child_access_name=self.child_access_name,
            element_to_child_space=self.element_to_child_space,
        )
    
    def __and__(self, other) -> "InvertibleSet":
        self.check_match_space_name(other)
        return self.to_my_space(fzs.__and__(self, other))
    
    def __or__(self, other) -> "InvertibleSet":
        self.check_match_space_name(other)
        return self.to_my_space(fzs.__or__(self, other))
    
    def __sub__(self, other) -> "InvertibleSet":
        self.check_match_space_name(other)
        return self.to_my_space(fzs.__sub__(self, other))
    
    def __xor__(self, other) -> "InvertibleSet":
        self.check_match_space_name(other)
        return self.to_my_space(fzs.__xor__(self, other))
    
    def __call__(self):
        return self
    
    def _cast_to_child_space(self, *args, **kwargs):
        if not self.full_space:
            raise ValueError(
                f"Full space is empty for set {self.space_name}."
            )
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
            
        first_child_space_item: InvertibleSet = next(iter(self.element_to_child_space.values()))
        return first_child_space_item.to_my_space(set.union(*(set(self.element_to_child_space[item]) for item in self), set()))
    
def eval_set_expression(
    expression: str,
    symbol_table: dict[str, InvertibleSet],
    expected_space_name: str,
    injective: bool = False
) -> InvertibleSet:
    try:
        result = eval(expression, {"__builtins__": {}}, symbol_table)
        if not isinstance(result, InvertibleSet):
            raise TypeError(
                f"Set expression \"{expression}\" returned a "
                f"non-set with type {type(result)}: {result}"
            )
        if injective and len(result) != 1:
            raise ValueError(
                f"injective=True, but set expression \"{expression}\" "
                f"returned a set with {len(result)} elements: {result}"
            )
            
        if not isinstance(result, InvertibleSet):
            raise TypeError(
                f"Set expression \"{expression}\" returned a "
                f"non-InvertibleSet with type {type(result)}: {result}"
            )
        if result.space_name != expected_space_name:
            raise ValueError(
                f"Set expression \"{expression}\" returned a "
                f"set with space name \"{result.space_name}\", "
                f"expected \"{expected_space_name}\""
            )
    except Exception as e:
        raise ValueError(
            f"{e}.Symbol table:\n\t" + "\n\t".join(f"{k}: {v}" for k, v in symbol_table.items())
        ) from e
    return result
