from symengine import Max, Min
from typing import Any


def MaxGeqZero(*args):
    # Fast paths for numeric args. 2-arg case dominates.
    if len(args) == 2:
        a, b = args
        if type(a) is int or type(a) is float:
            if type(b) is int or type(b) is float:
                return max(a, b, 0)
            if a == 0:
                return Max(0, b)
        elif type(b) is int or type(b) is float:
            if b == 0:
                return Max(0, a)
    return Max(0, *args)


def MinGeqZero(*args):
    if len(args) == 2:
        a, b = args
        if (type(a) is int or type(a) is float) and (
            type(b) is int or type(b) is float
        ):
            return min(a, b)
    return Min(*args)


# TODO: unsure if this is needed. If the sympy symbol is created with the
# correct assumption (e.g., positive), this should be automatic.
def min_nonzero(a: Any, b: Any) -> Any:
    if a == 0:
        return b
    if b == 0:
        return a
    return MinGeqZero(a, b)


def max_dict(a: dict[Any, Any], b: dict[Any, Any]) -> dict[Any, Any]:
    new = {**a}
    for key, value in b.items():
        new[key] = MaxGeqZero(new[key], value) if key in new else value
    assert isinstance(new, dict)
    return new
