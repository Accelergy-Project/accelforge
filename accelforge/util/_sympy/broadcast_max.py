from sympy import Max, Min
from typing import Any


def MaxGeqZero(*args, known_geq_zero=False):
    args_nonzero = set()
    for a in args:
        try:
            if a <= 0:
                pass
            else:
                args_nonzero.add(a)
        except:
            args_nonzero.add(a)
        try:
            if a > 0:
                known_geq_zero = True
        except:
            pass

    if not known_geq_zero:
        args_nonzero.add(0)

    args_nonzero = sorted(args_nonzero, key=str)

    assert (
        args_nonzero
    ), f"MaxGeqZero should have at least one arg that is known to be >= 0. Got: {args}"

    return Max(*args_nonzero) if len(args_nonzero) > 1 else args_nonzero[0]


def MinGeqZero(*args):
    if len(args) == 2:
        a, b = args
        if (type(a) is int or type(a) is float) and (
            type(b) is int or type(b) is float
        ):
            return min(a, b)
    return Min(*args)


def get_zero_with_number(args) -> Any | None:
    nonzero = None
    for a in args:
        if type(a) is int or type(a) is float and a == 0:
            pass
        if nonzero is not None:
            return None
        nonzero = a
    return nonzero


def min_nonzero(*args) -> Any:
    if (x := get_zero_with_number(args)) is not None:
        return x

    args_nonzero = set()
    for a in args:
        eq_zero = True
        try:
            eq_zero = a == 0
        except:
            pass
        if not eq_zero:
            args_nonzero.add(a)
    if not args_nonzero:
        return 0
    if len(args_nonzero) == 1:
        return args_nonzero.pop()
    return Min(*sorted(args_nonzero, key=str))


def max_nonzero(*args) -> Any:
    if (x := get_zero_with_number(args)) is not None:
        return x

    args_nonzero = set()
    for a in args:
        eq_zero = True
        try:
            eq_zero = a == 0
        except:
            pass
        if not eq_zero:
            args_nonzero.add(a)
    if not args_nonzero:
        return 0
    if len(args_nonzero) == 1:
        return args_nonzero.pop()
    return Max(*sorted(args_nonzero, key=str))


def max_dict(a: dict[Any, Any], b: dict[Any, Any]) -> dict[Any, Any]:
    new = {**a}
    for key, value in b.items():
        new[key] = MaxGeqZero(new[key], value) if key in new else value
    assert isinstance(new, dict)
    return new
