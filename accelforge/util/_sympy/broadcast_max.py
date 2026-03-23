import sympy
from sympy import Max
from sympy import Min
from numbers import Number

# MaxGeqZero = Max
# MinGeqZero = Min


_nonneg_true = lambda *args, **kwargs: True
_nonneg_false = lambda *args, **kwargs: False
_ZERO = sympy.Integer(0)


def MaxGeqZero(*args):
    # Fast path covers most calls, avoids sympy
    if len(args) == 2:
        a, b = args
        for (a, b) in ((args[0], args[1]), (args[1], args[0])):
            a_is_numeric = isinstance(a, (int, float))
            b_is_numeric = isinstance(b, (int, float))
            if a_is_numeric and b_is_numeric:
                return max(a, b, 0)
            if a_is_numeric and a == 0:
                if isinstance(b, sympy.core.numbers.Number):
                    return max(float(b), 0)
                if isinstance(b, Max) and _ZERO in b.args:
                    return b
                x = Max(b, 0)
                if isinstance(x, Max):
                    x._eval_is_nonnegative = _nonneg_true
                    x._eval_is_negative = _nonneg_false
                return x

    non_max_args = [a for a in args if not isinstance(a, Max)]
    max_args = [a for a in args if isinstance(a, Max)]
    for a in max_args:
        non_max_args.extend(a.args)

    if 0 not in non_max_args:
        non_max_args.append(0)

    x = Max(*non_max_args)
    if isinstance(x, Max):
        x._eval_is_nonnegative = _nonneg_true
        x._eval_is_negative = _nonneg_false
    return x


def MinGeqZero(*args):
    if len(args) == 2:
        a, b = args
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return min(a, b)

    non_min_args = [a for a in args if not isinstance(a, Min)]
    min_args = [a for a in args if isinstance(a, Min)]
    for a in min_args:
        non_min_args.extend(a.args)
    x = Min(*non_min_args)
    if isinstance(x, Min):
        x._eval_is_nonnegative = _nonneg_true
        x._eval_is_negative = _nonneg_false
    return x
