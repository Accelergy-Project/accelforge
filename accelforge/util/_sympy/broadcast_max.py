import sympy
from sympy import Max
from sympy import Min

# class MaxGeqZero(Max):
#     def _eval_is_nonnegative(self, *args, **kwargs):
#         return True

#     def _eval_is_negative(self, *args, **kwargs):
#         return False

# class MinGeqZero(Min):
#     def _eval_is_nonnegative(self, *args, **kwargs):
#         return True

#     def _eval_is_negative(self, *args, **kwargs):
#         return False

# MaxGeqZero = Max
# MinGeqZero = Min


def MaxGeqZero(*args):
    non_max_args = [a for a in args if not isinstance(a, Max)]
    max_args = [a for a in args if isinstance(a, Max)]
    for a in max_args:
        non_max_args.extend(a.args)

    if 0 not in non_max_args:
        non_max_args.append(0)

    x = Max(*non_max_args)
    if isinstance(x, Max):
        x._eval_is_nonnegative = lambda *args, **kwargs: True
        x._eval_is_negative = lambda *args, **kwargs: False
    return x


def MinGeqZero(*args):
    non_min_args = [a for a in args if not isinstance(a, Min)]
    min_args = [a for a in args if isinstance(a, Min)]
    for a in min_args:
        non_min_args.extend(a.args)
    x = Min(*non_min_args)
    if isinstance(x, Min):
        x._eval_is_nonnegative = lambda *args, **kwargs: True
        x._eval_is_negative = lambda *args, **kwargs: False
    return x


# MAX BUG FIX.
# def Min(a, *bs):
#     """More post-lambdify broadcast-friendly option than sympy.Min"""
#     result = a
#     for b in bs:
#         result = sympy.Piecewise((result, result < b), (b, True))
#     return result

# def Max(a, *bs):
#     """More post-lambdify broadcast-friendly option than sympy.Max"""
#     result = a
#     for b in bs:
#         result = sympy.Piecewise((result, result > b), (b, True))
#     return result
