import sympy


def Min(a, *bs):
    """More post-lambdify broadcast-friendly option than sympy.Min"""
    result = a
    for b in bs:
        result = sympy.Piecewise((result, result < b), (b, True))
    return result



def Max(a, *bs):
    """More post-lambdify broadcast-friendly option than sympy.Max"""
    result = a
    for b in bs:
        result = sympy.Piecewise((result, result > b), (b, True))
    return result

def Min(a, *bs):
    """More post-lambdify broadcast-friendly option than sympy.Min"""
    result = a
    for b in bs:
        result = sympy.Piecewise((result, result < b), (b, True))
    return result