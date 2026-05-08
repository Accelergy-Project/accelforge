from symengine import Max, Min


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
