import functools
from math import ceil, comb
import functools


@functools.lru_cache(maxsize=None)
def _count_factorizations_imperfect(n, into_n_parts):
    # Factorize n into into_n_parts parts
    RUBY_STYLE_IMPERFECT = True
    # RUBY_STYLE_IMPERFECT = True
    if n <= 1:
        return 1
    if into_n_parts <= 0:
        return 1

    shapes = list(range(1, ceil(n**0.5) + 1))
    shapes = shapes + [ceil(n / s) for s in shapes]
    shapes = sorted(set(shapes))

    if RUBY_STYLE_IMPERFECT:
        shapes = list(range(1, n + 1))

    total = 0
    for s in shapes:
        n = _count_factorizations_imperfect(ceil(n / s), into_n_parts - 1)
        total += _count_factorizations_imperfect(ceil(n / s), into_n_parts - 1)

    return total


def _prime_factorization(n):
    f = []
    i = 2
    while n > 1:
        if n % i == 0:
            f.append(i)
            n //= i
        else:
            i += 1
    return f


def _count_factorizations(n, into_n_parts, imperfect=False):
    if into_n_parts <= 1:
        return 1
    f = _prime_factorization(n)
    factors = {f2: f.count(f2) for f2 in set(f)}
    total = 1
    for exp in factors.values():
        total *= comb(exp + into_n_parts - 1, into_n_parts - 1)  # n choose k

    if imperfect:
        n = _count_factorizations_imperfect(n, into_n_parts)
        assert n >= total, f"n: {n} < total: {total}"
        return n

    return total
