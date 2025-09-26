from collections.abc import Iterable
from typing import TypeVar


T = TypeVar("T")
VT = TypeVar("VT")


def first(iterable: Iterable[T], default: VT = None) -> T | VT:
    """Return first element in `iterable` or `default` if empty.

    Equivalent to `next(iter(iterable), default)`.
    """
    return next(iter(iterable), default)
