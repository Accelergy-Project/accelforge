from typing import Generic, TypeVar

T = TypeVar("T")


def _sorted_iter(iterable):
    """Sort elements for deterministic iteration. Falls back to str() keys
    for elements that are not natively comparable."""
    items = list(iterable)
    if not items:
        return iter(items)
    try:
        return iter(sorted(items))
    except TypeError:
        return iter(sorted(items, key=lambda x: str(x)))


class fzs(frozenset[T], Generic[T]):
    __slots__ = ("_sorted_cache",)

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls, *args, **kwargs)
        object.__setattr__(obj, "_sorted_cache", None)
        return obj

    def __repr__(self):
        return f"{{{', '.join(sorted(x.__repr__() for x in self))}}}"

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        cached = object.__getattribute__(self, "_sorted_cache")
        if cached is not None:
            return iter(cached)
        items = list(frozenset.__iter__(self))
        if items:
            try:
                items.sort()
            except TypeError:
                items.sort(key=lambda x: str(x))
        result = tuple(items)
        try:
            object.__setattr__(self, "_sorted_cache", result)
        except (TypeError, AttributeError):
            pass  # frozenset subclass may resist setattr in some contexts
        return iter(result)

    def __or__(self, other: "fzs[T]") -> "fzs[T]":
        return fzs(super().__or__(other))

    def __and__(self, other: "fzs[T]") -> "fzs[T]":
        return fzs(super().__and__(other))

    def __sub__(self, other: "fzs[T]") -> "fzs[T]":
        return fzs(super().__sub__(other))

    def __xor__(self, other: "fzs[T]") -> "fzs[T]":
        return fzs(super().__xor__(other))

    def __lt__(self, other: "fzs[T]") -> bool:
        return sorted(self) < sorted(other)

    def __le__(self, other: "fzs[T]") -> bool:
        return sorted(self) <= sorted(other)

    def __gt__(self, other: "fzs[T]") -> bool:
        return sorted(self) > sorted(other)

    def __ge__(self, other: "fzs[T]") -> bool:
        return sorted(self) >= sorted(other)


class oset(set, Generic[T]):
    """Set that iterates in sorted order for deterministic behavior."""

    def __repr__(self):
        items = ", ".join(repr(x) for x in self)
        return f"oset({{{items}}})"

    def __iter__(self):
        return _sorted_iter(set.__iter__(self))

    def pop(self):
        val = min(self)
        self.discard(val)
        return val

    def __or__(self, other):
        return oset(set.__or__(self, other))

    def __ror__(self, other):
        return oset(set.__ror__(self, other))

    def __and__(self, other):
        return oset(set.__and__(self, other))

    def __rand__(self, other):
        return oset(set.__rand__(self, other))

    def __sub__(self, other):
        return oset(set.__sub__(self, other))

    def __rsub__(self, other):
        return oset(set.__rsub__(self, other))

    def __xor__(self, other):
        return oset(set.__xor__(self, other))

    def __rxor__(self, other):
        return oset(set.__rxor__(self, other))

    def copy(self):
        return oset(set.copy(self))

    def union(self, *others):
        return oset(set.union(self, *others))

    def intersection(self, *others):
        return oset(set.intersection(self, *others))

    def difference(self, *others):
        return oset(set.difference(self, *others))

    def symmetric_difference(self, other):
        return oset(set.symmetric_difference(self, other))
