from typing import Generic, TypeVar


T = TypeVar("T")


class fzs(frozenset[T], Generic[T]):
    def __repr__(self):
        return f"{{{', '.join(sorted(x.__repr__() for x in self))}}}"

    def __str__(self):
        return self.__repr__()

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
