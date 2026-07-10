from collections.abc import Iterable, Sequence
from typing import TypeVar

from accelforge.util import oset

T = TypeVar("T")


class Permutation:
    """
    A sequence on integers from the set of [0,N) representing a permutation
    of a sequence that has N elements.
    """
    def __init__(self, permutation: Iterable[int]):
        self.permutation = list(permutation)
        assert oset(self.permutation) == oset(range(len(self.permutation)))

    def __getitem__(self, idx: int):
        return self.permutation[idx]

    def __iter__(self) -> Iterable[int]:
        return iter(self.permutation)

    def __len__(self) -> int:
        return len(self.permutation)

    def extend_unpermuted(self, n: int):
        """
        Extend the permutation to n elements with remaining elements unpermuted.
        For example,
        ```
        p = Permutation([0, 2, 1])
        p.extend_unpermuted(5)
        p == Permutation([0, 2, 1, 3, 4])
        ```
        """
        assert n >= len(self)
        self.permutation.extend(list(range(len(self), n)))

    def apply(self, sequence: Sequence[T]) -> Iterable[T]:
        assert len(sequence) == len(self)
        yield from (sequence[self[i]] for i in range(len(sequence)))
