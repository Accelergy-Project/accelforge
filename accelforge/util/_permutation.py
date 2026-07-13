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
        permutation = list(permutation)
        assert oset(permutation) == oset(range(len(permutation)))
        self.permutation = permutation

    def __getitem__(self, idx: int):
        return self.permutation[idx]

    def __iter__(self) -> Iterable[int]:
        return iter(self.permutation)

    def __len__(self) -> int:
        return len(self.permutation)

    def apply(
        self,
        sequence: Sequence[T],
        include_remaining_unpermuted: bool=True
    ) -> Iterable[T]:
        """
        Apply permutation to sequence. If `include_remaining_unpermuted` and
        the sequence is longer than the permutation, then the remainder of the
        sequence is included unpermuted. Otherwise, the remainder of the
        sequence is omitted.
        """
        assert len(sequence) >= len(self)
        yield from (sequence[idx] for idx in self.permutation)
        if include_remaining_unpermuted and len(sequence) > len(self):
            yield from sequence[len(self):]

    def copy(self) -> "Permutation":
        return Permutation(self.permutation.copy())

    def get_prefix(self, n: int) -> "Permutation":
        return Permutation(self.permutation[:n])
