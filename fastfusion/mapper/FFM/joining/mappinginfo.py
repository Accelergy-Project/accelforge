from collections import defaultdict
from enum import Enum
from dataclasses import dataclass, replace
import itertools
from numbers import Number
from typing import Any, Generator, NamedTuple, Union, TypeVar

from fastfusion.frontend.workload.workload import RankName
from fastfusion.mapper.FFM.tags import Tags

from fastfusion.util import expfmt, fzs

# Abstractions:
# 1. Each tensor is stored above some loop index. 0 is the outermost loop, 1 the
#    next-innermost...
# 2. All loops above any shared tensor are co-tiled and must match between SIMs.

T = TypeVar('T', bound='Updatable')

class Updatable:
    def update(self: T, **kwargs) -> T:
        return replace(self, **kwargs)


class TilePattern(NamedTuple):
    stride: int
    initial: int
    def __str__(self) -> str:
        return f'<{self.stride}, {self.initial}>'


@dataclass(frozen=True, order=True, eq=True)
class Loop(Updatable):
    rank_name: RankName
    bound: Union[Number, TilePattern] | None
    is_spatial: bool

    def __post_init__(self):
        assert isinstance(self.rank_name, RankName)
        assert isinstance(self.bound, Number | TilePattern | None)
        assert isinstance(self.is_spatial, bool)

    def __repr__(self):
        return f"Loop({self.rank_name.__repr__()}, {self.bound}, {self.is_spatial})"

    def __str__(self):
        return (
            "S-" if self.is_spatial else ""
        ) + f"{self.rank_name}-{self.bound}"

    def pydot_str(self):
        if self.is_spatial:
            return f"S-for R{self.rank_name} size {expfmt(self.bound)}"
        return f"for {self.rank_name} size {expfmt(self.bound)}"

    def rename(self, rank_renaming: dict[str, str]) -> "Loop":
        return replace(
            self,
            rank_name=fzs(rank_renaming[r] for r in self.rank_name),
        )

    def to_yaml(self):
        return {"type": "loop", **self.__dict__}

    def merge_next(self, right: "Loop") -> "Loop":
        self_bound = self.bound if isinstance(self.bound, Number) else self.bound.stride
        right_bound = right.bound if isinstance(right.bound, Number) else right.bound.stride
        assert self_bound == right_bound
        assert self.is_spatial == right.is_spatial
        return Loop(
            self.rank_name | right.rank_name,
            right.bound,
            self.is_spatial,
        )

    def clear_loop_bound(self, value=0):
        return self.update(bound=value)


@dataclass(frozen=True)
class Reservation(Updatable):
    # This order is important. Above loop index should be before resource name
    # so when we sort reservations for tensors then the backing tensor holder comes
    # first.
    # Size is not included in hash or equality functions. This is because there
    # may be floating point rounding errors in reservation sizes. The other
    # attributes are sufficient to determine equality.
    name: str
    loops: tuple["Loop"]
    resource_name: str
    size: float

    @property
    def above_loop_index(self) -> int:
        return len(self.loops)

    def __hash__(self):
        return hash((self.name, self.resource_name, self.above_loop_index))

    def __str__(self):
        return f"[{self.resource_name}] {self.name} sz {expfmt(self.size)} above {self.above_loop_index}"

    def __repr__(self):
        return f"Reservation({repr(self.name)}, {self.above_loop_index}, {repr(self.resource_name)}, {self.size})"

    def pydot_str(self):
        return f"[{self.resource_name}] {self.name} size {expfmt(self.size)}"

    def __eq__(self, value):
        if not isinstance(value, Reservation):
            return False
        for k in self.__dict__:
            a, b = getattr(self, k), getattr(value, k)
            if k != "size" and a != "*" and b != "*" and a != b:
                return False
        return True

    def permute(self, permutation) -> "Reservation":
        new_loops = [self.loops[permutation[i]] for i in range(len(self.loops))]
        return self.update(loops=tuple(new_loops))

    def clear_loop_bounds(self) -> "Reservation":
        return self.update(loops=tuple(loop.clear_loop_bound()
                                       for loop in self.loops))


@dataclass(frozen=True)
class TensorReservation(Reservation):
    def rename(self,
               rank_renaming: dict[str, str],
               tensor_renaming: dict[str, str]) -> "TensorReservation":
        return replace(self,
                       name=tensor_renaming[self.name],
                       loops=tuple(l.rename(rank_renaming) for l in self.loops))

    @staticmethod
    def get_backing_tensors(all_tensors: set["TensorReservation"]) -> list["TensorReservation"]:
        id2tensor = defaultdict(lambda: [])
        for t in all_tensors:
            id2tensor[t.name].append(t)
        return sorted(sorted(v)[0] for v in id2tensor.values())
    
    def __eq__(self, other):
        return super().__eq__(other)
    
    def __hash__(self):
        return super().__hash__()

class SplitKind(Enum):
    SEQUENTIAL = 0
    PIPELINE = 1


@dataclass(frozen=True, order=True)
class Split:
    kind: SplitKind
    n_loops: int
    
MHA_PATCH = False

@dataclass(frozen=True)
class Compatibility(Updatable):
    tensors: fzs[TensorReservation]
    splits: fzs[Split] = fzs()
    tags: Tags = Tags(fzs())
    _n_loops_override: int | None = None
    _loops_override: tuple[Loop, ...] | None = None
    
    @property
    def n_loops(self) -> int:
        if self._n_loops_override is not None:
            return self._n_loops_override
        return max((len(s.loops) for s in self.tensors), default=0)

    @property
    def loops(self) -> tuple[Loop, ...]:
        if self._loops_override is not None:
            return self._loops_override
        if self.tensors:
            return max((t.loops for t in self.tensors), key=len)
        return tuple()
    
    def _get_hash_tuple(self):
        if MHA_PATCH:
            return self.n_loops, self.tensors, self.tags, self.loops
        return self.n_loops, self.tensors, self.tags
    
    def __hash__(self):
        return hash(self._get_hash_tuple())
    
    def __eq__(self, other):
        return self._get_hash_tuple() == other._get_hash_tuple()

    def __post_init__(self):
        assert isinstance(self.n_loops, int)
        assert isinstance(self.tensors, fzs)
        assert isinstance(self.splits, fzs)
        assert isinstance(self.tags, Tags)

    def get_backing_levels(self) -> dict[str, int]:
        backings = {}
        for t in self.tensors:
            prev = backings.get(t.name, t.above_loop_index)
            backings[t.name] = min(prev, t.above_loop_index)
        return backings

    @property
    def tensor_names(self) -> set[str]:
        return {t.name for t in self.tensors}

    @property
    def max_above_loop_index(self) -> int:
        if len(self.tensors) == 0:
            return 0
        return max(s.above_loop_index for s in self.tensors)

    def shared_loop_index(self, live_tensors: set[str]) -> int:
        n = [l for t, l in self.get_backing_levels().items() if t in live_tensors]
        return max(n) - 1 if n else -1

    def __len__(self) -> int:
        return self.max_above_loop_index

    def clear_dead_tensors(
        self,
        live_tensors: set[str],
        keep_loops: bool = False,
        drop_tags: bool = False,
    ) -> "Compatibility":
        """
        Return a new compatibility with "dead" tensors removed by:
        1. keeping only loops relevant to `live_tensors` and
        2. keeping only `live_tensors`.

        If `keep_loops` is `True`, then all loops are kept.
        If `keep_tensors` is a set, tensors in the set are kept.
        """
        remaining_tensors = fzs(s for s in self.tensors if s.name in live_tensors)
        new_n_loops = max((len(s.loops) for s in self.tensors), default=0)
        new_splits = fzs(split for split in self.splits if split.n_loops <= new_n_loops)
        tags = self.tags if not drop_tags else Tags(fzs())
        kwargs = dict(tensors=remaining_tensors, splits=new_splits, tags=tags)
        if keep_loops:
            kwargs["_n_loops_override"] = new_n_loops
            kwargs["_loops_override"] = self.loops
        return Compatibility(**kwargs)

    def __lt__(self, other):
        return (self.loops, self.tensors) < (other.loops, other.tensors)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"Compatibility(n_loops={self.n_loops}, tensors={repr(self.tensors)}), splits={repr(self.splits)}, tags={repr(self.tags)}"

    def merge_next(
        self, right: "Compatibility", live_tensors: set[str]
    ) -> "Compatibility":
        return Compatibility(
            tensors=fzs(s for s in (self.tensors | right.tensors) if s.name in live_tensors),
            tags=right.tags,
        )

    def rename(
        self, rank_renaming: dict[str, str], tensor_renaming: dict[str, str]
    ) -> "Compatibility":
        raise NotImplementedError()
        return replace(
            self,
            tensors=fzs(t.rename(rank_renaming, tensor_renaming)
                        for t in self.tensors),
            tags=self.tags,
        )

    def has_tensor(self, *tensors: TensorReservation) -> bool:
        return all(any(s == t for s in self.tensors) for t in tensors)

    def all_n_loops(self) -> list["Compatibility"]:
        min_n_loops = max(len(s.loops) for s in self.tensors)
        return [Compatibility(_n_loops_override=n_loops, tensors=self.tensors, tags=self.tags)
                for n_loops in range(min_n_loops, self.n_loops+1)]

    def _permute(
        self,
        loop_changes: list[int]
    ) -> "Compatibility":
        assert len(loop_changes) == self.n_loops
        new_tensors = fzs(s.permute(loop_changes) for s in self.tensors)
        return self.update(tensors=new_tensors)

    def make_equivalent_permutations(self, reservation_levels: set[int]) -> list["Compatibility"]:
        # Get contiguous blocks of loops with no tensor reservation between them
        blocks = []
        current_block = []
        for i in range(self.n_loops):
            # Can't permute loops if there's a reservation between them
            if any(s.above_loop_index == i for s in self.tensors) or i in reservation_levels:
                blocks.append(current_block)
                current_block = []
            current_block.append(i)
        if current_block:
            blocks.append(current_block)

        per_block_permutations = [
            list(itertools.permutations(block))
            for block in blocks
        ]
        all_permutations = list(itertools.product(*per_block_permutations))
        result = [self._permute(list(itertools.chain(*loop_changes))) for loop_changes in all_permutations]
        return result

    def get_tensors_by_name(self, tensor: str) -> TensorReservation:
        for s in self.tensors:
            if s.name == tensor:
                return s
        raise ValueError(f"No reservation found for {tensor}")

    def per_tensor_compatibility(self) -> dict[str, "Compatibility"]:
        result = {}
        for s in self.tensors:
            result[s.name] = self.clear_dead_tensors(set([s.name]))
        return result

    def clear_tags(self) -> "Compatibility":
        return self.update(tags=Tags(fzs()))
    
    def clear_loop_bounds(self) -> "Compatibility":
        return self.update(tensors=fzs(tensor.clear_loop_bounds()
                                       for tensor in self.tensors))

    def compatible_with(self, other: "Compatibility") -> bool:
        if not MHA_PATCH:
            return True
        for a in self.tensors:
            a = a.loops
            for b in other.tensors:
                b = b.loops
                if a[:len(b)] != b[:len(a)]:
                    return False
        return True