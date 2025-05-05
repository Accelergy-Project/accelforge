from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any
from fastfusion.tags import Tags
from fastfusion.util import expfmt, fzs

# Abstractions:
# 1. Each tensor is stored above some loop index. 0 is the outermost loop, 1 the
#    next-innermost...
# 2. All loops above any shared tensor are co-tiled and must match between SIMs.

class Updatable:
    def update(self, **kwargs) -> "Updatable":
        return replace(self, **kwargs)

@dataclass(frozen=True)
class Reservation(Updatable):
    # This order is important. Above loop index should be before resource name
    # so when we sort reservations for tensors then the backing storage comes
    # first.
    # Size is not included in hash or equality functions. This is because there
    # may be floating point rounding errors in reservation sizes. The other
    # attributes are sufficient to determine equality.
    name: str
    above_loop_index: int
    resource_name: str
    size: float

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


@dataclass(frozen=True, order=True, eq=True)
class Loop(Updatable):
    rank_names: fzs[str]
    bound: int
    is_spatial: bool

    def __post_init__(self):
        assert isinstance(self.rank_names, fzs)
        assert isinstance(self.bound, int)
        assert isinstance(self.is_spatial, bool)

    @property
    def rank_name(self):
        assert len(self.rank_names) == 1
        return next(iter(self.rank_names))

    def __repr__(self):
        # return ("S-" if self.is_spatial else "") + f"{self.rank_name}-{self.bound}"
        return f"Loop({self.rank_names.__repr__()}, {self.bound}, {self.is_spatial})"

    def __str__(self):
        return ("S-" if self.is_spatial else "") + f"{self.rank_names}-{self.bound}"

    def pydot_str(self):
        if self.is_spatial:
            return f"S-for R{self.rank_names} size {expfmt(self.bound)}"
        return f"for {self.rank_names} size {expfmt(self.bound)}"

    def rename(self, rank_renaming: dict[str, str]) -> "Loop":
        return replace(self, rank_names=fzs(rank_renaming[r] for r in self.rank_names))

    def to_yaml(self):
        return {"type": "loop", **self.__dict__}

    def merge_next(self, right: "Loop") -> "Loop":
        assert self.bound == right.bound
        assert self.is_spatial == right.is_spatial
        return Loop(
            self.rank_names | right.rank_names,
            self.bound,
            self.is_spatial,
        )

@dataclass(frozen=True, order=True)
class TensorStorage(Reservation):
    def rename(self, tensor_renaming: dict[str, str]) -> "TensorStorage":
        return replace(self, name=tensor_renaming[self.name])

    def to_yaml(self):
        return {"type": "storage", **self.__dict__}

    def get_backing_stores(all_tensors: set["TensorStorage"]) -> list["TensorStorage"]:
        id2tensor = defaultdict(lambda: [])
        for t in all_tensors:
            id2tensor[t.name].append(t)
        return sorted(sorted(v)[0] for v in id2tensor.values())


@dataclass(frozen=True)
class Mapping(Updatable):
    loops: tuple[Loop, ...]
    storage: fzs[TensorStorage]
    tags: Tags = Tags(fzs())

    def __post_init__(self):
        assert isinstance(self.storage, fzs)
        assert isinstance(self.loops, tuple)
        assert isinstance(self.tags, Tags)

    def get_backing_levels(self) -> dict[str, int]:
        backings = {}
        for t in self.storage:
            prev = backings.get(t.name, t.above_loop_index)
            backings[t.name] = min(prev, t.above_loop_index)
        return backings
    
    @property
    def tensor_names(self) -> set[str]:
        return {t.name for t in self.storage}

    def shared_loop_index(self, live_tensors: set[str]) -> int:
        n = [l for t, l in self.get_backing_levels().items() if t in live_tensors]
        return max(n) - 1 if n else -1
    
    def shared_storage(self, other: "Mapping") -> set[str]:
        return set(self.storage) & set(other.storage)

    def __len__(self) -> int:
        return len(self.loops)

    def clear_dead_tensors(
        self,
        live_tensors: set[str],
        keep_loops: bool = False,
        keep_tensors: set[str] = None,
        drop_tags: bool = False,
    ) -> "Mapping":
        loops = (
            self.loops
            if keep_loops
            else self.loops[: self.shared_loop_index(live_tensors) + 1]
        )
        keep_tensors = keep_tensors if keep_tensors is not None else live_tensors
        tensors = fzs(t for t in self.storage if t.name in keep_tensors)
        tags = self.tags if not drop_tags else Tags(fzs())
        return Mapping(loops, tensors, tags)

    def __lt__(self, other):
        return (self.loops, self.storage) < (other.loops, other.storage)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"Mapping(loops={self.loops.__repr__()}, storage={self.storage.__repr__()}, tags={self.tags.__repr__()})"

    def merge_next(self, right: "Mapping", live_tensors: set[str]) -> "Mapping":
        tensors = fzs(
            t for t in (right.storage | self.storage) if t.name in live_tensors
        )
        shared_loop_index = max(t.shared_loop_index(live_tensors) for t in [self, right])

        merged_loops = [l.merge_next(l2) for l, l2 in zip(self.loops, right.loops)]
        additional_loops = right.loops[len(merged_loops) : shared_loop_index + 1]

        return Mapping(
            tuple(merged_loops + list(additional_loops))[: shared_loop_index + 1],
            tensors,
            right.tags,
        )

    def rename(
        self, rank_renaming: dict[str, str], tensor_renaming: dict[str, str]
    ) -> "Mapping":
        return replace(
            self,
            loops=tuple(l.rename(rank_renaming) for l in self.loops),
            storage=fzs(t.rename(tensor_renaming) for t in self.storage),
        )

    def matches_permutation(self, permutation: list[str]) -> bool:
        i, j = 0, 0
        while True:
            if i == len(self.loops) and j == len(permutation):
                return True
            if j == len(permutation):
                return False

            # Mismatch!
            if i == len(self.loops) or self.loops[i].rank_name != permutation[j]:
                if permutation[j] != "*":
                    return False
                j += 1
                while i < len(self.loops) and (
                    j == len(permutation) or self.loops[i].rank_name != permutation[j]
                ):
                    i += 1
            else:
                i, j = i + 1, j + 1

    def has_tensor(self, *tensors: TensorStorage) -> bool:
        return all(any(t == tensor for t in self.storage) for tensor in tensors)

    def set_tags(self, *new_tags: Any) -> "Mapping":
        return self.update(tags=Tags(new_tags))

    def all_n_loops(self) -> list["Mapping"]:
        min_loops = max(t.above_loop_index for t in self.storage)
        return list(
            Mapping(self.loops[:i], self.storage, self.tags)
            for i in range(min_loops, len(self.loops) + 1)
        )
