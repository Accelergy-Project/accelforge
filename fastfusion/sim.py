from collections import defaultdict
import copy
from dataclasses import dataclass
from functools import cached_property
import struct
from typing import Any, Iterable, Optional

from joblib import delayed
import pandas as pd

from .pareto import Pareto, LOGSTRING, nameloop2col
from .tags import Tags
from .util import expfmt, fzs, parallel

# Abstractions:
# 1. Each tensor is stored above some loop index. 0 is the outermost loop, 1 the
#    next-innermost...
# 2. All loops above any shared tensor are co-tiled and must match between SIMs.


@dataclass(frozen=True)
class Loop:
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

    def __eq__(self, other):
        if not isinstance(other, Loop):
            return False
        return (
            self.rank_names == other.rank_names
            and self.bound == other.bound
            and self.is_spatial == other.is_spatial
        )

    def __lt__(self, other):
        return (
            self.rank_names < other.rank_names
            or self.bound < other.bound
            or self.is_spatial < other.is_spatial
        )

    def __hash__(self):
        return hash((self.rank_names, self.bound, self.is_spatial))

    def __repr__(self):
        # return ("S-" if self.is_spatial else "") + f"{self.rank_name}-{self.bound}"
        return f"Loop({self.rank_names.__repr__()}, {self.bound}, {self.is_spatial})"

    def __str__(self):
        return ("S-" if self.is_spatial else "") + f"{self.rank_names}-{self.bound}"

    def pydot_str(self):
        if self.is_spatial:
            return f"S-for R{self.rank_names} size {expfmt(self.bound)}"
        return f"for {self.rank_names} size {expfmt(self.bound)}"

    def rename(
        self, rank_renaming: dict[str, str], tensor_renaming: dict[str, str]
    ) -> "Loop":
        return Loop(
            fzs(rank_renaming[r] for r in self.rank_names), self.bound, self.is_spatial
        )

    def to_yaml(self):
        return {
            "type": "loop",
            "rank_name": self.rank_name,
            "bound": self.bound,
            "is_spatial": self.is_spatial,
        }

    def merge_next(self, other: "Loop") -> "Loop":
        assert self.bound == other.bound
        assert self.is_spatial == other.is_spatial
        return Loop(
            self.rank_names | other.rank_names,
            self.bound,
            self.is_spatial,
        )

    def update(self, **kwargs) -> "Loop":
        return Loop(**{**self.__dict__, **kwargs})

    def to_tuple(self):
        return (
            tuple(r for r in self.rank_names),
            self.bound,
            self.is_spatial,
        )

    @staticmethod
    def from_tuple(t: tuple) -> "Loop":
        return Loop(
            rank_names=fzs(r for r in t[0]),
            bound=t[1],
            is_spatial=t[2],
        )


@dataclass(frozen=True, order=True)
class TensorStorage:
    # This order is important. Make sure backing storage are the lowest.
    tensor_name: str
    above_loop_index: int
    memory_name: str
    # NOTE: Tile size is not included in hash or equality functions. This is
    # because inter-Einsum comparisons care about the loops and locations of
    # backing storage, and the tile sizes are derived from these. We don't want
    # rounding errors in the tile size to effect our inter-Einsum comparisons.
    tile_size: int
    # n_repititions: int = 1

    def __tuple__(self):
        return (
            self.tensor_name,
            self.memory_name,
            self.above_loop_index,
            self.tile_size,
        )

    def __hash__(self):
        return hash((self.tensor_name, self.memory_name, self.above_loop_index))

    def __str__(self):
        return f"[{self.memory_name}] {self.tensor_name} sz {expfmt(self.tile_size)} above {self.above_loop_index}"  # x{expfmt(self.n_repititions)}"

    def __repr__(self):
        return f"TensorStorage({repr(self.tensor_name)}, {self.above_loop_index}, {repr(self.memory_name)}, {self.tile_size})"

    def pydot_str(self):
        return f"[{self.memory_name}] {self.tensor_name} size {expfmt(self.tile_size)}"
        # *{expfmt(self.n_repititions)}={expfmt(self.tile_size)}"# * self.n_repititions)}"

    def rename(
        self, rank_renaming: dict[str, str], tensor_renaming: dict[str, str]
    ) -> "TensorStorage":
        return TensorStorage(
            tensor_renaming[self.tensor_name],
            self.above_loop_index,
            self.memory_name,
            self.tile_size,
            # self.n_repititions
        )

    def to_yaml(self):
        return {
            "type": "storage",
            "tensor_name": self.tensor_name,
            "memory_name": self.memory_name,
            "above_loop_index": self.above_loop_index,
            "tile_size": self.tile_size,
        }

    def get_backing_stores(all_tensors: set["TensorStorage"]) -> list["TensorStorage"]:
        id2tensor = defaultdict(lambda: [])
        for t in all_tensors:
            id2tensor[t.tensor_name].append(t)
        return sorted(sorted(v)[0] for v in id2tensor.values())

    def __eq__(self, value):
        if not isinstance(value, TensorStorage):
            return False
        for to_check in [
            "tensor_name",
            "memory_name",
            "above_loop_index",
        ]:  # $, "tile_size"]:
            a, b = getattr(self, to_check), getattr(value, to_check)
            if a != "*" and b != "*" and a != b:
                return False
        return True

    @staticmethod
    def from_tuple(t: tuple) -> "TensorStorage":
        return TensorStorage(
            tensor_name=t[0],
            memory_name=t[1],
            above_loop_index=t[2],
            tile_size=t[3],
        )

    def to_tuple(self):
        return (
            self.tensor_name,
            self.memory_name,
            self.above_loop_index,
            self.tile_size,
        )


@dataclass(frozen=True)
class Tiling:
    loops: tuple[Loop, ...]
    storage: fzs[TensorStorage]
    tags: Tags = Tags(fzs())

    def __post_init__(self):
        assert isinstance(self.storage, fzs)
        assert isinstance(self.loops, tuple)
        assert isinstance(self.tags, Tags)

    @property
    def tensor_names(self) -> set[str]:
        return {t.tensor_name for t in self.storage}

    def get_backing_levels(self) -> dict[str, int]:
        backings = {}
        for t in self.storage:
            prev = backings.get(t.tensor_name, t.above_loop_index)
            backings[t.tensor_name] = min(prev, t.above_loop_index)
        return backings

    def shared_loop_index(self, live_tensors: set[str]) -> int:
        n = [l for t, l in self.get_backing_levels().items() if t in live_tensors]
        return max(n) - 1 if n else -1

    def __eq__(self, other):
        return (
            self.loops == other.loops
            and self.tags == other.tags
            and self.storage == other.storage
        )

    def __len__(self) -> int:
        return len(self.loops)

    def __hash__(self):
        return hash((self.loops, self.storage, self.tags))

    def clear_dead_tensors(
        self,
        live_tensors: set[str],
        keep_loops: bool = False,
        keep_tensors: set[str] = None,
        drop_tags: bool = False,
    ) -> "Tiling":
        loops = (
            self.loops
            if keep_loops
            else self.loops[: self.shared_loop_index(live_tensors) + 1]
        )
        keep_tensors = keep_tensors if keep_tensors is not None else live_tensors
        tensors = fzs(t for t in self.storage if t.tensor_name in keep_tensors)
        tags = self.tags if not drop_tags else Tags(fzs())
        return Tiling(loops, tensors, tags)

    def __lt__(self, other):
        return (self.loops, self.storage) < (other.loops, other.storage)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"Tiling(loops={self.loops.__repr__()}, storage={self.storage.__repr__()}, tags={self.tags.__repr__()})"

    @staticmethod
    def from_repr(s: str) -> "Tiling":
        return eval(s)

    def merge_next(self, n: "Tiling", live_tensors: set[str]) -> "Tiling":
        tensors = fzs(
            t for t in (n.storage | self.storage) if t.tensor_name in live_tensors
        )
        shared_loop_index = max(t.shared_loop_index(live_tensors) for t in [self, n])

        merged_loops = [l.merge_next(l2) for l, l2 in zip(self.loops, n.loops)]
        additional_loops = n.loops[len(merged_loops) : shared_loop_index + 1]

        return Tiling(
            tuple(merged_loops + list(additional_loops))[: shared_loop_index + 1],
            tensors,
            n.tags,
        )

    def rename(
        self, rank_renaming: dict[str, str], tensor_renaming: dict[str, str]
    ) -> "Tiling":
        return Tiling(
            tuple(l.rename(rank_renaming, tensor_renaming) for l in self.loops),
            fzs(t.rename(rank_renaming, tensor_renaming) for t in self.storage),
            self.tags,
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

    def update(self, **kwargs) -> "Tiling":
        dict_self = {k: v for k, v in self.__dict__.items() if k != "tensor_names"}
        return Tiling(**{**dict_self, **kwargs})

    def set_tags(self, *new_tags: Any) -> "Tiling":
        return self.update(tags=Tags(new_tags))

    def all_n_loops(self) -> list["Tiling"]:
        min_loops = max(t.above_loop_index for t in self.storage)
        return list(
            Tiling(self.loops[:i], self.storage, self.tags)
            for i in range(min_loops, len(self.loops) + 1)
        )

    def add_tensors(self, *tensors: TensorStorage) -> "Tiling":
        return self.update(tensors=self.storage | fzs(tensors))

    def get_tensor_storage(self, tensor_name: str) -> TensorStorage:
        storage = [t for t in self.storage if t.tensor_name == tensor_name]
        assert len(storage) == 1
        return storage[0]

    def set_loop(self, index: int, loop: Loop) -> "Tiling":
        loops = list(self.loops)
        if index == len(loops):
            loops.append(loop)
        else:
            loops[index] = loop
        return self.update(loops=tuple(loops))

    def set_tensor_storage(self, tensor_name: str, storage: TensorStorage) -> "Tiling":
        storage = fzs(
            t if t.tensor_name != tensor_name else storage for t in self.storage
        )
        return self.update(storage=storage)

    def as_sorted_loopnest(self) -> tuple:
        loops = list(self.loops)
        for t in sorted(self.storage, key=lambda t: t.above_loop_index, reverse=True):
            loops.insert(t.above_loop_index, t)
        return tuple(loops)

    def to_tuple(self):
        return (
            tuple(l.to_tuple() for l in self.loops),
            tuple(t.to_tuple() for t in self.storage),
            tuple(t.to_tuple() for t in self.tags),
        )

    @staticmethod
    def from_tuple(t: tuple) -> "Tiling":
        loops = tuple(Loop.from_dict_small(l) for l in t[0])
        storage = fzs(TensorStorage.from_dict_small(t) for t in t[1])
        tags = Tags(fzs(t) for t in t[2])
        return Tiling(loops, storage, tags)


class SIM:
    def __init__(self, tiling: Tiling, mapping: Pareto):
        self.tiling: Tiling = tiling
        self.mapping: Pareto = mapping
        self.storage: dict[str, TensorStorage] = {
            t.tensor_name: t for t in self.tiling.storage
        }
        self.n_pre_prune_mappings = 0

    def tiling_str(self):
        tiling = ",".join(str(l) for l in self.tiling.loops)
        tiling += " || " + ", ".join(str(t) for t in self.storage.values())
        return tiling

    def mapping_str(self):
        return str(self.mapping.einsum_ids())

    @cached_property
    def tensor_names(self) -> set[str]:
        return set(self.storage)

    def copy(self) -> "SIM":
        return SIM(self.tiling, self.mapping.copy())

    def merge_next(
        self, n: "SIM", live_tensors: set[str], delay: bool = False
    ) -> "SIM":
        shared_loop_index = self.tiling.shared_loop_index(
            n.tiling.tensor_names | live_tensors
        )
        tiling = self.tiling.merge_next(n.tiling, live_tensors)
        next_shared_loop_index = tiling.shared_loop_index(live_tensors)
        mapping = self.mapping.merge_next(
            n.mapping, shared_loop_index, live_tensors, delay=delay
        )
        s = SIM(tiling, mapping)
        assert (
            len(tiling.loops) == next_shared_loop_index + 1
        ), f"{self.tiling} {n.tiling} {next_shared_loop_index + 1} -> {tiling} {len(tiling.loops)}"
        s.storage.update(n.storage)
        s.storage.update(self.storage)
        s.n_pre_prune_mappings = len(self.mapping.data) * len(n.mapping.data)
        return s

    def get_shared_loop_index(self, live_tensors: set[str]) -> int:
        live_tensors = list(self.tiling.tensor_names) + [live_tensors]
        return self.tiling.shared_loop_index(live_tensors)

    def free_squish(
        self,
        index: Optional[int],
        resource2capacity: dict[str, int] = None,
    ):
        needs_pareto = False
        needs_pareto = (
            self.mapping.free_to_loop_index(index, resource2capacity) or needs_pareto
        )
        needs_pareto = self.mapping.squish_left_right(index) or needs_pareto
        if needs_pareto:
            self.mapping.make_pareto()

    def _right_consolidate(
        self,
        live_tensors: set[str] = None,
        resource2capacity: dict[str, int] = None,
        shared_tensors: set[str] = None,
    ):
        dead_tensors = set(self.storage) - (live_tensors or set())
        check_tensors = (shared_tensors or set()) | (live_tensors or set())
        shared_loop_index = self.tiling.shared_loop_index(check_tensors)
        for t in dead_tensors:
            t = self.storage.pop(t)
            self.mapping.alloc(
                t.memory_name, t.tile_size, t.above_loop_index, resource2capacity
            )

        if live_tensors is None:
            self.free_squish(0, resource2capacity)
        else:
            self.free_squish(shared_loop_index + 1, resource2capacity)
        return self

    def _left_consolidate(
        self,
        live_tensors: set[str] = None,
        resource2capacity: dict[str, int] = None,
        shared_tensors: set[str] = None,
    ):
        check_tensors = (shared_tensors or set()) | (live_tensors or set())
        shared_loop_index = self.tiling.shared_loop_index(check_tensors)
        tensors_to_add = []
        for t in self.storage.values():
            if (
                t.above_loop_index > shared_loop_index
                or t.tensor_name not in check_tensors
            ):
                self.mapping.alloc(
                    t.memory_name, t.tile_size, t.above_loop_index, resource2capacity
                )
                tensors_to_add.append(t)
        if live_tensors is None:
            self.free_squish(-1, resource2capacity)
        else:
            self.free_squish(shared_loop_index + 1, resource2capacity)
        for t in tensors_to_add:
            self.mapping.add_tensor(t)
        return self

    @staticmethod
    def right_consolidate(
        sims: list["SIM"],
        live_tensors: set[str],
        resource2capacity: dict[str, int] = None,
        shared_tensors: set[str] = None,
        pbar: str = None,
    ) -> list["SIM"]:
        def job(s):
            return s._right_consolidate(live_tensors, resource2capacity, shared_tensors)

        return parallel([delayed(job)(s) for s in sims], pbar=pbar)

    @staticmethod
    def left_consolidate(
        sims: list["SIM"],
        live_tensors: set[str],
        resource2capacity: dict[str, int] = None,
        shared_tensors: set[str] = None,
        pbar: str = None,
    ) -> list["SIM"]:
        def job(s):
            return s._left_consolidate(live_tensors, resource2capacity, shared_tensors)

        return parallel([delayed(job)(s) for s in sims], pbar=pbar)

    def __eq__(self, other):
        return self.tiling == other.tiling and self.storage == other.storage

    def __hash__(self):
        return hash((self.tiling, fzs(self.storage.items())))

    @staticmethod
    def concat(sims: Iterable["SIM"], allow_different_tilings: bool = False) -> "SIM":
        sims = list(sims)
        assert len(sims) > 0, "Cannot concat empty list of SIMs"
        if not allow_different_tilings:
            s = set(fzs([(k, v) for k, v in s.storage.items()]) for s in sims)
            assert (
                len(s) == 1
            ), f"Cannot concat SIMs with different tensors:\n\t" + "\n\t".join(
                str(s2) for s2 in s
            )
        return SIM(sims[0].tiling, Pareto.concat([s.mapping for s in sims]))

    @staticmethod
    def _group(
        sims: list["SIM"],
        live_tensors: set[str],
        keep_loops: bool = False,
        variable_n_loops: bool = False,
        keep_tensors: set[str] = None,
        drop_tags: bool = False,
    ) -> dict[tuple[Tiling, ...], list["SIM"]]:
        grouped = defaultdict(list)
        for s in sims:
            tiling = s.tiling.clear_dead_tensors(
                live_tensors,
                keep_loops=keep_loops or variable_n_loops,
                keep_tensors=keep_tensors,
                drop_tags=drop_tags,
            )
            if variable_n_loops:
                tiling = tiling.all_n_loops()
            else:
                tiling = [tiling]
            for t in tiling:
                grouped[t].append(s)

        return grouped

    @staticmethod
    def combine_combineable(
        sims: list["SIM"],
        live_tensors: set[str],
        allow_different_tilings: bool = False,
        drop_tags: bool = False,
        combine_reservations: bool = True,
    ) -> list["SIM"]:
        no_combine = []
        if not combine_reservations:
            has_reservations = [s.mapping.has_reservations() for s in sims]
            no_combine = [s for s, h in zip(sims, has_reservations) if h]
            sims = [s for s, h in zip(sims, has_reservations) if not h]
        groups = list(SIM._group(sims, live_tensors, drop_tags=drop_tags).values())
        groups_with_one = [g[0] for g in groups if len(g) == 1]
        if len(groups_with_one) == len(groups):
            return groups_with_one + no_combine
        others = parallel(
            [
                delayed(SIM.concat)(g, allow_different_tilings)
                for g in groups
                if len(g) > 1
            ],
            pbar="Combining SIMs",
        )
        return groups_with_one + others + no_combine

    @staticmethod
    def filter_by_tensor_storage(
        sims: list["SIM"] | dict[Tiling, Any], tensors: set[str]
    ) -> list["SIM"]:
        def check(tensors_to_check):
            for t in tensors_to_check:
                for t2 in tensors:
                    if (
                        t2.tensor_name == "*" or t.tensor_name == t2.tensor_name
                    ) and t != t2:
                        return False
            return True

        tensors = set(tensors)
        if isinstance(sims, list):
            return [s for s in sims if check(s.tiling.storage)]
        if isinstance(sims, dict):
            return {k: v for k, v in sims.items() if check(k.storage)}
        raise ValueError(f"Invalid type {type(sims)}")

    def filter_by_mapping_hashes(self, hashes: set[str]) -> Optional["SIM"]:
        self = SIM(self.tiling, self.mapping.filter_by_mapping_hashes(hashes))
        return self if len(self.mapping.data) > 0 else None

    @staticmethod
    def group_left(
        sims: list["SIM"], live_tensors: set[str], drop_tags: bool = False
    ) -> dict[tuple[Tiling, ...], list["SIM"]]:
        return SIM._group(sims, live_tensors, keep_loops=True, drop_tags=drop_tags)

    @staticmethod
    def group_right(
        sims: list["SIM"], live_tensors: set[str], drop_tags: bool = False
    ) -> dict[tuple[Tiling, ...], list["SIM"]]:
        return SIM._group(
            sims, live_tensors, drop_tags=drop_tags, variable_n_loops=True
        )

    @staticmethod
    def remove_dead_tensors(sims: list["SIM"], live_tensors: set[str]) -> list["SIM"]:
        for s in sims:
            for t in list(s.storage):
                if t not in live_tensors:
                    del s.storage[t]

    def set_tags(self, *tags: Any) -> "SIM":
        self.tiling = self.tiling.set_tags(*tags)

    @property
    def tags(self) -> fzs[Any]:
        return self.tiling.tags

    @staticmethod
    def get_possibly_compatible(
        left: list["SIM"],
        right: list["SIM"],
        left_live_tensors: set[str],
        right_live_tensors: set[str],
    ):
        assert left and right, "Cannot check for compatibility with empty list"
        shared_tensors = left[0].tensor_names & right[0].tensor_names
        left = SIM._group(left, right_live_tensors, keep_tensors=shared_tensors)
        right = SIM._group(right, left_live_tensors, keep_tensors=shared_tensors)
        left_keys = set().union(*(l.all_n_loops() for l in left))
        right_keys = set(right)
        left_list = [s for k in left for s in left[k] if k in right_keys]
        right_list = [s for k in right for s in right[k] if k in left_keys]
        return left_list, right_list


import unittest


class TestSIM(unittest.TestCase):
    def test_reservations(self):
        a0 = TensorStorage("A0", 0, "GLB", 1)
        a1 = TensorStorage("A1", 1, "GLB", 2)
        a2 = TensorStorage("A2", 2, "GLB", 3)
        a1b1 = TensorStorage("A1B1", 2, "GLB", 2)  # Above loop 2 --> share 0, 1
        b0 = TensorStorage("B0", 0, "GLB", 4)
        b1 = TensorStorage("B1", 1, "GLB", 5)
        b2 = TensorStorage("B2", 2, "GLB", 6)
        b0c0 = TensorStorage("B0C0", 1, "GLB", 4)  # Above loop 1 --> share 0
        c0 = TensorStorage("C0", 0, "GLB", 7)
        c1 = TensorStorage("C1", 1, "GLB", 8)
        c2 = TensorStorage("C2", 2, "GLB", 9)
        c1d1 = TensorStorage("C1D1", 2, "GLB", 8)  # Above loop 2 --> share 0, 1
        d0 = TensorStorage("D0", 0, "GLB", 10)
        d1 = TensorStorage("D1", 1, "GLB", 11)
        d2 = TensorStorage("D2", 2, "GLB", 12)
        tensors0 = [a0, a1, a2, a1b1]
        tensors1 = [b0, b1, b2, a1b1, b0c0]
        tensors2 = [c0, c1, c2, b0c0, c1d1]
        tensors3 = [d0, d1, d2, c1d1]
        loops = [("R1", "R2"), ("R1", "R2"), ("R1", "R3"), ("R1", "R3")]

        sims: list[SIM] = []
        for tensors, loops in zip([tensors0, tensors1, tensors2, tensors3], loops):
            tiling = Tiling(tuple(Loop(fzs(r), 2, False) for r in loops), fzs(tensors))
            mapping = Pareto(pd.DataFrame({"Energy": [1]}))
            mapping.data[LOGSTRING] = [{}]
            for t in tensors:
                mapping.alloc(t.memory_name, t.tile_size, t.above_loop_index)
            sims.append(SIM(tiling, mapping))

        expected = [
            (1, 2, 5),
            (4, 9, 8),
            (7, 12, 17),
            (10, 11, 20),
        ]
        for e, s in zip(expected, sims):
            for i, e in enumerate(e):
                self.assertEqual(s.mapping.data[nameloop2col("GLB", i)].sum(), e)

    def test_all(self):
        a0 = TensorStorage("A0", 0, "GLB", 1)
        a1 = TensorStorage("A1", 1, "GLB", 2)
        a2 = TensorStorage("A2", 2, "GLB", 3)
        a1b1 = TensorStorage("A1B1", 2, "GLB", 2)  # Above loop 2 --> share 0, 1
        b0 = TensorStorage("B0", 0, "GLB", 4)  #
        b1 = TensorStorage("B1", 1, "GLB", 5)  #
        b2 = TensorStorage("B2", 2, "GLB", 6)  #
        b0c0 = TensorStorage("B0C0", 1, "GLB", 4)  #  # Above loop 1 --> share 0
        c0 = TensorStorage("C0", 0, "GLB", 7)  # 7
        c1 = TensorStorage("C1", 1, "GLB", 8)  # 8
        c2 = TensorStorage("C2", 2, "GLB", 9)  # 9
        c1d1 = TensorStorage("C1D1", 2, "GLB", 8)  # 8 # Above loop 2 --> share 0, 1
        d0 = TensorStorage("D0", 0, "GLB", 10)  #
        d1 = TensorStorage("D1", 1, "GLB", 11)  #
        d2 = TensorStorage("D2", 2, "GLB", 12)  #

        tensors0 = [a0, a1, a2, a1b1]
        tensors1 = [b0, b1, b2, a1b1, b0c0]
        tensors2 = [c0, c1, c2, b0c0, c1d1]
        tensors3 = [d0, d1, d2, c1d1]

        loops = [("R1", "R2"), ("R1", "R2"), ("R1", "R3"), ("R1", "R3")]

        sims: list[SIM] = []
        for tensors, loops in zip([tensors0, tensors1, tensors2, tensors3], loops):
            tiling = Tiling(tuple(Loop(fzs(r), 2, False) for r in loops), fzs(tensors))
            mapping = Pareto(pd.DataFrame({"Energy": [1]}))
            mapping.data[LOGSTRING] = [{}]
            for t in tensors:
                mapping.alloc(t.memory_name, t.tile_size, t.above_loop_index)
            sims.append(SIM(tiling, mapping))

        sims2 = copy.deepcopy(sims)
        while len(sims) > 1:
            sims[0].merge_next(sims.pop(1))
            sims2[0].merge_next(sims2.pop(1))
            sims2[0].consolidate(set().union(*[s.tensor_names for s in sims2]))
        sims[0].consolidate()
        sims2[0].consolidate()
        data0 = sims[0].mapping.data
        data1 = sims2[0].mapping.data
        for k in data0:
            self.assertTrue((data0[k] == data1[k]).all())

        expected_util = max(
            max(a2.tile_size, b2.tile_size)
            + a1.tile_size
            + b1.tile_size
            + a1b1.tile_size,
            max(c2.tile_size, d2.tile_size)
            + c1.tile_size
            + d1.tile_size
            + c1d1.tile_size,
        ) + (b0c0.tile_size + a0.tile_size + b0.tile_size + c0.tile_size + d0.tile_size)

        colname = nameloop2col("GLB", 0)
        self.assertEqual(sims[0].mapping.data[colname].sum(), expected_util)
        self.assertEqual(sims2[0].mapping.data[colname].sum(), expected_util)


if __name__ == "__main__":
    unittest.main()
