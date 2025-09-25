from collections import defaultdict
from enum import Enum
from dataclasses import dataclass, replace
import itertools
from numbers import Number
from typing import Any, Generator, NamedTuple, Union, TypeVar

from fastfusion.frontend.mapping import (
    Iteration,
    Mapping,
    Spatial,
    TensorHolder,
    Reservation as MappingReservation,
    Split as MappingSplit,
    TilePattern,
)
from fastfusion.frontend.renames import RankName, RankVariableName, TensorName
from fastfusion.mapper.FFM.deprecate_maybe.tags import Tags
from fastfusion.mapper.FFM._pmapping_group.df_convention import make_fused_loop_col

from fastfusion.util import expfmt, fzs

# Abstractions:
# 1. Each tensor is stored above some loop index. 0 is the outermost loop, 1 the
#    next-innermost...
# 2. All loops above any shared tensor are co-tiled and must match between SIMs.

T = TypeVar("T", bound="Updatable")


class Updatable:
    def update(self: T, **kwargs) -> T:
        return replace(self, **kwargs)


# class TilePattern(NamedTuple):
#     stride: int
#     initial: int
#     def __str__(self) -> str:
#         return f'<{self.stride}, {self.initial}>'


@dataclass(frozen=True, order=True, eq=True)
class Loop(Updatable):
    rank_name: RankName
    tile_pattern: TilePattern | None
    is_spatial: bool

    def __post_init__(self):
        assert isinstance(self.rank_name, RankName)
        assert isinstance(self.tile_pattern, Number | TilePattern | str | None)
        assert isinstance(self.is_spatial, bool)
        assert isinstance(
            self.tile_pattern.initial_tile_shape,
            Number | str | None,
        )
        assert isinstance(
            self.tile_pattern.stride,
            Number | str | None,
        )

    def __repr__(self):
        return (
            f"Loop({self.rank_name.__repr__()}, {self.tile_pattern}, {self.is_spatial})"
        )

    def __str__(self):
        return (
            "S-" if self.is_spatial else ""
        ) + f"{self.rank_name}-{self.tile_pattern}"

    def pydot_str(self):
        if self.is_spatial:
            return f"S-for R{self.rank_name} size {expfmt(self.tile_pattern)}"
        return f"for {self.rank_name} size {expfmt(self.tile_pattern)}"

    def rename(self, rank_renaming: dict[str, str]) -> "Loop":
        return replace(
            self,
            rank_name=fzs(rank_renaming[r] for r in self.rank_name),
        )

    def to_yaml(self):
        return {"type": "loop", **self.__dict__}

    def merge_next(self, right: "Loop") -> "Loop":
        assert self.tile_pattern == right.tile_pattern
        return Loop(
            self.rank_name | right.rank_name,
            right.tile_pattern,
            self.is_spatial,
        )

    def clear_loop_bound(self, value=0):
        return self.update(tile_pattern=value)

    def populate(self, loop: Iteration) -> "Loop":
        return self.update(tile_pattern=loop.tile_pattern.symbol2str())

    def prepend_symbols(self, prepend: str) -> "Loop":
        return self.update(tile_pattern=self.tile_pattern.prepend_symbols(prepend))

    def clear_tile_patterns(self) -> "Loop":
        return self.update(tile_pattern=TilePattern())

    def make_fused_loop_symbols(self) -> tuple[dict[str, str], "Loop"]:
        r = {}
        new = self

        def replace(attr, new):
            g = getattr(self.tile_pattern, attr)
            if not isinstance(g, str):
                return new
            g2 = make_fused_loop_col(g)
            r[g] = g2
            return new.update(tile_pattern=new.tile_pattern.update(**{attr: g2}))

        new = replace("initial_tile_shape", new)
        new = replace("stride", new)
        new = replace("calculated_n_iterations", new)
        return r, new

    def add_n_iteration_symbols(self) -> "Loop":
        return self.update(tile_pattern=self.tile_pattern.add_n_iteration_symbols())


@dataclass(frozen=True, eq=True, order=True)
class TensorReservation(Updatable):
    # This order is important. Above loop index should be before resource name
    # so when we sort reservations for tensors then the backing tensor holder comes
    # first.
    # Size is not included in hash or equality functions. This is because there
    # may be floating point rounding errors in reservation sizes. The other
    # attributes are sufficient to determine equality.
    loops: tuple[Loop]
    name: TensorName
    resource_name: str

    @property
    def above_loop_index(self) -> int:
        return len(self.loops)

    def __str__(self):
        return f"[{self.resource_name}] {self.name} below {self.loops}"

    def __repr__(self):
        return f"Reservation({repr(self.name)}, {repr(self.loops)}, {repr(self.resource_name)})"

    def pydot_str(self):
        return f"[{self.resource_name}] {self.name}"

    def permute(self, permutation) -> "Reservation":
        new_loops = [self.loops[permutation[i]] for i in range(len(self.loops))]
        return self.update(loops=tuple(new_loops))

    def clear_loop_bounds(self) -> "Reservation":
        return self.update(loops=tuple(loop.clear_loop_bound() for loop in self.loops))

    def populate_loops(self, loops: list[Iteration]) -> "TensorReservation":
        assert len(self.loops) <= len(loops)
        return self.update(
            loops=tuple(l.populate(loop) for l, loop in zip(self.loops, loops))
        )

    def rename(
        self, rank_renaming: dict[str, str], tensor_renaming: dict[str, str]
    ) -> "TensorReservation":
        return replace(
            self,
            name=tensor_renaming[self.name],
            loops=tuple(l.rename(rank_renaming) for l in self.loops),
        )

    @staticmethod
    def get_backing_tensors(
        all_tensors: set["TensorReservation"],
    ) -> list["TensorReservation"]:
        id2tensor = defaultdict(lambda: [])
        for t in all_tensors:
            id2tensor[t.name].append(t)
        return sorted(sorted(v)[0] for v in id2tensor.values())

    def drop_loop_indices(self, loop_indices: set[int]) -> "TensorReservation":
        loops = tuple(l for i, l in enumerate(self.loops) if i not in loop_indices)
        return self.update(loops=loops)

    def prepend_symbols(self, prepend: str) -> "TensorReservation":
        return self.update(
            loops=tuple(l.prepend_symbols(prepend) for l in self.loops),
        )

    def clear_tile_patterns(self) -> "TensorReservation":
        return self.update(
            loops=tuple(l.clear_tile_patterns() for l in self.loops),
        )

    def make_fused_loop_symbols(self) -> tuple[dict[str, str], "TensorReservation"]:
        result = {}
        loops = []
        for l in self.loops:
            r, l = l.make_fused_loop_symbols()
            result.update(r)
            loops.append(l)
        return result, self.update(loops=tuple(loops))

    def add_n_iteration_symbols(self) -> "TensorReservation":
        return self.update(
            loops=tuple(l.add_n_iteration_symbols() for l in self.loops),
        )


class SplitKind(Enum):
    SEQUENTIAL = 0
    PIPELINE = 1


@dataclass(frozen=True, order=True, eq=True)
class Split:
    split: MappingSplit
    above_loop_index: int


@dataclass(frozen=True)
class Compatibility(Updatable):
    tensors: fzs[TensorReservation]
    splits: fzs[Split] = fzs()
    tags: Tags = Tags(fzs())
    reservation_indices: fzs[int] = fzs()
    check_reservation_indices: bool = True

    @property
    def n_loops(self) -> int:
        return max([len(s.loops) for s in self.tensors], default=0)

    @property
    def loops(self) -> tuple[Loop, ...]:
        return max([t.loops for t in self.tensors], key=len) if self.tensors else ()

    def _get_hash_tuple(self):
        return self.n_loops, self.tensors, self.tags, self.reservation_indices

    def __hash__(self):
        return hash(self._get_hash_tuple())

    def __eq__(self, other):
        return self._get_hash_tuple() == other._get_hash_tuple()

    def __post_init__(self):
        assert isinstance(self.n_loops, int)
        assert isinstance(self.tensors, fzs)
        assert isinstance(self.splits, fzs)
        assert isinstance(self.tags, Tags)
        assert isinstance(self.reservation_indices, fzs)
        assert (
            max(self.reservation_indices, default=-1) <= self.n_loops
        ), f"Extra reservation indices {self.reservation_indices} are greater than n_loops {self.n_loops}"
        if self.check_reservation_indices:
            p = f"are not in reservation indices {self.reservation_indices}"
            assert all(
                i >= 0 for i in self.reservation_indices
            ), f"Reservation indices {self.reservation_indices} are not all >= 0"
            assert all(
                s.above_loop_index in self.reservation_indices for s in self.splits
            ), f"Split above loop indices {self.splits} {p}"
            assert all(
                len(s.loops) in self.reservation_indices for s in self.tensors
            ), f"Tensor loops {self.tensors} {p}"

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
        new_n_loops = max((len(s.loops) for s in remaining_tensors), default=0)
        new_splits = fzs(
            split for split in self.splits if split.above_loop_index < new_n_loops
        )
        reservation_indices = fzs(
            {min(i, new_n_loops) for i in self.reservation_indices}
        )
        reservation_indices = fzs(x for x in reservation_indices if x >= 0)

        tags = self.tags if not drop_tags else Tags(fzs())
        return self.update(
            tensors=remaining_tensors,
            splits=new_splits,
            tags=tags,
            reservation_indices=reservation_indices,
        )

    def __lt__(self, other):
        return self._get_hash_tuple() < other._get_hash_tuple()

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"Compatibility(n_loops={self.n_loops}, tensors={repr(self.tensors)}), splits={repr(self.splits)}, tags={repr(self.tags)}"

    def _and_tensors_with_names(self, names: set[str]) -> "Compatibility":
        return fzs(s for s in self.tensors if s.name in names)

    def merge_next(
        self,
        right: "Compatibility",
        live_tensors: set[str],
        mixable_ranks: dict[RankName, set[RankName]] = None,
    ) -> "Compatibility":
        self_freed = self.clear_dead_tensors(live_tensors)
        right_freed = right.clear_dead_tensors(live_tensors)
        if self_freed.n_loops > right_freed.n_loops:
            # This can be relaxed if we have a way to do order-independent joining
            # and/or non-looptree mappings.
            raise ValueError(
                f"Can't merge. I have more loops than the next, so my dataflow can't "
                f"be carried through a LoopTree to where it's needed."
            )

        live_minus_mine = live_tensors - {s.name for s in self.tensors}
        tensors_a = self._and_tensors_with_names(live_tensors)
        tensors_b = right._and_tensors_with_names(live_minus_mine)

        # TODO: tag and split handling?
        joined = Compatibility(
            tensors=tensors_a | tensors_b,
            tags=right.tags,
            reservation_indices=self_freed.reservation_indices
            | right_freed.reservation_indices,
        )

        if mixable_ranks is not None and not joined._is_valid(mixable_ranks):
            raise ValueError(f"Invalid rank mixing.")

        return joined

    def has_tensor(self, *tensors: TensorReservation) -> bool:
        return all(any(s == t for s in self.tensors) for t in tensors)

    def _permute_stops(self):
        stops = set(len(s.loops) for s in self.tensors)
        stops |= self.reservation_indices
        stops |= set(s.above_loop_index for s in self.splits)
        return stops

    def permute(
        self,
        loop_changes: list[int],
    ) -> "Compatibility":
        assert len(loop_changes) <= self.n_loops
        assert set(loop_changes) == set(
            range(len(loop_changes))
        ), f"Loop changes {loop_changes} are not a permutation of {range(len(loop_changes))}"
        if len(loop_changes) < len(self.loops):
            loop_changes = loop_changes + list(
                range(len(loop_changes), len(self.loops))
            )

        permute_stops = self._permute_stops()
        for i, c in enumerate(loop_changes):
            for r in permute_stops:
                assert (i < r) == (
                    c < r
                ), f"Loop changes {loop_changes} cross reservation {r}"
        new_tensors = fzs(s.permute(loop_changes) for s in self.tensors)
        return self.update(tensors=new_tensors)

    def make_equivalent_permutations(self) -> list[tuple["Compatibility", list[int]]]:
        # Get contiguous blocks of loops with no tensor reservation between them
        blocks = []
        current_block = []
        permute_stops = self._permute_stops()
        for i in range(self.n_loops):
            # Can't permute loops if there's a reservation between them
            if i in permute_stops:
                blocks.append(current_block)
                current_block = []
            current_block.append(i)
        if current_block:
            blocks.append(current_block)

        per_block_permutations = [
            list(itertools.permutations(block)) for block in blocks
        ]
        all_permutations = list(itertools.product(*per_block_permutations))
        all_permutations = [
            list(itertools.chain(*loop_changes)) for loop_changes in all_permutations
        ]
        return [(self.permute(p), p) for p in all_permutations]

    def get_tensor_by_name(self, tensor: str) -> TensorReservation:
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
        return self.update(tensors=fzs(t.clear_loop_bounds() for t in self.tensors))

    def compatible_with(self, other: "Compatibility") -> bool:
        return True
        # for a in self.tensors:
        #     a = a.loops
        #     for b in other.tensors:
        #         b = b.loops
        #         if a[:len(b)] != b[:len(a)]:
        #             return False
        # return True

    def populate_loops(self, mapping: Mapping):
        loops = [n for n in mapping.nodes if isinstance(n, Iteration)]
        return self.update(
            tensors=fzs(t.populate_loops(loops) for t in self.tensors),
        )

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping,
        tensors: set[TensorName],
        rank_variable_to_ranks: dict[TensorName, dict[RankVariableName, RankName]],
    ) -> "Compatibility":

        tensor_indices = []
        split_above_loop_indices = []
        reservation_indices = []
        backing_remaining = set(tensors)
        n_seen_loops = 0
        n_fused_loops = 0
        for i, n in enumerate(mapping.nodes):
            if isinstance(n, MappingReservation):
                reservation_indices.append(n_seen_loops)
                if not (backing := set(n.purposes) & backing_remaining):
                    continue
                backing_remaining -= backing
                assert (
                    len(n.purposes) == 1
                ), "Backing reservations should have only one purpose"
                tensor_indices.append(i)
            elif isinstance(n, MappingSplit):
                split_above_loop_indices.append(n_seen_loops)
            elif isinstance(n, Iteration):
                n_seen_loops += 1
                n_fused_loops += bool(backing_remaining)
            elif isinstance(n, TensorHolder):
                reservation_indices.append(n_seen_loops)

        reservation_indices = fzs(min(n, n_fused_loops) for n in reservation_indices)
        reservation_indices = fzs(x for x in reservation_indices if x >= 0)

        assert (
            not backing_remaining
        ), f"Tensors {backing_remaining} not found in mapping"

        def get_rank(rank_variable, tensor):
            rv = rank_variable_to_ranks[tensor][rank_variable]
            assert (
                len(rv) == 1
            ), f"Rank variable {rank_variable} indexes into multiple ranks {rv} for tensor {tensor} "
            return next(iter(rv))

        def make_loops(above_index: int, tensor_name: TensorName) -> list[Iteration]:
            loops = [n for n in mapping.nodes[:above_index] if isinstance(n, Iteration)]
            loops = [
                Loop(
                    rank_name=get_rank(n.rank_variable, tensor_name),
                    tile_pattern=n.tile_pattern.symbol2str(),
                    is_spatial=isinstance(n, Spatial),
                )
                for n in loops
            ]
            return tuple(loops)

        return cls(
            tensors=fzs(
                TensorReservation(
                    name=mapping.nodes[i].purpose,
                    loops=make_loops(i, mapping.nodes[i].purpose),
                    resource_name=mapping.nodes[i].resource,
                )
                for i in tensor_indices
            ),
            splits=fzs(
                Split(split=n, above_loop_index=i) for i in split_above_loop_indices
            ),
            reservation_indices=fzs(reservation_indices),
        )

    def symbols(self) -> list[str]:
        symbols = []

        def add(x: str):
            if isinstance(x, str) and x not in symbols:
                symbols.append(x)

        for t in self.tensors:
            for l in t.loops:
                add(l.tile_pattern.initial_tile_shape)
                add(l.tile_pattern.stride)
                add(l.tile_pattern.calculated_n_iterations)
        return symbols

    def drop_loop_indices(self, loop_indices: set[int]) -> "Compatibility":
        loop_indices = set(loop_indices)
        tensors = fzs(t.drop_loop_indices(loop_indices) for t in self.tensors)
        splits = fzs(s for s in self.splits if s.above_loop_index not in loop_indices)

        def adjust(i: int) -> int:
            return i - sum(x < i for x in loop_indices)

        reservation_indices = fzs(adjust(i) for i in self.reservation_indices)
        reservation_indices = fzs(x for x in reservation_indices if x >= 0)

        splits = fzs(
            s.update(above_loop_index=adjust(s.above_loop_index)) for s in self.splits
        )
        return Compatibility(
            tensors=tensors,
            splits=splits,
            reservation_indices=reservation_indices,
        )

    def prepend_symbols(self, prepend: str) -> "Compatibility":
        return self.update(
            tensors=fzs(t.prepend_symbols(prepend) for t in self.tensors)
        )

    def clear_tile_patterns_and_reservation_indices(self) -> "Compatibility":
        return self.update(
            tensors=fzs(t.clear_tile_patterns() for t in self.tensors),
            reservation_indices=fzs(),
            check_reservation_indices=False,
        )

    def make_fused_loop_symbols(self) -> tuple[dict[str, str], "Compatibility"]:
        result = {}
        tensors = []
        for t in self.tensors:
            r, t = t.make_fused_loop_symbols()
            tensors.append(t)
            result.update(r)

        return result, self.update(tensors=fzs(tensors))

    def add_n_iteration_symbols(self) -> "Compatibility":
        return self.update(
            tensors=fzs(t.add_n_iteration_symbols() for t in self.tensors)
        )

    def _is_valid(self, mixable_ranks: dict[RankName, set[RankName]]) -> bool:
        # Mixable ranks: Ranks that may be co-iterated by a single loop.
        ranks_at_each_loop_index = []
        for i in range(self.n_loops):
            ranks_at_each_loop_index.append(
                set(t.loops[i].rank_name for t in self.tensors if i < len(t.loops))
            )

        for ranks in ranks_at_each_loop_index:
            for r1, r2 in itertools.combinations(ranks, 2):
                if r1 not in mixable_ranks[r2]:
                    return False
        return True
