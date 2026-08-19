from collections import defaultdict
from numbers import Number
from dataclasses import dataclass, field, replace
import functools
import itertools
from typing import Callable, Iterable, Literal, TypeVar

from pandas import DataFrame

from accelforge._accelerated_imports import pandas as pd
from accelforge.frontend.mapping import (
    Mapping,
    Spatial,
    TensorHolder,
    Storage as MappingStorage,
    Reservation as MappingReservation,
    Split as MappingSplit,
    TilePattern,
    Loop as MappingLoop,
)
from accelforge.frontend.mapping.mapping import PatternSymbol
from accelforge.frontend.renames import Rank, RankVariable, TensorName
from accelforge.frontend.workload import Einsum, Workload
from accelforge.mapper.FFM._pareto_df.df_convention import (
    col2iterations,
    is_fused_loop_col,
    make_fused_loop_col,
    is_binding_col,
    make_binding_col,
    stride2col,
    initial2col,
    iterations2col,
)

from accelforge.util import _expfmt, fzs, oset

# Abstractions:
# 1. Each tensor is stored above some loop index. 0 is the outermost loop, 1 the
#    next-innermost...
# 2. All loops above any shared tensor are co-tiled and must match between
#    PmappingGroups.
# 3. Spatial loops *below* a physically-distributed storage (i.e., the data binding)
#    must match. These are in TensorReservations.physical_spatial_loops.

T = TypeVar("T", bound="Updatable")


class Updatable:
    def update(self: T, **kwargs) -> T:
        return replace(self, **kwargs)


def _update_rename_dict(
    renames: dict[str, str],
    new_renames: dict[str, str],
):
    for mine, other in new_renames.items():
        if mine not in renames:
            renames[mine] = other
        elif renames[mine] != other:
            raise ValueError(
                f"Renaming {mine} to {other} conflicts with {renames[mine]}"
            )


@dataclass(frozen=True, order=True, eq=True)
class Loop(Updatable):
    rank_name: Rank
    tile_pattern: TilePattern | None
    id: int = field(compare=False, hash=False)

    def __post_init__(self):
        assert isinstance(self.rank_name, Rank)
        assert isinstance(self.tile_pattern, TilePattern)
        assert isinstance(
            self.tile_pattern.initial_tile_shape,
            Number | str | None,
        ), f"instead is {self.tile_pattern.initial_tile_shape}"
        assert isinstance(
            self.tile_pattern.tile_shape,
            Number | str | None,
        ), f"tile pattern is {self.tile_pattern.tile_shape}"

    def __repr__(self):
        return f"Loop({self.rank_name.__repr__()}, {self.tile_pattern})"

    def __str__(self):
        return f"{self.rank_name}-{self.tile_pattern}"

    def pydot_str(self):
        return f"for {self.rank_name} size {_expfmt(self.tile_pattern)}"

    def to_yaml(self):
        return {"type": "loop", **self.__dict__}

    def populate(self, ranks_with_tile_pattern: set[Rank]) -> "Loop":
        initial = (
            initial2col(self.rank_name, self.id)
            if self.rank_name in ranks_with_tile_pattern
            else None
        )
        # NO RANK loops don't index the tensor, so they have no stride column
        tile_shape = (
            stride2col(self.rank_name, self.id)
            if self.rank_name != Rank("NO RANK. RECOMPUTED.")
            else None
        )
        tile_pattern = TilePattern(
            tile_shape=tile_shape,
            initial_tile_shape=initial,
            calculated_n_iterations=iterations2col(self.id),
        )
        return self.update(tile_pattern=tile_pattern)

    def _prepend_symbols(self, prepend: str) -> "Loop":
        return self.update(tile_pattern=self.tile_pattern._prepend_symbols(prepend))

    def clear_symbolic_tile_patterns(self) -> "Loop":
        return self.update(tile_pattern=self.tile_pattern._clear_symbols())

    def _make_symbols(
        self, prefix: str, make_col: Callable
    ) -> tuple[dict[str, str], "Loop"]:
        renames = {}
        pattern = self.tile_pattern
        for attr in pattern._symbol_attrs():
            val = getattr(pattern, attr)
            if isinstance(val, str):
                renames[val] = make_col(f"{prefix}<SEP>{val}")
                pattern = pattern.update(**{attr: renames[val]})
        return renames, self.update(tile_pattern=pattern)

    def _rename_to_match(self, other: "Loop") -> tuple["Loop", dict[str, str]]:
        new_tp, renames = self.tile_pattern._rename_to_match(other.tile_pattern)
        return self.update(rank_name=other.rank_name, tile_pattern=new_tp), renames


def _loop_sort_key(l: "Loop"):
    p = l.tile_pattern
    return (
        l.rank_name,
        repr((p.tile_shape, p.initial_tile_shape, p.calculated_n_iterations)),
    )


@dataclass(frozen=True, order=True, eq=True)
class PermutableLoopBlock(Updatable):
    loops: tuple[Loop, ...]
    is_spatial: bool
    # The architecture spatial dimension (e.g. "X", "Y") this loop fans out over.
    spatial_dim: str | None = None

    def __post_init__(self):
        assert isinstance(self.is_spatial, bool)
        assert isinstance(self.spatial_dim, str | None)
        for l in self.loops:
            assert isinstance(l, Loop)
        assert len(set(l.rank_name for l in self.loops)) == len(
            self.loops
        ), "Duplicate rank names in loops"
        # Loops in a block may be reordered freely, so store them in a
        # canonical order. Blocks with the same loops compare and hash equal.
        object.__setattr__(self, "loops", tuple(sorted(self.loops, key=_loop_sort_key)))

    def _update_loops(self, f) -> "PermutableLoopBlock":
        return self.update(loops=tuple(f(l) for l in self.loops))

    def __len__(self):
        return len(self.loops)

    def _loop_comma_join(self, f):
        return ", ".join(f(l) for l in self.loops)

    def __repr__(self):
        prepend = f"S-{self.spatial_dim}-" if self.is_spatial else "T-"
        return f"{prepend}PermutableLoopBlock(" + self._loop_comma_join(repr) + ")"

    def __str__(self):
        prepend = f"S-{self.spatial_dim}-" if self.is_spatial else "T-"
        return f"{prepend}(" + self._loop_comma_join(str) + ")"

    def to_yaml(self):
        return {"type": "loop", **self.__dict__}

    def populate(self, ranks_with_tile_pattern: set[Rank]) -> "PermutableLoopBlock":
        return self._update_loops(lambda l: l.populate(ranks_with_tile_pattern))

    def _prepend_symbols(self, prepend: str) -> "PermutableLoopBlock":
        return self._update_loops(lambda l: l._prepend_symbols(prepend))

    def clear_symbolic_tile_patterns(self) -> "PermutableLoopBlock":
        return self._update_loops(lambda l: l.clear_symbolic_tile_patterns())

    def _make_symbols(
        self, prefix: str, make_col
    ) -> tuple[dict[str, str], "PermutableLoopBlock"]:
        r = {}
        loops = []
        for l in self.loops:
            r2, l2 = l._make_symbols(prefix, make_col)
            r.update(r2)
            loops.append(l2)
        return r, self.update(loops=tuple(loops))

    def make_fused_loop_symbols(
        self, prefix: str
    ) -> tuple[dict[str, str], "PermutableLoopBlock"]:
        return self._make_symbols(prefix, make_fused_loop_col)

    def make_binding_symbols(
        self, prefix: str
    ) -> tuple[dict[str, str], "PermutableLoopBlock"]:
        assert self.is_spatial
        return self._make_symbols(prefix, make_binding_col)

    def _rename_to_match(
        self, other: "PermutableLoopBlock"
    ) -> tuple["PermutableLoopBlock", dict[str, str]]:
        r = {}
        loops = []
        assert len(self) == len(other), "PermutableLoopBlocks have different lengths"
        for l_mine, l_other in self._match_loops(other):
            l_mine, new_renames = l_mine._rename_to_match(l_other)
            _update_rename_dict(r, new_renames)
            loops.append(l_mine)
        return self.update(loops=tuple(loops)), r

    def _match_loops(self, others: "PermutableLoopBlock") -> list["Loop"]:
        r2loop = {l.rank_name: l for l in self.loops}
        matched = [(r2loop.get(l.rank_name, None), l) for l in others.loops]
        assert all(l is not None for l, _ in matched), f"{others} not found in {self}"
        assert len(self) == len(others), f"Loop counts differ: {self} vs {others}"
        # PermutableLoopBlock already asserts that rank names are unique
        return matched


@dataclass(frozen=True, eq=True, order=True)
class TensorReservation(Updatable):
    # This order is important. Above loop index should be before resource name
    # so when we sort reservations for tensors then the backing tensor holder comes
    # first.
    # Size is not included in hash or equality functions. This is because there
    # may be floating point rounding errors in reservation sizes. The other
    # attributes are sufficient to determine equality.
    loops: tuple[PermutableLoopBlock, ...]
    name: TensorName
    resource_name: str
    persistent: bool = False
    # Spatial loops *below* this storage that distribute the tensor across physical
    # instances
    physical_spatial_loops: tuple[PermutableLoopBlock, ...] = ()
    # prod(# iterations) of all fused loops above this reservationx
    n_iterations: PatternSymbol | None = None

    def __post_init__(self):
        if self.persistent:
            assert len(self.loops) == 0, "Persistent tensors be above all loops"
        assert all(
            isinstance(l, PermutableLoopBlock) and l.is_spatial
            for l in self.physical_spatial_loops
        ), "physical_spatial_loops must all be spatial Loops"
        for i in range(len(self.physical_spatial_loops)):
            for j in range(i + 1, len(self.physical_spatial_loops)):
                assert (
                    self.physical_spatial_loops[i].spatial_dim
                    != self.physical_spatial_loops[j].spatial_dim
                ), "physical_spatial_loops must have unique spatial dimensions"

    @property
    def n_loops(self) -> int:
        return sum(len(l) for l in self.loops)

    @property
    def above_loop_index(self) -> int:
        return -1 if self.persistent else self.n_loops

    def __str__(self):
        return f"[{self.resource_name}] {self.name} below {self.loops}"

    def __repr__(self):
        phys = (
            f", physical_spatial_loops={repr(self.physical_spatial_loops)}"
            if self.physical_spatial_loops
            else ""
        )
        return f"Reservation({repr(self.name)}, {repr(self.loops)}, {repr(self.resource_name)}{phys})"

    def pydot_str(self):
        return f"[{self.resource_name}] {self.name}"

    def populate_loops(self, ranks_with_tile_pattern: set[Rank]) -> "TensorReservation":
        updated_loops = self.update(
            loops=tuple(l.populate(ranks_with_tile_pattern) for l in self.loops),
            n_iterations=iterations2col(self.name),
        )
        if len(self.physical_spatial_loops) > 0:
            raise NotImplementedError()
            new_loop = []
            for compat_loop, mapping_loop in zip(
                updated_loops.physical_spatial_loops, reservation_node.binding
            ):
                mapping_tile_pattern = mapping_loop.tile_pattern._symbol2str()
                new_compat_tile_pattern = mapping_tile_pattern
                new_loop.append(
                    compat_loop.update(tile_pattern=new_compat_tile_pattern)
                )
            return updated_loops.update(physical_spatial_loops=tuple(new_loop))
        else:
            return updated_loops

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

    def _prepend_symbols(self, prepend: str) -> "TensorReservation":
        n_iterations = self.n_iterations
        if isinstance(n_iterations, str):
            n_iterations = prepend + n_iterations
        return self.update(
            loops=tuple(l._prepend_symbols(prepend) for l in self.loops),
            n_iterations=n_iterations,
        )

    def clear_symbolic_tile_patterns(self) -> "TensorReservation":
        return self.update(
            loops=tuple(l.clear_symbolic_tile_patterns() for l in self.loops),
            physical_spatial_loops=tuple(
                l.clear_symbolic_tile_patterns() for l in self.physical_spatial_loops
            ),
            n_iterations=None,
        )

    def make_fused_loop_symbols(
        self, prefix: str
    ) -> tuple[dict[str, str], "TensorReservation"]:
        result = {}
        loops = []
        for l in self.loops:
            r, l = l.make_fused_loop_symbols(prefix)
            result.update(r)
            loops.append(l)
        physical_loops = []
        for l in self.physical_spatial_loops:
            r, l = l.make_binding_symbols(prefix)
            result.update(r)
            physical_loops.append(l)
        n_iterations = self.n_iterations
        if isinstance(n_iterations, str):
            n_iterations = make_fused_loop_col(f"{prefix}<SEP>{n_iterations}")
            result[self.n_iterations] = n_iterations
        return (
            result,
            self.update(
                loops=tuple(loops),
                physical_spatial_loops=tuple(physical_loops),
                n_iterations=n_iterations,
            ),
        )

    def iter_fused_loops(self) -> Iterable[Loop]:
        for b in self.loops:
            for l in b.loops:
                yield l

    def _rename_to_match(
        self, other: "TensorReservation"
    ) -> tuple["TensorReservation", dict[str, str]]:
        renames = {}
        new_loops = []
        for l_mine, l_other in zip(self.loops, other.loops):
            l_mine, new_renames = l_mine._rename_to_match(l_other)
            _update_rename_dict(renames, new_renames)
            new_loops.append(l_mine)
        new_physical = []
        for l_mine, l_other in zip(
            self.physical_spatial_loops, other.physical_spatial_loops
        ):
            l_mine, new_renames = l_mine._rename_to_match(l_other)
            _update_rename_dict(renames, new_renames)
            new_physical.append(l_mine)
        if (
            isinstance(self.n_iterations, str)
            and isinstance(other.n_iterations, str)
            and self.n_iterations != other.n_iterations
        ):
            _update_rename_dict(renames, {self.n_iterations: other.n_iterations})
        return (
            self.update(
                loops=tuple(new_loops),
                physical_spatial_loops=tuple(new_physical),
                n_iterations=other.n_iterations,
            ),
            renames,
        )

    def shape(self) -> dict[Rank, tuple[PatternSymbol, ...]]:
        # Tile shape because iterations may not match up if there's multiple of a fused
        # loop (e.g., two "for m in [0..2)" loops versus one "for m in [0..4)" loop).
        result = {}
        for l in self.iter_fused_loops():
            p = l.tile_pattern
            if p.initial_tile_shape is not None:
                result[l.rank_name] = (p.tile_shape, p.initial_tile_shape)
            else:
                result[l.rank_name] = (p.tile_shape,)
        return result

    def physical_shape(self) -> dict[str, dict[Rank, tuple[PatternSymbol, ...]]]:
        # Calculated iterations (not tile shape like shape() above) because the shape
        # may change when we make equivalent permutations of fused loops. (e.g.,
        # "S-for-X m in [0..2), S-for-Y m in [0..2)" is identical no matter what order
        # the loops are in, but the tile shapes would be different if the loops were
        # swapped). We don't have the multiple-loops problem like shape() does because
        # there's at most one loop per rank per spatial dimension.
        return {
            b.spatial_dim: {
                l.rank_name: (l.tile_pattern.calculated_n_iterations,) for l in b.loops
            }
            for b in self.physical_spatial_loops
        }

    def _sorted_rank_tuples(self):
        ranks = []
        for r, shape in sorted(self.shape().items()):
            ranks.append((r, shape))

        physical_ranks = []
        for dim, shape in sorted(self.physical_shape().items()):
            cur_ranks = []
            for r, shape in sorted(shape.items()):
                cur_ranks.append((r, shape))
            physical_ranks.append((dim, cur_ranks))
        return ranks, physical_ranks

    def loop_structure(self) -> tuple[tuple, ...]:
        # Format: tuple of tuples
        # - Ranks
        # - Spatial dims
        # - For each spatial dim
        #    - Ranks
        ranks, physical_ranks = self._sorted_rank_tuples()
        structure = [
            tuple(r[0] for r in ranks),
            tuple(p[0] for p in physical_ranks),
        ]
        for _, ranks in physical_ranks:
            structure.append(tuple(r[0] for r in ranks))
        return tuple(structure)

    def compatibility_shape_symbols(self) -> tuple[PatternSymbol, ...]:
        ranks, physical_ranks = self._sorted_rank_tuples()
        all_shapes = [r[1] for r in ranks]
        all_shapes += [p[1] for _, ps in physical_ranks for p in ps]
        # Nones come from irrelevant ranks, and don't affect shape
        symbols = tuple(
            s for s in itertools.chain.from_iterable(all_shapes) if s is not None
        )
        # Irrelevant ranks do affect the total iterations though, so it's caught here
        if self.n_iterations is not None:
            symbols += (self.n_iterations,)
        return symbols

    def _compatibility_values_df(self, df: DataFrame) -> DataFrame:
        return pd.DataFrame(
            {
                f"__compatibility_col{i}": _symbol_values(s, df)
                for i, s in enumerate(self.compatibility_shape_symbols())
            },
            index=df.index,
        )

    def iter_compatibility_tuples(self, df: pd.DataFrame):
        frame = self._compatibility_values_df(df).drop_duplicates()
        yield from frame.itertuples(index=False, name=None)


@dataclass(frozen=True)
class Compatibility(Updatable):
    tensors: fzs[TensorReservation]

    @property
    def n_loops(self) -> int:
        try:
            return object.__getattribute__(self, "_n_loops_cached")
        except AttributeError:
            val = max((s.n_loops for s in frozenset.__iter__(self.tensors)), default=0)
            object.__setattr__(self, "_n_loops_cached", val)
            return val

    @property
    def loops(self) -> tuple[PermutableLoopBlock, ...]:
        key = lambda t: t.n_loops
        return max(self.tensors, key=key, default=None).loops if self.tensors else ()

    def _get_hash_tuple(self):
        return self.n_loops, self.tensors

    def __hash__(self):
        try:
            return object.__getattribute__(self, "_hash_cached")
        except AttributeError:
            val = hash(self._get_hash_tuple())
            object.__setattr__(self, "_hash_cached", val)
            return val

    def __eq__(self, other):
        if self is other:
            return True
        return self._get_hash_tuple() == other._get_hash_tuple()

    def __post_init__(self):
        assert isinstance(self.tensors, fzs)

    def get_backing_levels(self) -> dict[str, int]:
        backings = {}
        for t in self.tensors:
            prev = backings.get(t.name, t.above_loop_index)
            backings[t.name] = min(prev, t.above_loop_index)
        return backings

    @property
    def tensor_names(self) -> set[str]:
        return oset(t.name for t in self.tensors)

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

    def _rename_to_match(
        self, other: "Compatibility"
    ) -> tuple["Compatibility", dict[str, str]]:
        renames = {}
        assert (
            self.clear_symbolic_tile_patterns() == other.clear_symbolic_tile_patterns()
        )
        tensors = []
        for t in self.tensors:
            other_t = other.get_reservation_of_tensor(t.name)
            t, new_renames = t._rename_to_match(other_t)
            tensors.append(t)
            _update_rename_dict(renames, new_renames)

        return self.update(tensors=fzs(tensors)), renames

    def clear_dead_tensors(
        self,
        live_tensors: set[str] | Literal["All"],
    ) -> "Compatibility":
        if live_tensors == "All":
            live_tensors = self.tensor_names
        return _clear_dead_tensors_cached(self, fzs(live_tensors))

    def _clear_dead_tensors_uncached(self, live_tensors: set[str]) -> "Compatibility":
        remaining_tensors = fzs(s for s in self.tensors if s.name in live_tensors)
        return self.update(tensors=remaining_tensors)

    def __lt__(self, other):
        return self._get_hash_tuple() < other._get_hash_tuple()

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"Compatibility(n_loops={self.n_loops}, tensors={repr(self.tensors)})"

    def _and_tensors_with_names(self, names: set[str]) -> "Compatibility":
        return fzs(s for s in self.tensors if s.name in names)

    def merge_next(
        self,
        right: "Compatibility",
        live_tensors: set[str],
    ) -> list[tuple["Compatibility", list[tuple]]]:

        # =====================================================================================
        # VIBE CODE WARNING VIBE CODE WARNING VIBE CODE WARNING VIBE CODE WARNING VIBE CODE
        # WARNING VIBE CODE WARNING VIBE CODE WARNING VIBE CODE WARNING VIBE CODE WARNING VIBE
        # CODE WARNING VIBE CODE WARNING VIBE CODE WARNING VIBE CODE WARNING VIBE CODE WARNING
        # VIBE CODE WARNING VIBE CODE WARNING VIBE CODE WARNING VIBE CODE WARNING VIBE CODE
        # WARNING VIBE CODE WARNING VIBE CODE WARNING VIBE CODE WARNING VIBE CODE WARNING!!!!!
        # Merge_next is vibe coded.
        # =====================================================================================
        """Returns [(joined compatibility, pattern match pairs)]."""
        result = _merge_next_cached(
            self,
            right,
            fzs(live_tensors),
        )
        if isinstance(result, ValueError):
            raise result
        return result

    def _merge_next_uncached(
        self,
        right: "Compatibility",
        live_tensors: set[str],
    ) -> list[tuple["Compatibility", list[tuple]]]:
        self_freed = self.clear_dead_tensors(live_tensors)
        right_freed = right.clear_dead_tensors(live_tensors)
        if self_freed.n_loops > right_freed.n_loops:
            # This can be relaxed if we have a way to do order-independent joining
            # and/or non-looptree mappings.
            raise ValueError(
                f"Can't merge. I have more loops than the next, so my dataflow can't "
                f"be carried through a LoopTree to where it's needed."
            )

        for a, b in itertools.product(self.tensors, right.tensors):
            if a.name == b.name and a.resource_name != b.resource_name:
                raise ValueError(
                    f"Can't merge. Tensor {a.name} is stored in different resources "
                    f"({a.resource_name} vs {b.resource_name})"
                )

        # One global pairing aligns my whole nest with the top of the right
        # side's nest. The same pairing makes the DataFrame column matches
        # AND the joined block structure, so a surviving row always has a
        # loop order satisfying both sides' reservations — which mapping
        # reconstruction later depends on.
        shared = self.tensor_names & right.tensor_names
        left_loop_to_right_loop = []
        for name in shared:
            lt = self.get_reservation_of_tensor(name)
            rt = right.get_reservation_of_tensor(name)
            if _reservation_join_key(lt) != _reservation_join_key(rt):
                raise ValueError(
                    f"Can't merge. Tensor {name} is backed at different depths."
                )
            physical = _pair_physical_loops(lt, rt)
            if physical is None:
                raise ValueError(
                    f"Can't merge. Tensor {name}'s physical spatial loops "
                    f"don't pair."
                )
            for ba, bb in physical:
                for a, b in ba._match_loops(bb):
                    left_loop_to_right_loop += _tile_left_loop_to_right_loop(a, b)
        paired = _pair_loops(self, right, shared)
        if paired is None:
            raise ValueError(
                "Can't merge. No pairing aligns the loops and reservations."
            )

        pairs, stops, leftover_blocks = paired
        left_loop_to_right_loop += pairs
        left_col2right_col = dict(pairs)
        for name in shared:
            rt = right.get_reservation_of_tensor(name)
            right_loops = {
                l.tile_pattern.calculated_n_iterations: l for l in rt.iter_fused_loops()
            }
            for l in self.get_reservation_of_tensor(name).iter_fused_loops():
                col = left_col2right_col[l.tile_pattern.calculated_n_iterations]
                left_loop_to_right_loop += _tile_left_loop_to_right_loop(
                    l, right_loops[col]
                )

        joined = self._merge_paired_loops(
            right, live_tensors, pairs, stops, leftover_blocks
        )
        return [(joined, list(dict.fromkeys(left_loop_to_right_loop)))]

    def _merge_paired_loops(
        self,
        right: "Compatibility",
        live_tensors: set[str],
        pairs: list[tuple[str, str]],
        stops: set[int],
        leftover_blocks: list[list[str]],
    ) -> "Compatibility":
        # The joined loop order is the paired loops in pairing order, then
        # the right side's unpaired loops in their order. Right tensors keep
        # their own tile shape columns but take the left loop's
        # calculated_n_iterations, so every tensor's loops share one order.
        order = {}
        for i, (lc, rc) in enumerate(pairs):
            order[lc] = order[rc] = i
        stops = set(stops) | {len(pairs)}
        n = len(pairs)
        for cols in leftover_blocks:
            for c in cols:
                order[c] = n
                n += 1
            stops.add(n)

        def rebuild(t: TensorReservation, col2col: dict[str, str]) -> TensorReservation:
            singles = []
            for b in t.loops:
                for l in b.loops:
                    col = l.tile_pattern.calculated_n_iterations
                    col = col2col.get(col, col)
                    l = l.update(
                        tile_pattern=l.tile_pattern.update(calculated_n_iterations=col)
                    )
                    singles.append((order[col], b.is_spatial, l))
            singles.sort(key=lambda x: x[0])
            blocks = []
            for run in _cut_at_counts(singles, stops):
                spatials = oset(sp for _, sp, _ in run)
                assert (
                    len(spatials) == 1
                ), "Merged block mixes spatial and temporal loops"
                blocks.append(
                    PermutableLoopBlock(
                        loops=tuple(l for _, _, l in run),
                        is_spatial=next(iter(spatials)),
                    )
                )
            return t.update(loops=tuple(blocks))

        right_col2left_col = {rc: lc for lc, rc in pairs}
        live_minus_mine = live_tensors - self.tensor_names
        return Compatibility(
            tensors=fzs(
                [rebuild(t, {}) for t in self._and_tensors_with_names(live_tensors)]
                + [
                    rebuild(t, right_col2left_col)
                    for t in right._and_tensors_with_names(live_minus_mine)
                ]
            )
        )

    def per_tensor_compatibility(self) -> dict[str, "Compatibility"]:
        result = {}
        for s in self.tensors:
            result[s.name] = self.clear_dead_tensors(oset([s.name]))
        return result

    def populate_loops(self, ranks_with_tile_pattern: set[Rank], mapping: Mapping):
        return self.update(
            tensors=fzs(
                t.populate_loops(ranks_with_tile_pattern) for t in self.tensors
            ),
        )

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping,
        tensors: set[TensorName],
        einsum: Einsum,
    ) -> "Compatibility":
        """
        Create Compatibility from a mapping, a set of fusable tensors, and the
        workload.
        """
        if not isinstance(einsum, Einsum):
            raise TypeError(f"einsum should be an Einsum, but {type(einsum)} instead")
        rank_variable_to_ranks = {
            t.name: t.rank_variable2ranks for t in einsum.tensor_accesses
        }

        tensor_indices = []
        split_above_loop_indices = []
        permutation_stops = []
        backing_remaining = oset(tensors)
        n_seen_loops = 0
        n_fused_loops = 0
        prev_loop = None
        for i, n in enumerate(mapping.nodes):
            if isinstance(n, MappingReservation):
                permutation_stops.append(n_seen_loops)
                if not (backing := oset(n.purposes) & backing_remaining):
                    continue
                backing_remaining -= backing
                assert (
                    len(n.purposes) == 1
                ), "Backing reservations should have only one purpose"
                tensor_indices.append(i)
            elif isinstance(n, MappingSplit):
                split_above_loop_indices.append(n_seen_loops)
            elif isinstance(n, MappingLoop):
                if prev_loop is not None and isinstance(
                    prev_loop, Spatial
                ) != isinstance(n, Spatial):
                    permutation_stops.append(n_seen_loops)
                prev_loop = n
                n_seen_loops += 1
                n_fused_loops += bool(backing_remaining)
            elif isinstance(n, TensorHolder):
                permutation_stops.append(n_seen_loops)

        permutation_stops = fzs(
            min(n, n_fused_loops) for n in permutation_stops + split_above_loop_indices
        )

        assert (
            not backing_remaining
        ), f"Tensors {backing_remaining} not found in mapping"

        id2idx = mapping._get_node_ids()

        def make_pattern(n: MappingLoop) -> TilePattern:
            return n.tile_pattern._symbol2str().update(
                calculated_n_iterations=iterations2col(id2idx[id(n)])
            )

        def get_rank(rank_variable, tensor):
            """
            Return rank in tensor indexed by rank_variable or
            Rank("NO RANK.RECOMPUTED") if rank not in tensor.
            """
            # TODO: shouldn't this whole logic use relevancy from workload?
            rv = rank_variable_to_ranks[tensor].get(rank_variable, oset())
            assert (
                len(rv) <= 1
            ), f"Rank variable {rank_variable} indexes into multiple ranks {rv} for tensor {tensor} "
            return next(iter(rv), Rank("NO RANK. RECOMPUTED."))

        def make_loops(above_index: int, tensor_name: TensorName):
            loop_nodes = [
                n for n in mapping.nodes[:above_index] if isinstance(n, MappingLoop)
            ]
            return tuple(
                PermutableLoopBlock(
                    loops=tuple(
                        Loop(
                            rank_name=get_rank(n.rank_variable, tensor_name),
                            tile_pattern=make_pattern(n),
                            id=id2idx[id(n)],
                        )
                        for n in run
                    ),
                    is_spatial=isinstance(run[0], Spatial),
                )
                for run in _cut_at_counts(loop_nodes, permutation_stops)
            )

        def make_physical_spatial_loops(above_index: int) -> tuple[PermutableLoopBlock]:
            """Make data binding of physically distributed storages."""
            assert isinstance(
                reservation_node := mapping.nodes[above_index], MappingReservation
            )
            assert isinstance(
                storage_node := reservation_node._tensor_holder, MappingStorage
            )
            memory = reservation_node._component_object
            if memory is None or not memory._is_distributed():
                return ()
            return tuple(
                PermutableLoopBlock(
                    loops=(
                        Loop(
                            rank_name=loop.rank_variable,
                            tile_pattern=loop.tile_pattern._symbol2str(),
                            id=id2idx[id(loop)],
                        ),
                    ),
                    is_spatial=True,
                    spatial_dim=loop.name,
                )
                for loop in storage_node.binding
            )

        made_tensors = fzs(
            TensorReservation(
                name=mapping.nodes[i].purpose,
                loops=make_loops(i, mapping.nodes[i].purpose),
                resource_name=mapping.nodes[i].resource,
                persistent=mapping.nodes[i].persistent,
                physical_spatial_loops=make_physical_spatial_loops(i),
                n_iterations=iterations2col(mapping.nodes[i].purpose),
            )
            for i in tensor_indices
        )
        return cls(tensors=made_tensors)

    def symbols(self) -> list[str]:
        symbols = []

        def add(x: str):
            if isinstance(x, str) and x not in symbols:
                symbols.append(x)

        for t in self.tensors:
            add(t.n_iterations)
            for b in itertools.chain(t.loops, t.physical_spatial_loops):
                for l in b.loops:
                    add(l.tile_pattern.initial_tile_shape)
                    add(l.tile_pattern.tile_shape)
                    add(l.tile_pattern.calculated_n_iterations)
        return symbols

    def drop_loops(self, ids: set[int]) -> "Compatibility":
        ids = oset(ids)

        def drop(t: TensorReservation) -> TensorReservation:
            blocks = []
            for b in t.loops:
                if kept := tuple(l for l in b.loops if l.id not in ids):
                    blocks.append(b.update(loops=kept))
            return t.update(loops=tuple(blocks))

        return Compatibility(tensors=fzs(drop(t) for t in self.tensors))

    def _prepend_symbols(self, prepend: str) -> "Compatibility":
        return self.update(
            tensors=fzs(t._prepend_symbols(prepend) for t in self.tensors)
        )

    def clear_symbolic_tile_patterns(self) -> "Compatibility":
        return _clear_symbolic_cached(self)

    def _clear_symbolic_tile_patterns_uncached(self) -> "Compatibility":
        return self.update(
            tensors=fzs(t.clear_symbolic_tile_patterns() for t in self.tensors)
        )

    def make_fused_loop_symbols(
        self, prefix: str
    ) -> tuple[dict[str, str], "Compatibility"]:
        result = {}
        tensors = []
        for t in self.tensors:
            r, t = t.make_fused_loop_symbols(prefix)
            tensors.append(t)
            result.update(r)

        return result, self.update(tensors=fzs(tensors))

    def clear_unrelated_columns(self, mappings: pd.DataFrame) -> "Compatibility":
        my_symbols = oset(self.symbols())
        for c in my_symbols:
            assert c in mappings.columns, f"Column {c} not found in mappings"
        keep = [
            c
            for c in mappings.columns
            if not (is_fused_loop_col(c) and c not in my_symbols)
        ]
        return mappings if len(keep) == len(mappings.columns) else mappings[keep]

    # =====================================================================================
    # VIBE CODE WARNING VIBE CODE WARNING VIBE CODE WARNING VIBE CODE WARNING VIBE CODE
    # WARNING VIBE CODE WARNING VIBE CODE WARNING VIBE CODE WARNING VIBE CODE WARNING VIBE
    # CODE WARNING VIBE CODE WARNING VIBE CODE WARNING VIBE CODE WARNING VIBE CODE WARNING
    # VIBE CODE WARNING VIBE CODE WARNING VIBE CODE WARNING VIBE CODE WARNING VIBE CODE
    # WARNING VIBE CODE WARNING VIBE CODE WARNING VIBE CODE WARNING VIBE CODE WARNING!!!!!
    # Vibe coded below. Because after this I'm implementing a joining algo that doesn't do
    # this. This is just an intermedaite placeholder.
    # =====================================================================================

    def get_reservation_of_tensor(self, tensor: str) -> TensorReservation:
        try:
            name2reservation = object.__getattribute__(self, "_name2reservation")
        except AttributeError:
            name2reservation = {t.name: t for t in self.tensors}
            object.__setattr__(self, "_name2reservation", name2reservation)
        if tensor not in name2reservation:
            raise ValueError(f"No reservation found for {tensor}")
        return name2reservation[tensor]


def _cut_at_counts(items: list, counts) -> list[list]:
    """Cut an ordered list into runs at the given cumulative-count boundaries."""
    if not items:
        return []
    bounds = sorted({c for c in counts if 0 < c < len(items)} | {len(items)})
    return [items[a:b] for a, b in zip([0] + bounds, bounds)]


@functools.lru_cache(maxsize=None)
def _multi_tensor_shared_loop_structure(c: "Compatibility"):
    """
    Which (tensor, block index, rank) slots hold each n_iterations column.
    Cleared compatibilities lose this cross-tensor correlation, so two
    equal-cleared compatibilities are renameable to each other iff this also
    matches.
    """
    col2slots = defaultdict(oset)
    for t in c.tensors:
        for i, b in enumerate(t.loops):
            for l in b.loops:
                col = l.tile_pattern.calculated_n_iterations
                if not isinstance(col, str):
                    return None
                col2slots[col].add((t.name, i, l.rank_name))
    return fzs(fzs(slots) for slots in col2slots.values())


@functools.lru_cache(maxsize=None)
def _reservation_join_key(t: "TensorReservation"):
    # above_loop_index is -1 for persistent tensors, so this also matches
    # persistence.
    return (t.resource_name, t.above_loop_index)


@functools.lru_cache(maxsize=None)
def _reservation_structure(t: "TensorReservation") -> tuple[tuple, ...]:
    return t.loop_structure()


def _symbol_values(s: PatternSymbol | tuple, df: pd.DataFrame) -> pd.Series:
    """A shape symbol's per-row values: a column, a product of iteration
    columns, or a broadcast constant."""
    if isinstance(s, tuple):
        values = df[s[0]]
        for c in s[1:]:
            values = values * df[c]
        return values
    if isinstance(s, str):
        return df[s]
    if s is None:
        # As a list of objects: a scalar None would be stored as NaN, and NaN
        # breaks set membership (each NaN boxes to a distinct object).
        return pd.Series([None] * len(df), index=df.index, dtype=object)
    return pd.Series(s, index=df.index)


def _pair_physical_loops(
    left: "TensorReservation", right: "TensorReservation"
) -> list[tuple[PermutableLoopBlock, PermutableLoopBlock]] | None:
    """Pair two reservations' physical spatial loops by cleared equality.
    Loops with the same cleared form are interchangeable. None if the loops
    don't match up."""
    groups = defaultdict(lambda: ([], []))
    for i, t in enumerate((left, right)):
        for b in t.physical_spatial_loops:
            groups[b.clear_symbolic_tile_patterns()][i].append(b)
    pairs = []
    for ls, rs in groups.values():
        if len(ls) != len(rs):
            return None
        pairs += zip(ls, rs)
    return pairs


def _tile_left_loop_to_right_loop(a: Loop, b: Loop) -> list[tuple]:
    """Pairs of the two loops' tile pattern fields, to be matched in a join."""
    return [
        (getattr(a.tile_pattern, attr), getattr(b.tile_pattern, attr))
        for attr in a.tile_pattern._symbol_attrs()
    ]


def _col_labels(c: "Compatibility", shared: set[str]) -> dict[str, tuple]:
    """One label per loop column: the loop's spatiality and how every shared
    tensor sees it. Loops pair at merge time iff their labels are equal;
    loops no shared tensor holds are told apart only by spatiality."""
    views = defaultdict(oset)
    for t in c.tensors:
        if t.name in shared:
            for l in t.iter_fused_loops():
                views[l.tile_pattern.calculated_n_iterations].add(
                    (t.name, l.clear_symbolic_tile_patterns())
                )
    labels = {}
    for b in c.loops:
        for l in b.loops:
            col = l.tile_pattern.calculated_n_iterations
            labels[col] = (b.is_spatial, fzs(views[col]))
    return labels


def _pair_loops(
    left: "Compatibility", right: "Compatibility", shared: set[str]
) -> tuple[list[tuple[str, str]], set[int], list[list[str]]] | None:
    """
    Pair every left loop with a right loop, walking both sides' blocks from
    the top. When a block runs out on one side, move to that side's next
    block. Loops pair when their labels match; same-label loops are
    interchangeable. None if the current blocks on both sides have loops and
    none can pair, or if left loops remain when the right side runs out.

    Returns (pairs in joined loop order, counts where either side starts a
    new block, leftover right blocks that sit below the shared prefix).
    """
    l_labels = _col_labels(left, shared)
    r_labels = _col_labels(right, shared)

    def block_cols(c: "Compatibility") -> list[list[str]]:
        return [
            [l.tile_pattern.calculated_n_iterations for l in b.loops] for b in c.loops
        ]

    l_blocks = block_cols(left)
    r_blocks = block_cols(right)
    pairs = []
    stops = set()
    while True:
        while l_blocks and not l_blocks[0]:
            l_blocks.pop(0)
            stops.add(len(pairs))
        if not l_blocks:
            break
        while r_blocks and not r_blocks[0]:
            r_blocks.pop(0)
            stops.add(len(pairs))
        if not r_blocks:
            return None
        pair = next(
            (
                (lc, rc)
                for lc in l_blocks[0]
                for rc in r_blocks[0]
                if r_labels[rc] == l_labels[lc]
            ),
            None,
        )
        if pair is None:
            return None
        l_blocks[0].remove(pair[0])
        r_blocks[0].remove(pair[1])
        pairs.append(pair)
    return pairs, stops, [b for b in r_blocks if b]


# Compatibilities are immutable, and the joining strategy repeats whole join
# rounds over the same compatibilities, so cache these by value. Results are
# interned (one canonical instance per value) so that equal compatibilities are
# usually the same object and cache probes hit the identity fast path instead
# of deep equality.
@functools.lru_cache(maxsize=None)
def _intern(c: Compatibility) -> Compatibility:
    return c


@functools.lru_cache(maxsize=None)
def _clear_symbolic_cached(c: Compatibility) -> Compatibility:
    return _intern(c._clear_symbolic_tile_patterns_uncached())


@functools.lru_cache(maxsize=None)
def _clear_dead_tensors_cached(c, live_tensors):
    return _intern(c._clear_dead_tensors_uncached(live_tensors))


@functools.lru_cache(maxsize=None)
def _merge_next_cached(left, right, live_tensors):
    # lru_cache does not cache exceptions, and incompatible pairs are the
    # common case in the join scans, so cache the error as a value.
    try:
        return [
            (_intern(joined), pairing)
            for joined, pairing in left._merge_next_uncached(right, live_tensors)
        ]
    except ValueError as e:
        return e
