import copy
import random
from typing import Callable, Generator

import joblib

from accelforge import Spec, util
from accelforge._accelerated_imports import pd
from accelforge.frontend.mapper.metrics import Metrics
from accelforge.frontend.renames import EinsumName, Rank
from accelforge.mapper.FFM import PmappingGroup
from accelforge.mapper.FFM._join_pmappings.compatibility import Compatibility
from accelforge.mapper.FFM._join_pmappings.compress_pmappings import (
    compress_einsum2pmappings,
)
from accelforge.mapper.FFM._join_pmappings.pmapping_dataframe import PmappingDataframe
from accelforge.mapper.FFM._pareto_df.df_convention import col2reservation
from accelforge.mapper.FFM.pmappings import MultiEinsumPmappings
from accelforge.mapper._simanneal2.tracking import EvaluationsScoreTracker
from accelforge.util._frozenset import oset
from accelforge.util.parallel import get_n_parallel_jobs


class FailedMutation(Exception):
    pass


MAX_COMBOS = 100000


def clear_compatibility(
    c: Compatibility | PmappingGroup, shared_tensors: set[str]
) -> Compatibility:
    if isinstance(c, PmappingGroup):
        c = c.compatibility
    return c.clear_dead_tensors(
        shared_tensors
    ).clear_tile_patterns_and_reservation_indices()


class MapspaceGlobals:
    def __init__(
        self,
        einsum2sims: dict[EinsumName, list[PmappingGroup]],
        aliased_tensors: dict[str, set[str]],
        objective_function: Callable[[pd.Series], float],
        tracker: EvaluationsScoreTracker,
        mixable_ranks: dict[Rank, set[Rank]] | None = None,
    ) -> None:
        self.einsum2sims: dict[EinsumName, list[PmappingGroup]] = einsum2sims
        self.aliased_tensors: dict[str, set[str]] = aliased_tensors
        self.objective_function: Callable[[pd.Series], float] = objective_function
        self.tracker: EvaluationsScoreTracker = tracker
        self.pick_lowest_usage_first: bool = False
        self.mixable_ranks: dict[Rank, set[Rank]] | None = mixable_ranks
        self.valid_combos: list[tuple[PmappingGroup, ...]] = []

    def _enumerate_valid_combos(self) -> list[list[tuple[PmappingGroup, ...]]]:
        """Enumerate all valid combinations of compatibility groups across
        einsums. Returns a list of "blocks" — each block covers a contiguous
        range of einsums. When combos exceed MAX_COMBOS, the current block is
        finalized and a new block starts from the next einsum."""
        einsum2sims = self.einsum2sims
        einsums = list(einsum2sims.keys())
        if not einsums:
            return []

        from tqdm import tqdm

        all_blocks = []
        block_start = 0
        combos = [
            (g, g.compatibility)
            for g in einsum2sims[einsums[0]]
            if self.satisfies_mixable_ranks(g.compatibility)
        ]
        print(f"  Einsum {einsums[0]}: {len(combos)} valid groups")

        for i in range(1, len(einsums)):
            e = einsums[i]
            shared_tensors = self._einsum2tensors(
                range(block_start, i)
            ) & self._einsum2tensors(i)
            future_live = (
                self._einsum2tensors(range(i + 1, len(einsums)))
                if i + 1 < len(einsums)
                else oset()
            )
            next_groups = [
                g
                for g in einsum2sims[e]
                if self.satisfies_mixable_ranks(g.compatibility)
            ]

            freed_to_groups = {}
            for g in next_groups:
                freed_to_groups.setdefault(
                    clear_compatibility(g, shared_tensors), []
                ).append(g)

            new_combos = []
            for *groups, joined_compat in tqdm(
                combos, desc=f"  Einsum {e}", leave=False
            ):
                joined_cleared = joined_compat.clear_dead_tensors(
                    future_live | shared_tensors
                )
                shared = joined_cleared.tensor_names & shared_tensors
                if not shared:
                    for g in next_groups:
                        try:
                            new_joined = joined_cleared.merge_next(
                                g.compatibility, future_live
                            )
                        except ValueError:
                            continue
                        new_combos.append((*groups, g, new_joined))
                    continue

                joined_freed = clear_compatibility(joined_cleared, shared)
                for g in freed_to_groups.get(joined_freed, []):
                    try:
                        new_joined = joined_cleared.merge_next(
                            g.compatibility, future_live
                        )
                    except ValueError:
                        continue
                    new_combos.append((*groups, g, new_joined))

            if not new_combos:
                for g in combos:
                    print(f"    LEFT: No match for {g[-1]}")
                for g in next_groups:
                    print(f"    RIGHT: No match for {g.compatibility}")
            combos = new_combos
            print(f"  Einsum {e}: {len(combos)} valid combos")
            assert combos

            # If too many combos, finalize this block and start fresh
            if len(combos) > MAX_COMBOS:
                stripped = [tuple(c[:-1]) for c in combos]
                all_blocks.append((block_start, stripped))
                print(
                    f"  Block {len(all_blocks)}: einsums {block_start}-{i} -> {len(stripped)} combos (capped)"
                )
                # Start new block from this einsum
                block_start = i
                combos = [
                    (g, g.compatibility)
                    for g in einsum2sims[einsums[i]]
                    if self.satisfies_mixable_ranks(g.compatibility)
                ]

        # Finalize last block
        stripped = [tuple(c[:-1]) for c in combos]
        all_blocks.append((block_start, stripped))
        print(
            f"  Block {len(all_blocks)}: einsums {block_start}-{len(einsums)-1} -> {len(stripped)} combos"
        )

        total = sum(len(b[1]) for b in all_blocks)
        print(f"  Total: {len(all_blocks)} blocks, {total} combos")
        return all_blocks

    def _einsum2tensors(
        self, e: EinsumName | int | Generator[EinsumName | int, None, None]
    ) -> set[str]:
        if isinstance(e, Generator) or isinstance(e, range):
            return oset.union(oset(), *(self._einsum2tensors(i) for i in e))
        if isinstance(e, int):
            e = list(self.einsum2sims.keys())[e]
        return self.einsum2sims[e][0].compatibility.tensor_names

    def _filter_einsum2sims(
        self,
        einsum2sims: dict[EinsumName, list[PmappingGroup]],
    ) -> dict[EinsumName, list[PmappingGroup]]:
        """Remove PmappingGroups that have no compatible partner in any
        neighbor einsum. Iterates until no more groups are removed."""
        einsum2sims = {e: list(sims) for e, sims in einsum2sims.items()}

        prev_len = 0
        while sum(len(sims) for sims in einsum2sims.values()) != prev_len:
            prev_len = sum(len(sims) for sims in einsum2sims.values())
            print(f"Filtering einsum2sims. Current length: {prev_len}")

            einsums = list(einsum2sims.keys())
            for i, e in enumerate(einsums):
                for e2 in einsums[i + 1 :]:
                    shared_tensors = self._einsum2tensors(e) & self._einsum2tensors(e2)
                    if not shared_tensors:
                        continue

                    freed_e = {
                        clear_compatibility(s, shared_tensors) for s in einsum2sims[e]
                    }
                    freed_e2 = {
                        clear_compatibility(s, shared_tensors) for s in einsum2sims[e2]
                    }
                    einsum2sims[e] = [
                        s
                        for s in einsum2sims[e]
                        if clear_compatibility(s, shared_tensors) in freed_e2
                    ]
                    einsum2sims[e2] = [
                        s
                        for s in einsum2sims[e2]
                        if clear_compatibility(s, shared_tensors) in freed_e
                    ]

        print(f"Filtered einsum2sims. Final length: {prev_len}")
        return einsum2sims

    def satisfies_mixable_ranks(
        self,
        compatibility: Compatibility,
    ) -> bool:
        if self.mixable_ranks is None:
            return True
        for i in range(len(compatibility.loops)):
            ranks = oset()
            for t in compatibility.tensors:
                if i < len(t.loops):
                    ranks.update(t.loops[i].rank_name)
            for r in ranks:
                if r not in self.mixable_ranks:
                    continue
                for r2 in ranks:
                    if r2 not in self.mixable_ranks[r]:
                        return False
        return True

    def joined_satisfies_mixable_ranks(
        self,
        left: PmappingGroup,
        right: PmappingGroup,
    ) -> bool:
        shared_tensors = (
            left.compatibility.tensor_names & right.compatibility.tensor_names
        )
        joined_compatibility = left.compatibility.merge_next(
            right.compatibility,
            shared_tensors,
        )
        return self.satisfies_mixable_ranks(joined_compatibility)


class SimAnnealMapping:
    def __init__(self, mapspace_globals: MapspaceGlobals) -> None:
        # self.einsum2sim: dict[EinsumName, PmappingGroup] = {
        #     e: random.choice(s) for e, s in mapspace_globals.einsum2sims.items()
        # }
        self.mapspace_globals: MapspaceGlobals = mapspace_globals
        self.einsum2sim: dict[EinsumName, PmappingGroup] = {
            e: random.choice(s) for e, s in mapspace_globals.einsum2sims.items()
        }
        self.einsum2index: dict[EinsumName, int] = {e: 0 for e in self.einsum2sim}
        self.init_choices()
        self._prev_score = None

    def mutate(self) -> None:
        # Pick a random einsum
        e = random.choice(list(self.einsum2sim.keys()))

        random.choice(
            [
                self._randomize_index,
                self._randomize_sim,
            ]
        )(e)
        try:
            self.ensure_match(e)
        except FailedMutation:
            raise
        finally:
            self.add_evaluations()

    def add_evaluations(self) -> None:
        s = sum(sim.mappings.n_total_pmappings for sim in self.einsum2sim.values())
        s *= len(self.mapspace_globals.einsum2sims) / len(self.einsum2sim)
        self.mapspace_globals.tracker.add_evaluation(s, float("inf"))

    def _randomize_index(self, e: EinsumName) -> None:
        self._prev_score = None
        self.einsum2index[e] = random.randint(0, 10000000000000)

    def _randomize_sim(self, e: EinsumName) -> None:
        self.einsum2sim[e] = random.choice(self.mapspace_globals.einsum2sims[e])
        self._randomize_index(e)

    def _einsum_position_in_list(self, e: EinsumName) -> int:
        return list(self.einsum2sim.keys()).index(e)

    def _randomize_lowest_reservation(
        self, e: EinsumName, valid_indices: list[int] | None = None
    ):
        """Pick the row index with the lowest reservations"""
        group = self.einsum2sim[e]
        res_cols = [c for c in group.mappings.data.columns if "reservation" in c]
        if not res_cols:
            self._randomize_index(e)
            return

        data = (
            group.mappings.data
            if valid_indices is None
            else group.mappings.data.iloc[valid_indices]
        )
        self.einsum2index[e] = data[res_cols].max(axis=1).idxmin()

    def _get_valid_right_compatibilities(
        self,
        left: PmappingGroup,
        possible_rights: list[PmappingGroup],
        live_tensors: set[str],
        future_einsums: list[EinsumName],
    ) -> list[PmappingGroup]:
        """Filter possible right PmappingGroups to those whose joined
        compatibility with the left has at least one valid match in every
        future einsum (lookahead)."""
        if not future_einsums:
            return possible_rights

        # Precompute: for each future einsum that shares tensors with left,
        # build the set of freed compatibilities once (not per candidate).
        left_tensors = left.compatibility.tensor_names
        future_checks = []  # list of (shared_tensors, freed_compat_set)
        for future_name in future_einsums:
            future_tensors = self._einsum2tensors(future_name)
            shared = left_tensors & future_tensors
            if not shared:
                continue
            freed = frozenset(
                clear_compatibility(g, shared)
                for g in self.mapspace_globals.einsum2sims[future_name]
            )
            future_checks.append((shared, freed))

        if not future_checks:
            return possible_rights

        result = []
        for right in possible_rights:
            try:
                joined_compat = left.compatibility.merge_next(
                    right.compatibility, live_tensors
                )
            except ValueError:
                continue

            ok = True
            for shared, future_freed in future_checks:
                if clear_compatibility(joined_compat, shared) not in future_freed:
                    ok = False
                    break
            if ok:
                result.append(right)
        return result

    def _get_valid_right_indices(
        self,
        left: PmappingGroup,
        right: PmappingGroup,
        live_tensors: set[str],
        right_tensors: set[str],
        left_row_idx: int | None = None,
    ) -> list[int]:
        """Find which right-side row indices survive the inner join with the
        left side. Tags right with _INDEX, calls PmappingDataframe.merge_next
        directly with ALL resources ignored, then reads surviving _INDEX.

        If left_row_idx is None, looks up the index from self.einsum2index."""
        import copy as _copy

        # Reduce left to just the chosen row
        if left_row_idx is None:
            left_row_idx = self.einsum2index.get(
                next((e for e, v in self.einsum2sim.items() if v is left), None), 0
            )
        left_data = left.mappings.data
        left_row_idx = left_row_idx % len(left_data)
        left_df = PmappingDataframe(
            left_data.iloc[left_row_idx : left_row_idx + 1].copy(),
            n_total_pmappings=1,
            n_valid_pmappings=1,
            ignored_resources=left.mappings.ignored_resources,
            drop_valid_reservations=left.mappings.drop_valid_reservations,
        )

        # Tag right rows with _INDEX
        right_mappings = _copy.copy(right.mappings)
        right_mappings._data = right.mappings.data.assign(
            _INDEX=range(len(right.mappings.data))
        )

        # Collect all resource names so we can ignore them all
        all_resources = oset(
            col.split("<SEP>")[1]
            for df in [left_data, right.mappings.data]
            for col in df.columns
            if col.startswith("reservation<SEP>")
        )

        try:
            joined_compat = left.compatibility.merge_next(
                right.compatibility, live_tensors
            )
            merged = left_df.merge_next(
                right_mappings,
                duplicated_aliased_tensors=oset(),
                compatibility_left=left.compatibility,
                compatibility_right=right.compatibility,
                compatibility_joined=joined_compat,
                ignored_resources=all_resources,
            )
            if "_INDEX" in merged.data.columns:
                return list(oset(merged.data["_INDEX"]))
            return list(range(len(right.mappings.data)))
        except (ValueError, FailedMutation):
            return []

    def init_choices(self) -> None:
        if not self.mapspace_globals.valid_combos:
            raise FailedMutation("No valid compatibility combinations exist")

        all_einsums = list(self.mapspace_globals.einsum2sims.keys())
        self.einsum2sim = {}

        # Pick one combo from each block
        for block_start, block_combos in self.mapspace_globals.valid_combos:
            combo = random.choice(block_combos)
            block_einsums = all_einsums[block_start : block_start + len(combo)]
            for einsum_name, group in zip(block_einsums, combo):
                self.einsum2sim[einsum_name] = group

        # Pick row indices — use _get_valid_right_indices to pre-filter
        for idx, einsum_name in enumerate(all_einsums):
            if self.mapspace_globals.pick_lowest_usage_first:
                self._randomize_lowest_reservation(einsum_name)
            else:
                self._randomize_index(einsum_name)

            if idx > 0:
                prev_name = all_einsums[idx - 1]
                right_tensors = self._einsum2tensors(prev_name)
                live_tensors = (
                    self._einsum2tensors(range(idx, len(all_einsums)))
                    if idx < len(all_einsums)
                    else oset()
                )
                valid = self._get_valid_right_indices(
                    self.einsum2sim[prev_name],
                    self.einsum2sim[einsum_name],
                    live_tensors,
                    right_tensors,
                )
                if valid:
                    if self.mapspace_globals.pick_lowest_usage_first:
                        self._randomize_lowest_reservation(einsum_name, valid)
                    else:
                        self.einsum2index[einsum_name] = random.choice(valid)

        self.add_evaluations()

    def ensure_match(
        self,
        lock_choice_for_einsum: EinsumName,
    ) -> None:

        new_einsum2sim: dict[EinsumName, PmappingGroup] = {}

        # Grab all the compatibilities that match
        for i, (e, s) in enumerate[tuple[EinsumName, PmappingGroup]](
            list(self.einsum2sim.items())
        ):
            if e == lock_choice_for_einsum:
                new_einsum2sim[e] = s
                continue

            following_tensors = self._einsum2tensors(range(i + 1, len(self.einsum2sim)))

            to_check = [(s2, s) for s2 in new_einsum2sim.values()]

            if i < self._einsum_position_in_list(lock_choice_for_einsum):
                to_check.append((s, self.einsum2sim[lock_choice_for_einsum]))
            else:
                to_check.append((self.einsum2sim[lock_choice_for_einsum], s))

            for left, right in to_check:
                shared = (
                    left.compatibility.tensor_names & right.compatibility.tensor_names
                )
                if clear_compatibility(left, shared) != clear_compatibility(
                    right, shared
                ):
                    break
                cl = clear_compatibility(left, following_tensors)
                cr = clear_compatibility(right, following_tensors)
                if cl.n_loops > cr.n_loops:
                    break

            else:
                new_einsum2sim[e] = s

        # Grab compatibilities that don't match
        def _matches(s: PmappingGroup, c: Compatibility) -> bool:
            shared = s.compatibility.tensor_names & c.tensor_names
            return clear_compatibility(s, shared) == clear_compatibility(c, shared)

        for e, pmapping_groups in self.mapspace_globals.einsum2sims.items():
            if e in new_einsum2sim:
                continue

            for s in new_einsum2sim.values():
                pmapping_groups = [
                    s2 for s2 in pmapping_groups if _matches(s2, s.compatibility)
                ]

            if not pmapping_groups:
                # print(f"No compatible PmappingGroups found for {e}")
                raise FailedMutation(f"No compatible PmappingGroups found for {e}")

            new_einsum2sim[e] = random.choice(pmapping_groups)
            self._randomize_index(e)

            # pmapping_groups = self.mapspace_globals.einsum2sims[e]
            # [s.compatibility for s in self.einsum2sim.values()]
            # [s.compatibility for s in new_einsum2sim.values()]
            # {e: s.compatibility for e, s in new_einsum2sim.items()}

        assert len(new_einsum2sim) == len(self.einsum2sim)
        assert oset(new_einsum2sim.keys()) == oset(self.einsum2sim.keys())
        self.einsum2sim = {k: new_einsum2sim[k] for k in self.einsum2sim.keys()}

    def _einsum2tensors(
        self, e: EinsumName | int | Generator[EinsumName | int, None, None]
    ) -> set[str]:
        return self.mapspace_globals._einsum2tensors(e)

    def _access_index(self, e: EinsumName, index_override: int | None = None):
        s = self.einsum2sim[e]
        data = s.mappings.data
        i = self.einsum2index[e] if index_override is None else index_override
        i %= len(data)
        return PmappingGroup(
            compatibility=s.compatibility,
            mappings=PmappingDataframe(
                data.iloc[i : i + 1].copy(),
                n_total_pmappings=s.mappings.n_total_pmappings,
                n_valid_pmappings=s.mappings.n_valid_pmappings,
                ignored_resources=s.mappings.ignored_resources,
                drop_valid_reservations=s.mappings.drop_valid_reservations,
                n_concurrent_threads=s.mappings.n_concurrent_threads,
            ),
        )

    def get_score(self) -> float:
        if self._prev_score is not None:
            return self._prev_score

        all_einsum_names = list(self.einsum2sim.keys())
        items: list[tuple[EinsumName, PmappingGroup]] = list(self.einsum2sim.items())
        joined: PmappingGroup = items.pop(0)[1]
        joined = copy.deepcopy(joined)
        for i, (e, s) in enumerate(items):
            # We're starting from 1, so i in items means einsum index i+1 in the full
            # list
            full_idx = i + 1
            right_tensors = self._einsum2tensors(full_idx)
            live_tensors = (
                self._einsum2tensors(range(full_idx + 1, len(all_einsum_names)))
                if full_idx + 1 < len(all_einsum_names)
                else oset()
            )

            joined.compatibility = joined.compatibility.clear_dead_tensors(
                live_tensors | right_tensors
            )

            def _merge_next(
                left: PmappingGroup,
                right: PmappingGroup,
            ) -> PmappingGroup:
                try:
                    return left.merge_next(
                        right,
                        live_tensors_post_join=live_tensors,
                        live_tensors_with_right=live_tensors | right_tensors,
                        aliased_tensors=self.mapspace_globals.aliased_tensors,
                        compatibility_joined=joined.compatibility.merge_next(
                            s.compatibility,
                            live_tensors,
                        ),
                        permuted_compatibility_left=left.compatibility,
                        permuted_compatibility_right=right.compatibility,
                        delay=False,
                        ignored_resources=oset(),
                    )
                except (ValueError, AssertionError) as err:
                    raise FailedMutation(f"No valid pmappings: {err}")

            # Try to merge using the index we already have set
            joined_new = _merge_next(joined, self._access_index(e))
            if len(joined_new.mappings.data) >= 1:
                # If multiple valid joined mappings, keep only the first one
                if len(joined_new.mappings.data) > 1:
                    joined_new.mappings._data = joined_new.mappings.data.iloc[:1].copy()
                joined = joined_new
                continue

            # No valid pmappings with current index. Use a cheap inner join
            # to find which right-side indices are valid, then pick one.
            self.add_evaluations()

            valid_indices = self._get_valid_right_indices(
                joined,
                self.einsum2sim[e],
                live_tensors,
                right_tensors,
                left_row_idx=0,
            )
            if valid_indices:
                i = random.choice(valid_indices)
                joined_new = _merge_next(joined, self._access_index(e, i))
                if len(joined_new.mappings.data) >= 1:
                    self.einsum2index[e] = i
                    if len(joined_new.mappings.data) > 1:
                        joined_new.mappings._data = joined_new.mappings.data.iloc[
                            :1
                        ].copy()
                    joined = joined_new
                    continue

            # Full fallback: merge all rows and pick a valid one
            s = self.einsum2sim[e]
            s.mappings.data["_INDEX"] = list(range(len(s.mappings.data)))
            joined_new = _merge_next(
                joined,
                s,
            )
            s.mappings._data = s.mappings.data.drop(columns=["_INDEX"])
            try:
                i = random.choice(list(oset(joined_new.mappings.data["_INDEX"])))
            except IndexError:
                raise FailedMutation(f"No valid pmappings for {e}")

            # Now that we've picked, merge with the index we just set
            joined_new = _merge_next(joined, self._access_index(e, i))

            if len(joined_new.mappings.data) >= 1:
                # If multiple valid joined mappings, keep only the first one
                if len(joined_new.mappings.data) > 1:
                    joined_new.mappings._data = joined_new.mappings.data.iloc[:1].copy()
                # If it worked, set the index
                self.einsum2index[e] = i
                joined = joined_new
                continue

            raise FailedMutation(
                f"Got {len(joined_new.mappings.data)} pmappings for {e}"
            )

        assert len(joined.mappings.data) == 1
        score = self.mapspace_globals.objective_function(joined.mappings.data.iloc[0])
        self.mapspace_globals.tracker.add_evaluation(0, score)
        self._prev_score = score
        return score

    def crossover(self, other: "SimAnnealMapping") -> "SimAnnealMapping":
        child = other.copy()
        e = random.choice(list(self.einsum2sim.keys()))
        child.einsum2sim[e] = self.einsum2sim[e]
        child.einsum2index[e] = self.einsum2index[e]

        try:
            child.ensure_match(e)
        except FailedMutation:
            child = other.copy()

        child.add_evaluations()
        return child

    def copy(self) -> "SimAnnealMapping":
        s = SimAnnealMapping.__new__(SimAnnealMapping)
        s.mapspace_globals = self.mapspace_globals
        s.einsum2sim = self.einsum2sim.copy()
        s.einsum2index = self.einsum2index.copy()
        s._prev_score = self._prev_score
        return s


def get_random_mapping(mapspace_globals: MapspaceGlobals) -> SimAnnealMapping:
    while True:
        try:
            s = SimAnnealMapping(mapspace_globals)
            s.get_score()
            return s
        except FailedMutation:
            if mapspace_globals.tracker.finished():
                return None
            continue


def _make_mapspace_globals(
    pmapping_groups: dict[EinsumName, list[PmappingGroup]],
    spec: Spec,
    tracker: EvaluationsScoreTracker,
    pick_lowest_usage_first: bool = False,
    valid_combos: list[tuple[PmappingGroup, ...]] | None = None,
) -> MapspaceGlobals:
    objective = spec.mapper.metrics
    if objective == Metrics.ENERGY:
        objective_function = lambda x: x["Total<SEP>energy"]
    elif objective == Metrics.LATENCY:
        objective_function = lambda x: x["Total<SEP>latency"]
    elif objective == (Metrics.ENERGY | Metrics.LATENCY):
        objective_function = lambda x: x["Total<SEP>energy"] * x["Total<SEP>latency"]
    else:
        raise ValueError(f"Unknown objective {objective}")
    mg = MapspaceGlobals(
        einsum2sims=pmapping_groups,
        aliased_tensors=spec.workload.get_tensor_copies(),
        objective_function=objective_function,
        tracker=tracker,
    )
    mg.pick_lowest_usage_first = pick_lowest_usage_first
    if valid_combos is not None:
        mg.valid_combos = valid_combos
    return mg


def _make_initial_population(
    mapspace_globals: MapspaceGlobals,
    pop_size: int,
) -> list[SimAnnealMapping]:
    tracker = mapspace_globals.tracker
    mappings = []
    while len(mappings) < pop_size:
        m = get_random_mapping(mapspace_globals)
        if m is None:
            return mappings
        mappings.append(m)
        if tracker.finished():
            print(f"Timed out after {len(mappings)}/{pop_size} mappings.")
            return mappings
    print(
        f"Completed making initial population of {len(mappings)} mappings after {tracker.evaluations}  evaluations."
    )
    return mappings


def _join_pmappings(
    pmapping_groups: dict[EinsumName, list[PmappingGroup]],
    spec: Spec,
    tracker: EvaluationsScoreTracker,
    pop_size_per_thread: int,
    algorithm: str = "simanneal",
    pick_lowest_usage_first: bool = False,
    valid_combos: list[tuple[PmappingGroup, ...]] | None = None,
) -> PmappingGroup:
    mapspace_globals = _make_mapspace_globals(
        pmapping_groups, spec, tracker, pick_lowest_usage_first, valid_combos
    )
    mappings = _make_initial_population(mapspace_globals, pop_size_per_thread)
    if not mappings or tracker.finished():
        return

    if algorithm == "simanneal":
        _run_simanneal(mappings, tracker)
    elif algorithm == "genetic":
        _run_genetic(mappings, tracker, pop_size_per_thread)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def _run_simanneal(
    mappings: list[SimAnnealMapping],
    tracker: EvaluationsScoreTracker,
) -> None:
    while not tracker.finished():
        for i, m in enumerate(list(mappings)):
            try:
                new = m.copy()
                new.mutate()
                if new.get_score() < m.get_score():
                    mappings[i] = new
                if tracker.finished():
                    break
            except FailedMutation:
                continue


def _run_genetic(
    mappings: list[SimAnnealMapping],
    tracker: EvaluationsScoreTracker,
    pop_size: int,
    crossover_rate: float = 0.7,
    mutation_rate: float = 0.3,
) -> None:
    while not tracker.finished():
        fitness = []
        for m in mappings:
            try:
                fitness.append(m.get_score())
            except FailedMutation:
                fitness.append(float("inf"))

        best_idx = min(range(len(fitness)), key=lambda i: fitness[i])
        best_mapping = mappings[best_idx]

        inv_fitness = [1.0 / (f + 1e-9) for f in fitness]
        total = sum(inv_fitness)
        n = len(inv_fitness)
        probs = [f / total for f in inv_fitness] if total > 0 else [1.0 / n] * n
        selected = random.choices(range(len(mappings)), probs, k=pop_size)

        new_pop: list[SimAnnealMapping] = []
        for i in range(0, len(selected), 2):
            p1 = mappings[selected[i]]
            p2 = mappings[selected[(i + 1) % len(selected)]]
            if random.random() < crossover_rate:
                new_pop.append(p1.crossover(p2))
                new_pop.append(p2.crossover(p1))
            else:
                new_pop.append(p1.copy())
                new_pop.append(p2.copy())

        for m in new_pop:
            if random.random() < mutation_rate:
                try:
                    m.mutate()
                except FailedMutation:
                    pass

        new_pop.append(best_mapping)
        mappings = new_pop[:pop_size]

        if tracker.finished():
            break


def get_n_tile_shapes(sim: PmappingGroup) -> int:
    df = sim.mappings.data
    symbols = sim.compatibility.symbols()
    cols = [c for c in df.columns if c in symbols]
    if not cols:
        return 1
    return len(df.groupby(cols).size())


def join_pmappings(
    pmappings: MultiEinsumPmappings,
    max_evaluations: int = 1,
    population_size=1000,
    score_target: float | None = None,
    algorithm: str = "simanneal",
    optimal: float | None = None,
    pick_lowest_usage_first: bool = False,
) -> EvaluationsScoreTracker:

    spec = pmappings.spec
    tracker = EvaluationsScoreTracker(
        max_evaluations=max_evaluations / get_n_parallel_jobs(),
        stop_at_score=None,
        print_period=1,
        optimal=optimal,
    )

    # Disable validation in the compatibility class to avoid errors when joining
    # pmappings. We use a simplified version of the joining process in this code that
    # doesn't do all the reservation cleaning that we do in FFM.
    Compatibility.__post_init__ = lambda *args, **kwargs: None

    compressed, decompress_data = compress_einsum2pmappings(pmappings.einsum2pmappings)

    if population_size != float("inf"):
        pop_size_per_thread = max(1, population_size // get_n_parallel_jobs())
    else:
        pop_size_per_thread = float("inf")

    # Multiply by the number of einsums
    # print(
    #     f"Multiplying scale by {len(pmappings.einsum2pmappings)} for number of einsums"
    # )
    # tracker._scale_by *= len(pmappings.einsum2pmappings)

    # We allow FFM to query n_pareto_optimal_pmappings mappings from the mapspace, so we
    # scale by 1 / n_pareto_optimal_pmappings to get simanneal 1 evaluation = same
    # number of mappings as FFM
    # print(
    #     f"Multiplying scale by {1 / pmappings.n_pareto_optimal_pmappings()} for Pareto-optimal mapspace size"
    # )
    # tracker._scale_by *= 1 / pmappings.n_pareto_optimal_pmappings()

    for einsum_name, einsum_pmappings in pmappings.einsum2pmappings.items():
        total = sum(len(p.mappings.data) for p in einsum_pmappings)
        n_compatibilities = len(einsum_pmappings)
        print(
            f"Einsum {einsum_name} has {total} pmappings with {n_compatibilities} compatibilities"
        )
        if total == 0:
            raise ValueError(f"Einsum {einsum_name} has no pmappings")

    permuted = {}
    n_evaluated = 1
    for einsum_name, einsum_sims in compressed.items():
        for s in einsum_sims:
            n_pmappings = s.mappings.n_total_pmappings
            n_evaluated += n_pmappings
            # Count all permutations as separate choices
            for c_perm, _ in s.compatibility.make_equivalent_permutations():
                permuted.setdefault(einsum_name, []).append(
                    PmappingGroup(
                        compatibility=c_perm,
                        mappings=s.mappings,
                    )
                )

    print(f"Multiplying scale by {1 / n_evaluated} for number of evaluated pmappings")
    tracker._scale_by *= 1 / n_evaluated

    mg = MapspaceGlobals(
        permuted,
        {},
        lambda x: 0,
        tracker,
        mixable_ranks=spec.workload._get_ranks_that_share_indexing_rank_variables(),
    )
    permuted = mg._filter_einsum2sims(permuted)
    mg.einsum2sims = permuted
    mg.einsum2sims = permuted
    valid_combos = mg._enumerate_valid_combos()

    def parallel_join(
        permuted: dict[EinsumName, list[PmappingGroup]],
        spec: Spec,
        tracker: EvaluationsScoreTracker,
        pop_size_per_thread: int,
        algorithm: str,
        pick_lowest_usage_first: bool,
        valid_combos: list[tuple[PmappingGroup, ...]],
    ) -> EvaluationsScoreTracker:
        _join_pmappings(
            permuted,
            spec,
            tracker,
            pop_size_per_thread,
            algorithm,
            pick_lowest_usage_first,
            valid_combos,
        )
        return tracker

    trackers = util.parallel(
        joblib.delayed(parallel_join)(
            permuted,
            spec,
            tracker,
            pop_size_per_thread,
            algorithm,
            pick_lowest_usage_first,
            valid_combos,
        )
        for _ in range(get_n_parallel_jobs())
    )

    t0 = trackers[0]
    for t in trackers[1:]:
        t0.merge_with(t)

    # for einsum_name in pmappings.einsum2pmappings:
    #     col = f"{einsum_name}<SEP>{MAPPING_COLUMN}"
    #     joined.data[col] = joined.data[col].apply(
    #         lambda x: pmappings.pmapping_objects[einsum_name][x]
    #     )

    # rank_variable_bounds = get_rank_variable_bounds_for_all_einsums(spec)
    # joined.data[f"Total<SEP>{MAPPING_COLUMN}"] = joined.data.apply(
    #     lambda row: MappingFromRow(row, spec, rank_variable_bounds), axis=1
    # )
    # # Fill nans with 0. We might get missing columns for some mapping entries if there
    # # are energy entries for some pmappings but not others (e.g., one pmapping accesses
    # # DRAM while another doesn't.)
    # joined._data = joined.data._fillna_and__numeric_cast(0)
    return t0  # Mappings(spec, list(pmappings.einsum2pmappings.keys()), joined.data)
