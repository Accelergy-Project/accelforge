import inspect
import os
import random
from typing import Callable, Generator
from fastfusion import arch, util
from fastfusion import Specification
from fastfusion.frontend.mapper.metrics import Metrics
from fastfusion.mapper.FFM._interface.pmappings import MultiEinsumPmappings
from fastfusion.mapper.FFM._interface.mappings import Mappings
from fastfusion.mapper.FFM._join_pmappings.compress_pmappings import (
    compress_einsum2pmappings,
    decompress_pmappings,
)
from fastfusion.frontend.workload import EinsumName
from fastfusion.frontend.mapping import Mapping
from fastfusion.mapper.FFM._join_pmappings.sim import SIM
from fastfusion.mapper.FFM._pmapping_group.df_convention import (
    MAPPING_COLUMN,
    col2nameloop,
)
from fastfusion.mapper.FFM._pmapping_group.pmapping_group import (
    PmappingGroup,
    row2pmappings,
)
from fastfusion.mapper.FFM._make_pmappings.mapper_multi_einsum import (
    get_rank_variable_bounds_for_all_einsums,
)
from fastfusion.accelerated_imports import pd
import joblib
from fastfusion.mapper.FFM._join_pmappings.compatibility import Compatibility
from fastfusion.mapper.simanneal2.tracking import EvaluationsScoreTracker

# Simulated annealing algorithm
# -----------------------------
# Given:
# - Pmappings for each Einsum

# 1. Make a compatibility -> SIMs dict for each Einsum
# 2. While True:
#    a. Randomly change a compatibility choice for one Einsum


# Functions:
# - Given compatibility choices & pmapping index numbers, return a score
# - Given compatibility choices & pmapping index numbers, make sure all compatibilities
#   & indices match


class FailedMutation(Exception):
    pass


class MapspaceGlobals:
    def __init__(
        self,
        einsum2sims: dict[EinsumName, list[SIM]],
        resource2capacity: dict[str, int],
        aliased_tensors: dict[str, set[str]],
        objective_function: Callable[[pd.Series], float],
        tracker: EvaluationsScoreTracker,
    ) -> None:
        self.einsum2sims: dict[EinsumName, list[SIM]] = einsum2sims
        self.resource2capacity: dict[str, int] = resource2capacity
        self.aliased_tensors: dict[str, set[str]] = aliased_tensors
        self.objective_function: Callable[[pd.Series], float] = objective_function
        self.tracker: EvaluationsScoreTracker = tracker


class SimAnnealMapping:
    def __init__(self, mapspace_globals: MapspaceGlobals) -> None:
        # self.einsum2sim: dict[EinsumName, SIM] = {
        #     e: random.choice(s) for e, s in mapspace_globals.einsum2sims.items()
        # }
        self.mapspace_globals: MapspaceGlobals = mapspace_globals
        self.einsum2sim: dict[EinsumName, SIM] = {
            e: random.choice(s) for e, s in mapspace_globals.einsum2sims.items()
        }
        self.einsum2index: dict[EinsumName, int] = {e: 0 for e in self.einsum2sim}
        self.ensure_match(list(self.einsum2sim.keys())[0])
        for e in self.einsum2sim:
            self.randomize_index(e)
        self._prev_score = None

    def mutate(self) -> None:
        # Pick a random einsum
        e = random.choice(list(self.einsum2sim.keys()))

        random.choice(
            [
                self.randomize_index,
                self.randomize_sim,
            ]
        )(e)

        self.ensure_match(e)

    def randomize_index(self, e: EinsumName) -> None:
        self._prev_score = None
        self.einsum2index[e] = random.randint(0, 10000000000000)
        self.mapspace_globals.tracker.add_evaluation(1, float("inf"))

    def randomize_sim(self, e: EinsumName) -> None:
        self.einsum2sim[e] = random.choice(self.mapspace_globals.einsum2sims[e])
        self.randomize_index(e)

    def _einsum_position_in_list(self, e: EinsumName) -> int:
        return list(self.einsum2sim.keys()).index(e)

    def ensure_match(
        self,
        lock_choice_for_einsum: EinsumName,
    ) -> None:

        new_einsum2sim: dict[EinsumName, SIM] = {}

        # Grab all the compatibilities that match
        for i, (e, s) in enumerate(list(self.einsum2sim.items())):
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
                c = left.compatibility.clear_dead_tensors(
                    right.compatibility.tensor_names
                ).clear_tile_patterns_and_reservation_indices()
                c2 = right.compatibility.clear_dead_tensors(
                    left.compatibility.tensor_names
                ).clear_tile_patterns_and_reservation_indices()
                if c != c2:
                    break

                c = left.compatibility.clear_dead_tensors(
                    following_tensors
                ).clear_tile_patterns_and_reservation_indices()
                c2 = right.compatibility.clear_dead_tensors(
                    following_tensors
                ).clear_tile_patterns_and_reservation_indices()

                # Can't merge. I have more loops than the next, so my dataflow can't be
                # carried through a LoopTree to where it's needed.
                if c.n_loops > c2.n_loops:
                    break

            else:
                new_einsum2sim[e] = s

        # Grab compatibilities that don't match
        def _matches(s: SIM, c: Compatibility) -> bool:
            cs = s.compatibility.clear_dead_tensors(
                c.tensor_names
            ).clear_tile_patterns_and_reservation_indices()
            cn = c.clear_dead_tensors(
                s.compatibility.tensor_names
            ).clear_tile_patterns_and_reservation_indices()
            return cs == cn

        for e, sims in self.mapspace_globals.einsum2sims.items():
            if e in new_einsum2sim:
                continue

            for s in new_einsum2sim.values():
                sims = [s2 for s2 in sims if _matches(s2, s.compatibility)]

            if not sims:
                # print(f"No compatible SIMs found for {e}")
                raise FailedMutation(f"No compatible SIMs found for {e}")

            new_einsum2sim[e] = random.choice(sims)
            self.randomize_index(e)

            # sims = self.mapspace_globals.einsum2sims[e]
            # [s.compatibility for s in self.einsum2sim.values()]
            # [s.compatibility for s in new_einsum2sim.values()]
            # {e: s.compatibility for e, s in new_einsum2sim.items()}

        assert len(new_einsum2sim) == len(self.einsum2sim)
        assert set(new_einsum2sim.keys()) == set(self.einsum2sim.keys())
        self.einsum2sim = {k: new_einsum2sim[k] for k in self.einsum2sim.keys()}

    def _einsum2tensors(
        self, e: EinsumName | int | Generator[EinsumName | int, None, None]
    ) -> set[str]:
        if isinstance(e, Generator) or isinstance(e, range):
            return set.union(set(), *(self._einsum2tensors(i) for i in e))
        if isinstance(e, int):
            e = list(self.einsum2sim.keys())[e]
        return self.einsum2sim[e].compatibility.tensor_names

    def _access_index(self, e: EinsumName, index_override: int | None = None):
        s = self.einsum2sim[e]
        data = s.mappings.data
        i = self.einsum2index[e] if index_override is None else index_override
        i %= len(data)
        return SIM(
            compatibility=s.compatibility, mappings=PmappingGroup(data.iloc[i : i + 1])
        )

    def get_score(self) -> float:
        if self._prev_score is not None:
            return self._prev_score

        items: list[tuple[EinsumName, SIM]] = list(self.einsum2sim.items())
        joined: SIM = items.pop(0)[1]
        for i, (e, s) in enumerate(items):
            right_tensors = self._einsum2tensors(i)
            live_tensors = self._einsum2tensors(range(i + 1, len(items)))

            joined.compatibility = joined.compatibility.clear_dead_tensors(
                live_tensors | right_tensors
            )

            def _merge_next(
                left: SIM, right: SIM, apply_resource_limit: bool = True
            ) -> SIM:
                try:
                    return left.merge_next(
                        right,
                        live_tensors=live_tensors,
                        live_tensors_with_right=live_tensors | right_tensors,
                        aliased_tensors=self.mapspace_globals.aliased_tensors,
                        compatibility_joined=joined.compatibility.merge_next(
                            s.compatibility, live_tensors
                        ),
                        resource2capacity=(
                            self.mapspace_globals.resource2capacity
                            if apply_resource_limit
                            else None
                        ),
                        drop_valid_reservations=True,
                        delay=False,
                    )
                except ValueError as err:
                    # print(err)
                    raise FailedMutation(f"No valid pmappings: {err}")

            # Try to merge using the index we already have set
            joined_new = _merge_next(joined, self._access_index(e))
            if len(joined_new.mappings.data) == 1:
                joined = joined_new
                # print(' '.join(f'{k}={v}' for k, v in dict(joined.mappings.data.iloc[0]).items() if col2nameloop(k)))
                continue
            if len(joined_new.mappings.data) > 1:
                raise ValueError(
                    f"Got {len(joined_new.mappings.data)} pmappings for {e}"
                )

            # No valid pmappings! Merge all possible, then pick one
            self.mapspace_globals.tracker.add_evaluation(1, float("inf"))
            s = self.einsum2sim[e]
            s.mappings.data["_INDEX"] = list(range(len(s.mappings.data)))
            joined_new = _merge_next(
                joined,
                s,
                apply_resource_limit=False,
            )
            s.mappings._data = s.mappings.data.drop(columns=["_INDEX"])
            try:
                i = random.choice(list(set(joined_new.mappings.data["_INDEX"])))
            except IndexError:
                raise FailedMutation(f"No valid pmappings for {e}")

            # Now that we've picked, merge with the index we just set
            joined_new = _merge_next(joined, self._access_index(e, i))

            if len(joined_new.mappings.data) == 1:
                # If it worked, set the index
                self.einsum2index[e] = i
                joined = joined_new
                # print(' '.join(f'{k}={v}' for k, v in dict(joined.mappings.data.iloc[0]).items() if col2nameloop(k)))
                continue

            if len(joined_new.mappings.data) > 1:
                raise ValueError(
                    f"Got {len(joined_new.mappings.data)} pmappings for {e}"
                )

            raise FailedMutation(
                f"Got {len(joined_new.mappings.data)} pmappings for {e}"
            )

        assert len(joined.mappings.data) == 1
        score = self.mapspace_globals.objective_function(joined.mappings.data.iloc[0])
        self.mapspace_globals.tracker.add_evaluation(0, score)
        self._prev_score = score
        return score

    def copy(self) -> "SimAnnealMapping":
        s = SimAnnealMapping(self.mapspace_globals)
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


def join_sims(
    sims: dict[EinsumName, list[SIM]],
    spec: Specification,
    resource2capacity: dict[str, int],
    tracker: EvaluationsScoreTracker,
    pop_size_per_thread: int,
) -> SIM:
    objective = spec.mapper.ffm.metrics
    if objective == Metrics.ENERGY:
        objective_function = lambda x: x["Total<SEP>energy"]
    elif objective == Metrics.LATENCY:
        objective_function = lambda x: x["Total<SEP>latency"]
    elif objective == (Metrics.ENERGY | Metrics.LATENCY):
        objective_function = lambda x: x["Total<SEP>energy"] * x["Total<SEP>latency"]
    else:
        raise ValueError(f"Unknown objective {objective}")
    # print(f'Resource2capacity: {resource2capacity}')
    mapspace_globals = MapspaceGlobals(
        einsum2sims=sims,
        resource2capacity=resource2capacity,
        aliased_tensors=spec.workload.get_tensor_copies(),
        objective_function=objective_function,
        tracker=tracker,
    )

    mappings = []
    while len(mappings) < pop_size_per_thread:
        mappings.append(get_random_mapping(mapspace_globals))
        if tracker.finished():
            return
    print(f"Completed making initial population of {len(mappings)} mappings")

    i = 0
    while True:
        if i > 1e6:
            break
        i += 1
        for i, m in enumerate(list(mappings)):
            try:
                new = m.copy()
                new.mutate()
                if new.get_score() < m.get_score():
                    mappings[i] = new

                # 0 evaluations because they've been accounted for in the mutation and
                # score calculation functions
                if tracker.finished():
                    break
            except FailedMutation:
                continue

            # else:
            #     for einsum_name, sim in simanneal_mapping.einsum2sim.items():
            #         print(f"Einsum {einsum_name}, index {simanneal_mapping.einsum2index[einsum_name]}")
            #         for c in sim.compatibility.tensors:
            #             print(f'\t{c}')

            #         df = sim.mappings.data.iloc[simanneal_mapping.einsum2index[einsum_name] % len(sim.mappings.data)]
            #         for s in sim.compatibility.symbols():
            #             print(f'\t{s} = {df[s]}')
            # print(f"Iteration {i}: Score {new_score} (prev {prev_score})")
    # raise ValueError("No valid mapping found")


def get_n_tile_shapes(sim: SIM) -> int:
    df = sim.mappings.data
    symbols = sim.compatibility.symbols()
    cols = [c for c in df.columns if c in symbols]
    if not cols:
        return 1
    return len(df.groupby(cols).size())


def join_pmappings(
    spec: Specification,
    pmappings: MultiEinsumPmappings,
    max_evaluations: int = 1,
    population_size=100,
    score_target: float | None = None,
) -> EvaluationsScoreTracker:
    tracker = EvaluationsScoreTracker(
        max_evaluations=max_evaluations / util.N_PARALLEL_PROCESSES,
        stop_at_score=None,
        print_period=1,
    )

    if score_target is not None:
        tracker.multiply_score_by(1 / score_target)

    pop_size_per_thread = population_size // util.N_PARALLEL_PROCESSES

    # Multiply by the number of einsums
    tracker.multiply_scale_by(len(pmappings.einsum2pmappings))

    # Expected #pmappings before a Pareto-optimal one is found
    # tracker.multiply_scale_by(pmappings._evaluated_pmappings_for_simanneal_baseline_compare() / pmappings.pareto_optimal_pmappings())
    tracker.multiply_scale_by(
        pmappings.evaluated_pmappings() / pmappings.pareto_optimal_pmappings()
    )

    # Normalize to the speed of the intra-Einsum pmapper
    tracker.multiply_scale_by(1 / pmappings.evaluated_pmappings())

    for einsum_name, einsum_pmappings in pmappings.einsum2pmappings.items():
        total = sum(len(p.mappings.data) for p in einsum_pmappings)
        n_compatibilities = len(einsum_pmappings)
        print(
            f"Einsum {einsum_name} has {total} pmappings with {n_compatibilities} compatibilities"
        )
        if total == 0:
            raise ValueError(f"Einsum {einsum_name} has no pmappings")

    print(f"TODO: Populate SIMs with all permutations")

    compressed, decompress_data = compress_einsum2pmappings(pmappings.einsum2pmappings)

    permuted = {}
    for einsum_name, einsum_sims in compressed.items():
        for s in einsum_sims:
            for c_perm, _ in s.compatibility.make_equivalent_permutations():
                permuted.setdefault(einsum_name, []).append(
                    SIM(
                        compatibility=c_perm,
                        mappings=s.mappings,
                    )
                )

    tile_shapes = [get_n_tile_shapes(s) for sims in compressed.values() for s in sims]

    # average_tile_shapes = sum(tile_shapes) / len(tile_shapes)
    # print(f"Average tile shapes: {average_tile_shapes}")
    # tracker.multiply_scale_by(average_tile_shapes)

    def parallel_join(
        permuted: dict[EinsumName, list[SIM]],
        spec: Specification,
        resource2capacity: dict[str, int],
        tracker: EvaluationsScoreTracker,
        pop_size_per_thread: int,
    ) -> EvaluationsScoreTracker:
        join_sims(permuted, spec, resource2capacity, tracker, pop_size_per_thread)
        return tracker

    trackers = util.parallel(
        joblib.delayed(parallel_join)(
            permuted,
            spec,
            pmappings.resource2capacity,
            tracker,
            pop_size_per_thread,
        )
        for _ in range(util.N_PARALLEL_PROCESSES)
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
    # joined._data = joined.data.fillna(0)
    return t0  # Mappings(spec, list(pmappings.einsum2pmappings.keys()), joined.data)
