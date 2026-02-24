from accelforge.frontend.renames import EinsumName
from accelforge.mapper.FFM._join_pmappings.pmapping_group import PmappingGroup
import inspect
import os
import random
from typing import Callable, Generator
from accelforge import arch, util
from accelforge import Spec
from accelforge.frontend.mapper.metrics import Metrics
from accelforge.mapper.FFM.pmappings import MultiEinsumPmappings
from accelforge.mapper.FFM._join_pmappings.compress_pmappings import (
    compress_einsum2pmappings,
    decompress_pmappings,
)
from accelforge.frontend.workload import EinsumName
from accelforge.frontend.mapping import Mapping
from accelforge.mapper.FFM import PmappingGroup
from accelforge.mapper.FFM._pareto_df.df_convention import (
    MAPPING_COLUMN,
    col2nameloop,
)
from accelforge.mapper.FFM._join_pmappings.pmapping_group import PmappingDataframe
from accelforge.mapper.FFM._make_pmappings.make_pmappings import (
    get_rank_variable_bounds_for_all_einsums,
)
from accelforge._accelerated_imports import pd
import joblib
from accelforge.mapper.FFM._join_pmappings.compatibility import Compatibility
from accelforge.mapper._simanneal2.tracking import EvaluationsScoreTracker
from accelforge.util.parallel import get_n_parallel_jobs

# Simulated annealing algorithm
# -----------------------------
# Given:
# - Pmappings for each Einsum

# 1. Make a compatibility -> PmappingGroups dict for each Einsum
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
        einsum2sims: dict[EinsumName, list[PmappingGroup]],
        aliased_tensors: dict[str, set[str]],
        objective_function: Callable[[pd.Series], float],
        tracker: EvaluationsScoreTracker,
    ) -> None:
        self.einsum2sims: dict[EinsumName, list[PmappingGroup]] = einsum2sims
        self.aliased_tensors: dict[str, set[str]] = aliased_tensors
        self.objective_function: Callable[[pd.Series], float] = objective_function
        self.tracker: EvaluationsScoreTracker = tracker


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

    def init_choices(self) -> None:
        try:
            self.einsum2sim = {}
            for einsum_name in self.mapspace_globals.einsum2sims:
                possible_choices = self.mapspace_globals.einsum2sims[einsum_name]
                cur_tensors = self._einsum2tensors(einsum_name)

                for prev_einsum, prev_sim in self.einsum2sim.items():
                    prev_tensors = self._einsum2tensors(prev_einsum)
                    shared_tensors = prev_tensors & cur_tensors
                    c = prev_sim.compatibility.clear_dead_tensors(
                        shared_tensors
                    ).clear_tile_patterns_and_reservation_indices()

                    possible_choices = [
                        s
                        for s in possible_choices
                        if s.compatibility.clear_dead_tensors(
                            shared_tensors
                        ).clear_tile_patterns_and_reservation_indices()
                        == c
                    ]
                if not possible_choices:

                    raise FailedMutation(
                        f"No compatible PmappingGroups found for {einsum_name}"
                    )
                self.einsum2sim[einsum_name] = random.choice(possible_choices)
                self._randomize_index(einsum_name)
        except FailedMutation:
            raise
        finally:
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
        def _matches(s: PmappingGroup, c: Compatibility) -> bool:
            cs = s.compatibility.clear_dead_tensors(
                c.tensor_names
            ).clear_tile_patterns_and_reservation_indices()
            cn = c.clear_dead_tensors(
                s.compatibility.tensor_names
            ).clear_tile_patterns_and_reservation_indices()
            return cs == cn

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
        assert set(new_einsum2sim.keys()) == set(self.einsum2sim.keys())
        self.einsum2sim = {k: new_einsum2sim[k] for k in self.einsum2sim.keys()}

    def _einsum2tensors(
        self, e: EinsumName | int | Generator[EinsumName | int, None, None]
    ) -> set[str]:
        if isinstance(e, Generator) or isinstance(e, range):
            return set.union(set(), *(self._einsum2tensors(i) for i in e))
        if isinstance(e, int):
            e = list(self.mapspace_globals.einsum2sims.keys())[e]
        return self.mapspace_globals.einsum2sims[e][0].compatibility.tensor_names

    def _access_index(self, e: EinsumName, index_override: int | None = None):
        s = self.einsum2sim[e]
        data = s.mappings.data
        i = self.einsum2index[e] if index_override is None else index_override
        i %= len(data)
        return PmappingGroup(
            compatibility=s.compatibility,
            mappings=PmappingDataframe(
                data.iloc[i : i + 1],
                n_total_pmappings=s.mappings.n_total_pmappings,
                n_valid_pmappings=s.mappings.n_valid_pmappings,
            ),
        )

    def get_score(self) -> float:
        if self._prev_score is not None:
            return self._prev_score

        items: list[tuple[EinsumName, PmappingGroup]] = list(self.einsum2sim.items())
        joined: PmappingGroup = items.pop(0)[1]
        for i, (e, s) in enumerate(items):
            right_tensors = self._einsum2tensors(i)
            live_tensors = self._einsum2tensors(range(i + 1, len(items)))

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
                        live_tensors=live_tensors,
                        live_tensors_with_right=live_tensors | right_tensors,
                        aliased_tensors=self.mapspace_globals.aliased_tensors,
                        compatibility_joined=joined.compatibility.merge_next(
                            s.compatibility,
                            live_tensors,
                        ),
                        permuted_compatibility_left=left.compatibility,
                        permuted_compatibility_right=right.compatibility,
                        drop_valid_reservations=True,
                        delay=False,
                        ignored_resources=set(),
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

            # No valid pmappings! Merge all possible, then pick one. We'll charge for
            # evaluations again because we're picking a new mapping.

            self.add_evaluations()

            s = self.einsum2sim[e]
            s.mappings.data["_INDEX"] = list(range(len(s.mappings.data)))
            joined_new = _merge_next(
                joined,
                s,
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
    return MapspaceGlobals(
        einsum2sims=pmapping_groups,
        aliased_tensors=spec.workload.get_tensor_copies(),
        objective_function=objective_function,
        tracker=tracker,
    )


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
    print(f"Completed making initial population of {len(mappings)} mappings")
    return mappings


def _join_pmappings(
    pmapping_groups: dict[EinsumName, list[PmappingGroup]],
    spec: Spec,
    tracker: EvaluationsScoreTracker,
    pop_size_per_thread: int,
    algorithm: str = "simanneal",
) -> PmappingGroup:
    mapspace_globals = _make_mapspace_globals(pmapping_groups, spec, tracker)
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
        probs = [f / total for f in inv_fitness]
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
) -> EvaluationsScoreTracker:

    spec = pmappings.spec
    tracker = EvaluationsScoreTracker(
        max_evaluations=max_evaluations / get_n_parallel_jobs(),
        stop_at_score=None,
        print_period=1,
    )

    # Disable validation in the compatibility class to avoid errors when joining
    # pmappings. We use a simplified version of the joining process in this code that
    # doesn't do all the reservation cleaning that we do in FFM.
    Compatibility.__post_init__ = lambda *args, **kwargs: None

    compressed, decompress_data = compress_einsum2pmappings(pmappings.einsum2pmappings)

    if score_target is not None:
        tracker._scale_score_by *= 1 / score_target

    pop_size_per_thread = max(1, population_size // get_n_parallel_jobs())

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

    def parallel_join(
        permuted: dict[EinsumName, list[PmappingGroup]],
        spec: Spec,
        tracker: EvaluationsScoreTracker,
        pop_size_per_thread: int,
        algorithm: str,
    ) -> EvaluationsScoreTracker:
        _join_pmappings(permuted, spec, tracker, pop_size_per_thread, algorithm)
        return tracker

    trackers = util.parallel(
        joblib.delayed(parallel_join)(
            permuted,
            spec,
            tracker,
            pop_size_per_thread,
            algorithm,
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
    # joined._data = joined.data.fillna(0)
    return t0  # Mappings(spec, list(pmappings.einsum2pmappings.keys()), joined.data)
