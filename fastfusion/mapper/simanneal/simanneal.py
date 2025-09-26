from collections.abc import Mapping
import copy
from math import ceil, exp, prod
import random
import threading
import time
from fastfusion.accelerated_imports import pd
from fastfusion.mapper.simanneal.evalmapping import quick_join
from fastfusion.mapper.simanneal.tracking import EvaluationsScoreTracker
from fastfusion.mapper.FFM._join_pmappings.join_pmappings import SIM
from fastfusion.mapper.FFM._join_pmappings.compatibility import TensorReservation, Compatibility
from fastfusion.mapper.FFM._pmapping_group import MAPPING_COLUMN, PmappingGroup
from fastfusion.util import fzs
from fastfusion.mapper.simanneal.mapspaceglobals import MapspaceGlobals
OBJECTIVE_COLUMN = None # None -> Product

class FailedMutation(Exception):
    pass

class Mapping:
    def __init__(self, sims: dict[str, list[SIM]]):
        self.einsum_names = list(sims.keys())
        self.einsum2intra_choice = {einsum_name: None for einsum_name in self.einsum_names}
        self.einsum2tiling = {}
        for einsum_name, sim_list in sims.items():
            tensor_names = sim_list[0].tensor_names
            tensors = fzs(TensorReservation(t, 0, 0, 0) for t in tensor_names)
            self.set_einsum2tiling(einsum_name, Compatibility(tuple(), tensors))
        # self.history = []
        class dummy_appender:
            def append(*args, **kwargs):
                pass
        self.history = dummy_appender()
        self.n_crossovers = 0
        self.n_mutations = 0
        
        self.n_changes = 0
        self.prev_eval_result = float("inf")
        self.prev_eval_at_n_changes = -1
        
    def set_einsum2tiling(self, einsum_name: str, tiling: Compatibility):
        prev = self.einsum2tiling.get(einsum_name, None)
        if prev is not None and prev == tiling:
            return
        self.einsum2tiling[einsum_name] = tiling
        self.einsum2intra_choice[einsum_name] = None

    def fix_loops(self, mapspace_globals: MapspaceGlobals):
        """ Ensure that all tilings have the correct number of loops """
        self.n_changes += 1
        self.history.append("Fixing loops")

        try: 
            for einsum in self.einsum_names:
                tiling = self.einsum2tiling[einsum]
                n_loops = max(t.above_loop_index for t in tiling.tensors)

                # If there's too many loops then drop the extra ones
                if n_loops < len(tiling.loops):
                    self.set_einsum2tiling(einsum, tiling.update(loops=tiling.loops[:n_loops]))

                # If there's not enough loops then add some
                if n_loops > len(tiling.loops):
                    for tensor in tiling.tensors:
                        for loop in range(len(tiling.loops), tensor.above_loop_index):
                            self.mutate_loop(mapspace_globals, tensor, loop, einsum)
                            self.force_loop_match(mapspace_globals, loop, einsum)
                assert n_loops == len(self.einsum2tiling[einsum].loops)
                
                tiling = self.einsum2tiling[einsum]
                tensors = tiling.tensors
                for i in range(len(tiling.loops)):
                    tensors = list(t for t in tensors if t.above_loop_index > i)
                    if not tensors:
                        continue
                    possible_loops = set.intersection(
                        *(mapspace_globals.tensor2possible_loops_above_set[einsum][t] for t in tensors)
                    )
                    if not possible_loops:
                        raise FailedMutation(f"No possible loops above {i} for {einsum}")
                    if tiling.loops[i] not in possible_loops:
                        new_loop = random.choice(list(possible_loops))
                        self.history.append(f"Fixing loop {i} for {einsum} to {new_loop}")
                        new_loops = tiling.loops[:i] + (new_loop,) + tiling.loops[i+1:]
                        self.set_einsum2tiling(einsum, tiling.update(loops=new_loops))

        except FailedMutation:
            self.history.append(f"Failed to fix loops")
            raise FailedMutation("Failed to fix loops")


    def match_loops(
        self, index: int, einsum_name: str, mapspace_globals: MapspaceGlobals
    ):
        """ Ensure that loops match across Einsums """
        self.n_changes += 1
        tiling = self.einsum2tiling[einsum_name]
        for einsum_name2, tiling2 in self.einsum2tiling.items():
            if einsum_name2 == einsum_name:
                continue
            shared_loop_index = max(
                tiling.shared_loop_index(tiling2.tensor_names),
                tiling2.shared_loop_index(tiling.tensor_names),
            )
            for i in range(min(shared_loop_index, index) + 1):
                # Translate loop from einsum_name to einsum_name2
                loop = tiling.loops[i]
                translations = mapspace_globals.rank_translations[einsum_name][
                    einsum_name2
                ][loop.rank_variable_name]
                if not translations:
                    raise FailedMutation(
                        f"Failed to translate loop {loop} from {einsum_name} to {einsum_name2}"
                    )
                rank_variable_name = random.choice(translations)
                new_loops = tiling2.loops[:i] + (loop.update(rank_variable_names=fzs((rank_variable_name,))),) + tiling2.loops[i+1:]
                tiling2 = tiling2.update(loops=new_loops)
            self.set_einsum2tiling(einsum_name2, tiling2)


    def mutate_loop(
        self,
        mapspace_globals: MapspaceGlobals,
        tensor: TensorReservation=None,
        index: int=None,
        einsum_name: str=None,
    ):
        self.n_changes += 1
        if tensor is None:
            memories = set().union(*(t.tensors for t in self.einsum2tiling.values()))
            memories = [m for m in memories if m.above_loop_index > 0]
            if not memories:
                raise FailedMutation("No memories to mutate")
            tensor = random.choice(list(memories))
        if index is None:
            index = random.randint(0, tensor.above_loop_index - 1)
        if einsum_name is None:
            possible_einsums = [e for e, t in self.einsum2tiling.items() if tensor in t.tensors]
            assert possible_einsums
            einsum_name = random.choice(possible_einsums)

        tiling = self.einsum2tiling[einsum_name]
        prev_loop = None

        choice = random.choice(["Increasing", "Decreasing", "Randomizing"])
        if len(tiling.loops) <= index:
            choice = "Randomizing"

        candidates = mapspace_globals.tensor2possible_loops_above[einsum_name][tensor]
        if choice == "Randomizing":
            new_loop = random.choice(candidates)
        else:
            prev_loop = tiling.loops[index]
            rank, bound = prev_loop.rank_variable_name, prev_loop.bound
            comparison = lambda x, y: x > y if choice == "Increasing" else x < y

            candidates = [
                c
                for c in candidates
                if comparison(c.bound, bound) and c.rank_variable_name == rank
            ]
            if not candidates:
                raise FailedMutation(
                    f"{choice} {prev_loop} for {einsum_name} at {index} failed"
                )
            new_loop = random.choice(candidates)

        self.history.append(f"{choice} loop {index} for {einsum_name} to {new_loop}")
        new_loops = tiling.loops[:index] + (new_loop,) + tiling.loops[index+1:]
        self.set_einsum2tiling(einsum_name, tiling.update(loops=new_loops))

    def get_shared_loop_index(
            self, 
            mapspace_globals: MapspaceGlobals, 
            einsum_name0: int, 
            einsum_name1: int
        ):
        einsum_names = list(self.einsum2tiling.keys())
        if einsum_name0 == einsum_name1:
            einsum_name = einsum_names[einsum_index0]
            return len(self.einsum2tiling[einsum_name].loops) - 1
        
        einsum_index0 = einsum_names.index(einsum_name0)
        einsum_index1 = einsum_names.index(einsum_name1)
        
        if einsum_index0 > einsum_index1:
            einsum_index0, einsum_index1 = einsum_index1, einsum_index0
            
        tiling0 = self.einsum2tiling[einsum_names[einsum_index0]]
        tiling1 = self.einsum2tiling[einsum_names[einsum_index1]]
        left_tensors = mapspace_globals.get_tensors(*einsum_names[:einsum_index0 + 1])
        right_tensors = mapspace_globals.get_tensors(*einsum_names[einsum_index1:])
        return max(
            tiling0.shared_loop_index(right_tensors),
            tiling1.shared_loop_index(left_tensors),
        )

    def force_loop_match(
        self, mapspace_globals: MapspaceGlobals, index: int, einsum_name: str, 
    ):
        self.n_changes += 1
        tiling = self.einsum2tiling[einsum_name]
        for einsum_name2, tiling2 in self.einsum2tiling.items():
            if einsum_name2 == einsum_name:
                continue
            shared_loop_index = self.get_shared_loop_index(mapspace_globals, einsum_name, einsum_name2)
            rank_translations = mapspace_globals.rank_translations[einsum_name][einsum_name2]
            for i in range(min(shared_loop_index, index) + 1):
                loop = tiling.loops[i]
                translations = rank_translations[loop.rank_variable_name]
                if not translations:
                    raise FailedMutation(
                        f"Failed to translate loop {loop} from {einsum_name} to {einsum_name2}"
                    )
                rank_variable_name = random.choice(translations)
                new_loops = tiling2.loops[:i] + (loop.update(rank_variable_names=fzs((rank_variable_name,))),) + tiling2.loops[i+1:]
                tiling2 = tiling2.update(loops=new_loops)
            self.set_einsum2tiling(einsum_name2, tiling2)

    def mutate_backing_tensor(self, mapspace_globals: MapspaceGlobals):
        self.n_changes += 1
        tensor = random.choice(list(mapspace_globals.intermediate_tensor_names))
        memories = random.choice(mapspace_globals.tensor2memories[tensor])
        for t in self.einsum2tiling.values():
            if memories in t.tensors:
                raise FailedMutation(
                    f"Moving tensor {tensor} to tensor {memories} failed"
                )
        self.history.append(f"Moving tensor {tensor} to tensor {memories}")
        for einsum, tiling in self.einsum2tiling.items():
            if not any(r.name == tensor for r in tiling.tensors):
                continue
            new_tensors = [memories] + [r for r in tiling.tensors if r.name != tensor]
            self.set_einsum2tiling(einsum, tiling.update(tensors=fzs(new_tensors)))
        self.fix_loops(mapspace_globals)

    def mutate_order(self, mapspace_globals: MapspaceGlobals):
        return
        self.n_changes += 1
        e0, e1 = random.sample(self.einsum_names, 2)
        print(f"Switching {e0} and {e1}")
        self.einsum2tiling[e0], self.einsum2tiling[e1] = (
            self.einsum2tiling[e1],
            self.einsum2tiling[e0],
        )
        self.fix_loops(mapspace_globals)

    def evaluate(self, mapspace_globals: MapspaceGlobals, return_df=False) -> float:
        if self.n_changes == self.prev_eval_at_n_changes and not return_df:
            return self.prev_eval_result, 1
        
        chosen_sims = []
        chosen_mappings = {}
        n_evaluations = mapspace_globals.size_scale * mapspace_globals.find_pmapping_scale
        
        if self.n_changes == self.prev_eval_at_n_changes and not return_df:
            return self.prev_eval_result, 1
        self.prev_eval_at_n_changes = self.n_changes
        self.prev_eval_result = float("inf")

        for einsum_name, t in self.einsum2tiling.items():
            if t not in mapspace_globals.einsum_tiling_2_sim[einsum_name]:
                assert not return_df
                return float("inf"), n_evaluations

            sim = mapspace_globals.einsum_tiling_2_sim[einsum_name][t]
            chosen_sims.append(sim)
            intra_mappings = sim.mappings.data
            
            if self.einsum2intra_choice[einsum_name] is not None:
                mapping = intra_mappings.iloc[self.einsum2intra_choice[einsum_name] % len(intra_mappings)]
                chosen_mappings[einsum_name] = mapping
                continue
            
            self.einsum2intra_choice[einsum_name] = random.randint(0, 1000000000000)
            choice = self.einsum2intra_choice[einsum_name] % len(sim.mappings.data)
            self.einsum2intra_choice[einsum_name] = choice
            n_evaluations += mapspace_globals.size_scale * mapspace_globals.find_pmapping_scale
            mapping = intra_mappings.iloc[choice]
            chosen_mappings[einsum_name] = mapping

        try:
            new_sims = {}
            for einsum_name, tiling in self.einsum2tiling.items():
                sim = mapspace_globals.einsum_tiling_2_sim[einsum_name][tiling]
                mapping_index = self.einsum2intra_choice[einsum_name] % len(sim.mappings.data)
                new_sims[einsum_name] = [
                    SIM(
                        compatibility=sim.compatibility,
                        mappings=PmappingGroup(sim.mappings.data.iloc[mapping_index:mapping_index+1].copy()),
                    )
                ]
                chosen_mappings = quick_join(new_sims, mapspace_globals)
                assert len(chosen_mappings.data) == 1
                chosen_mappings = chosen_mappings.data.iloc[0]
        except Exception as e:
            assert not return_df
            return float("inf"), n_evaluations
            
        obj_cols = mapspace_globals.objective_function_cols
        score = prod(chosen_mappings[col] for col in obj_cols)
        # if score < 4.7770043942936216e+20:
        #     print("AHH")
        # import pydot
        # graph = pydot.Dot(graph_type="digraph", ranksep="0.2", nodesep="0.2")
        # tree.to_pydot(graph)
        # with open(f"test.png", "wb") as f:
        #     f.write(graph.create_png())

            

        if return_df:
            d = {col: sum(c[col] for c in chosen_mappings.values()) for col in obj_cols}
            d[MAPPING_COLUMN] = mapping
            self.prev_eval_result = score
            return pd.DataFrame([d]), n_evaluations
        self.prev_eval_result = score
        return score, n_evaluations
    
    def mutate_intra_mapping(self, mapspace_globals: MapspaceGlobals):
        self.n_changes += 1
        einsum_name = random.choice(self.einsum_names)
        self.history.append(f"Choosing intra-layer mapping for {einsum_name}")
        self.einsum2intra_choice[einsum_name] = None
    
    def get_mutation_functions(self):
        return [self.mutate_loop, self.mutate_backing_tensor, self.mutate_order, self.mutate_intra_mapping]

    def crossover(self, other: Mapping, mapspace_globals: MapspaceGlobals):
        child = copy.deepcopy(other)
        einsum_name = random.choice(child.einsum_names)
        try:
            child.set_einsum2tiling(einsum_name, self.einsum2tiling[einsum_name])
            child.einsum2intra_choice[einsum_name] = self.einsum2intra_choice[einsum_name]
            child.n_changes += 1
            for i in range(len(child.einsum2tiling[einsum_name].loops)):
                child.match_loops(i, einsum_name, mapspace_globals)
            child.fix_loops(mapspace_globals)
            child.n_crossovers += 1
        except FailedMutation:
            return copy.deepcopy(other)
        return child
    
    @staticmethod
    def create_random_mapping(mapspace_globals: MapspaceGlobals):
        mapping = Mapping(mapspace_globals.sims)
        prev_compatibility: Compatibility = None
        einsum_names = list(mapping.einsum2tiling.keys())
        for i, einsum_name in enumerate(einsum_names):
            sim_list = mapspace_globals.sims[einsum_name]
            if prev_compatibility is None:
                sim = random.choice(sim_list)
                mapping.set_einsum2tiling(einsum_name, sim.compatibility)
                if len(einsum_names) == 1:
                    break
                prev_compatibility = mapspace_globals.compatibility2rightcompatibility[einsum_name][sim.compatibility]
                live_tensors = mapspace_globals.get_live_tensors(*einsum_names[i+1:])
                prev_compatibility = prev_compatibility.clear_dead_tensors(live_tensors=live_tensors)
                continue

            tilings = []
            compatiblity_options = mapspace_globals.leftcompatibility2tiling[einsum_name]
            cur_tensors = mapspace_globals.get_tensors(einsum_name)
            for translation in mapspace_globals.get_possible_translations(
                prev_compatibility,
                einsum_name
            ):
                translation = translation.clear_dead_tensors(live_tensors=cur_tensors, keep_loops=True)
                if translation in compatiblity_options:
                    tilings.extend(compatiblity_options[translation])
                
            if not tilings:
                raise FailedMutation(f"No tilings for {einsum_name} with {prev_compatibility}")
            sim_choices = [mapspace_globals.einsum_tiling_2_sim[einsum_name][t] for t in tilings]
            sim = random.choice(sim_choices)
            tiling = sim.compatibility
            mapping.set_einsum2tiling(einsum_name, tiling)
            if i == len(einsum_names) - 1:
                break
            new_compatibility: Compatibility = mapspace_globals.compatibility2rightcompatibility[einsum_name][tiling]
            # Combine prev_compatibility and new_compatibility
            live_tensors = mapspace_globals.get_live_tensors(*einsum_names[i+1:])
            prev_compatibility = prev_compatibility.merge_next(new_compatibility, live_tensors)
        return mapping
    
def get_accept_function(temperature, cooling_rate, evaluations_tracker):
    proportion = evaluations_tracker.evaluations / evaluations_tracker.max_evaluations
    new_temp = (
        temperature
        * (1 - proportion)
        / (1 + cooling_rate * proportion)
    )
    # Assume prescient knowledge of the best score with which to scale by
    def accept(prev_eval_result, new_score):
        if new_score == float("inf"):
            return False
        if new_score <= prev_eval_result:
            return True
        scaleby = new_temp * evaluations_tracker.stop_at_score
        if scaleby > 0 and random.random() < exp((prev_eval_result - new_score) / scaleby):
            return True
        return False
    return accept

def mutate(mapping: Mapping, mapspace_globals: MapspaceGlobals, accept_function: callable):
    prev_mapping = copy.deepcopy(mapping)
    prev_eval_result = mapping.prev_eval_result
    n_evaluations = 1
    try:
        choice = random.choice(mapping.get_mutation_functions())
        choice(mapspace_globals)
    except FailedMutation:
        return prev_mapping, n_evaluations
    new_score, n_evaluations = mapping.evaluate(mapspace_globals)
    if new_score == float("inf"):
        return prev_mapping, n_evaluations
    if accept_function(prev_eval_result, new_score):
        return mapping, n_evaluations
    return prev_mapping, n_evaluations

def _fuse_sims(
    mapspace_globals: MapspaceGlobals,
    n_threads: int,
    evaluations_tracker: EvaluationsScoreTracker,
    algorithm: str
):
    random.seed(time.time() + hash(threading.get_ident()))  # Seed with thread ID
    evaluations_tracker.multiply_scale_by(len(mapspace_globals.einsum_names))
    evaluations_tracker.print_period *= n_threads
    evaluations_tracker.max_evaluations //= n_threads
    def anneal_population(population, mapspace_globals: MapspaceGlobals, n_rounds):
        temperature = 0.07
        cooling_rate = 8
        while True:
            accept_function = get_accept_function(temperature, cooling_rate, evaluations_tracker)
            # population = parallel([delayed(mutate)(m, mapspace_globals, accept_function) for m in population])
            for j, mapping in enumerate(population):
                population[j], evaluations = mutate(mapping, mapspace_globals, accept_function)
                if evaluations_tracker.add_evaluation(evaluations, population[j].prev_eval_result):
                    return population

    def genetic_algorithm_population(population, mapspace_globals: MapspaceGlobals, n_rounds):
        population_size = len(population)
        crossover_rate = 0.7
        mutation_rate = 0.2

        def crossover(parent1: Mapping, parent2: Mapping):
            if random.random() > crossover_rate:
                return copy.deepcopy(parent1)
            return parent1.crossover(parent2, mapspace_globals)

        def mutate_individual(individual):
            individual = copy.deepcopy(individual)
            prev_mapping = copy.deepcopy(individual)
            if random.random() > mutation_rate:
                return individual
            try:
                mutation_function = random.choice(individual.get_mutation_functions())
                mutation_function(mapspace_globals)
                individual.n_mutations += 1
                return individual
            except FailedMutation:
                return prev_mapping

        best_fitness = float("inf")
        while True:
            # Evaluate fitness
            fitness = [0] * len(population)
            for i, individual in enumerate(population):
                f, evaluations = individual.evaluate(mapspace_globals)
                fitness[i] = f
                best_fitness = min(best_fitness, f)
                if evaluations_tracker.add_evaluation(evaluations, best_fitness):
                    return population

            best_score = min(fitness)
            best_mapping = population[fitness.index(best_score)]

            # Selection (roulette wheel selection)
            total_fitness = sum(1.0 / (f + 1e-9) for f in fitness)
            probabilities = [(1.0 / (f + 1e-9)) / total_fitness for f in fitness]
            selected_indices = random.choices(range(len(population)), probabilities, k=population_size)

            # Crossover
            new_population = list(population[i] for i in selected_indices)
            for i in range(0, population_size, 2):
                parent1 = population[selected_indices[i]]
                parent2 = population[selected_indices[(i + 1) % population_size]]
                child1 = crossover(parent1, parent2)
                child2 = crossover(parent2, parent1)
                new_population.extend([child1, child2])

            # Mutation
            for i, individual in enumerate(new_population):
                new_population[i] = mutate_individual(individual)

            new_population.append(best_mapping) # Keep the best mapping around
            population = new_population

        return population
    
    def random_sample_population(population, mapspace_globals: MapspaceGlobals, n_rounds, prune=False):
        best_mapping = population[0]
        best_score = float("inf")
        while True:
            try:
                mapping = Mapping.create_random_mapping(mapspace_globals)
            except FailedMutation:
                if not prune:
                    if evaluations_tracker.add_evaluation(1, float("inf")):
                        return [best_mapping]
                continue
            score, evaluations = mapping.evaluate(mapspace_globals)
            if score < best_score:
                best_mapping = mapping
                best_score = score
            if evaluations_tracker.add_evaluation(evaluations, score):
                return [best_mapping]
        return [best_mapping]

    extra_args = {}
    if algorithm == "genetic":
        population_size = 1000
        callfunc = genetic_algorithm_population
    elif algorithm == "simulated_anneal":
        population_size = 100 // n_threads
        callfunc = anneal_population
    elif "random" in algorithm:
        population_size = 1
        callfunc = random_sample_population
        extra_args["prune"] = "pruned" in algorithm
        
    # Randomly intialize the population
    def get_random_mapping():
        while True:
            try:
                mapping = Mapping.create_random_mapping(mapspace_globals)
                score, evaluations = mapping.evaluate(mapspace_globals)
                evaluations_tracker.add_evaluation(evaluations, score)
                if score == float("inf"):
                    raise FailedMutation("Random mapping failed")
                return mapping
            except FailedMutation:
                pass
            
    population = []
    while len(population) < population_size:
        try:
            mapping = Mapping.create_random_mapping(mapspace_globals)
            score, evaluations = mapping.evaluate(mapspace_globals)
            if evaluations_tracker.add_evaluation(evaluations, score):
                break
            if score == float("inf"):
                raise FailedMutation("Random mapping failed")
            population.append(mapping)
        except FailedMutation:
            if evaluations_tracker.add_evaluation(1, float("inf")):
                break

    n_rounds = 9999999999999999999999999
    results = callfunc(population, mapspace_globals, n_rounds)
    eval_results = []
    for m in results:
        try:
            eval_results.append(m.evaluate(mapspace_globals, return_df=True)[0])
        except:
            pass
    try:
        return pd.DataFrame(), evaluations_tracker
        assert False, "Not saving chosen mappings to avoid big files"
        return pd.concat(eval_results), evaluations_tracker # <- Resulted in large files bc it's not pareto pruned
    except Exception as e:
        for i in range(30):
            print(f'Failed to concatenate results. Exception: {e}')
        return pd.DataFrame(), evaluations_tracker

