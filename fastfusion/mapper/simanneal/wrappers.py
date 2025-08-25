from collections import defaultdict
import copy
import itertools
import time
from typing import TypeAlias, Union
from joblib import delayed
from fastfusion.accelerated_imports import pd
from fastfusion.frontend import architecture
from fastfusion.frontend.specification import Specification
from fastfusion.mapper.FFM.joining.sim import SIM, Loop, Compatibility
from fastfusion.mapper.FFM.pareto import PartialMappings, is_reservation_col
from fastfusion.mapper.simanneal.simanneal import MapspaceGlobals, _fuse_sims
from fastfusion.mapper.simanneal.tracking import EvaluationsScoreTracker
from fastfusion.util import fzs, parallel, util


def mapping2sims(einsum_to_result: Compatibility):
    r = {}
    for einsum_name, compat_dict in einsum_to_result.items():
        r[einsum_name] = [paretofy(k, v) for k, v in compat_dict.items()]
    return list(r.values())


def paretofy(k, v):
    return SIM(k, PartialMappings(pd.DataFrame(v).fillna(0)))


def get_possible_translations(
    t: Compatibility,
    pairwise_equivalent_rank_variables: dict[str, set[str]],
    full_equivalent_rank_variables: dict[str, set[str]],
    right_rank_variables: set[str],
):
    # Fused ranks should be transitive, but if a fused loop indexes into two
    # different ranks in the next Einsum, we can't fuse becuase it will tile in
    # multiple directions.
    #
    # The first union checks what loops we CAN fuse with in the next Einsum. The
    # second union checks what loops MUST index into in the next
    #
    # Einsum. If we alias into multiple ranks, we can't fuse. Otherwise, try out
    # each possible rank.
    def translate_loop(l: Loop):
        compatible_rank_variables = (
            set.union(*(full_equivalent_rank_variables[n] for n in l.rank_variable_names))
            & right_rank_variables
        )
        pairwise_compatible_rank_variables = (
            set.union(*(pairwise_equivalent_rank_variables[n] for n in l.rank_variable_names))
            & right_rank_variables
        )
        if len(pairwise_compatible_rank_variables) > 1:
            return
        for n in compatible_rank_variables:
            yield Loop(fzs((n,)), l.bound, l.is_spatial)

    for loops in itertools.product(*map(translate_loop, t.loops)):
        yield t.update(loops=loops)


prev_time = 0
total_time = defaultdict(int)


def init_print_time():
    global prev_time, total_time
    prev_time = time.time()
    total_time = defaultdict(int)


def print_time(what: str):
    global prev_time
    t = time.time() - prev_time
    print(f"{what}: {t:.2f} seconds")
    total_time[what] += t
    prev_time = time.time()


def print_total_time():
    print(f"\n======== Total time ========")
    for k, v in total_time.items():
        print(f"{k}: {v:.2f} seconds")
    total = sum(total_time.values())
    if total > 60:
        print(f"\nTotal: {total:.2f} seconds ({total/60:.2f} minutes)")
    else:
        print(f"\nTotal: {total:.2f} seconds")
    print(f"============================\n")


class GroupOfSIMsHolder:
    def __init__(self, einsum_name: str, sim_list: list[SIM]):
        self.einsum_name: str = einsum_name
        self.sims: list[SIM] = sim_list
        self.tensor_names: set[str] = set(sim_list[0].tensor_names)

    def __getitem__(self, i):
        return self.sims[i]


def make_full_equivalent_rank_variables(pairwise_equivalent_rank_variables):
    full_equivalent_rank_variables = {
        k: set(v) for k, v in pairwise_equivalent_rank_variables.items()
    }
    changed = True
    while changed:
        changed = False
        for r in full_equivalent_rank_variables:
            for r2 in list(full_equivalent_rank_variables[r]):
                for r3 in list(full_equivalent_rank_variables[r2]):
                    if r3 in full_equivalent_rank_variables[r]:
                        continue
                    changed = True
                    full_equivalent_rank_variables[r].add(r3)
    return full_equivalent_rank_variables



def get_sims_data(
    sims: dict[str, list[SIM]],
    evaluations_tracker,
    spec: Specification = None,
    flattened_architecture: list[architecture.Leaf] = None,
):
    resource2capacity = {}
    flattened_architecture = flattened_architecture or spec.get_flattened_architecture()
    for l in flattened_architecture:
        if isinstance(l, architecture.Memory):
            resource2capacity[l.name] = l.attributes.size

    pairwise_equivalent_rank_variables = (
        spec.workload.get_pairwise_equivalent_rank_variables()
    )

    aliased_tensors = spec.workload.get_tensor_copies()

    full_equivalent_rank_variables = make_full_equivalent_rank_variables(
        pairwise_equivalent_rank_variables
    )

    return (
        sims,
        evaluations_tracker,
        spec,
        flattened_architecture,
        resource2capacity,
        pairwise_equivalent_rank_variables,
        aliased_tensors,
        full_equivalent_rank_variables,
    )


def join_sims(
    sims: dict[str, list[SIM]],
    evaluations_tracker: EvaluationsScoreTracker,
    algorithm: str,
    spec: Specification = None,
    flattened_architecture: list[architecture.Leaf] = None,
) -> PartialMappings:
    objective_function_cols = None
    cols = next(iter(sims.values()))[0].mappings.data.columns
    if objective_function_cols is None:
        objective_function_cols = [c for c in cols if "Total" in c]
    keepcols = []

    for sim_list in sims.values():
        for sim in sim_list:
            for col in objective_function_cols:
                if col not in sim.mappings.data.columns:
                    sim.mappings.data[col] = 0
            reservations = [c for c in sim.mappings.data.columns if is_reservation_col(c)]
            sim.mappings._data = sim.mappings.data[objective_function_cols + keepcols + reservations]

    mapspace_globals = MapspaceGlobals(
        sims,
        spec,
        objective_function_cols,
        flattened_architecture,
    )

    n_threads = util.N_PARALLEL_PROCESSES
    while n_threads >= 1:
        try:
            results_and_trackers = parallel([delayed(_fuse_sims)(
                mapspace_globals,
                n_threads=n_threads,
                evaluations_tracker=copy.deepcopy(evaluations_tracker),
                algorithm=algorithm,
            ) for _ in range(n_threads)], n_jobs=n_threads if util.PARALLELIZE else 1)
            results = pd.concat([r[0] for r in results_and_trackers])
            break
        except OSError as e:
            if n_threads == 1:
                raise OSError("Failed to fuse sims with 1 thread") from e
            print(f"Failed to fuse sims with {n_threads} threads, trying with {n_threads // 2}")
            n_threads //= 2
            
    for t in results_and_trackers:
        evaluations_tracker.merge_with(t[1])
    return results
