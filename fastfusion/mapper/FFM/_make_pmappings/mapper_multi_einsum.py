import logging
from math import prod

from typing import Callable, Optional
import uuid

from joblib import delayed


from fastfusion.frontend import arch
from fastfusion.frontend.specification import Specification
from fastfusion.frontend.mapping import Iteration, Mapping, TensorHolder
from fastfusion.frontend.workload._isl import get_rank_variable_bounds
from fastfusion.frontend.workload.workload import EinsumName, TensorName

from fastfusion.mapper.FFM._make_pmappings.mapper_one_einsum.mapper_one_einsum import (
    generate_pmappings,
)
from fastfusion.frontend.mapper.metrics import Metrics
from fastfusion.mapper.FFM._make_pmappings.mapper_one_einsum import (
    get_single_einsum_jobs,
)
from fastfusion.mapper.FFM._join_pmappings.compatibility import Compatibility
from fastfusion.mapper.FFM._join_pmappings.sim import SIM
from fastfusion.mapper.FFM._make_pmappings.mapper_one_einsum.tile_shape_exploration import (
    EXPERIMENTAL_TILE_SHAPE_EXPLORATION,
)
from fastfusion.mapper.FFM.deprecate_maybe.tags import Tags
from fastfusion.util.util import parallel
from fastfusion.util import util
from fastfusion.mapper.FFM._make_pmappings.mapper_one_einsum.mapper_job import (
    Job,
    SameCompatibilityJobs,
)


def get_rank_variable_bounds_for_all_einsums(spec: Specification):
    rank_variable_bounds = {
        einsum_name: get_rank_variable_bounds(spec.workload, einsum_name)
        for einsum_name in spec.workload.einsum_names
    }
    result = {}
    for e1, rv1 in rank_variable_bounds.items():
        result.update(rv1)
        for e2, rv2 in rank_variable_bounds.items():
            for r in set(rv1.keys()) & set(rv2.keys()):
                if rv1[r] != rv2[r]:
                    raise ValueError(
                        f"Rank variable {r} has different bounds for "
                        f"einsum {e1} and {e2}: {rv1[r]} and {rv2[r]}"
                    )
    return result


def get_num_computes(spec: Specification, einsum_name: EinsumName | None = None) -> int:
    rank_variable_bounds = get_rank_variable_bounds_for_all_einsums(spec)
    einsums = spec.workload.einsums
    einsums = [einsums[einsum_name]] if einsum_name is not None else einsums
    return sum(prod(rank_variable_bounds[r] for r in e.rank_variables) for e in einsums)


def get_per_tensor_size(spec: Specification) -> dict[TensorName, int]:
    rank_variable_bounds = get_rank_variable_bounds_for_all_einsums(spec)
    sizes = {}
    for t in spec.workload.tensor_names:
        einsum = next(iter(spec.workload.einsums_with_tensor(t)))
        size = 1
        access = einsum.tensor_accesses[t]
        for r in access.fully_relevant_rank_variables:
            size *= rank_variable_bounds[r]
        if access.partially_relevant_rank_variables:
            raise ValueError(
                f"Tensor {t} has partially relevant rank variables."
                f"This function only works for fully-relevant rank variables."
            )
        sizes[t] = size
    return sizes


def get_jobs(
    spec: Specification,
    flattened_arches: list[list[arch.Leaf]],
    tagger: Callable[[Mapping], Tags] | None = None,
    metrics: Metrics = Metrics.ENERGY | Metrics.LATENCY,
    einsum_names: Optional[list[EinsumName]] = None,
    except_from_imperfect: set = set(),
    fail_if_no_pmappings_for_einsum: bool = False,
) -> dict[EinsumName, dict[Compatibility, SameCompatibilityJobs]]:

    einsum2jobs = {}
    intermediate_tensors = spec.workload.intermediate_tensor_names
    rank_variable_bounds = get_rank_variable_bounds_for_all_einsums(spec)

    def make_jobs_for_einsum(einsum_name: EinsumName, flattened_arch: list[arch.Leaf]):
        workload_einsum = spec.workload.einsums[einsum_name]
        # Create jobs for each Einsum
        jobs = {}
        job = Job(
            spec=spec,
            einsum_name=einsum_name,
            metrics=metrics,
            rank_variable_bounds=rank_variable_bounds,
            flattened_arch=flattened_arch,
            tensor2compatibilties={},  # tensor2compatibilties
            tensor2boundless_compatibilities={},  # tensor2boundless_compatibilities
            tagger=tagger,
            job_id=uuid.uuid4(),
            except_from_imperfect=except_from_imperfect,
            intermediate_tensors=intermediate_tensors & workload_einsum.tensor_names,
        )
        for j in get_single_einsum_jobs(job):
            jobs.setdefault(j.compatibility, SameCompatibilityJobs()).append(j)

        return einsum_name, jobs

    einsum2jobs = {}
    for einsum_name, jobs in parallel(
        [
            delayed(make_jobs_for_einsum)(einsum_name, flattened_arch)
            for einsum_name in einsum_names
            for flattened_arch in flattened_arches
        ],
        pbar="Generating jobs",
        return_as="generator",
    ):
        # n_jobs = sum(len(j) for j in jobs.values())
        # print(f"Generated {n_jobs} job{'s'[:n_jobs != 1]} for {einsum_name}")
        einsum2jobs.setdefault(einsum_name, {})
        for compatibility, job_list in jobs.items():
            einsum2jobs[einsum_name].setdefault(
                compatibility, SameCompatibilityJobs()
            ).extend(job_list)

    if fail_if_no_pmappings_for_einsum:
        for einsum_name, jobs in einsum2jobs.items():
            if len(jobs) == 0:
                raise ValueError(
                    f"No pmappings for {einsum_name}. Was the mapspace overconstrained?"
                )

    total_jobs = sum(len(jobs) for jobs in einsum2jobs.values())
    n_procs = util.N_PARALLEL_PROCESSES if util.PARALLELIZE else 1
    memory_limit = min(
        spec.mapper.ffm.memory_limit, spec.mapper.ffm.memory_limit_per_process / n_procs
    )
    time_limit = min(
        spec.mapper.ffm.time_limit * n_procs / max(total_jobs, 1),
        spec.mapper.ffm.time_limit_per_pmapping_template,
    )
    for einsum_name, compatibility_jobs in einsum2jobs.items():
        total_jobs = sum(len(j) for j in compatibility_jobs.values())
        logging.warning(f"Einsum {einsum_name} has {total_jobs} pmapping templates:")
        for job_list in compatibility_jobs.values():
            for job in job_list:
                logging.warning(f"\t{job.mapping.compact_str()}")
                job.memory_limit = memory_limit
                job.time_limit = time_limit

    return einsum2jobs


def get_memory_to_size(
    flattened_arches: list[list[arch.Leaf]],
) -> dict[str, tuple[arch.Memory, int]]:
    result = {}
    for flattened_arch in flattened_arches:
        for l in flattened_arch:
            if not isinstance(l, arch.Memory):
                continue
            size = l.attributes.size
            result.setdefault(l.name, (l, size))
            if result[l.name][1] != size:
                raise ValueError(
                    f"Memory {l.name} has different sizes in different flattened "
                    f"architectures: {result[l.name]} and {size}. Size may not depend "
                    f"on which compute node is used."
                )
    return result


def get_memories_to_track(
    spec: Specification,
    flattened_arches: list[list[arch.Leaf]],
    jobs: list[Job],
    metrics: Metrics,
    can_combine_multiple_runs: bool,
) -> tuple[list[str], list[str]]:

    memory_to_size = get_memory_to_size(flattened_arches)
    memories_track_all = list(memory_to_size.keys())
    memories_track_pmappings_only = []

    # If we're combining the pmappings from multiple runs, we can't conclude anything
    # about the metrics to track
    if can_combine_multiple_runs:
        return memories_track_all, memories_track_pmappings_only

    if metrics.RESOURCE_USAGE in metrics:
        return memories_track_all, memories_track_pmappings_only

    total_tensor_sizes = sum(get_per_tensor_size(spec).values())

    # If the memory is big enough to hold all the tensors then we don't need to consider
    # it
    for m in list(memories_track_all):
        memory, size = memory_to_size[m]
        if size >= total_tensor_sizes * max(memory.attributes.datawidth.values()):
            memories_track_all.remove(m)
            logging.info(
                f"Not tracking memory {m}. It is big enough to hold "
                f"every tensor in the workload."
            )

    # If the memory is below every backing tensor holder node, then we need it for the
    # pmapping exploration but can drop it immediately
    for m in list(memories_track_all):
        must_track = False
        for job in jobs:
            seen = False
            for node in job.mapping.nodes:
                if isinstance(node, TensorHolder) and node.component == m:
                    seen = True
                if isinstance(node, Iteration) and node._fused and seen:
                    must_track = True

        if not must_track:
            memories_track_all.remove(m)
            memories_track_pmappings_only.append(m)
            logging.info(
                f"Not tracking memory {m} across joining stages. It is never "
                f"reserved across fused loop iterations."
            )

    return memories_track_all, memories_track_pmappings_only


def get_sims(
    spec: Specification,
    flattened_arches: list[list[arch.Leaf]],
    can_combine_multiple_runs: bool,
    tagger: Callable[[Mapping], Tags] | None = None,
    metrics: Metrics = Metrics.ENERGY | Metrics.LATENCY,
    einsum_names: Optional[list[EinsumName]] = None,
    fail_if_no_pmappings_for_einsum: bool | None = None,
) -> tuple[
    dict[EinsumName, list[SIM]],
    dict[EinsumName, dict[uuid.UUID, Mapping]],
    dict[EinsumName, list[Job]],
]:
    """
    Explores pmapspace of `einsum_names` (default: all Einsums in workload).
    """
    if einsum_names is None:
        einsum_names = spec.workload.einsum_names

    if fail_if_no_pmappings_for_einsum is None:
        fail_if_no_pmappings_for_einsum = not can_combine_multiple_runs

    einsum2jobs = {}
    new_einsum2jobs = get_jobs(
        spec,
        flattened_arches,
        tagger,
        metrics,
        einsum_names,
        fail_if_no_pmappings_for_einsum,
    )
    _fill_jobs_with_memories_to_track(
        new_einsum2jobs, spec, flattened_arches, metrics, can_combine_multiple_runs
    )
    for einsum_name, jobs in new_einsum2jobs.items():
        einsum2jobs.setdefault(einsum_name, {})
        for compatibility, job_list in jobs.items():
            einsum2jobs[einsum_name].setdefault(
                compatibility, SameCompatibilityJobs()
            ).extend(job_list)

    calls = _allocate_jobs(einsum2jobs)

    if EXPERIMENTAL_TILE_SHAPE_EXPLORATION:
        # Sort the calls by the length of the longest mapping in each job. We get long
        # poles with the long mappings, so we want to get them done early so we don't
        # have one or two procs slowing us down at the end.
        def get_longest_mapping_length(call):
            j: SameCompatibilityJobs = call[2]["jobs_with_similar_compatibilities"]
            return max([len(j2.mapping.nodes) for j2 in j])

        calls = sorted(calls, key=get_longest_mapping_length, reverse=True)

    pmapping_objects = {}
    sims = {einsum_name: [] for einsum_name in spec.workload.einsum_names}
    return_jobs = {}
    for einsum_name, new_sims, pmappings, jobs_with_similar_compatibilities in parallel(
        calls,
        pbar=f"Generating pmappings",
        return_as="generator_unordered",
    ):
        sims[einsum_name].extend(new_sims)
        pmapping_objects.setdefault(einsum_name, {}).update(pmappings)
        return_jobs.setdefault(einsum_name, []).extend(
            jobs_with_similar_compatibilities
        )
    return sims, pmapping_objects, return_jobs


def _raise_error_if_no_pmappings(einsum2jobs):
    for einsum_name, jobs in einsum2jobs.items():
        if len(jobs) == 0:
            raise ValueError(
                f"No pmappings for {einsum_name}. " f"Was the mapspace overconstrained?"
            )


def _allocate_jobs(einsum2jobs):
    calls = []
    for einsum_name, jobs in einsum2jobs.items():
        calls.extend(
            delayed(generate_pmappings)(
                jobs_with_similar_compatibilities=job_list,
            )
            for job_list in jobs.values()
        )

    split = False
    if not split and util.PARALLELIZE and len(calls) < util.N_PARALLEL_PROCESSES * 4:
        logging.warning(
            f"Insufficient jobs available to utilize available threads. "
            f"Splitting jobs into smaller chunks."
        )
        split = True

    if split:
        calls = []
        for einsum_name, jobs in einsum2jobs.items():
            for job_list in jobs.values():
                calls.extend(
                    delayed(generate_pmappings)(
                        jobs_with_similar_compatibilities=job,
                    )
                    for job in job_list.split()
                )
    return calls


def _fill_jobs_with_memories_to_track(
    einsum2jobs,
    spec,
    flattened_arches,
    metrics,
    can_combine_multiple_runs,
):
    jobs_flattened = [
        j
        for compatibility2joblist in einsum2jobs.values()
        for job_list in compatibility2joblist.values()
        for j in job_list
    ]
    memories_track_all, memories_track_pmappings_only = get_memories_to_track(
        spec,
        flattened_arches,
        jobs_flattened,
        metrics,
        can_combine_multiple_runs,
    )
    for j in jobs_flattened:
        j.memories_track_all = memories_track_all
        j.memories_track_pmappings_only = memories_track_pmappings_only
