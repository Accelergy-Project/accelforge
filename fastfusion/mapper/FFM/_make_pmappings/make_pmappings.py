import logging
from math import prod

from typing import Callable, Optional
import uuid
import copy

from joblib import delayed
from tqdm import tqdm


from fastfusion.frontend import arch
from fastfusion.frontend.spec import Spec
from fastfusion.frontend.mapping import Loop, Mapping, TensorHolder
from fastfusion.frontend._workload_isl._isl import (
    get_rank_variable_bounds,
    get_tensor_size,
    get_operation_space_size,
)
from fastfusion.frontend.workload import EinsumName, SymbolTable, TensorName

from fastfusion.mapper.FFM._make_pmappings.make_pmapping_templates import (
    make_pmapping_templates,
)
from fastfusion.frontend.mapper.metrics import Metrics
from fastfusion.mapper.FFM._make_pmappings.make_pmappings_from_templates import (
    make_pmappings_from_templates,
)
from fastfusion.mapper.FFM._join_pmappings.compatibility import Compatibility
from fastfusion.mapper.FFM._join_pmappings.pmapping_group import PmappingGroup
from fastfusion.util.parallel import (
    parallel,
    _memmap_read,
    get_n_parallel_jobs,
    is_using_parallel_processing,
)
from fastfusion.mapper.FFM._make_pmappings.pmapper_job import (
    Job,
    SameCompatibilityJobs,
)


def get_rank_variable_bounds_for_all_einsums(spec: Spec):
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


def get_num_computes(spec: Spec, einsum_name: EinsumName | None = None) -> int:
    einsums = spec.workload.einsums
    einsums = [einsum_name] if einsum_name is not None else spec.workload.einsum_names
    return sum(get_operation_space_size(spec.workload, e) for e in einsums)


def get_per_tensor_size(spec: Spec) -> dict[TensorName, int]:
    return {
        tensor: get_tensor_size(spec.workload, tensor)
        for tensor in spec.workload.tensor_names
    }


def get_jobs(
    spec: Spec,
    metrics: Metrics,
    einsum_names: list[EinsumName],
    fail_if_no_pmappings_for_einsum: bool,
) -> dict[EinsumName, dict[Compatibility, SameCompatibilityJobs]]:

    spec = spec

    einsum2jobs = {}
    fusable_tensors = spec.workload.tensor_names_used_in_multiple_einsums
    rank_variable_bounds = get_rank_variable_bounds_for_all_einsums(spec)

    einsum2spec: dict[EinsumName, Spec] = {}
    s = f"Getting energy, latency, and leak power for components running "
    pbar = tqdm(einsum_names, desc=s)
    for einsum_name in pbar:
        pbar.set_description(s + einsum_name)
        einsum2spec[einsum_name] = spec._spec_parse_expressions(
            einsum_name=einsum_name,
            _parse_arch=True,
            _parse_non_arch=False,
        ).calculate_component_area_energy_latency_leak(
            einsum_name=einsum_name,
            area=False,
        )
        einsum2spec[einsum_name] = _memmap_read(einsum2spec[einsum_name])

    def make_jobs_for_einsum(einsum_name: EinsumName, spec: Spec):
        jobs = {}
        workload_einsum = spec.workload.einsums[einsum_name]
        for flattened_arch in spec._get_flattened_architecture():
            # Create jobs for each Einsum
            job = Job(
                spec=spec,
                einsum_name=einsum_name,
                metrics=metrics,
                rank_variable_bounds=rank_variable_bounds,
                flattened_arch=_memmap_read(flattened_arch),
                job_id=uuid.uuid4(),
                fusable_tensors=fusable_tensors & workload_einsum.tensor_names,
            )
            for j in make_pmapping_templates(job):
                jobs.setdefault(j.compatibility, SameCompatibilityJobs()).append(j)

        return einsum_name, jobs

    for einsum_name, jobs in parallel(
        [
            delayed(make_jobs_for_einsum)(einsum_name, spec)
            for einsum_name, spec in einsum2spec.items()
        ],
        pbar="Generating jobs",
        return_as="generator",
    ):
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
    n_procs = get_n_parallel_jobs()
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
    jobs: list[Job],
) -> dict[str, tuple[arch.Memory, int]]:
    resource2capacity = {}
    who_set = {}
    for job in jobs:
        memories = [m for m in job.flattened_arch if isinstance(m, arch.Memory)]
        for m in memories:
            resource2capacity.setdefault(m.name, m.size)
            who_set.setdefault(m.name, job.einsum_name)
            if resource2capacity[m.name] != m.size:
                raise ValueError(
                    f"Memory {m.name} has different sizes depending on which Einsum "
                    f"is being mapped. Memory sizes should not depend on which Einsum "
                    f"is being mapped. Einsum {who_set[m.name]} set the size to "
                    f"{resource2capacity[m.name]}, but Einsum {job.einsum_name} set "
                    f"the size to {m.size}."
                )
            resource2capacity[m.name] = m.size
    return resource2capacity


def get_memories_to_track(
    spec: Spec,
    jobs: list[Job],
    metrics: Metrics,
    can_combine_multiple_runs: bool,
) -> tuple[list[str], list[str]]:

    memory_to_size = get_memory_to_size(jobs)
    memories_track_all = list(memory_to_size.keys())
    memories_track_pmappings_only = []
    ignored_resources = set()

    # If we're combining the pmappings from multiple runs, we can't conclude anything
    # about the metrics to track
    if can_combine_multiple_runs:
        ignored_resources = set(memory_to_size.keys())
        return (
            memories_track_all,
            memories_track_pmappings_only,
            ignored_resources,
        )

    if metrics.RESOURCE_USAGE in metrics:
        ignored_resources = set(memory_to_size.keys())
        return (
            memories_track_all,
            memories_track_pmappings_only,
            ignored_resources,
        )

    def _get_scale(tensor_name: TensorName) -> int:
        scale = 1
        for einsum in spec.workload.einsums_with_tensor(tensor_name):
            if einsum.tensor_accesses[tensor_name].persistent:
                scale = max(scale, spec.workload.n_instances * einsum.n_instances)
        return scale

    total_tensor_sizes = sum(
        s * _get_scale(t) for t, s in get_per_tensor_size(spec).items()
    )

    # If the memory is big enough to hold all the tensors then we don't need to consider
    # it
    max_bits_per_value = max(t.bits_per_value for e in spec.workload.einsums for t in e.tensor_accesses)
    for m, size in memory_to_size.items():
        max_bits_per_value_scale = 0
        for job in jobs:
            mem = job.spec.arch.find(m)
            max_bits_per_value_scale = max(max_bits_per_value_scale, max(mem.bits_per_value_scale.values()))

        if size >= total_tensor_sizes * max_bits_per_value * max_bits_per_value_scale:
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
                    if node.persistent:
                        ignored_resources.add(m)
                if isinstance(node, Loop) and node._fused and seen:
                    must_track = True

        if not must_track:
            memories_track_all.remove(m)
            memories_track_pmappings_only.append(m)
            logging.info(
                f"Not tracking memory {m} across joining stages. It is never "
                f"reserved across fused loop iterations."
            )

    return memories_track_all, memories_track_pmappings_only, ignored_resources


def make_pmappings(
    spec: Spec,
    can_combine_multiple_runs: bool,
    metrics: Metrics = Metrics.ENERGY | Metrics.LATENCY,
    einsum_names: Optional[list[EinsumName]] = None,
    fail_if_no_pmappings_for_einsum: bool | None = None,
) -> tuple[
    dict[EinsumName, list[PmappingGroup]],
    dict[EinsumName, dict[uuid.UUID, Mapping]],
    dict[EinsumName, list[Job]],
]:
    """
    Explores pmapspace of `einsum_names` (default: all Einsums in workload).
    """
    spec = copy.deepcopy(spec)

    if einsum_names is None:
        einsum_names = spec.workload.einsum_names

    if fail_if_no_pmappings_for_einsum is None:
        fail_if_no_pmappings_for_einsum = not can_combine_multiple_runs

    spec = spec._spec_parse_expressions(
        _parse_arch=False,
        _parse_non_arch=True,
    )

    einsum2jobs = {}
    new_einsum2jobs = get_jobs(
        spec,
        metrics,
        einsum_names,
        fail_if_no_pmappings_for_einsum,
    )
    _fill_jobs_with_memories_to_track(
        new_einsum2jobs, spec, metrics, can_combine_multiple_runs
    )
    for einsum_name, jobs in new_einsum2jobs.items():
        einsum2jobs.setdefault(einsum_name, {})
        for compatibility, job_list in jobs.items():
            einsum2jobs[einsum_name].setdefault(
                compatibility, SameCompatibilityJobs()
            ).extend(job_list)

    calls = _allocate_jobs(einsum2jobs)

    # Sort the calls by the length of the longest mapping in each job. We get long
    # poles with the long mappings, so we want to get them done early so we don't
    # have one or two procs slowing us down at the end.
    def get_longest_mapping_length(call):
        j: SameCompatibilityJobs = call[2]["jobs_with_similar_compatibilities"]
        return max([len(j2.mapping.nodes) for j2 in j])

    calls = sorted(calls, key=get_longest_mapping_length, reverse=True)
    # # Randomly permute the calls
    # import random
    # random.shuffle(calls)

    pmapping_objects = {}
    pmapping_groups = {einsum_name: [] for einsum_name in spec.workload.einsum_names}
    return_jobs = {}
    for (
        einsum_name,
        new_pmapping_groups,
        pmappings,
        jobs_with_similar_compatibilities,
    ) in parallel(
        calls,
        pbar=f"Generating pmappings",
        return_as="generator_unordered",
    ):
        pmapping_groups[einsum_name].extend(new_pmapping_groups)
        pmapping_objects.setdefault(einsum_name, {}).update(pmappings)
        return_jobs.setdefault(einsum_name, []).extend(
            jobs_with_similar_compatibilities
        )

    for einsum_name in list(pmapping_groups.keys()):
        pmapping_groups[einsum_name] = PmappingGroup.combine_combineable(
            pmapping_groups[einsum_name],
            "All",
            pbar_postfix=f" for {einsum_name}",
        )

    return pmapping_groups, pmapping_objects, return_jobs


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
            delayed(make_pmappings_from_templates)(
                jobs_with_similar_compatibilities=job_list,
            )
            for job_list in jobs.values()
        )

    split = False
    if (
        not split
        and is_using_parallel_processing()
        and len(calls) < get_n_parallel_jobs() * 4
    ):
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
                    delayed(make_pmappings_from_templates)(
                        jobs_with_similar_compatibilities=job,
                    )
                    for job in job_list.split()
                )
    return calls


def _fill_jobs_with_memories_to_track(
    einsum2jobs,
    spec,
    metrics,
    can_combine_multiple_runs,
):
    jobs_flattened = [
        j
        for compatibility2joblist in einsum2jobs.values()
        for job_list in compatibility2joblist.values()
        for j in job_list
    ]
    memories_track_all, memories_track_pmappings_only, ignored_resources = (
        get_memories_to_track(
            spec,
            jobs_flattened,
            metrics,
            can_combine_multiple_runs,
        )
    )
    for j in jobs_flattened:
        j.memories_track_all = memories_track_all
        j.memories_track_pmappings_only = memories_track_pmappings_only
        j.ignored_resources = ignored_resources
