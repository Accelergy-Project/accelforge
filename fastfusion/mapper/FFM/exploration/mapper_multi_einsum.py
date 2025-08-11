import logging
from math import prod
from pathlib import Path
from typing import Callable, Optional
import uuid

from joblib import delayed
from fastfusion.accelerated_imports import pd

from fastfusion.frontend import architecture
from fastfusion.frontend.specification import Specification
from fastfusion.frontend.mapping import Iteration, Mapping, TensorHolder
from fastfusion.frontend.workload.isl import get_rank_variable_bounds
from fastfusion.frontend.workload.workload import EinsumName, TensorName

from fastfusion.mapper.FFM.exploration.mapper_one_einsum.mapper_one_einsum import generate_pmappings
from fastfusion.mapper.metrics import Metrics
from fastfusion.mapper.FFM.exploration.mapper_one_einsum import get_single_einsum_jobs
from fastfusion.mapper.FFM.joining.mappinginfo import Compatibility
from fastfusion.mapper.FFM.joining.sim import SIM
from fastfusion.mapper.FFM.deprecate_maybe.tags import Tags
from fastfusion.util.util import parallel
from fastfusion.util import util
from fastfusion.mapper.FFM.exploration.mapper_one_einsum.mapper_job import Job, SameCompatibilityJobs

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

def get_num_computes(spec: Specification) -> int:
    rank_variable_bounds = get_rank_variable_bounds_for_all_einsums(spec)
    return sum(
        prod(
            rank_variable_bounds[r] for r in einsum.rank_variables
        )
        for einsum in spec.workload.einsums
    )
    
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
    flattened_arch: Optional[list[architecture.Leaf]] = None,
    tagger: Callable[[Mapping], Tags] | None = None,
    metrics: Metrics = Metrics.ENERGY | Metrics.LATENCY,
    einsum_names: Optional[list[EinsumName]] = None,
    except_from_imperfect: set = set(),
    fail_if_no_pmappings_for_einsum: bool = True,
) -> dict[EinsumName, dict[Compatibility, SameCompatibilityJobs]]:

    einsum2jobs = {}
    intermediate_tensors = spec.workload.intermediate_tensor_names
    rank_variable_bounds = get_rank_variable_bounds_for_all_einsums(spec)

    def make_jobs_for_einsum(einsum_name: EinsumName):
        workload_einsum = spec.workload.einsums[einsum_name]
        # Create jobs for each Einsum
        jobs = {}
        job = Job(
            spec=spec,
            einsum_name=einsum_name,
            metrics=metrics,
            rank_variable_bounds=rank_variable_bounds,
            flattened_arch=flattened_arch,
            tensor2compatibilties={},#tensor2compatibilties
            tensor2boundless_compatibilities={},#tensor2boundless_compatibilities
            tagger=tagger,
            job_id=uuid.uuid4(),
            except_from_imperfect=except_from_imperfect,
            intermediate_tensors=intermediate_tensors & workload_einsum.tensor_names
        )
        for j in get_single_einsum_jobs(job):
            jobs.setdefault(j.compatibility, SameCompatibilityJobs()).append(j)

        return einsum_name, jobs

    einsum2jobs = {}
    for einsum_name, jobs in parallel(
        [delayed(make_jobs_for_einsum)(einsum_name) for einsum_name in einsum_names],
        pbar="Generating jobs",
        return_as="generator",
    ):
        n_jobs = sum(len(j) for j in jobs.values())
        print(f"Generated {n_jobs} job{'s'[:n_jobs != 1]} for {einsum_name}")
        einsum2jobs[einsum_name] = jobs

    if fail_if_no_pmappings_for_einsum:
        for einsum_name, jobs in einsum2jobs.items():
            if len(jobs) == 0:
                raise ValueError(
                    f"No pmappings for {einsum_name}. Was the mapspace overconstrained?"
                )

    return einsum2jobs

def get_memories_to_track(
    spec: Specification,
    flattened_arch: Optional[list[architecture.Leaf]],
    jobs: list[Job],
    metrics: Metrics,
) -> tuple[list[str], list[str]]:
    memories_track_all = [m.name for m in flattened_arch if isinstance(m, architecture.Memory)]
    memories_track_pmappings_only = []
    
    if metrics.RESOURCE_USAGE in metrics:
        return memories_track_all, memories_track_pmappings_only
    
    total_tensor_sizes = sum(get_per_tensor_size(spec).values())
    
    # If the memory is big enough to hold all the tensors then we don't need to consider
    # it
    for m in list(memories_track_all):
        memory = [n for n in flattened_arch if n.name == m]
        assert len(memory) == 1
        memory = memory[0]
        if memory.attributes.size >= total_tensor_sizes * max(memory.attributes.datawidth.values()):
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
    flattened_arch: Optional[list[architecture.Leaf]] = None,
    tagger: Callable[[Mapping], Tags] | None = None,
    metrics: Metrics = Metrics.ENERGY | Metrics.LATENCY,
    einsum_names: Optional[list[EinsumName]] = None,
    fail_if_no_pmappings_for_einsum: bool = True
) -> tuple[dict[EinsumName, list[SIM]], dict[EinsumName, dict[uuid.UUID, Mapping]]]:
    """
    Explores pmapspace of `einsum_names` (default: all Einsums in workload).
    """
    if flattened_arch is None:
        flattened_arch = spec.get_flattened_architecture()

    if einsum_names is None:
        einsum_names = spec.workload.einsum_names

    einsum2jobs = get_jobs(spec,
                           flattened_arch,
                           tagger,
                           metrics,
                           einsum_names,
                           fail_if_no_pmappings_for_einsum)

    _fill_jobs_with_memories_to_track(einsum2jobs, spec, flattened_arch, metrics)

    calls = _allocate_jobs(einsum2jobs)

    pmapping_objects = {}
    sims = {einsum_name: [] for einsum_name in spec.workload.einsum_names}
    for einsum_name, new_sims, pmappings in parallel(
        calls,
        pbar=f"Generating pmappings",
        return_as="generator_unordered",
    ):
        sims[einsum_name].extend(new_sims)
        pmapping_objects.setdefault(einsum_name, {}).update(pmappings)
    return sims, pmapping_objects


def _raise_error_if_no_pmappings(einsum2jobs):
    for einsum_name, jobs in einsum2jobs.items():
        if len(jobs) == 0:
            raise ValueError(
                f"No pmappings for {einsum_name}. "
                f"Was the mapspace overconstrained?"
            )


def _allocate_jobs(einsum2jobs):
    calls = []
    for einsum_name, jobs in einsum2jobs.items():
        calls.extend(delayed(generate_pmappings)(job_list)
                    for job_list in jobs.values())

    if util.PARALLELIZE and len(calls) < util.N_PARALLEL_THREADS * 4:
        logging.warning(
            f"Insufficient jobs available to utilize available threads. "
            f"Splitting jobs into smaller chunks."
        )
        calls = []
        for einsum_name, jobs in einsum2jobs.items():
            for job_list in jobs.values():
                calls.extend(delayed(generate_pmappings)(job)
                            for job in job_list.split())
    return calls

def _fill_jobs_with_memories_to_track(
    einsum2jobs,
    spec,
    flattened_arch,
    metrics,
):
    jobs_flattened = [
        j for compatibility2joblist in einsum2jobs.values() 
        for job_list in compatibility2joblist.values()
        for j in job_list
    ]
    memories_track_all, memories_track_pmappings_only = get_memories_to_track(
        spec,
        flattened_arch,
        jobs_flattened, 
        metrics
    )
    for j in jobs_flattened:
        j.memories_track_all = memories_track_all
        j.memories_track_pmappings_only = memories_track_pmappings_only
