import logging
from math import prod
from pathlib import Path
from typing import Callable, Optional

from joblib import delayed
from fastfusion.accelerated_imports import pd

from fastfusion.frontend import architecture
from fastfusion.frontend.specification import Specification
from fastfusion.frontend.mapping import Iteration, Mapping, Reservation, Storage
from fastfusion.frontend.workload.isl import get_rank_variable_bounds
from fastfusion.frontend.workload.workload import EinsumName, TensorName

from fastfusion.mapper.FFM.exploration.mapper_one_einsum.mapper_one_einsum import generate_pmappings
from fastfusion.mapper.FFM.exploration.metrics import Metrics
from fastfusion.mapper.FFM.exploration.mapper_one_einsum import get_single_einsum_jobs
from fastfusion.mapper.FFM.joining.mappinginfo import Compatibility
from fastfusion.mapper.FFM.joining.sim import SIM
from fastfusion.mapper.FFM.compress_pmappings import DecompressData, GroupedDecompressData
from fastfusion.mapper.FFM.tags import Tags
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
            job_id=0,
            except_from_imperfect=except_from_imperfect,
            intermediate_tensors=intermediate_tensors & workload_einsum.tensor_names
        )
        for j in get_single_einsum_jobs(job):
            jobs.setdefault(j.compatibility, SameCompatibilityJobs()).append(j)

        return einsum_name, jobs

    einsum2jobs = {}
    for einsum_name, jobs in parallel(
        [delayed(make_jobs_for_einsum)(einsum_name) for einsum_name in einsum_names],
        pbar="Generating Jobs",
        return_as="generator",
    ):
        print(f"Generated {sum(len(j) for j in jobs.values())} jobs for {einsum_name}")
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
        if memory.attributes.size >= total_tensor_sizes * memory.attributes.datawidth: # max(m.attributes.datawidth.values()):
            memories_track_all.remove(m)
            logging.info(
                f"Not tracking memory {m}. It is big enough to hold "
                f"every tensor in the workload."
            )

    # If the memory is below every backing storage node, then we need it for the
    # pmapping exploration but can drop it immediately
    for m in list(memories_track_all):
        must_track = False
        for job in jobs:
            seen = False
            for node in job.mapping.nodes:
                if isinstance(node, Storage) and node.memory == m:
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


class PmappingExplorer:
    """
    Explores pmapspace of `einsum_names` (default: all Einsums in workload).

    Call `generate_unshaped_pmappings` first to explore dataflow and storage
    nodes order. 
    """
    def __init__(
        self,
        spec: Specification,
        flattened_arch: Optional[list[architecture.Leaf]] = None,
        tagger: Callable[[Mapping], Tags] | None = None,
        metrics: Metrics = Metrics.ENERGY | Metrics.LATENCY,
        einsum_names: Optional[list[EinsumName]] = None,
        except_from_imperfect: set = set(),
        fail_if_no_pmappings_for_einsum: bool = True
    ):
        self.spec = spec

        if flattened_arch is None:
            flattened_arch = spec.get_flattened_architecture()
        self.flattened_arch = flattened_arch

        self.tagger = tagger
        self.metrics = metrics

        if einsum_names is None:
            einsum_names = spec.workload.einsum_names
        self.einsum_names = einsum_names

        self.except_from_imperfect = except_from_imperfect
        self.fail_if_no_pmappings_for_einsum = fail_if_no_pmappings_for_einsum

        self._einsum2jobs = None
        self._sims = None
        self._grouped_decompress_data = None

    def generate_unshaped_pmappings(self):
        """Generate pmappings without tile shapes. Caches results."""
        if self._einsum2jobs is not None:
            return
        self._einsum2jobs = get_jobs(self.spec,
                                    self.flattened_arch,
                                    self.tagger,
                                    self.metrics,
                                    self.einsum_names,
                                    self.except_from_imperfect,
                                    self.fail_if_no_pmappings_for_einsum)
        if self.fail_if_no_pmappings_for_einsum:
            for einsum_name, jobs in self._einsum2jobs.items():
                if len(jobs) == 0:
                    raise ValueError(
                        f"No pmappings for {einsum_name}. "
                        f"Was the mapspace overconstrained?"
                    )

    @property
    def einsum2unshaped_compatibilities(self):
        self.generate_unshaped_pmappings()
        return {einsum_name: compatibility
                for einsum_name, compatibilities in self._einsum2jobs.items()
                for compatibility in compatibilities}

    def prune_unshaped_pmappings(self):
        raise NotImplementedError()

    def generate_complete_pmappings(self):
        """Generate pmappings complete with shapes. Caches results."""
        if self._sims is not None:
            return self._sims, self._grouped_decompress_data

        self.generate_unshaped_pmappings()

        self._fill_jobs_with_memories_to_track()

        calls = self._allocate_jobs()

        grouped_decompress_data = GroupedDecompressData(prefix2datalist={})
        sims = {einsum_name: [] for einsum_name in self.spec.workload.einsum_names}
        for einsum_name, new_sims, decompress_data, job_ids in parallel(
            calls,
            pbar=f"Generating Pmappings",
            return_as="generator_unordered",
        ):
            grouped_decompress_data.register_decompress_data(
                einsum_name,
                job_ids,
                decompress_data,
            )
            sims[einsum_name].extend(new_sims)
            # for sim_group in new_sims:
            #     for sim in sim_group._equivalent_sims:
            #         # if sim.compatibility in seen_compatibilities[einsum_name]:
            #         #     print(f'Einsum: {einsum_name}')
            #         #     job_a = job
            #         #     job_b = seen_compatibilities[einsum_name][sim.compatibility]
            #         #     print(f'\tJob {id(job_a)} compatibility: {job_a.compatibility}')
            #         #     print(f'\tJob {id(job_b)} compatibility: {job_b.compatibility}')
            #         #     print(f'\tDuplicate compatibility {sim.compatibility}')
            #         #     raise ValueError(f"Duplicate compatibility {sim.compatibility} for {einsum_name}")
            #         seen_compatibilities[einsum_name][sim.compatibility] = job
        
        self.sims = sims
        self.grouped_decompress_data = grouped_decompress_data

        return sims, grouped_decompress_data

    def _allocate_jobs(self):
        calls = []
        for einsum_name, jobs in self._einsum2jobs.items():
            calls.extend(delayed(generate_pmappings)(job_list)
                        for job_list in jobs.values())
            
        if util.PARALLELIZE and len(calls) < util.N_PARALLEL_THREADS * 4:
            logging.warning(
                f"Insufficient jobs available to utilize available threads. "
                f"Splitting jobs into smaller chunks."
            )
            calls = []
            for einsum_name, jobs in self._einsum2jobs.items():
                for job_list in jobs.values():
                    calls.extend(delayed(generate_pmappings)(job)
                                for job in job_list.split())

        return calls

    def _fill_jobs_with_memories_to_track(self):
        jobs_flattened = [
            j for compatibility2joblist in self._einsum2jobs.values() 
            for job_list in compatibility2joblist.values()
            for j in job_list
        ]
        memories_track_all, memories_track_pmappings_only = get_memories_to_track(
            self.spec,
            self.flattened_arch,
            jobs_flattened, 
            self.metrics
        )
        for j in jobs_flattened:
            j.memories_track_all = memories_track_all
            j.memories_track_pmappings_only = memories_track_pmappings_only
