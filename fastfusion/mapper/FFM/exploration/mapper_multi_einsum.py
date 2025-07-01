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

    # If the memory is below every backing storage node, then we need it for the
    # pmapping exploration but can drop it immediately
    for m in list(memories_track_all):
        seen = False
        must_track = False
        for job in jobs:
            for node in job.mapping.nodes:
                if isinstance(node, Storage) and node.memory == m:
                    seen = True
                if isinstance(node, Iteration) and node._fused and seen:
                    must_track = True                    

        if not must_track:
            memories_track_all.remove(m)
            memories_track_pmappings_only.append(m)
            
    print(f'Memories to track: \n\t' + '\n\t'.join(memories_track_all))
    print(f'Memories to track pmappings only: \n\t' + '\n\t'.join(memories_track_pmappings_only))

    return memories_track_all, memories_track_pmappings_only

def get_sims(
    spec: Specification,
    flattened_arch: Optional[list[architecture.Leaf]] = None,
    tagger: Callable[[Mapping], Tags] | None = None,
    metrics: Metrics = Metrics.ENERGY | Metrics.LATENCY,
    einsum_names: Optional[list[EinsumName]] = None,
    except_from_imperfect: set = set(),
) -> tuple[dict[EinsumName, list[SIM]], DecompressData]:
    
    print(
        f'By default metrics optimizes for energy and latency.'
        f'We should change to just energy or just latency at '
        f'some point.'
    )

    if flattened_arch is None:
        flattened_arch = spec.get_flattened_architecture()
        
    einsum2jobs = {}
    tensor2compatibilties = {}
    intermediate_tensors = spec.workload.intermediate_tensor_names
    einsum_names = einsum_names or spec.workload.einsum_names
    # for einsum_name in spec.workload.einsum_names:
    #     workload_einsum = spec.workload.einsums[einsum_name]
    #     # Create jobs for each Einsum
    #     jobs = {}
    #     job = Job(
    #         spec=spec,
    #         einsum_name=einsum_name,
    #         metrics=metrics,
    #         rank_variable_bounds=rank_variable_bounds,
    #         flattened_arch=flattened_arch,
    #         tensor2compatibilties={},#tensor2compatibilties,
    #         tensor2boundless_compatibilities={},#tensor2boundless_compatibilities,
    #         tagger=tagger,
    #         job_id=0,
    #         except_from_imperfect=except_from_imperfect,
    #         intermediate_tensors=intermediate_tensors & workload_einsum.tensor_names
    #     )
    #     for j in get_single_einsum_jobs(job):
    #         jobs.setdefault(j.compatibility, []).append(j)
    #     einsum2jobs[einsum_name] = jobs
        
    #     # Update tensor2compatibilties. This will store only tensor compatibilities that
    #     # have a matching job for every Einsum that touches the tensor.
    #     compatibilities = list(jobs.keys())
    #     cur_tensor2compatibilties = {}
    #     for compatibility in compatibilities:
    #         for t, c in compatibility.per_tensor_compatibility().items():
    #             for c2 in c.subsets_of_loops():
    #                 cur_tensor2compatibilties.setdefault(t, set()).add(c2)
    #     for t in cur_tensor2compatibilties:
    #         if t in tensor2compatibilties:
    #             tensor2compatibilties[t] &= cur_tensor2compatibilties[t]
        
    # # Drop jobs that won't be compatible with other Einsums
    # for einsum_name, jobs in einsum2jobs.items():
    #     for compatibility in list(jobs.keys()):
    #         for t, c in compatibility.per_tensor_compatibility().items():
    #             if t not in tensor2compatibilties:
    #                 raise RuntimeError(f"BUG")
    #             if c in tensor2compatibilties[t]:
    #                 break
    #         else:
    #             print(f'Dropping {einsum_name} compatibility {compatibility}')
    #             del jobs[compatibility]
    # einsum_names = ["V"]
    einsum2jobs = get_jobs(spec, flattened_arch, tagger, metrics, einsum_names, except_from_imperfect)
            
    # Allocate jobs
    calls = []
    grouped_decompress_data = GroupedDecompressData(prefix2datalist={})
    for einsum_name, jobs in einsum2jobs.items():
        calls.extend(delayed(generate_pmappings)(job_list) for job_list in jobs.values())
        
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

    seen_compatibilities = {einsum_name: {} for einsum_name in spec.workload.einsum_names}
    sims = {einsum_name: [] for einsum_name in spec.workload.einsum_names}
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
    
    return sims, grouped_decompress_data
    
     
        
    # sims = {}
    # grouped_decompress_data: GroupedDecompressData = GroupedDecompressData(prefix2datalist={})
    # tensor2compatibilties = {}
    # intermediate_tensors = spec.workload.intermediate_tensor_names
    # for einsum_name in spec.workload.einsum_names:
    #     cur_sims: list[SIM] = []
    #     workload_einsum = spec.workload.einsums[einsum_name]
    #     relevant_tensor2compatibilties = {
    #         t: s
    #         for t, s in tensor2compatibilties.items()
    #         if t in workload_einsum.tensor_names
    #     }
    #     relevant_tensor2boundless_compatibilities = {
    #         t: set(
    #             c.clear_loop_bounds()
    #             for c in relevant_tensor2compatibilties[t]
    #         )
    #         for t in relevant_tensor2compatibilties
    #     }
    #     job = Job(
    #         spec=spec,
    #         einsum_name=einsum_name,
    #         metrics=metrics,
    #         rank_variable_bounds=rank_variable_bounds,
    #         flattened_arch=flattened_arch,
    #         tensor2compatibilties=relevant_tensor2compatibilties,
    #         tensor2boundless_compatibilities=relevant_tensor2boundless_compatibilities,
    #         tagger=tagger,
    #         job_id=0,
    #         intermediate_tensors= intermediate_tensors & workload_einsum.tensor_names,
    #         except_from_imperfect=except_from_imperfect,
    #     )
        
    #     jobs = get_single_einsum_jobs(job)
        
    #     for einsum_name, new_sims, decompress_data, job_id in parallel(
    #         [delayed(_per_proc_compatibility2sim)(job) for job in jobs],
    #         pbar=f"Generating Pmappings for {einsum_name}",
    #         return_as="generator_unordered",
    #     ):
    #         if len(new_sims) == 0:
    #             continue
    #         for sim in new_sims:
    #             for equivalent_sim in sim._equivalent_sims:
    #                 equivalent_sim.mappings = sim.mappings
    #                 cur_sims.append(equivalent_sim)
    #                 break
    #         # cur_sims.extend(new_sims)
    #         grouped_decompress_data.register_decompress_data(
    #             einsum_name,
    #             job_id,
    #             decompress_data,
    #         )
            
    #     cur_tensor2compatibilties = {}
    #     for s in cur_sims:
    #         for t, c in s.compatibility.per_tensor_compatibility().items():
    #             cur_tensor2compatibilties.setdefault(t, set()).add(c.clear_tags())
                
    #     for t in cur_tensor2compatibilties:
    #         if t in tensor2compatibilties:
    #             tensor2compatibilties[t] &= cur_tensor2compatibilties[t]
    #         else:
    #             tensor2compatibilties[t] = cur_tensor2compatibilties[t]
            
    #     sims[einsum_name] = cur_sims
        
    # intermediate_tensors = spec.workload.intermediate_tensor_names
    # for einsum_name, sims2 in sims.items():
    #     sims[einsum_name] = SIM.combine_combineable(sims2, live_tensors=intermediate_tensors, pbar_postfix = f" for {einsum_name}")

    # return sims, grouped_decompress_data


    single_einsum_jobs = []
    einsum_names = einsum_names or spec.workload.einsum_names
    intermediate_tensors = spec.workload.intermediate_tensor_names
    for einsum_name in einsum_names:
        job = Job(
            spec=spec,
            einsum_name=einsum_name,
            metrics=metrics,
            flattened_arch=flattened_arch,
            except_from_imperfect=except_from_imperfect,
            intermediate_tensors=intermediate_tensors & spec.workload.einsums[einsum_name].tensor_names,
            tagger=tagger,
            job_id=len(single_einsum_jobs),
            tensor2compatibilties ={},
        )
        single_einsum_jobs.extend(get_single_einsum_jobs(
            job
        ))

    sims = {einsum_name: [] for einsum_name in spec.workload.einsum_names}
    grouped_decompress_data: GroupedDecompressData = GroupedDecompressData(prefix2datalist={})
    for einsum_name, new_sims, decompress_data, job_id in parallel(
        [delayed(_per_proc_compatibility2sim)(job) for job in single_einsum_jobs],
        pbar="Generating Partial Mappings",
        return_as="generator_unordered",
    ):
        grouped_decompress_data.register_decompress_data(
            einsum_name,
            job_id,
            decompress_data,
        )
        sims[einsum_name].extend(new_sims)

    intermediate_tensors = spec.workload.intermediate_tensor_names
    for einsum_name, sims2 in sims.items():
        sims[einsum_name] = SIM.combine_combineable(sims2, live_tensors=intermediate_tensors, pbar_postfix = f" for {einsum_name}")

    return sims, grouped_decompress_data

