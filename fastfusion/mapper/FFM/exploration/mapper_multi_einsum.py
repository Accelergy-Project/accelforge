from math import prod
from pathlib import Path
from typing import Callable, Optional

from joblib import delayed
import pandas as pd

from fastfusion.frontend import architecture
from fastfusion.frontend.specification import Specification
from fastfusion.frontend.mapping import Mapping
from fastfusion.frontend.workload.isl import get_rank_variable_bounds
from fastfusion.frontend.workload.workload import EinsumName, TensorName

from fastfusion.mapper.FFM.exploration.metrics import Metrics
from fastfusion.mapper.FFM.exploration.mapper_one_einsum import get_single_einsum_jobs
from fastfusion.mapper.FFM.joining.sim import SIM
from fastfusion.mapper.FFM.pareto import DecompressData, GroupedDecompressData
from fastfusion.mapper.FFM.tags import Tags
from fastfusion.util.util import parallel

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

def get_sims(
    spec: Specification,
    flattened_arch: Optional[list[architecture.Leaf]] = None,
    tagger: Callable[[Mapping], Tags] | None = None,
    metrics: Metrics = Metrics.ENERGY | Metrics.LATENCY,
    einsum_names: Optional[list[EinsumName]] = None,
    except_from_imperfect: set = set(),
    df_lambda: Callable[[pd.DataFrame], pd.DataFrame] = lambda df: df
) -> tuple[dict[EinsumName, list[SIM]], DecompressData]:
    
    print(
        f'By default metrics optimizes for energy and latency.'
        f'We should change to just energy or just latency at '
        f'some point.'
    )
    
    rank_variable_bounds = get_rank_variable_bounds_for_all_einsums(spec)

    if flattened_arch is None:
        flattened_arch = spec.get_flattened_arch()
        
    sims = {}
    grouped_decompress_data: GroupedDecompressData = GroupedDecompressData(prefix2datalist={})
    tensor2compatibilties = {}
    for einsum_name in spec.workload.einsum_names:
        cur_sims: list[SIM] = []
        relevant_tensor2compatibilties = {
            t: s
            for t, s in tensor2compatibilties.items()
            if t in spec.workload.einsums[einsum_name].tensor_names
        }
        relevant_tensor2boundless_compatibilities = {
            t: set(
                c.clear_loop_bounds()
                for c in relevant_tensor2compatibilties[t]
            )
            for t in relevant_tensor2compatibilties
        }
        
        jobs = get_single_einsum_jobs(
            spec=spec,
            einsum_name=einsum_name,
            metrics=metrics,
            rank_variable_bounds=rank_variable_bounds,
            flattened_arch=flattened_arch,
            tensor2compatibilties=relevant_tensor2compatibilties,
            tensor2boundless_compatibilities=relevant_tensor2boundless_compatibilities,
            df_lambda=df_lambda,
        )
        for einsum_name, new_sims, decompress_data, job_id in parallel(
            jobs,
            pbar=f"Generating Pmappings for {einsum_name}",
            return_as="generator_unordered",
        ):
            cur_sims.extend(new_sims)
            grouped_decompress_data.register_decompress_data(
                einsum_name,
                job_id,
                decompress_data,
            )
            
        cur_tensor2compatibilties = {}
        for s in cur_sims:
            for t, c in s.compatibility.per_tensor_compatibility().items():
                cur_tensor2compatibilties.setdefault(t, set()).add(c.clear_tags())
                
        for t in cur_tensor2compatibilties:
            if t in tensor2compatibilties:
                tensor2compatibilties[t] &= cur_tensor2compatibilties[t]
            else:
                tensor2compatibilties[t] = cur_tensor2compatibilties[t]
            
        sims[einsum_name] = cur_sims
        
    intermediate_tensors = spec.workload.intermediate_tensor_names
    for einsum_name, sims2 in sims.items():
        sims[einsum_name] = SIM.combine_combineable(sims2, live_tensors=intermediate_tensors, pbar_postfix = f" for {einsum_name}")

    return sims, grouped_decompress_data
        
        
        
        
        
        
        
        
        
        
        
        

    # single_einsum_jobs = []
    # einsum_names = einsum_names or spec.workload.einsum_names
    # for einsum_name in einsum_names:
    #     single_einsum_jobs.extend(get_single_einsum_jobs(
    #         spec,
    #         einsum_name,
    #         metrics,
    #         rank_variable_bounds,
    #         flattened_arch,
    #         start_index=len(single_einsum_jobs),
    #         tagger=tagger,
    #         except_from_imperfect=except_from_imperfect,
    #     ))

    # sims = {einsum_name: [] for einsum_name in spec.workload.einsum_names}
    # grouped_decompress_data: GroupedDecompressData = GroupedDecompressData(prefix2datalist={})
    # for einsum_name, new_sims, decompress_data, job_id in parallel(
    #     single_einsum_jobs,
    #     pbar="Generating Partial Mappings",
    #     return_as="generator_unordered",
    # ):
    #     grouped_decompress_data.register_decompress_data(
    #         einsum_name,
    #         job_id,
    #         decompress_data,
    #     )
    #     sims[einsum_name].extend(new_sims)

    # intermediate_tensors = spec.workload.intermediate_tensor_names
    # for einsum_name, sims2 in sims.items():
    #     sims[einsum_name] = SIM.combine_combineable(sims2, live_tensors=intermediate_tensors, pbar_postfix = f" for {einsum_name}")

    # return sims, grouped_decompress_data

