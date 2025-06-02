from pathlib import Path
import pickle
from typing import Optional
from fastfusion.frontend import architecture
from fastfusion.frontend.specification import Specification
from fastfusion.frontend.workload.isl import get_rank_variable_bounds
from fastfusion.frontend.workload.workload import EinsumName, RankVariableName
from fastfusion.mapper.FFM.exploration.mapper_one_einsum import get_single_einsum_sims, add_to_compatibility2sim
from fastfusion.mapper.FFM.joining.mappinginfo import Compatibility
from fastfusion.mapper.FFM.joining.sim import SIM
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

def get_sims(
    spec: Specification,
    flattened_architecture: Optional[list[architecture.Leaf]] = None,
    parallelize_einsums = True,
) -> dict[EinsumName, list[SIM]]:
    rank_variable_bounds = get_rank_variable_bounds_for_all_einsums(spec)
    if not parallelize_einsums:
        sims = {}
        if flattened_architecture is None:
            flattened_architecture = spec.get_flattened_architecture()
        for einsum_name in spec.workload.einsum_names:
            sims[einsum_name] = get_single_einsum_sims(
                spec,
                einsum_name,
                rank_variable_bounds,
                flattened_architecture,
            )
        return sims


    sims = {einsum_name: {} for einsum_name in spec.workload.einsum_names}
    jobs = []
    for einsum_name in spec.workload.einsum_names:
        jobs.extend(get_single_einsum_sims(
            spec,
            einsum_name,
            rank_variable_bounds,
            flattened_architecture,
            return_jobs=True,
        ))
    
    for einsum_name, new_sims in parallel(
        jobs,
        pbar="Generating SIMs",
        return_as="generator"
    ):
        target = sims[einsum_name]
        for sim in new_sims:
            add_to_compatibility2sim(target, sim)
        
    return {einsum_name: list(sims.values()) for einsum_name, sims in sims.items()}
        
