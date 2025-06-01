from pathlib import Path
import pickle
from typing import Optional
from fastfusion.frontend import architecture
from fastfusion.frontend.specification import Specification
from fastfusion.frontend.workload.workload import EinsumName, RankVariableName
from fastfusion.mapper.FFM.exploration.mapper_one_einsum import get_single_einsum_sims, add_to_compatibility2sim
from fastfusion.mapper.FFM.joining.mappinginfo import Compatibility
from fastfusion.mapper.FFM.joining.sim import SIM
from fastfusion.util.util import parallel



def get_sims(
    spec: Specification,
    rank_variable_to_size: dict[RankVariableName, int],
    flattened_architecture: Optional[list[architecture.Leaf]] = None,
    parallelize_einsums = True,
) -> dict[EinsumName, list[SIM]]:
    if not parallelize_einsums:
        sims = {}
        if flattened_architecture is None:
            flattened_architecture = spec.get_flattened_architecture()
        for einsum_name in spec.workload.einsum_names:
            sims[einsum_name] = get_single_einsum_sims(
                spec,
                einsum_name,
                rank_variable_to_size,
                flattened_architecture,
            )
        return sims


    sims = {einsum_name: {} for einsum_name in spec.workload.einsum_names}
    jobs = []
    for einsum_name in spec.workload.einsum_names:
        if einsum_name != "K":
            continue
        jobs.extend(get_single_einsum_sims(
            spec,
            einsum_name,
            rank_variable_to_size,
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
        
