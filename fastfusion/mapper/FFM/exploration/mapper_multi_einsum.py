from pathlib import Path
import pickle
from typing import Optional
from fastfusion.frontend import architecture
from fastfusion.frontend.specification import Specification
from fastfusion.frontend.workload.workload import EinsumName, RankVariableName
from fastfusion.mapper.FFM.exploration.mapper_one_einsum import get_single_einsum_sims
from fastfusion.mapper.FFM.joining.mappinginfo import Compatibility
from fastfusion.mapper.FFM.joining.sim import SIM



def get_sims(
    spec: Specification,
    rank_variable_to_size: dict[RankVariableName, int],
    flattened_architecture: Optional[list[architecture.Leaf]] = None,
    # pkl_cache: Optional[Path] = None,
) -> dict[EinsumName, list[SIM]]:
    # if pkl_cache is not None:
    #     try:
    #         with open(pkl_cache, "rb") as f:
    #             return pickle.load(f)
    #     except FileNotFoundError:
    #         pass
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
    # if pkl_cache is not None:
    #     with open(pkl_cache, "wb") as f:
    #         pickle.dump(sims, f)
    return sims
