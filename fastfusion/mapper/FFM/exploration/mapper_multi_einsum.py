from pathlib import Path
from typing import Callable, Optional

from joblib import delayed
from fastfusion.frontend import architecture
from fastfusion.frontend.specification import Specification
from fastfusion.frontend.mapping import Mapping
from fastfusion.frontend.workload.isl import get_rank_variable_bounds
from fastfusion.frontend.workload.workload import EinsumName
from fastfusion.mapper.FFM.exploration.mapper_one_einsum import concat_sims, get_single_einsum_sims
from fastfusion.mapper.FFM.joining.sim import SIM
from fastfusion.mapper.FFM.pareto import DecompressData
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

def get_sims(
    spec: Specification,
    flattened_architecture: Optional[list[architecture.Leaf]] = None,
    parallelize_einsums = True,
    tagger: Callable[[Mapping], Tags] | None = None,
    einsum_names: Optional[list[EinsumName]] = None,
) -> tuple[dict[EinsumName, list[SIM]], DecompressData]:
    rank_variable_bounds = get_rank_variable_bounds_for_all_einsums(spec)

    if flattened_architecture is None:
        flattened_architecture = spec.get_flattened_architecture()

    if not parallelize_einsums:
        sims = {}
        for einsum_name in spec.workload.einsum_names:
            sims[einsum_name] = get_single_einsum_sims(
                spec,
                einsum_name,
                rank_variable_bounds,
                flattened_architecture,
                tagger=tagger,
            )
        return sims


    jobs = []
    einsum_names = einsum_names or spec.workload.einsum_names
    for einsum_name in einsum_names:
        jobs.extend(get_single_einsum_sims(
            spec,
            einsum_name,
            rank_variable_bounds,
            flattened_architecture,
            tagger=tagger,
            return_jobs=True,
        ))

    sims = {einsum_name: {} for einsum_name in spec.workload.einsum_names}
    id2mapping = {}
    for einsum_name, new_sims, id, mapping in parallel(
        jobs,
        pbar="Generating Partial Mappings",
        return_as="generator_unordered",
    ):
        id2mapping[id] = mapping
        target = sims[einsum_name]
        for compatibility, ns in new_sims.items():
            target.setdefault(compatibility, []).extend(ns)

    return concat_sims(sims, id2mapping)
        
