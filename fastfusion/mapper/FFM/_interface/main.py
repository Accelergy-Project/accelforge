import inspect
import os
from typing import Callable
from fastfusion import arch
from fastfusion import Specification
from fastfusion.mapper.FFM._interface.pmappings import MultiEinsumPmappings
from fastfusion.mapper.FFM._interface.mappings import Mappings
from fastfusion.mapper.FFM._join_pmappings.compress_pmappings import (
    compress_einsum2pmappings,
    decompress_pmappings,
)
from fastfusion.mapper.FFM._make_pmappings.mapper_multi_einsum import get_sims
from fastfusion.frontend.workload import EinsumName
from fastfusion.frontend.mapping import Mapping
from fastfusion.mapper.FFM._join_pmappings.join_pmappings import join_sims
from fastfusion.mapper.FFM._pmapping_group.df_convention import MAPPING_COLUMN
from fastfusion.mapper.FFM._pmapping_group.pmapping_group import (
    PmappingGroup,
    row2pmappings,
)
from fastfusion.mapper.FFM._make_pmappings.mapper_multi_einsum import (
    get_rank_variable_bounds_for_all_einsums,
)
from fastfusion.accelerated_imports import pd
import joblib


class MappingFromRow:
    def __init__(
        self,
        row: pd.Series,
        spec: Specification,
        rank_variable_bounds: dict[str, dict[str, int]],
    ):
        self.row = row
        self.spec = spec
        self.rank_variable_bounds = rank_variable_bounds

    def __call__(self) -> Mapping:
        return row2mapping(self.row, self.spec, self.rank_variable_bounds)

    def render(self) -> str:
        return self().render()

def _make_pmappings(
    spec: Specification,
    einsum_names: list[EinsumName] | None = None,
    can_combine_multiple_runs: bool = False,
) -> MultiEinsumPmappings:
    parsed_spec, _ = spec._parse_expressions()
    parsed_spec.calculate_component_energy_area(area=False)
    flattened_arches = parsed_spec.get_flattened_architecture()
    sims, pmapping_objects, einsum2jobs = get_sims(
        parsed_spec,
        flattened_arches,
        metrics=spec.mapper.ffm.metrics,
        einsum_names=einsum_names,
        can_combine_multiple_runs=can_combine_multiple_runs,
    )
    resource2capacity = {}
    for flattened_arch in flattened_arches:
        for l in flattened_arch:
            if isinstance(l, arch.Memory):
                resource2capacity[l.name] = l.attributes.size
    return MultiEinsumPmappings(
        sims,
        pmapping_objects,
        resource2capacity,
        einsum2jobs,
        can_combine_multiple_runs=can_combine_multiple_runs,
    )


def make_pmappings(
    spec: Specification,
    einsum_names: list[EinsumName] | None = None,
    can_combine_multiple_runs: bool = False,
    cache_dir: str | None = None,
) -> MultiEinsumPmappings:
    """
    Creates pmappings for a specification. Pmappings must be joined together using
    `join_pmappings` to create a full mapping.

    Args:
        spec: The Specification to generate pmappings for.
        einsum_names: The einsum names to generate pmappings for. If None, all einsums will be included.
        can_combine_multiple_runs: Whether we would like to be able to combine multiple
        make_pmappings runs. Haivng this as True allows you to do things like
            `pmappings = make_pmappings(*args_a) | make_pmappings(*args_b)`
        but slows down execution.
        cache_dir: The directory to cache pmappings in. If None, no caching will be done.

    Returns:
        A MultiEinsumPmappings object.
    """

    kwargs = dict(
        spec=spec,
        einsum_names=einsum_names,
        can_combine_multiple_runs=can_combine_multiple_runs,
    )
    assert len(kwargs) == len(inspect.signature(_make_pmappings).parameters)

    if cache_dir is None:
        return _make_pmappings(**kwargs)

    @joblib.Memory(location=os.path.join(cache_dir), compress=True).cache
    def _make_pmappings_cached(**kwargs) -> MultiEinsumPmappings:
        return _make_pmappings(**kwargs)

    return _make_pmappings_cached(**kwargs)



def row2mapping(
    row: pd.Series, spec: Specification, rank_variable_bounds: dict[str, dict[str, int]]
) -> Mapping:
    return Mapping.from_pmappings(
        row2pmappings(row, spec.workload.einsum_names, rank_variable_bounds),
        rank_variable_bounds=rank_variable_bounds,
    )


def join_pmappings(
    spec: Specification,
    pmappings: MultiEinsumPmappings, 
    pmapping_row_filter_function: Callable[[pd.Series], bool] | None = None
) -> Mappings:
    for einsum_name, einsum_pmappings in pmappings.einsum2pmappings.items():
        total = sum(len(p.mappings.data) for p in einsum_pmappings)
        n_compatibilities = len(einsum_pmappings)
        print(
            f"Einsum {einsum_name} has {total} pmappings with {n_compatibilities} compatibilities"
        )
        if total == 0:
            raise ValueError(f"Einsum {einsum_name} has no pmappings")

    compressed, decompress_data = compress_einsum2pmappings(pmappings.einsum2pmappings)
    joined = join_sims(
        compressed,
        spec,
        pmappings.resource2capacity,
        pmapping_row_filter_function=pmapping_row_filter_function,
    )
    joined = decompress_pmappings(joined, decompress_data)

    for einsum_name in pmappings.einsum2pmappings:
        col = f"{einsum_name}<SEP>{MAPPING_COLUMN}"
        joined.data[col] = joined.data[col].apply(
            lambda x: pmappings.pmapping_objects[einsum_name][x]
        )
    joined._data = joined.data.fillna(0).reset_index(drop=True)

    rank_variable_bounds = get_rank_variable_bounds_for_all_einsums(spec)
    joined.data[f"Total<SEP>{MAPPING_COLUMN}"] = [
        MappingFromRow(r, spec, rank_variable_bounds) for _, r in joined.data.iterrows()
    ]
    # Fill nans with 0. We might get missing columns for some mapping entries if there
    # are energy entries for some pmappings but not others (e.g., one pmapping accesses
    # DRAM while another doesn't.)
    return Mappings(spec, list(pmappings.einsum2pmappings.keys()), joined.data)
