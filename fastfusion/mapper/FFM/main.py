import inspect
import os
from typing import Callable
from fastfusion import arch
from fastfusion import Spec
from fastfusion.mapper.FFM.pmappings import MultiEinsumPmappings
from fastfusion.mapper.FFM.mappings import Mappings
from fastfusion.mapper.FFM._join_pmappings.compress_pmappings import (
    compress_einsum2pmappings,
    decompress_pmappings,
)
import fastfusion.mapper.FFM._make_pmappings.make_pmappings as pmapper
from fastfusion.frontend.workload import EinsumName
from fastfusion.frontend.mapping import Mapping
from fastfusion.mapper.FFM._join_pmappings.join_pmappings import (
    join_pmappings as _join_pmappings,
)
from fastfusion.mapper.FFM._pareto_df.df_convention import MAPPING_COLUMN
from fastfusion.mapper.FFM._join_pmappings.pmapping_dataframe import (
    PmappingDataframe,
    row2pmappings,
)
from fastfusion.mapper.FFM._make_pmappings.make_pmappings import (
    get_rank_variable_bounds_for_all_einsums,
)
from fastfusion._accelerated_imports import pd
import joblib
import logging


class MappingFromRow:
    def __init__(
        self,
        row: pd.Series,
        rank_variable_bounds: dict[str, dict[str, int]],
        einsum_names: list[EinsumName] | None = None,
    ):
        self.row = row
        self.rank_variable_bounds = rank_variable_bounds
        self.einsum_names = einsum_names

    def __call__(self) -> Mapping:
        return row2mapping(self.row, self.rank_variable_bounds, self.einsum_names)

    def _repr_svg_(self) -> str:
        return self.render()

    def render(self) -> str:
        return self().render()


def _make_pmappings(
    spec: Spec,
    einsum_names: list[EinsumName] | None = None,
    can_combine_multiple_runs: bool = False,
) -> MultiEinsumPmappings:
    parsed_spec = spec.calculate_component_energy_area(area=False)
    flattened_arches = parsed_spec.get_flattened_architecture()
    for i, flattened_arch in enumerate(flattened_arches):
        logging.info(f'Flattened arch {i} uses compute {flattened_arch[-1].name}')
    pmapping_groups, pmapping_objects, einsum2jobs = pmapper.make_pmappings(
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

    m = MultiEinsumPmappings(
        pmapping_groups,
        pmapping_objects,
        resource2capacity,
        einsum2jobs,
        can_combine_multiple_runs=can_combine_multiple_runs,
        einsums_with_pmappings_generated=set(
            einsum_names if einsum_names else spec.workload.einsum_names
        ),
    )

    return m


def make_pmappings(
    spec: Spec,
    einsum_names: list[EinsumName] | None = None,
    can_combine_multiple_runs: bool = False,
    cache_dir: str | None = None,
    print_number_of_pmappings: bool = True,
) -> MultiEinsumPmappings:
    """
    Creates pmappings for a spec. Pmappings must be joined together using
    `join_pmappings` to create a full mapping.

    Args:
        spec:
            The Spec to generate pmappings for.
        einsum_names:
            The einsum names to generate pmappings for. If None, all einsums will be
            included.
        can_combine_multiple_runs: Whether we would like to be able to combine multiple
            make_pmappings runs. Having this as True allows you to do things like
            `pmappings = make_pmappings(*args_a) | make_pmappings(*args_b)` but slows
            down execution.
        cache_dir:
            The directory to cache pmappings in. If None, no caching will be done.
        print_number_of_pmappings:
            Whether to print the number of pmappings for each einsum.

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
        result = _make_pmappings(**kwargs)
    else:

        @joblib.Memory(location=os.path.join(cache_dir), compress=True).cache
        def _make_pmappings_cached(**kwargs) -> MultiEinsumPmappings:
            return _make_pmappings(**kwargs)

        result = _make_pmappings_cached(**kwargs)

    if print_number_of_pmappings:
        print(result.n_pmapping_string())

    return result


def row2mapping(
    row: pd.Series,
    rank_variable_bounds: dict[str, dict[str, int]],
    einsum_names: list[EinsumName],
) -> Mapping:
    return Mapping._from_pmappings(
        row2pmappings(row, einsum_names, rank_variable_bounds),
        rank_variable_bounds=rank_variable_bounds,
    )


def join_pmappings(
    spec: Spec,
    pmappings: MultiEinsumPmappings,
    require_all_einsums: bool = True,
    _pmapping_row_filter_function: Callable[[pd.Series], bool] | None = None,
) -> Mappings:
    """
    Joins pmappings into a full mappings for the entire workload. Pmappings can
    be generated using `make_pmappings`.

    Parameters
    ----------
    spec:
        The complete specifications for the workload and architecture.
    pmappings:
        The pmappings to join.
    require_all_einsums:
        If True, all einsums in the workload must have pmappings. If False, only
        einsums that have pmappings will be included in the final mappings.
    _pmapping_row_filter_function:
        A function that takes in a row of the pmapping dataframe and returns
        True if the row should be included in the final mappings, and False
        otherwise. If None, all rows will be included.

    Returns
    -------
    A Mappings object containing all valid, optimal mappings for the workload.
    """
    einsum2pmappings = pmappings.einsum2pmappings
    if not require_all_einsums:
        einsum2pmappings = {
            k: v
            for k, v in pmappings.einsum2pmappings.items()
            if k in pmappings.einsums_with_pmappings_generated
        }

    for einsum_name, einsum_pmappings in einsum2pmappings.items():
        total = sum(len(p.mappings.data) for p in einsum_pmappings)
        n_compatibilities = len(einsum_pmappings)
        print(
            f"Einsum {einsum_name} has {total} pmappings with {n_compatibilities} compatibilities"
        )
        if total == 0:
            if einsum_name in pmappings.einsums_with_pmappings_generated:
                raise ValueError(
                    f"Einsum {einsum_name} has no pmappings. This likely means that "
                    f"no pmappings satisfied constraints for the Einsum. Please check "
                    f"the stats outputs from the MultiEinsumPmappings object."
                )

            raise ValueError(
                f"Einsum {einsum_name} has no pmappings generated. It looks like you "
                "may have used `make_pmappings` with `einsum_names` set. You may set "
                "`require_all_einsums=False` to ignore this error and map only the "
                "Einsums that have pmappings."
            )

    compressed, decompress_data = compress_einsum2pmappings(einsum2pmappings)
    joined = _join_pmappings(
        compressed,
        spec,
        pmappings.resource2capacity,
        _pmapping_row_filter_function=_pmapping_row_filter_function,
    )
    joined = decompress_pmappings(joined, decompress_data)

    for einsum_name in einsum2pmappings:
        col = f"{einsum_name}<SEP>{MAPPING_COLUMN}"
        joined.data[col] = joined.data[col].apply(
            lambda x: pmappings.pmapping_objects[einsum_name][x]
        )
    joined._data = joined.data.fillna(0).reset_index(drop=True)

    rank_variable_bounds = get_rank_variable_bounds_for_all_einsums(spec)
    einsum_names = list(einsum2pmappings.keys())
    joined.data[f"Total<SEP>{MAPPING_COLUMN}"] = [
        MappingFromRow(r, rank_variable_bounds, einsum_names)
        for _, r in joined.data.iterrows()
    ]
    # Fill nans with 0. We might get missing columns for some mapping entries if there
    # are energy entries for some pmappings but not others (e.g., one pmapping accesses
    # DRAM while another doesn't.)
    return Mappings(
        spec,
        list(einsum2pmappings.keys()),
        joined.data,
        total_mappings=joined.n_total_pmappings,
        valid_mappings=joined.n_valid_pmappings,
    )
