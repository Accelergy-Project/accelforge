from copy import deepcopy
import inspect
import os
from typing import Callable
import joblib
import logging

from fastfusion import arch
from fastfusion import Spec
from fastfusion.mapper.FFM.pmappings import MultiEinsumPmappings
from fastfusion.mapper.FFM.mappings import Mappings
import fastfusion.mapper.FFM._make_pmappings.make_pmappings as pmapper
from fastfusion.frontend.workload import EinsumName
from fastfusion.mapper.FFM._join_pmappings.join_pmappings import (
    clean_compress_and_join_pmappings,
)
from fastfusion._accelerated_imports import pd


logger = logging.getLogger(__name__)


def map_workload_to_arch(
    spec: Spec,
    einsum_names: list[EinsumName] | None = None,
    can_combine_multiple_runs: bool = False,
    cache_dir: str | None = None,
    print_number_of_pmappings: bool = True,
    _pmapping_row_filter_function: Callable[[pd.Series], bool] | None = None,
) -> Mappings:
    """
    Maps a workload to an architecture using the FastFusion FFM mapper.

    Parameters
    ----------
    spec:
        The Spec to map.
    einsum_names:
        The einsum names to map. If None, all einsums will be mapped.
    can_combine_multiple_runs: Whether we would like to be able to combine multiple
        make_pmappings runs. Having this as True allows you to do things like
        `pmappings = make_pmappings(*args_a) | make_pmappings(*args_b)` but slows
        down execution.
    cache_dir:
        The directory to cache pmappings in. If None, no caching will be done.
    print_number_of_pmappings:
        Whether to print the number of pmappings for each einsum.
    _pmapping_row_filter_function:
        A function that takes in a row of the pmapping dataframe and returns
        True if the row should be included in the final mappings, and False
        otherwise. If None, all rows will be included.
    """
    from fastfusion.model.main import evaluate_mapping

    pmappings = make_pmappings(
        spec,
        einsum_names=einsum_names,
        can_combine_multiple_runs=can_combine_multiple_runs,
        cache_dir=cache_dir,
        print_number_of_pmappings=print_number_of_pmappings,
    )
    mappings = join_pmappings(
        spec,
        pmappings,
        require_all_einsums=einsum_names is not None,
        _pmapping_row_filter_function=_pmapping_row_filter_function,
    )

    new_mapping_data = []
    for i in range(len(mappings.data)):
        local_spec = deepcopy(spec)
        local_spec.model.metrics = local_spec.mapper.ffm.info_metrics
        local_spec.mapping = mappings.data.iloc[i]["Total<SEP>mapping"]()
        # BUG: Mapping._from_pmappings create mappings that cannot be evaluated!
        this_mapping = evaluate_mapping(local_spec)
        new_mapping_data.append(this_mapping.data)

    mappings.data = pd.concat(new_mapping_data)

    return mappings


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

    Parameters
    ----------
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
    return clean_compress_and_join_pmappings(
        spec, pmappings, require_all_einsums, _pmapping_row_filter_function
    )


def _make_pmappings(
    spec: Spec,
    einsum_names: list[EinsumName] | None = None,
    can_combine_multiple_runs: bool = False,
) -> MultiEinsumPmappings:
    if einsum_names is None:
        einsum_names = [e.name for e in spec.workload.einsums]

    pmapping_groups, pmapping_objects, einsum2jobs = pmapper.make_pmappings(
        spec,
        metrics=spec.mapper.ffm.metrics,
        einsum_names=einsum_names,
        can_combine_multiple_runs=can_combine_multiple_runs,
    )

    jobs_flattened = [j for jobs in einsum2jobs.values() for j in jobs]
    resource2capacity = pmapper.get_memory_to_size(jobs_flattened)

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
