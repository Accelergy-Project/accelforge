from copy import deepcopy
import inspect
import os
from typing import Callable
import joblib
import logging

from accelforge import arch
from accelforge import Spec
from accelforge.mapper.FFM.pmappings import MultiEinsumPmappings
from accelforge.mapper.FFM.mappings import Mappings
import accelforge.mapper.FFM._make_pmappings.make_pmappings as pmapper
from accelforge.frontend.workload import EinsumName
from accelforge.mapper.FFM._join_pmappings.join_pmappings import (
    clean_compress_and_join_pmappings,
)
from accelforge.mapper.FFM._join_pmappings.join_pmappings import (
    _check_einsum2pmappings_not_empty,
)
from accelforge._accelerated_imports import pd


logger = logging.getLogger(__name__)


def map_workload_to_arch(
    spec: Spec,
    einsum_names: list[EinsumName] | None = None,
    can_combine_multiple_runs: bool = False,
    cache_dir: str | None = None,
    one_pbar_only: bool = False,
    print_progress: bool = True,
    print_number_of_pmappings: bool = True,
    _pmapping_row_filter_function: Callable[[pd.Series], bool] | None = None,
) -> Mappings:
    """
    Maps a workload to an architecture using the AccelForge Fast and Fusiest Mapper
    (FFM).

    Parameters
    ----------
    spec:
        The Spec to map.
    einsum_names:
        The einsum names to map. If None, all einsums will be mapped.
    can_combine_multiple_runs:
        If True, allows combining multiple make_pmappings runs (e.g. ``pmappings =
        make_pmappings(*args_a) | make_pmappings(*args_b)``). Set to False for faster
        execution.
    cache_dir:
        The directory to cache pmappings in. If None, no caching will be done.
    one_pbar_only:
        Whether to only print only a single progress bar. If this is True, then only a
        progress bar will be created for making tile shapes, which is generally the
        longest-running part of the mapping process.
    print_progress:
        Whether to print progress of the mapping process, including progress bars.
    print_number_of_pmappings:
        Whether to print the number of pmappings for each einsum.
    _pmapping_row_filter_function:
        A function that takes in a row of the pmapping dataframe and returns True if the
        row should be included in the final mappings, and False otherwise. If None, all
        rows will be included.
    """
    from accelforge.model.main import evaluate_mapping

    if one_pbar_only:
        print_progress = False
        print_number_of_pmappings = False

    pmappings = make_pmappings(
        spec,
        einsum_names=einsum_names,
        print_progress=print_progress,
        can_combine_multiple_runs=can_combine_multiple_runs,
        cache_dir=cache_dir,
        print_number_of_pmappings=print_number_of_pmappings,
        one_pbar_only=one_pbar_only,
    )

    mappings = join_pmappings(
        pmappings,
        require_all_einsums=False,
        _pmapping_row_filter_function=_pmapping_row_filter_function,
        print_progress=print_progress,
    )

    new_mapping_data = []
    for i in range(len(mappings.data)):
        local_spec = deepcopy(spec)
        local_spec.model.metrics = local_spec.mapper.info_metrics
        local_spec.mapping = mappings.data.iloc[i]["Total<SEP>mapping"]()
        # BUG: Mapping._from_pmappings create mappings that cannot be evaluated!
        this_mapping = evaluate_mapping(
            local_spec,
            flattened_arches=mappings.flattened_arches,
            evaluated_specs=mappings.evaluated_specs,
        )
        new_mapping_data.append(this_mapping.data)

    mappings.data = pd.concat(new_mapping_data).fillna(0)

    return mappings


def make_pmappings(
    spec: Spec,
    einsum_names: list[EinsumName] | None = None,
    can_combine_multiple_runs: bool = False,
    cache_dir: str | None = None,
    print_progress: bool = True,
    print_number_of_pmappings: bool = True,
    one_pbar_only: bool = False,
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
    can_combine_multiple_runs:
        If True, allows combining multiple make_pmappings runs (e.g. ``pmappings =
        make_pmappings(*args_a) | make_pmappings(*args_b)``). Set to False for faster
        execution.
    cache_dir:
        The directory to cache pmappings in. If None, no caching will be done.
    one_pbar_only:
        Whether to only print only a single progress bar. If this is True, then only a
        progress bar will be created for making tile shapes, which is generally the
        longest-running part of the mapping process.
    print_progress:
        Whether to print progress of the mapping process, including progress bars.
    print_number_of_pmappings:
        Whether to print the number of pmappings for each einsum.

    Returns:
        A MultiEinsumPmappings object.
    """
    kwargs = dict(
        spec=spec,
        einsum_names=einsum_names,
        can_combine_multiple_runs=can_combine_multiple_runs,
        print_progress=print_progress,
        one_pbar_only=one_pbar_only,
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
    pmappings: MultiEinsumPmappings,
    require_all_einsums: bool = True,
    _pmapping_row_filter_function: Callable[[pd.Series], bool] | None = None,
    print_progress: bool = True,
) -> Mappings:
    """
    Joins pmappings into a full mappings for the entire workload. Pmappings can be
    generated using `make_pmappings`.

    Parameters
    ----------
    pmappings:
        The pmappings to join.
    require_all_einsums:
        If True, all einsums in the workload must have pmappings. If False, only einsums
        that have pmappings will be included in the final mappings.
    _pmapping_row_filter_function:
        A function that takes in a row of the pmapping dataframe and returns True if the
        row should be included in the final mappings, and False otherwise. If None, all
        rows will be included.
    print_progress:
        Whether to print progress of the mapping process, including progress bars.
    Returns
    -------
    Mappings
        A Mappings object containing all valid, optimal mappings for the workload.
    """
    return clean_compress_and_join_pmappings(
        pmappings,
        require_all_einsums,
        _pmapping_row_filter_function,
        print_progress=print_progress,
    )


def _make_pmappings(
    spec: Spec,
    einsum_names: list[EinsumName] | None = None,
    can_combine_multiple_runs: bool = False,
    print_progress: bool = True,
    one_pbar_only: bool = False,
) -> MultiEinsumPmappings:
    if einsum_names is None:
        einsum_names = [e.name for e in spec.workload.einsums]

    pmapping_groups, pmapping_objects, einsum2jobs = pmapper.make_pmappings(
        spec,
        metrics=spec.mapper.metrics,
        einsum_names=einsum_names,
        can_combine_multiple_runs=can_combine_multiple_runs,
        print_progress=print_progress,
        one_pbar_only=one_pbar_only,
    )

    flattened_arches = {}
    evaluated_specs = {}
    for einsum_name, jobs in einsum2jobs.items():
        for job in jobs:
            compute_name = job.flattened_arch[-1].name
            flattened_arches[(einsum_name, compute_name)] = job.flattened_arch
            evaluated_specs[einsum_name] = job.spec

    m = MultiEinsumPmappings(
        spec,
        pmapping_groups,
        pmapping_objects,
        einsum2jobs,
        can_combine_multiple_runs=can_combine_multiple_runs,
        einsums_with_pmappings_generated=set(
            einsum_names if einsum_names else spec.workload.einsum_names
        ),
        flattened_arches=flattened_arches,
        evaluated_specs=evaluated_specs,
    )

    return m
