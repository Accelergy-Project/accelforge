"""
Drop pmappings that are worse, in all objective metrics and reservations, than a
less-fused pmapping.
"""

from collections import defaultdict

import numpy as np
import pandas as pd

from accelforge.frontend import arch
from accelforge.mapper.FFM._join_pmappings.compatibility import Compatibility
from accelforge.mapper.FFM._join_pmappings.pmapping_dataframe import (
    PmappingDataframe,
    get_reservation_or_parent,
)
from accelforge.mapper.FFM._join_pmappings.pmapping_group import PmappingGroup
from accelforge.mapper.FFM._pareto_df.df_convention import (
    col2reservation,
    is_left_col,
    is_objective_col,
)
from accelforge.util import fzs, oset


def _tile_shape_columns_to_match_and_can_compare(
    more_fused: Compatibility, less_fused: Compatibility, memory2index: dict[str, int]
) -> list[tuple[str, str]] | None:
    """
    Returns (more_fused, less_fused) tile shape columns that must be equal for two
    pmappings to be compared. Returns None if the pmappings can't be compared.
    """
    columns = []
    any_further = False
    n_compared_loops = 0
    for t in more_fused.tensors:
        l = less_fused.get_reservation_of_tensor(t.name)

        # A different backing storage must be farther from compute
        if t.resource_name != l.resource_name:
            if memory2index[l.resource_name] >= memory2index[t.resource_name]:
                return None
            any_further = True
            continue

        # The same backing storage must have the same loops above it
        t_loop_ranks = [(x.rank_name, x.is_spatial, x.spatial_dim) for x in t.loops]
        l_loop_ranks = [(x.rank_name, x.is_spatial, x.spatial_dim) for x in l.loops]
        if t.persistent != l.persistent or t_loop_ranks != l_loop_ranks:
            return None

        # A rank without a tile pattern has None here in every pmapping
        for t_loop, l_loop in zip(t.loops, l.loops):
            for attr in t_loop.tile_pattern._symbol_attrs():
                t_column = getattr(t_loop.tile_pattern, attr)
                l_column = getattr(l_loop.tile_pattern, attr)
                if t_column is not None:
                    columns.append((t_column, l_column))
        n_compared_loops = max(n_compared_loops, len(t.loops))

    if not any_further:
        return None

    # Reservation indices within the compared loops must match
    compared_loops = set(range(n_compared_loops + 1))
    more_fused_indices = more_fused.reservation_indices & compared_loops
    less_fused_indices = less_fused.reservation_indices & compared_loops
    if more_fused_indices != less_fused_indices:
        return

    return columns


def _is_worse(more_fused: np.ndarray, less_fused: np.ndarray) -> np.ndarray:
    """VIBE CODE WARNING"""
    worse = np.zeros(len(more_fused), dtype=bool)
    if not len(more_fused) or not len(less_fused):
        return worse
    # No row can be worse if some column of less_fused is above all of more_fused
    if (less_fused.min(axis=0) > more_fused.max(axis=0)).any():
        return worse
    # Sorted by the first column, only a prefix of less_fused can be <= a row
    less_fused = less_fused[np.argsort(less_fused[:, 0], kind="stable")]
    order = np.argsort(more_fused[:, 0], kind="stable")
    n_candidates = np.searchsorted(less_fused[:, 0], more_fused[order, 0], side="right")
    # Blocks bound the memory of the all-pairs comparison
    for start in range(0, len(order), 256):
        rows = order[start : start + 256]
        candidates = less_fused[: n_candidates[start : start + 256].max()]
        if not len(candidates):
            continue
        ok = np.ones((len(rows), len(candidates)), dtype=bool)
        for k in range(more_fused.shape[1]):
            ok &= candidates[:, k][None, :] <= more_fused[rows, k][:, None]
        worse[rows] = ok.any(axis=1)
    return worse


def _metrics(
    mappings: PmappingDataframe, objectives: list[str], reservations: oset
) -> np.ndarray:
    df = mappings.data
    levels = mappings._make_reservations()
    columns = [df[c].to_numpy(dtype=float) for c in objectives]
    for name, nloops, left in reservations:
        column = get_reservation_or_parent(name, nloops, *levels, left=left)
        if column is None:
            columns.append(np.zeros(len(df)))
        else:
            columns.append(df[column].to_numpy(dtype=float))
    return np.column_stack(columns)


def _tile_shapes_to_rows(mappings: PmappingDataframe, columns: tuple[str, ...]) -> dict:
    df = mappings.data
    keys = pd.DataFrame({i: df[c].to_numpy() for i, c in enumerate(columns)})
    return keys.groupby(list(range(len(columns))), sort=False).indices


def _worse_pmappings(
    more_fused_metrics: np.ndarray,
    more_fused_rows: dict,
    less_fused_metrics: np.ndarray,
    less_fused_rows: dict,
) -> np.ndarray:
    """Returns a mask for dominated pmappings"""
    worse = np.zeros(len(more_fused_metrics), dtype=bool)
    for tile_shapes, rows in more_fused_rows.items():
        if tile_shapes in less_fused_rows:
            matching = less_fused_metrics[less_fused_rows[tile_shapes]]
            worse[rows] = _is_worse(more_fused_metrics[rows], matching)
    return worse


def drop_more_fused_pmappings_that_are_worse(
    pmapping_groups: dict[str, list[PmappingGroup]],
    einsum2jobs: dict,
    print_progress: bool = True,
    _einsum_name: str | None = None,
) -> None:
    """
    Drop each Einsum's pmappings that are worse than a less-fused pmapping. Mutates
    pmapping_groups in place.
    """
    if isinstance(pmapping_groups, dict):
        for k, v in pmapping_groups.items():
            drop_more_fused_pmappings_that_are_worse(v, einsum2jobs, print_progress, k)
        return

    if not pmapping_groups:
        return

    group = pmapping_groups
    jobs = [j for job_list in einsum2jobs[_einsum_name].values() for j in job_list]

    # Copy Einsums are not pruned
    if jobs[0].spec_one_einsum.workload.einsums[_einsum_name].is_copy_operation:
        return

    # Farther from compute = earlier in the flattened architecture
    memory2index = {}
    for job in jobs:
        for i, node in enumerate(job.flattened_arch):
            if isinstance(node, arch.Memory):
                memory2index.setdefault(node.name, i)

    n_before = sum(len(g.mappings.data) for g in group)
    assert len(set((fzs(g.compatibility.tensor_names) for g in group))) == 1

    kept = []
    objectives = [c for c in group[0].mappings.data.columns if is_objective_col(c)]
    reservations = oset(
        (key.name, key.nloops, is_left_col(c))
        for g in group
        for c in g.mappings.data.columns
        if (key := col2reservation(c)) is not None
    )
    compatibility_and_metrics = [
        (g.compatibility, _metrics(g.mappings, objectives, reservations)) for g in group
    ]
    tile_shape_to_rows = {}

    def rows(i: int, columns: tuple[str, ...]) -> dict:
        if (i, columns) not in tile_shape_to_rows:
            tile_shape_to_rows[i, columns] = _tile_shapes_to_rows(
                group[i].mappings, columns
            )
        return tile_shape_to_rows[i, columns]

    for i, g in enumerate(group):
        worse = np.zeros(len(g.mappings.data), dtype=bool)
        for j, (less_fused, less_fused_metrics) in enumerate(compatibility_and_metrics):
            columns_to_match = _tile_shape_columns_to_match_and_can_compare(
                g.compatibility, less_fused, memory2index
            )
            if columns_to_match is None:
                continue
            if not columns_to_match:
                worse |= _is_worse(compatibility_and_metrics[i][1], less_fused_metrics)
                continue
            more_fused_columns, less_fused_columns = zip(*columns_to_match)
            worse |= _worse_pmappings(
                compatibility_and_metrics[i][1],
                rows(i, more_fused_columns),
                less_fused_metrics,
                rows(j, less_fused_columns),
            )
        if worse.any():
            g.mappings = g.mappings.filter_rows(lambda _: ~worse)
        if len(g.mappings.data):
            kept.append(g)

    pmapping_groups[:] = kept
    n_after = sum(len(g.mappings.data) for g in kept)
    if print_progress and n_after != n_before:
        print(
            f"Einsum {_einsum_name}: dropped {n_before - n_after}/{n_before} "
            f"pmappings worse than less-fused pmappings"
        )
