"""
Drop pmappings that are worse, in all objective metrics and reservations, than a
less-fused pmapping.
"""

from collections import defaultdict

import numpy as np

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
from accelforge.util import oset


def _tile_shape_columns_to_match(
    more_fused: Compatibility, less_fused: Compatibility, memory2index: dict[str, int]
) -> list[tuple[str, str]] | None:
    """
    Returns (more_fused, less_fused) tile shape columns that must be equal for two
    pmappings to be compared. Returns None if the pmappings can't be compared.
    """
    if more_fused.tensor_names != less_fused.tensor_names:
        return None

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
    """
    For each row of more_fused, whether some row of less_fused is <= in every column.
    Returns one bool per row of more_fused.
    """
    # Blocks bound the memory of the all-pairs comparison
    blocks = np.array_split(more_fused, range(1024, len(more_fused), 1024))
    return np.concatenate(
        [(less_fused[None] <= block[:, None]).all(-1).any(1) for block in blocks]
    )


def _worse_pmappings(
    more_fused: PmappingDataframe,
    less_fused: PmappingDataframe,
    columns_to_match: list[tuple[str, str]],
) -> np.ndarray:
    """
    For each pmapping of more_fused, whether a less_fused pmapping with matching tile
    shapes is same-or-better in every objective metric and reservation.
    """
    objectives = [c for c in more_fused.data.columns if is_objective_col(c)]
    reservations = oset(
        (key.name, key.nloops, is_left_col(c))
        for mappings in (more_fused, less_fused)
        for c in mappings.data.columns
        if (key := col2reservation(c)) is not None
    )

    def metrics(mappings: PmappingDataframe) -> np.ndarray:
        df = mappings.data
        levels = mappings._make_reservations()
        columns = [df[c].to_numpy(dtype=float) for c in objectives]
        # A missing reservation is taken from the parent level, or is 0 without one
        for name, nloops, left in reservations:
            column = get_reservation_or_parent(name, nloops, *levels, left=left)
            if column is None:
                columns.append(np.zeros(len(df)))
            else:
                columns.append(df[column].to_numpy(dtype=float))
        return np.column_stack(columns)

    def rows_by_tile_shapes(mappings: PmappingDataframe, columns) -> dict:
        rows = defaultdict(list)
        tile_shape_columns = [mappings.data[c].to_numpy() for c in columns]
        for i, tile_shapes in enumerate(zip(*tile_shape_columns)):
            rows[tile_shapes].append(i)
        return rows

    more_fused_metrics = metrics(more_fused)
    less_fused_metrics = metrics(less_fused)

    # Without tile shapes to match, every pair of pmappings is compared
    if not columns_to_match:
        return _is_worse(more_fused_metrics, less_fused_metrics)

    more_fused_columns, less_fused_columns = zip(*columns_to_match)
    more_fused_rows = rows_by_tile_shapes(more_fused, more_fused_columns)
    less_fused_rows = rows_by_tile_shapes(less_fused, less_fused_columns)
    worse = np.zeros(len(more_fused.data), dtype=bool)
    for tile_shapes, rows in more_fused_rows.items():
        if tile_shapes in less_fused_rows:
            matching = less_fused_metrics[less_fused_rows[tile_shapes]]
            worse[rows] = _is_worse(more_fused_metrics[rows], matching)
    return worse


def drop_less_fused_pmappings_that_are_worse(
    pmapping_groups: dict[str, list[PmappingGroup]],
    einsum2jobs: dict,
    print_progress: bool = True,
) -> None:
    """
    Drop each Einsum's pmappings that are worse than a less-fused pmapping. Mutates
    pmapping_groups in place.
    """
    for einsum_name, groups in pmapping_groups.items():
        # Einsums with no pmappings may have no jobs either
        if not groups:
            continue
        jobs = [j for job_list in einsum2jobs[einsum_name].values() for j in job_list]

        # Copy Einsums are not pruned
        if jobs[0].spec_one_einsum.workload.einsums[einsum_name].is_copy_operation:
            continue

        # Farther from compute = earlier in the flattened architecture
        memory2index = {}
        for job in jobs:
            for i, node in enumerate(job.flattened_arch):
                if isinstance(node, arch.Memory):
                    memory2index.setdefault(node.name, i)

        # Compare against the pmappings before any are dropped
        before_dropping = [(g.compatibility, g.mappings) for g in groups]
        n_before = sum(len(g.mappings.data) for g in groups)

        kept = []
        for g in groups:
            worse = np.zeros(len(g.mappings.data), dtype=bool)
            for less_fused, less_fused_mappings in before_dropping:
                columns_to_match = _tile_shape_columns_to_match(
                    g.compatibility, less_fused, memory2index
                )
                if columns_to_match is None:
                    continue
                worse |= _worse_pmappings(
                    g.mappings, less_fused_mappings, columns_to_match
                )
            if worse.any():
                g.mappings = g.mappings.filter_rows(lambda _: ~worse)
            if len(g.mappings.data):
                kept.append(g)
        pmapping_groups[einsum_name] = kept

        n_after = sum(len(g.mappings.data) for g in kept)
        if print_progress and n_after != n_before:
            print(
                f"Einsum {einsum_name}: dropped {n_before - n_after}/{n_before} "
                f"pmappings worse than less-fused pmappings"
            )
