from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from combinatorics.integer import *

import numpy as np


@dataclass
class TilingSegment:
    n_loops: int
    shape: int


def explore_tile_shapes(pmapping, shape, constraints: list[tuple]):
    pmapping = pmapping.nodes

    rank_var_to_tiling_segments = collect_tiling_segments(pmapping, shape)

    for rank_var, n_loops in rank_var_to_n_loops.items():
        choices = make_shapes_for_one_rank(shape[rank_var], n_loops)
        print(rank_var, choices)


def collect_tiling_segments(
    pmapping,
    rank_shape: dict
) -> dict[str, list[TilingSegment]]:
    rank_var_to_tiling_segments = {}
    for node in pmapping:
        if node['type'] in ['temporal', 'spatial']:
            rank_var = node['rank']
            tile_shape = node['tile_shape']

            if rank_var not in rank_var_to_tiling_segments:
                rank_var_to_tiling_segments[rank_var] = \
                    [TilingSegment(0, rank_shape[rank_var])]

            if tile_shape == 'symbolic':
                rank_var_to_tiling_segments[rank_var][-1].n_loops += 1
            elif isinstance(tile_shape, int):
                rank_var_to_tiling_segments[rank_var].append(
                    TilingSegment(0, tile_shape)
                )
            else:
                raise NotImplementedError(f'Unsupported tile shape {tile_shape}')

    for rank_var, tiling_segments in rank_var_to_tiling_segments.items():
        if tiling_segments[-1].shape != 1:
            if tiling_segments[-1].n_loops == 0:
                raise ValueError(
                    f'Last tile shape provided for rank {rank_var}, but not 1'
                )
            tiling_segments[-1].n_loops -= 1
            tiling_segments.append(TilingSegment(0, 1))

    return rank_var_to_tiling_segments


def make_shapes_for_one_rank(rank_shape, n_loops):
    if n_loops == 1:
        return np.array([[1]])
    factorizations = integer_factorizations_to_n_parts(rank_shape, n_loops-1)
    choices = np.array(list(factorizations))
    return np.concatenate([choices, [[1]*choices.shape[0]]], axis=1)