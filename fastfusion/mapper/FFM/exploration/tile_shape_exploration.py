from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from combinatorics.integer import *

import numpy as np


class GivenShape(int):
    def __str__(self):
        return f'GivenShape({super().__repr__()})'

    def __repr__(self):
        return f'GivenShape({super().__repr__()})'


class NumLoops(int):
    def __str__(self):
        return f'NumLoops({super().__repr__()})'

    def __repr__(self):
        return f'NumLoops({super().__repr__()})'


class TilingSegment:
    def __init__(self, full_shape):
        self.data: list[GivenShape | NumLoops] = [GivenShape(full_shape), NumLoops(0)]

    def add_symbol(self):
        self.data[-1] = NumLoops(self.data[-1] + 1)

    def add_tile_shape(self, shape):
        self.data.append(GivenShape(shape))
        self.data.append(NumLoops(1))
        self.data.append(GivenShape(shape))
        self.data.append(NumLoops(0))

    def finish(self):
        if self.data[-1] == NumLoops(0):
            self.data.pop()
        else:
            self.data.append(GivenShape(1))

        assert self.data[-2] > 0
        self.data[-2] = NumLoops(self.data[-2] - 1)
        self.data.append(NumLoops(1))
        self.data.append(GivenShape(1))

    def iterate_segments(self):
        """Returns iterator over tuples (n_loops, max_shape, min_shape)."""
        for i in range(0, len(self.data)-1, 2):
            if self.data[i+1] == NumLoops(0):
                continue
            max_shape = self.data[i]
            n_loops = self.data[i+1]
            min_shape = self.data[i+2]
            yield (n_loops, max_shape, min_shape)



def explore_tile_shapes(pmapping, shape, constraints: list[tuple]):
    pmapping = pmapping.nodes

    set_last_tile_shape_to_one(pmapping)

    rank_var_to_tiling_segments = collect_tiling_segments(pmapping, shape)

    rank_var_to_choices = {}
    for rank_var, tiling_segments in rank_var_to_tiling_segments.items():
        choices = make_shapes_for_one_rank(tiling_segments)


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
                    TilingSegment(rank_shape[rank_var])
            tiling_segment = rank_var_to_tiling_segments[rank_var]

            if tile_shape == 'symbol':
                tiling_segment.add_symbol()
            elif isinstance(tile_shape, int):
                tiling_segment.add_tile_shape(tile_shape)
            else:
                raise NotImplementedError(f'Unsupported tile shape {tile_shape}')

    for rank_var, tiling_segment in rank_var_to_tiling_segments.items():
        tiling_segment.finish()

    return rank_var_to_tiling_segments


def make_shapes_for_one_rank(tiling_segments):
    all_tile_shapes = None
    total_loops = 0
    for n_loops, max_shape, min_shape in tiling_segments.iterate_segments():
        total_loops += n_loops

        factors = integer_factorizations_to_n_parts(max_shape, n_loops+1)
        factors = np.asarray(list(factors))[:,:-1]
        tile_shape = max_shape // np.cumprod(factors, axis=1)
        tile_shape = tile_shape[np.all(tile_shape >= min_shape, axis=1), :]

        if all_tile_shapes is None:
            all_tile_shapes = tile_shape
        else:
            all_tile_shapes_n_rows = all_tile_shapes.shape[0]
            all_tile_shapes = np.tile(all_tile_shapes, (tile_shape.shape[0], 1))
            tile_shape = np.repeat(tile_shape, repeats=all_tile_shapes_n_rows, axis=0)
            all_tile_shapes = np.concatenate((all_tile_shapes, tile_shape), axis=1)
    return all_tile_shapes


def set_last_tile_shape_to_one(pmapping):
    rank_var_to_last_node = {}
    for node in pmapping:
        if node['type'] in ['temporal', 'spatial']:
            rank_var_to_last_node[node['rank']] = node

    for last_node in rank_var_to_last_node.values():
        last_node['tile_shape'] = 1