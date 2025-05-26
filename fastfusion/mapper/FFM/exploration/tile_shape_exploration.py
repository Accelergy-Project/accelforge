from combinatorics.integer import *

import numpy as np
import pandas as pd

from fastfusion.frontend.workload.isl import get_rank_variable_bounds
from fastfusion.frontend.mapping import Temporal, Spatial, Storage


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
        self.indices = list[int] = []

    def add_symbol(self, loop_idx: int):
        self.data[-1] = NumLoops(self.data[-1] + 1)
        self.indices.append(loop_idx)

    def add_tile_shape(self, shape, loop_idx: int):
        self.data.append(GivenShape(shape))
        self.data.append(NumLoops(1))
        self.data.append(GivenShape(shape))
        self.data.append(NumLoops(0))
        self.indices.append(loop_idx)

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


def dummy_tile_shape_exploration(pmapping, workload, constraints):
    N_ROWS = 1000
    n_loops = 0
    memory_levels = []
    for node in pmapping:
        if isinstance(node, Temporal) or isinstance(node, Spatial):
            n_loops += 1
        elif isinstance(node, Storage):
            memory_levels.append(node.memory.name)

    result = {
        f"tile_shape{n}": np.random.randint(1, 1024, size=N_ROWS)
        for n in range(n_loops)
    }

    result["energy"] = np.random.random(size=N_ROWS)
    result["latency"] = np.random.random(size=N_ROWS)
    result["latency"] = np.random.random(size=N_ROWS)

    for mem_level in memory_levels:
        result[f"{mem_level}_Occupancy"] = np.random.random(size=N_ROWS)

    return pd.DataFrame(result)


def explore_tile_shapes(pmapping, workload, constraints: list[tuple]):
    shape = get_rank_variable_bounds(workload, pmapping[-1].einsum)

    set_last_tile_shape_to_one(pmapping)

    rank_var_to_tiling_segments = collect_tiling_segments(pmapping, shape)

    rank_var_and_choices: list[tuple[frozenset[str], list[int], np.array]] = {}
    for rank_var, tiling_segments in rank_var_to_tiling_segments.items():
        choices = make_shapes_for_one_rank(tiling_segments)
        n_rows = choices.shape[0]
        n_loops = choices.shape[1]
        indices = tiling_segments.indices.copy()
        for other_rank, other_segments in rank_var_to_tiling_segments.items():
            if rank_var == other_rank:
                continue
            other_n_loops = len(other_segments)
            indices.extend(other_segments.indices)
            choices = np.concatenate(np.ones((n_rows, other_n_loops)),
                                     axis=1)
        # TODO: select out of choices to put into constraints
        # TODO: mask out bad choices
        rank_var_and_choices.append((
            frozenset(rank_var),
            tiling_segments.indices.copy(),
            choices[:,:n_loops]
        ))

    while len(rank_var_and_choices) > 1:
        rank_a, index_a, choices_a = rank_var_and_choices.pop()
        rank_b, index_b, choices_b = rank_var_and_choices.pop()

        combined_choices = np.concatenate(
            np.tile(choices_a, (choices_b.shape[0], 1)),
            np.repeat(choices_b, choices_a.shape[0], axis=0),
            axis=1
        )

        # TODO: insert ones and constrain

        rank_var_and_choices.append((
            rank_a | rank_b,
            index_a + index_b,
            combined_choices
        ))

    # TODO: insert to model


def collect_tiling_segments(
    pmapping,
    rank_shape: dict
) -> dict[str, TilingSegment]:
    rank_var_to_tiling_segments = {}
    loop_idx = 0
    for node in pmapping:
        if node['type'] in ['temporal', 'spatial']:
            rank_var = node['rank']
            tile_shape = node['tile_shape']

            if rank_var not in rank_var_to_tiling_segments:
                rank_var_to_tiling_segments[rank_var] = \
                    TilingSegment(rank_shape[rank_var])
            tiling_segment = rank_var_to_tiling_segments[rank_var]

            if tile_shape == 'symbol':
                tiling_segment.add_symbol(loop_idx)
            elif isinstance(tile_shape, int):
                tiling_segment.add_tile_shape(tile_shape, loop_idx)
            else:
                raise NotImplementedError(f'Unsupported tile shape {tile_shape}')

            loop_idx += 1

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