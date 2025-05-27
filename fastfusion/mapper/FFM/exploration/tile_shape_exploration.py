from numbers import Number

from combinatorics.integer import *

import numpy as np

import sympy

import pandas as pd

from fastfusion.frontend.workload.isl import get_rank_variable_bounds
from fastfusion.frontend.mapping import Temporal, Spatial, Storage
from fastfusion.frontend.specification import Specification

from fastfusion.model.looptree.reuse.summarized.symbolic import analyze_reuse
from fastfusion.model.looptree.energy import compute_energy_from_actions, gather_actions
from fastfusion.model.looptree.latency import get_latency



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
        self.indices: list[int] = []
        self.is_symbol: list[bool] = []

    def add_symbol(self, loop_idx: int):
        self.data[-1] = NumLoops(self.data[-1] + 1)
        self.indices.append(loop_idx)
        self.is_symbol.append(True)

    def add_tile_shape(self, shape, loop_idx: int):
        self.data.append(GivenShape(shape))
        self.data.append(NumLoops(1))
        self.data.append(GivenShape(shape))
        self.data.append(NumLoops(0))
        self.indices.append(loop_idx)
        self.is_symbol.append(False)

    def finish(self):
        if self.data[-1] == NumLoops(0):
            self.data.pop()
        assert self.data[-1] == GivenShape(1)

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
            memory_levels.append(node.memory)

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


def explore_tile_shapes(pmapping, constraints, specification: Specification):
    workload = specification.workload
    architecture = specification.architecture
    ert = specification.component_energy

    set_last_tile_shape_to_one(pmapping)

    tile_shapes, is_symbol = generate_tile_shapes(pmapping, workload, constraints)
    tile_shapes = tile_shapes[:, is_symbol]
    n_shapes = tile_shapes.shape[0]
    tile_shapes = [
        tile_shapes[:,i]
        for i in range(tile_shapes.shape[1])
    ]

    reuse = analyze_reuse(pmapping, workload)
    overall_latency, _, _ = get_latency(reuse, pmapping, workload, architecture)
    actions = gather_actions(reuse, pmapping, workload, None, is_path=True, use_name=True)
    # TODO: energy
    # energy = compute_energy_from_actions(actions, ert)

    df = {}
    total_occupancy = {}
    compute_unit = pmapping.nodes[-1].compute
    for buffet, stats in reuse.buffet_stats.items():
        if buffet.level == compute_unit:
            continue
        occupancy = stats.occupancy
        if isinstance(occupancy, Number):
            occupancy = np.repeat(occupancy, n_shapes)
        else:
            occupancy = sympy.lambdify(reuse.symbols, stats.occupancy)(*tile_shapes)
        if buffet.level not in total_occupancy:
            total_occupancy[buffet.level] = occupancy
        else:
            total_occupancy[buffet.level] += occupancy

    for memory, occupancy in total_occupancy.items():
        df[f'{memory}_Occupancy'] = occupancy

    if isinstance(overall_latency, Number):
        overall_latency = np.repeat(overall_latency, n_shapes)
    else:
        overall_latency = sympy.lambdify(reuse.symbols, overall_latency)(*tile_shapes)
    df['Latency'] = overall_latency

    return df



def generate_tile_shapes(pmapping, workload, constraints: list[tuple]):
    pmapping = pmapping.nodes

    shape = get_rank_variable_bounds(workload, pmapping[-1].einsum)

    rank_var_to_tiling_segments = collect_tiling_segments(pmapping, shape)

    rank_var_and_choices: list[tuple[frozenset[str], list[int], list[bool], np.array]] = []
    for rank_var, tiling_segments in rank_var_to_tiling_segments.items():
        choices = make_shapes_for_one_rank(tiling_segments)
        n_rows = choices.shape[0]
        n_loops = choices.shape[1]

        # Insert ones
        indices = tiling_segments.indices.copy()
        for other_rank, other_segments in rank_var_to_tiling_segments.items():
            if rank_var == other_rank:
                continue
            other_n_loops = len(other_segments.indices)
            indices.extend(other_segments.indices)
            choices = np.concatenate(
                (choices, np.ones((n_rows, other_n_loops), dtype=np.int32)),
                axis=1
            )
        # TODO: select out of choices to put into constraints
        # TODO: mask out bad choices

        rank_var_and_choices.append((
            frozenset(rank_var),
            tiling_segments.indices.copy(),
            tiling_segments.is_symbol.copy(),
            choices[:,:n_loops]
        ))

    while len(rank_var_and_choices) > 1:
        rank_a, index_a, is_symbol_a, choices_a = rank_var_and_choices.pop()
        rank_b, index_b, is_symbol_b, choices_b = rank_var_and_choices.pop()

        combined_choices = np.concatenate(
            (
                np.tile(choices_a, (choices_b.shape[0], 1)),
                np.repeat(choices_b, choices_a.shape[0], axis=0)
            ),
            axis=1
        )

        # TODO: insert ones and constrain

        rank_var_and_choices.append((
            rank_a | rank_b,
            index_a + index_b,
            is_symbol_a + is_symbol_b,
            combined_choices
        ))

    # TODO: insert to model
    all_rank_variables, inverted_indices, is_symbol, choices = rank_var_and_choices[0]
    is_symbol = np.asarray(is_symbol)

    # Invert indices
    indices = [0]*len(inverted_indices)
    for inverted_idx, idx in enumerate(inverted_indices):
        indices[idx] = inverted_idx

    return choices[:,indices], is_symbol[indices]


def collect_tiling_segments(
    pmapping,
    rank_shape: dict
) -> dict[str, TilingSegment]:
    rank_var_to_tiling_segments = {}
    loop_idx = 0
    for node in pmapping:
        if isinstance(node, Temporal) or isinstance(node, Spatial):
            rank_var = node.rank_variable
            tile_shape = node.tile_shape

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
        factors = np.asarray(list(factors), dtype=np.int32)[:,:-1]
        tile_shape = max_shape // np.cumprod(factors, axis=1)
        tile_shape = tile_shape.astype(np.int32)
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
    pmapping = pmapping.nodes

    rank_var_to_last_node = {}
    for node in pmapping:
        if isinstance(node, Temporal) or isinstance(node, Spatial):
            rank_var_to_last_node[node.rank_variable] = node

    for last_node in rank_var_to_last_node.values():
        last_node.tile_shape = 1