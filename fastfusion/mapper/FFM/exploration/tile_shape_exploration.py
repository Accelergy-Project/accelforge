from numbers import Number

from combinatorics.integer import *

import numpy as np

import sympy

import pandas as pd

import fastfusion.frontend.architecture as architecture
from fastfusion.frontend.architecture import Memory
from fastfusion.frontend.workload.isl import get_rank_variable_bounds
from fastfusion.frontend.mapping import Temporal, Spatial, Storage
from fastfusion.frontend.specification import Specification

from fastfusion.model.looptree.reuse.summarized.symbolic import analyze_reuse
from fastfusion.model.looptree.energy import compute_energy_from_actions, gather_actions
from fastfusion.model.looptree.latency import get_latency

from fastfusion.mapper.FFM.pareto import nameloop2col, col2nameloop



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


def explore_tile_shapes(pmapping, constraints, specification: Specification, flattened_arch):
    set_last_tile_shape_to_one(pmapping)

    symbols, symbolic_df, per_memory_occupancy_df, utilization_df = run_model(pmapping, specification, flattened_arch)
    compiled_df = compile_dict(symbols, symbolic_df)
    compiled_per_memory_occupancy_df = compile_dict(symbols, per_memory_occupancy_df)
    compiled_utilization_df = compile_dict(symbols, utilization_df)

    tile_shapes, is_symbol = generate_tile_shapes(pmapping, constraints, compiled_per_memory_occupancy_df, compiled_utilization_df, specification)

    df = {}
    for i in range(tile_shapes.shape[1]):
        df[f'__tile_shape{i}'] = tile_shapes[:,i]

    tile_shapes = tile_shapes[:, is_symbol]
    tile_shapes = [
        tile_shapes[:,i]
        for i in range(tile_shapes.shape[1])
    ]

    for key in compiled_df:
        df[key] = compiled_df[key](*tile_shapes)

    rval = pd.DataFrame(df)
    print(f'Returning df: {rval.shape}')
    return rval

def generate_tile_shapes(pmapping, constraints, usage_df, utilization_df, specification):
    pmapping = pmapping.nodes
    workload = specification.workload

    shape = get_rank_variable_bounds(workload, pmapping[-1].einsum)

    rank_var_to_tiling_segments = collect_tiling_segments(pmapping, shape)

    rank_var_and_choices: list[tuple[frozenset[str], list[int], list[bool], np.array]] = []
    for rank_var, tiling_segments in rank_var_to_tiling_segments.items():
        choices = make_shapes_for_one_rank(tiling_segments)
        n_rows = choices.shape[0]
        n_loops = choices.shape[1]

        # Insert ones
        indices = tiling_segments.indices.copy()
        is_symbols = tiling_segments.is_symbol.copy()
        for other_rank, other_segments in rank_var_to_tiling_segments.items():
            if rank_var == other_rank:
                continue
            other_n_loops = len(other_segments.indices)
            indices.extend(other_segments.indices)
            is_symbols.extend(other_segments.is_symbol)
            choices = np.concatenate(
                (choices, np.ones((n_rows, other_n_loops), dtype=np.int64)),
                axis=1
            )

        # TODO: there may be a more efficient order
        corrected_indices = np.asarray(invert_indices(indices))
        corrected_choices = choices[:,corrected_indices]
        is_symbols = np.asarray(is_symbols)[corrected_indices]
        corrected_choices = corrected_choices[:,is_symbols]

        # Check if capacity is overused
        mask = np.ones(corrected_choices.shape[0], dtype=np.bool)
        for memory, usage_model in usage_df.items():
            usage = usage_model(*[
                corrected_choices[:,i] for i in range(corrected_choices.shape[1])
            ])
            mask = mask & (usage <= 1.0)

        utilization = {}
        for (component, dim), utilization_model in utilization_df.items():
            utilization[(component, dim)] = utilization_model(*[
                corrected_choices[:,i] for i in range(corrected_choices.shape[1])
            ])
            mask = mask & (utilization[(component, dim)] <= 1.0)

        good_choices = choices[mask,:]

        if good_choices.shape[0] == 0:
            return good_choices, is_symbols

        rank_var_and_choices.append((
            frozenset(rank_var),
            tiling_segments.indices.copy(),
            tiling_segments.is_symbol.copy(),
            good_choices[:,:n_loops]
        ))

    while len(rank_var_and_choices) > 1:
        rank_a, index_a, is_symbol_a, choices_a = rank_var_and_choices.pop()
        rank_b, index_b, is_symbol_b, choices_b = rank_var_and_choices.pop()

        # print(f"With {len(rank_var_and_choices)} choices left, combining lengths {choices_a.shape[0]} and {choices_b.shape[0]} -> {choices_a.shape[0] * choices_b.shape[0]}")
        combined_choices = np.concatenate(
            (
                np.tile(choices_a, (choices_b.shape[0], 1)),
                np.repeat(choices_b, choices_a.shape[0], axis=0)
            ),
            axis=1
        )
        n_rows = combined_choices.shape[0]
        n_loops = combined_choices.shape[1]

        is_symbols = is_symbol_a + is_symbol_b
        indices = index_a + index_b

        # Insert ones
        combined_choices_with_ones = combined_choices
        for other_ranks, other_indices, other_is_symbol, other_choices in rank_var_and_choices:
            indices.extend(other_indices)
            is_symbols.extend(other_is_symbol)

            other_n_loops = len(other_indices)
            combined_choices_with_ones = np.concatenate(
                (
                    combined_choices_with_ones,
                    np.ones((n_rows, other_n_loops), dtype=np.int64)
                ),
                axis=1
            )

        # TODO: there may be a more efficient order
        corrected_indices = np.asarray(invert_indices(indices))
        corrected_choices = combined_choices_with_ones[:,corrected_indices]
        is_symbols = np.asarray(is_symbols)[corrected_indices]
        corrected_choices = corrected_choices[:,is_symbols]

        # Check if capacity is overused
        mask = np.ones(corrected_choices.shape[0], dtype=np.bool)
        for memory, usage_model in usage_df.items():
            usage = usage_model(*[
                corrected_choices[:,i] for i in range(corrected_choices.shape[1])
            ])
            mask = mask & (usage <= 1.0)

        # Compute utilization
        utilization = {}
        for (component, dim), utilization_model in utilization_df.items():
            utilization[(component, dim)] = utilization_model(*[
                corrected_choices[:,i] for i in range(corrected_choices.shape[1])
            ])

        # Insert largest value
        combined_choices_with_largest = combined_choices
        for other_ranks, other_indices, other_is_symbol, other_choices in rank_var_and_choices:
            largest_other_choices = np.max(other_choices, axis=0, keepdims=True)
            combined_choices_with_largest = np.concatenate(
                (
                    combined_choices_with_largest,
                    np.repeat(largest_other_choices, n_rows, axis=0)
                ),
                axis=1
            )
        corrected_choices = combined_choices_with_largest[:,corrected_indices]
        corrected_choices = corrected_choices[:,is_symbols]

        for (component, dim), utilization_model in utilization_df.items():
            utilization[(component, dim)] = np.minimum(
                utilization[(component, dim)],
                utilization_model(*[
                    corrected_choices[:,i]
                    for i in range(corrected_choices.shape[1])
                ])
            )
            # TODO: Remove this constraint
            mask = mask & (utilization[(component, dim)] <= 1.0)
            if np.any(mask & (utilization[(component, dim)] == 1)):
                mask = mask & (utilization[(component, dim)] == 1)
            else:
                min_check = utilization[(component, dim)] + ~mask
                min_utilization = np.min(min_check)
                mask = mask & (min_check == min_utilization)

        good_choices = combined_choices[mask,:]

        # print(choices_a.shape[0], choices_b.shape[0], combined_choices.shape[0], np.sum(mask)/combined_choices.shape[0])

        if good_choices.shape[0] == 0:
            print(f"No good choices found")
            return combined_choices_with_largest[:,corrected_indices], is_symbols

        rank_var_and_choices.append((
            rank_a | rank_b,
            index_a + index_b,
            is_symbol_a + is_symbol_b,
            good_choices
        ))

    _, inverted_indices, is_symbol, choices = rank_var_and_choices[0]
    is_symbol = np.asarray(is_symbol)

    # Invert indices
    indices = invert_indices(inverted_indices)

    print(f"Returning choices of shape {choices[:,indices].shape}")

    return choices[:,indices], is_symbol[indices]

def invert_indices(inverted_indices):
    return np.argsort(inverted_indices)

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
            tiling_segment: TilingSegment = rank_var_to_tiling_segments[rank_var]

            if tile_shape == 'symbol' or isinstance(tile_shape, sympy.Symbol):
                tiling_segment.add_symbol(loop_idx)
            elif isinstance(tile_shape, int):
                tiling_segment.add_tile_shape(tile_shape, loop_idx)
            else:
                raise NotImplementedError(f'Unsupported tile shape {tile_shape}')

            loop_idx += 1

    for rank_var, tiling_segment in rank_var_to_tiling_segments.items():
        tiling_segment.finish()

    return rank_var_to_tiling_segments


def make_shapes_for_one_rank(tiling_segments: TilingSegment):
    all_tile_shapes = None
    total_loops = 0
    for n_loops, max_shape, min_shape in tiling_segments.iterate_segments():
        total_loops += n_loops

        factors = integer_factorizations_to_n_parts(max_shape, n_loops+1)
        factors = np.asarray(list(factors), dtype=np.int64)[:,:-1]
        tile_shape = max_shape // np.cumprod(factors, axis=1)
        tile_shape = tile_shape.astype(np.int64)
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


def run_model(pmapping, spec, flattened_arch: list[architecture.Leaf]):
    workload = spec.workload
    ert = spec.component_energy

    component_to_max_fanout = {}
    memory_to_datawidth = {}
    memory_to_size = {}
    for node in flattened_arch:
        if isinstance(node, Memory):
            memory_to_datawidth[node.name] = node.attributes.datawidth
            memory_to_size[node.name] = node.attributes.size
        component_to_max_fanout[node.name] = node.spatial.fanout

    df = {}

    reuse = analyze_reuse(pmapping, workload)
    overall_latency, _, _ = get_latency(reuse, pmapping, workload, flattened_arch)
    actions = gather_actions(reuse, pmapping, workload, None, is_path=True, use_name=True)
    energy = compute_energy_from_actions(actions, ert)

    total_occupancy = {}
    compute_unit = pmapping.nodes[-1].compute
    max_n_loops = 0
    for buffet, stats in reuse.buffet_stats.items():
        if buffet.level == compute_unit:
            continue
        occupancy = stats.occupancy*memory_to_datawidth[buffet.level]

        if buffet.level not in total_occupancy:
            total_occupancy[buffet.level] = {stats.n_loops_above: occupancy}
        else:
            total_occupancy[buffet.level][stats.n_loops_above] = occupancy

        max_n_loops = max(max_n_loops, stats.n_loops_above+1)

    for memory, occupancies in total_occupancy.items():
        running_total = 0
        for n_loop in range(max_n_loops):
            if n_loop in occupancies:
                running_total += occupancies[n_loop]
                df[nameloop2col(memory, n_loop)] = running_total

    df['metric_Latency'] = overall_latency

    total_energy = sum(energy.values())
    df['metric_Energy'] = total_energy

    per_memory_usage_df = {}
    for memory, occupancies in total_occupancy.items():
        per_memory_usage_df[memory] = sum(occupancies.values()) / memory_to_size[memory]

    utilization_df = {}
    for (component, einsum), per_dim_fanout in reuse.fanout.items():
        for dim, fanout in per_dim_fanout.items():
            utilization_df[(component, dim)] = \
                fanout / component_to_max_fanout[component][dim]


    return reuse.symbols, df, per_memory_usage_df, utilization_df

def compile_dict(symbols, dictionary):
    return {
        key: sympy.lambdify(symbols, value)
        for key, value in dictionary.items()
    }
