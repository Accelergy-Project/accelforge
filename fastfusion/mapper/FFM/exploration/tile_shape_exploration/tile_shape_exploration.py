from collections import defaultdict
from combinatorics.integer import *
from dataclasses import dataclass, field

import numpy as np

import sympy

import pandas as pd

import fastfusion.frontend.architecture as architecture
from fastfusion.frontend.architecture import Memory
from fastfusion.frontend.workload import Workload, Einsum
from fastfusion.frontend.workload.isl import get_rank_variable_bounds
from fastfusion.frontend.workload.symbolic import get_stride_and_halo
from fastfusion.frontend.mapping import Temporal, Spatial, Storage, Pattern
from fastfusion.frontend.specification import Specification

from fastfusion.mapper.FFM.exploration import metrics
from fastfusion.model.looptree.reuse.summarized.symbolic import analyze_reuse
from fastfusion.model.looptree.energy import compute_energy_from_actions, gather_actions
from fastfusion.model.looptree.latency import get_latency

from fastfusion.mapper.FFM.pareto import nameloop2col, tensor2col
from fastfusion.mapper.FFM.exploration.metrics import Metrics



class GivenShape(int):
    def __str__(self):
        return f'GivenShape({super().__repr__()})'

    def __repr__(self):
        return f'GivenShape({super().__repr__()})'


@dataclass(repr=True)
class NumLoops:
    n_loops: int = 0
    initial_delta_choices: dict[int, set[int]] = field(default_factory=dict)


class TilingSegment:
    def __init__(self, full_shape):
        self.data: list[GivenShape | NumLoops] = [GivenShape(full_shape), NumLoops(0)]
        self.indices: list[int] = []
        self.is_symbol: list[bool] = []

    def add_symbol(self, loop_idx: int):
        self.data[-1].n_loops += 1
        self.indices.append(loop_idx)
        self.is_symbol.append(True)

    def add_pattern(self, loop_idx: int, initial_delta_choices):
        last_data = self.data[-1]
        last_data.initial_delta_choices[last_data.n_loops] = \
            initial_delta_choices
        last_data.n_loops += 1
        self.indices.append(loop_idx)
        self.indices.append(loop_idx+1)
        self.is_symbol.append(True)
        self.is_symbol.append(True)

    def add_tile_shape(self, shape, loop_idx: int):
        self.data.append(GivenShape(shape))
        self.data.append(NumLoops(1))
        self.data.append(GivenShape(shape))
        self.data.append(NumLoops(0))
        self.indices.append(loop_idx)
        self.is_symbol.append(False)

    def finish(self):
        if isinstance(self.data[-1], NumLoops) and self.data[-1].n_loops == 0:
            self.data.pop()
        assert self.data[-1] == GivenShape(1)

    def iterate_segments(self):
        """Returns iterator over tuples (n_loops, max_shape, min_shape)."""
        for i in range(0, len(self.data)-1, 2):
            if self.data[i+1].n_loops == 0:
                continue
            max_shape = self.data[i]
            n_loops = self.data[i+1].n_loops
            initial_delta_choices = self.data[i+1].initial_delta_choices
            min_shape = self.data[i+2]
            yield (n_loops, initial_delta_choices, max_shape, min_shape)


def explore_tile_shapes(pmapping,
                        constraints: "MappingConstraints",
                        specification: Specification,
                        flattened_arch,
                        metrics: metrics.Metrics,
                        _fix_me=False):
    set_last_tile_shape_to_one(pmapping)

    symbols, symbolic_df, per_memory_occupancy_df, utilization_df = run_model(
        pmapping,
        specification,
        flattened_arch,
        metrics
    )

    if _fix_me:
        return
    try:
        compiled_df = compile_dict(symbols, symbolic_df)
        compiled_per_memory_occupancy_df = compile_dict(symbols, per_memory_occupancy_df)
        compiled_utilization_df = compile_dict(symbols, utilization_df)
    except:
        print('Compilation failed for this mapping:')
        for node in pmapping.nodes:
            if hasattr(node, 'compact_string'):
                print(node.compact_string())
        print(symbolic_df)
        raise RuntimeError('Compilation failed')

    tile_shapes, is_symbol, total_pmappings = generate_tile_shapes(
        pmapping,
        constraints,
        compiled_per_memory_occupancy_df,
        compiled_utilization_df,
        specification
    )

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

    df = pd.DataFrame(df)
    return df, total_pmappings


def generate_tile_shapes(pmapping, constraints, usage_df, utilization_df, specification):
    pmapping = pmapping.nodes
    workload = specification.workload

    initial_delta_choices = get_initial_delta_choices(pmapping[-1].einsum, workload)

    shape = get_rank_variable_bounds(workload, pmapping[-1].einsum)

    rank_var_to_tiling_segments = collect_tiling_segments(pmapping, shape, initial_delta_choices)

    def check_valid_tile_shape(combined_choices, is_symbols, other_rank_var_and_choices, indices, ranks):
        # print(f'\t Combined rank {rank_a} and {rank_b}: {choices_a.shape[0]} x {choices_b.shape[0]} -> {combined_choices.shape[0]}')
        if combined_choices.shape[0] == 0:
            return combined_choices
        
        n_rows = combined_choices.shape[0]
        n_loops = combined_choices.shape[1]

        # Insert ones
        combined_choices_with_ones = combined_choices
        combined_choices_with_largest = combined_choices
        for other_ranks, other_indices, other_is_symbol, other_choices in other_rank_var_and_choices:
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
            
            largest_other_choices = np.max(other_choices) # np.max(other_choices, axis=0, keepdims=True)
            combined_choices_with_largest = np.concatenate(
                (
                    combined_choices_with_largest,
                    np.ones((n_rows, other_n_loops), dtype=np.int64) * largest_other_choices
                ),
                axis=1
            )
            
        corrected_indices = np.asarray(invert_indices(indices))
        is_symbols = np.asarray(is_symbols)[corrected_indices]
        corrected_choices = combined_choices_with_ones[:,corrected_indices]
        corrected_choices = corrected_choices[:,is_symbols]
        corrected_choices_2 = combined_choices_with_largest[:,corrected_indices]
        corrected_choices_2 = corrected_choices_2[:,is_symbols]

        # TODO: there may be a more efficient order
        mask = constraints.check_tile_shape_constraints(corrected_choices, ranks)

        # Check if capacity is overused
        for memory, usage_model in usage_df.items():
            usage = usage_model(*[
                corrected_choices[:,i] for i in range(corrected_choices.shape[1])
            ])
            mask &= (usage <= 1.0)
            # if mask.sum() == 0:
            #     print(f'No valid memory usage for {rank_var}')

        # Compute utilization
        utilization = {}
        utilization2 = {}
        for (component, dim), utilization_model in utilization_df.items():
            utilization[(component, dim)] = utilization_model(*[
                corrected_choices[:,i] for i in range(corrected_choices.shape[1])
            ])
            utilization2[(component, dim)] = utilization_model(*[
                corrected_choices_2[:,i] for i in range(corrected_choices_2.shape[1])
            ])
            util = np.minimum(utilization[(component, dim)], utilization2[(component, dim)])
            mask &= util <= 1.0
            util = util * mask
            mask &= constraints.check_min_utilization_constraints(component, dim, util * mask, ranks)
            # if mask.sum() == 0:
            #     print(f'No valid utilization for {rank_var}')

        good_choices = combined_choices[mask,:]

        return good_choices

    rank_var_and_choices: list[tuple[frozenset[str], list[int], list[bool], np.array]] = []
    for rank_var, tiling_segments in rank_var_to_tiling_segments.items():
        choices = make_shapes_for_one_rank(tiling_segments)
        n_rows = choices.shape[0]
        n_loops = choices.shape[1]
        
        # other_rank_var_and_choices = []
        # for other_rank, other_segments in rank_var_to_tiling_segments.items():
        #     if rank_var == other_rank:
        #         continue
        #     other_rank_var_and_choices.append((
        #         frozenset((other_rank,)),
        #         other_segments.indices.copy(),
        #         other_segments.is_symbol.copy(),
        #         make_ones_for_one_rank(other_segments)
        #     ))
        
        # good_choices = check_valid_tile_shape(
        #     choices, 
        #     tiling_segments.is_symbol.copy(), 
        #     other_rank_var_and_choices, 
        #     tiling_segments.indices.copy(), 
        #     set((rank_var,))
        # )
        good_choices = choices
        is_symbols = tiling_segments.is_symbol.copy()

        # Insert ones
        indices = tiling_segments.indices.copy()
        is_symbols = tiling_segments.is_symbol.copy()
        choices2 = choices.copy()
        for other_rank, other_segments in rank_var_to_tiling_segments.items():
            if rank_var == other_rank:
                continue
            indices.extend(other_segments.indices)
            is_symbols.extend(other_segments.is_symbol)
            choices = np.concatenate(
                (
                    choices,
                    np.repeat(make_ones_for_one_rank(other_segments),
                              repeats=choices.shape[0],
                              axis=0)
                ),
                axis=1
            )
            choices2 = np.concatenate(
                (
                    choices2,
                    np.repeat(make_biggest_for_one_rank(other_segments),
                              repeats=choices2.shape[0],
                              axis=0)
                ),
                axis=1
            )

        # TODO: there may be a more efficient order
        corrected_indices = np.asarray(invert_indices(indices))
        corrected_choices = choices[:,corrected_indices]
        corrected_choices2 = choices2[:,corrected_indices]
        
        mask = constraints.check_tile_shape_constraints(corrected_choices, set((rank_var,)))
        # if mask.sum() == 0:
        #     print(f'No valid tile shapes for {rank_var}')
        is_symbols = np.asarray(is_symbols)[corrected_indices]
        corrected_choices = corrected_choices[:,is_symbols]
        corrected_choices2 = corrected_choices2[:,is_symbols]
        # Check if capacity is overused
        for memory, usage_model in usage_df.items():
            usage = usage_model(*[
                corrected_choices[:,i] for i in range(corrected_choices.shape[1])
            ])
            mask &= (usage <= 1.0)
        # if mask.sum() == 0:
        #     print(f'No valid memory usage for {rank_var}')

        utilization = {}
        utilization2 = {}
        for (component, dim), utilization_model in utilization_df.items():
            utilization[(component, dim)] = utilization_model(*[
                corrected_choices[:,i] for i in range(corrected_choices.shape[1])
            ])
            utilization2[(component, dim)] = utilization_model(*[
                corrected_choices2[:,i] for i in range(corrected_choices2.shape[1])
            ])
            util = np.minimum(utilization[(component, dim)], utilization2[(component, dim)])
            mask &= util <= 1.0
            util = util * mask
            mask &= constraints.check_min_utilization_constraints(component, dim, util, set((rank_var,)))
            # if mask.sum() == 0:
            #     print(f'No valid utilization for {rank_var}')

        good_choices = choices[mask,:]

        if good_choices.shape[0] == 0:
            return good_choices, is_symbols, 0

        rank_var_and_choices.append((
            frozenset((rank_var,)),
            tiling_segments.indices.copy(),
            tiling_segments.is_symbol.copy(),
            good_choices[:,:n_loops]
        ))

    # for (i, (rank, index, is_symbol, choices)) in enumerate(rank_var_and_choices):
    #     other_rank_var_and_choices = [x for k, x in enumerate(rank_var_and_choices) if k != i]
    #     good_choices = check_valid_tile_shape(
    #         choices, 
    #         is_symbol, 
    #         other_rank_var_and_choices, 
    #         index, 
    #         rank
    #     )
    #     rank_var_and_choices[i] = (rank, index, is_symbol, good_choices)

    total_pmappings = 1
    for rv in rank_var_and_choices:
        total_pmappings *= rv[-1].shape[0]

    def get_combined_choices(rank_var_and_choices_a, rank_var_and_choices_b, other_rank_var_and_choices, tile_shape=128):
        rank_a, index_a, is_symbol_a, choices_a = rank_var_and_choices_a
        rank_b, index_b, is_symbol_b, choices_b = rank_var_and_choices_b

        new_rank = rank_a | rank_b
        new_index = index_a + index_b
        new_is_symbol = is_symbol_a + is_symbol_b
        if any(x.shape[0] == 0 for x in [choices_a, choices_b] + [x[-1] for x in other_rank_var_and_choices]):
            return (
                new_rank,
                new_index,
                new_is_symbol,
                np.empty((0, len(new_index)), dtype=np.int64)
            )

        all_good_choices = []
        for a_idx in range(0, choices_a.shape[0], tile_shape):
            a_idx_max = min(a_idx+tile_shape, choices_a.shape[0])
            tile_occupancy_a = a_idx_max - a_idx
            
            for b_idx in range(0, choices_b.shape[0], tile_shape):
                b_idx_max = min(b_idx+tile_shape, choices_b.shape[0])
                tile_occupancy_b = b_idx_max - b_idx
                
                combined_choices = np.concatenate(
                    (
                        np.tile(choices_a[a_idx:a_idx_max,:], (tile_occupancy_b, 1)),
                        np.repeat(choices_b[b_idx:b_idx_max,:],
                                  tile_occupancy_a,
                                  axis=0)
                    ),
                    axis=1
                )

                good_choices = check_valid_tile_shape(combined_choices, is_symbol_a + is_symbol_b, other_rank_var_and_choices, index_a + index_b, rank_a | rank_b)
                all_good_choices.append(good_choices)

                # # print(f'\t Combined rank {rank_a} and {rank_b}: {choices_a.shape[0]} x {choices_b.shape[0]} -> {combined_choices.shape[0]}')
                # n_rows = combined_choices.shape[0]
                # n_loops = combined_choices.shape[1]

                # is_symbols = is_symbol_a + is_symbol_b
                # indices = index_a + index_b

                # # Insert ones
                # combined_choices_with_ones = combined_choices
                # for other_ranks, other_indices, other_is_symbol, other_choices in other_rank_var_and_choices:
                #     indices.extend(other_indices)
                #     is_symbols.extend(other_is_symbol)

                #     other_n_loops = len(other_indices)
                #     combined_choices_with_ones = np.concatenate(
                #         (
                #             combined_choices_with_ones,
                #             np.ones((n_rows, other_n_loops), dtype=np.int64)
                #         ),
                #         axis=1
                #     )

                # # TODO: there may be a more efficient order
                # corrected_indices = np.asarray(invert_indices(indices))
                # corrected_choices = combined_choices_with_ones[:,corrected_indices]
                # mask = constraints.check_tile_shape_constraints(corrected_choices, rank_a | rank_b)
                # is_symbols = np.asarray(is_symbols)[corrected_indices]
                # corrected_choices = corrected_choices[:,is_symbols]

                # # Check if capacity is overused
                # for memory, usage_model in usage_df.items():
                #     usage = usage_model(*[
                #         corrected_choices[:,i] for i in range(corrected_choices.shape[1])
                #     ])
                #     mask &= (usage <= 1.0)

                # # Compute utilization
                # utilization = {}
                # for (component, dim), utilization_model in utilization_df.items():
                #     utilization[(component, dim)] = utilization_model(*[
                #         corrected_choices[:,i] for i in range(corrected_choices.shape[1])
                #     ])
                #     util = utilization[(component, dim)]
                #     mask &= util <= 1.0
                #     util = util * mask
                #     mask &= constraints.check_min_utilization_constraints(component, dim, util * mask, rank_a | rank_b)

                # # Insert largest value
                # combined_choices_with_largest = combined_choices
                # for other_ranks, other_indices, other_is_symbol, other_choices in other_rank_var_and_choices:
                #     largest_other_choices = np.max(other_choices, axis=0, keepdims=True)
                #     combined_choices_with_largest = np.concatenate(
                #         (
                #             combined_choices_with_largest,
                #             np.repeat(largest_other_choices, n_rows, axis=0)
                #         ),
                #         axis=1
                #     )
                # corrected_choices = combined_choices_with_largest[:,corrected_indices]
                # corrected_choices = corrected_choices[:,is_symbols]

                # # for (component, dim), utilization_model in utilization_df.items():
                # #     utilization[(component, dim)] = np.minimum(
                # #         utilization[(component, dim)],
                # #         utilization_model(*[
                # #             corrected_choices[:,i]
                # #             for i in range(corrected_choices.shape[1])
                # #         ])
                # #     )
                # #     mask &= (utilization[(component, dim)] <= 1.0)

                # good_choices = combined_choices[mask,:]
                # all_good_choices.append(good_choices)

        return (
            new_rank,
            new_index,
            new_is_symbol,
            np.concatenate(all_good_choices, axis=0)
        )

    while len(rank_var_and_choices) > 1:
        best_reduction = 2
        best_indices = None
        best_combined = None
        for i, rank_var_and_choices_a in enumerate(rank_var_and_choices):
            for j, rank_var_and_choices_b in enumerate(rank_var_and_choices):
                if i >= j:
                    continue

                choices_a = rank_var_and_choices_a[-1]
                choices_b = rank_var_and_choices_b[-1]

                # If we're going to have too many choices, skip
                if choices_a.shape[0] * choices_b.shape[0] > 10000:
                    continue

                other_rank_var_and_choices = [x for k, x in enumerate(rank_var_and_choices) if k not in (i, j)]
                combined = get_combined_choices(rank_var_and_choices_a, rank_var_and_choices_b, other_rank_var_and_choices)
                choices_combined = combined[-1]
                if choices_a.shape[0] == 0 or choices_b.shape[0] == 0:
                    reduction = 0
                else:
                    reduction = choices_combined.shape[0] / choices_a.shape[0] / choices_b.shape[0]

                # Encourage combining choices with 1
                reduction -= choices_a.shape[0] == 1
                reduction -= choices_b.shape[0] == 1

                # Combine the two choices that lead to the most reduction in the number of choices
                if reduction < best_reduction:
                    best_reduction = reduction
                    best_indices = (i, j)
                    best_combined = combined

        if best_indices is None:
            break
    
        rank_var_and_choices.pop(best_indices[1])
        rank_var_and_choices.pop(best_indices[0])
        rank_var_and_choices.append(best_combined)

    # Now pick the smallest choices
    rank_var_and_choices.sort(key=lambda x: x[-1].shape[0], reverse=True)
    while len(rank_var_and_choices) > 1:
        def get_smallest():
            smallest, smallest_idx = None, None
            for i, rv in enumerate(rank_var_and_choices):
                if smallest is None or rv[-1].shape[0] < smallest[-1].shape[0]:
                    smallest_idx, smallest = i, rv
            rank_var_and_choices.pop(smallest_idx)
            return smallest
        smallest = get_smallest()
        smallest2 = get_smallest()
        rank_var_and_choices.append(get_combined_choices(smallest, smallest2, rank_var_and_choices))

    # Sort rank var and choices by the rank variable name
    # rank_var_and_choices.sort(key=lambda x: x[0])
    # while len(rank_var_and_choices) > 1:
    #     a = rank_var_and_choices.pop(0)
    #     b = rank_var_and_choices.pop(0)
    #     rank_var_and_choices.insert(0, get_combined_choices(a, b, rank_var_and_choices))

    combined_choices = rank_var_and_choices[0][-1]
    is_symbol = rank_var_and_choices[0][2]
    indices = rank_var_and_choices[0][1]
    ranks = rank_var_and_choices[0][0]
    other_rank_var_and_choices = []  # All rank variables have been combined, so no others remain
    good_choices = check_valid_tile_shape(combined_choices, is_symbol, other_rank_var_and_choices, indices, ranks)


    _, inverted_indices, is_symbol, choices = rank_var_and_choices[0]
    is_symbol = np.asarray(is_symbol)

    # Invert indices
    indices = invert_indices(inverted_indices)

    # print(f"Returning choices of shape {choices[:,indices].shape}")
    # for node in pmapping:
    #     if isinstance(node, Iteration):
    #         print(node)
    # print(choices[:,indices])

    return choices[:,indices], is_symbol[indices], total_pmappings

def invert_indices(inverted_indices):
    return np.argsort(inverted_indices)

def collect_tiling_segments(
    pmapping,
    rank_shape: dict,
    initial_delta_choices: dict[str, set[int]]={}
) -> dict[str, TilingSegment]:
    rank_var_to_tiling_segments = {}
    loop_idx = 0
    for node in pmapping:
        if isinstance(node, Temporal) or isinstance(node, Spatial):
            rank_var = node.rank_variable
            tile_shape = node.tile_shape
            tile_pattern = node.tile_pattern

            if rank_var not in rank_var_to_tiling_segments:
                rank_var_to_tiling_segments[rank_var] = \
                    TilingSegment(rank_shape[rank_var])
            tiling_segment: TilingSegment = rank_var_to_tiling_segments[rank_var]

            if tile_shape == 'symbol' or isinstance(tile_shape, sympy.Symbol):
                tiling_segment.add_symbol(loop_idx)
            elif isinstance(tile_shape, int):
                tiling_segment.add_tile_shape(tile_shape, loop_idx)
            elif isinstance(tile_pattern, Pattern):
                stride = tile_pattern.stride
                initial_tile_shape = tile_pattern.initial_tile_shape
                pattern_shape = tile_pattern.tile_shape
                if pattern_shape is not None:
                    raise ValueError('Recomputation not yet supported')
                assert stride is not None and initial_tile_shape is not None
                if isinstance(stride, int):
                    tiling_segment.add_tile_shape(stride, loop_idx)
                elif isinstance(stride, sympy.Symbol):
                    tiling_segment.add_pattern(loop_idx,
                                               initial_delta_choices[rank_var])
                else:
                    raise RuntimeError('BUG')
            else:
                raise NotImplementedError(f'Unsupported tile shape {tile_shape}')

            loop_idx += 1

    for rank_var, tiling_segment in rank_var_to_tiling_segments.items():
        tiling_segment.finish()

    return rank_var_to_tiling_segments

def make_shapes_for_one_rank(tiling_segments: TilingSegment):
    all_tile_shapes = None
    total_loops = 0
    for n_loops, initial_delta_choices, max_shape, min_shape in tiling_segments.iterate_segments():
        total_loops += n_loops

        factors = integer_factorizations_to_n_parts(max_shape, n_loops+1)
        factors = np.asarray(list(factors), dtype=np.int64)[:,:-1]
        tile_shape = max_shape // np.cumprod(factors, axis=1)
        tile_shape = tile_shape.astype(np.int64)
        tile_shape = tile_shape[np.all(tile_shape >= min_shape, axis=1), :]

        for i in sorted(initial_delta_choices, reverse=True):
            choices = np.array(list(initial_delta_choices[i])).reshape(-1, 1).astype(np.int64)
            tile_shape = np.concatenate((
                    np.tile(tile_shape[:,:i+1], (choices.shape[0], 1)),
                    np.repeat(choices, repeats=tile_shape.shape[0], axis=0),
                    np.tile(tile_shape[:,i+1:], (choices.shape[0], 1))
                ), axis=1)
            tile_shape[:,i+1] += tile_shape[:,i]

            if i >= 1:
                mask_full_tile = tile_shape[:,i] == tile_shape[:,i-1]
            else:
                mask_full_tile = tile_shape[:,i] == max_shape
            tile_shape[mask_full_tile,i+1] = tile_shape[mask_full_tile,i]

            if i >= 1:
                mask_valid_initial = tile_shape[:,i+1] <= tile_shape[:,i-1]
            else:
                mask_valid_initial = tile_shape[:,i+1] <= max_shape
            tile_shape = tile_shape[mask_valid_initial,:]

        if all_tile_shapes is None:
            all_tile_shapes = tile_shape
        else:
            all_tile_shapes_n_rows = all_tile_shapes.shape[0]
            all_tile_shapes = np.tile(all_tile_shapes, (tile_shape.shape[0], 1))
            tile_shape = np.repeat(tile_shape, repeats=all_tile_shapes_n_rows, axis=0)
            all_tile_shapes = np.concatenate((all_tile_shapes, tile_shape), axis=1)

    return all_tile_shapes

def make_ones_for_one_rank(tiling_segments: TilingSegment):
    total_cols = 0
    for n_loops, initial_delta_choices, max_shape, min_shape in tiling_segments.iterate_segments():
        total_cols += n_loops + len(initial_delta_choices.keys())
    return np.ones((1, total_cols))

def make_biggest_for_one_rank(tiling_segments: TilingSegment):
    total_cols = 0
    max_max_shape = 0
    for n_loops, initial_delta_choices, max_shape, min_shape in tiling_segments.iterate_segments():
        total_cols += n_loops + len(initial_delta_choices.keys())
        max_max_shape = max(max_max_shape, max_shape)
    return np.ones((1, total_cols)) * max_max_shape

def set_last_tile_shape_to_one(pmapping):
    pmapping = pmapping.nodes

    rank_var_to_last_node = {}
    for node in pmapping:
        if isinstance(node, Temporal) or isinstance(node, Spatial):
            rank_var_to_last_node[node.rank_variable] = node

    for last_node in rank_var_to_last_node.values():
        last_node.tile_shape = 1


def run_model(pmapping, spec, flattened_arch: list[architecture.Leaf], metrics: metrics.Metrics):
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
    overall_latency, comp_latency, mem_latency = get_latency(reuse, pmapping, workload, flattened_arch)
    actions = gather_actions(reuse, pmapping, workload, None, is_path=True, use_name=True)
    energy = compute_energy_from_actions(actions, ert)

    intermediate_tensors = workload.intermediate_tensor_names
    tensor_to_backing = {}
    for node in pmapping.nodes:
        if isinstance(node, Storage):
            for tensor in node.tensors:
                if (
                    tensor not in tensor_to_backing
                    and tensor in intermediate_tensors
                ):
                    tensor_to_backing[tensor] = node.memory

    total_occupancy = {}
    compute_unit = pmapping.nodes[-1].compute
    max_n_loops = 0
    for buffet, stats in reuse.buffet_stats.items():
        if buffet.level == compute_unit:
            continue

        occupancy = stats.occupancy*memory_to_datawidth[buffet.level]

        if (
            buffet.tensor in tensor_to_backing
            and tensor_to_backing[buffet.tensor] == buffet.level
        ):
            df[tensor2col(buffet.tensor)] = occupancy

        if buffet.level not in total_occupancy:
            total_occupancy[buffet.level] = {stats.n_loops_above: occupancy}
        else:
            occupancy_per_level = total_occupancy[buffet.level]
            if stats.n_loops_above not in occupancy_per_level:
                occupancy_per_level[stats.n_loops_above] = occupancy
            else:
                occupancy_per_level[stats.n_loops_above] += occupancy

        max_n_loops = max(max_n_loops, stats.n_loops_above+1)

    for memory, occupancies in total_occupancy.items():
        running_total = 0
        for n_loop in range(max_n_loops):
            if n_loop in occupancies:
                running_total += occupancies[n_loop]
                df[nameloop2col(memory, n_loop)] = running_total

    if metrics & Metrics.LATENCY:
        df['metric_Latency'] = overall_latency
        df['compute_Latency'] = comp_latency
        for mem, latency in mem_latency.items():
            df[f'{mem}_Latency'] = latency

    if metrics & Metrics.ENERGY:
        df['metric_Energy'] = sum(energy.values())

    if metrics & Metrics.PER_COMPONENT_ENERGY:
        for component, component_energy in energy.items():
            df[f'{component}_Energy'] = component_energy

    if metrics & Metrics.RESERVATIONS:
        for memory, occupancies in total_occupancy.items():
            df[f'{memory}_Reservations'] = sum(occupancies.values())

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


def get_initial_delta_choices(einsum_name: str, workload: Workload):
    stride_and_halo = get_stride_and_halo(workload)
    einsum = workload.einsums[einsum_name]

    choices = defaultdict(lambda: set([0]))
    consumer_chains = []
    stack = [[(None, einsum)]]
    while stack:
        cur_chain = stack.pop()
        last_tensor, last_einsum = cur_chain[-1]
        for tensor in last_einsum.output_tensors():
            einsums_that_read_tensor = workload.einsums_that_read_tensor(tensor)

            if len(einsums_that_read_tensor) == 0:
                consumer_chains.append(cur_chain)

            for next_einsum in einsums_that_read_tensor:
                stack.append(cur_chain + [(tensor, next_einsum)])

    for chain in consumer_chains:
        for (_, producer), (tensor, consumer) in zip(list(reversed(chain))[1:],
                                                     reversed(chain)):
            rank_stride_and_halo = stride_and_halo[(consumer.name, tensor)]
            if tensor is None:
                break  # done

            for cons_rank_var in consumer.rank_variables:
                for prod_rank_var in producer.rank_variables:
                    for cons_choice in choices[cons_rank_var]:
                        if (prod_rank_var, cons_rank_var) not in rank_stride_and_halo:
                            continue
                        stride, halo = rank_stride_and_halo[(prod_rank_var, cons_rank_var)]
                        choices[prod_rank_var].add(cons_choice*stride + halo)

    return choices
