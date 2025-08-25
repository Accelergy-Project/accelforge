from collections import defaultdict
import itertools
from math import ceil
import random
import resource
from typing import Callable
from combinatorics.integer import *
from dataclasses import dataclass, field

from fastfusion import util
from fastfusion.accelerated_imports import np

import sympy

from fastfusion.accelerated_imports import pd

import fastfusion.frontend.architecture as architecture
from fastfusion.frontend.architecture import Memory
from fastfusion.frontend.specification import Specification
from fastfusion.frontend.workload import Workload
from fastfusion.frontend.workload.isl import get_rank_variable_bounds
from fastfusion.frontend.workload.symbolic import get_stride_and_halo
from fastfusion.frontend.mapping import MappingNode, Temporal, Spatial, TensorHolder, Pattern

from fastfusion.mapper import metrics
from fastfusion.mapper.FFM.exploration.contraints.constraints import MappingConstraints
from fastfusion.mapper.FFM.exploration.mapper_one_einsum.mapper_job import Job
from fastfusion.model.looptree.reuse.summarized.symbolic import analyze_reuse_and_add_reservations_to_mapping
from fastfusion.model.looptree.energy import compute_energy_from_actions, gather_actions
from fastfusion.model.looptree.latency import get_latency

from fastfusion.mapper.FFM.pareto import nameloop2col, tensor2col
from fastfusion.mapper.metrics import Metrics

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
    def __init__(self, full_shape, max_fused_loops: int):
        self.data: list[GivenShape | NumLoops] = [GivenShape(full_shape), NumLoops(0)]
        self.indices: list[int] = []
        self.is_symbol: list[bool] = []
        self.max_fused_loops: int = max_fused_loops
        self.is_fused: list[bool] = []

    def add_symbol(self, loop_idx: int, fused: bool):
        self.data[-1].n_loops += 1
        self.indices.append(loop_idx)
        self.is_symbol.append(True)
        self.is_fused.append(fused)

    def add_pattern(self, loop_idx: int, initial_delta_choices, fused: bool):
        last_data = self.data[-1]
        last_data.initial_delta_choices[last_data.n_loops] = \
            initial_delta_choices
        last_data.n_loops += 1
        self.indices.append(loop_idx)
        self.indices.append(loop_idx+1)
        self.is_symbol.append(True)
        self.is_symbol.append(True)
        self.is_fused.append(fused)

    def add_tile_shape(self, shape, loop_idx: int, fused: bool):
        self.data.append(GivenShape(shape))
        self.data.append(NumLoops(1))
        self.data.append(GivenShape(shape))
        self.data.append(NumLoops(0))
        self.indices.append(loop_idx)
        self.is_symbol.append(False)
        self.is_fused.append(fused)

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


def _explore_tile_shapes(job: "Job"):
    pmapping = job.mapping
    constraints = job.constraints
    specification = job.spec

    constraints.set_loop_indices(pmapping.nodes)

    set_last_tile_shape_to_one(pmapping)

    symbols, symbolic_df, per_memory_occupancy_df, utilization_df = run_model(job)

    try:
        compiled_df = compile_dict(symbols, symbolic_df)
        compiled_per_memory_occupancy_df = compile_dict(symbols, per_memory_occupancy_df)
        compiled_utilization_df = compile_dict(symbols, utilization_df)
    except Exception as e:
        print('Compilation failed for this mapping:')
        for node in pmapping.nodes:
            if hasattr(node, 'compact_str'):
                print(node.compact_str())
        print(symbolic_df)
        raise RuntimeError('Compilation failed') from e

    tile_shapes, is_symbol, total_pmappings = generate_tile_shapes(
        pmapping,
        constraints,
        compiled_per_memory_occupancy_df,
        compiled_utilization_df,
        specification,
        job
    )

    df = {}
    for i in range(tile_shapes.shape[1]):
        df[f"tile_shape\0{i}"] = tile_shapes[:, i]

    tile_shapes = tile_shapes[:, is_symbol]
    tile_shapes = [
        tile_shapes[:, i]
        for i in range(tile_shapes.shape[1])
    ]

    for key in compiled_df:
        df[key] = compiled_df[key](*tile_shapes)
        
    df = pd.DataFrame(df, columns=df.keys())
    assert not df.isna().any().any()
    return df, total_pmappings

def explore_tile_shapes(job: "Job"):
    memory_limit = job.memory_limit // 8 # Bytes -> bits
    if job.memory_limit != float('inf'):
        try:
            resource.setrlimit(resource.RLIMIT_AS, (job.memory_limit, job.memory_limit))
        except (ValueError, OSError):
            # Ignore permission errors when trying to set memory limits
            pass

    if job.time_limit != float('inf'):
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (ceil(job.time_limit), ceil(job.time_limit)))
        except (ValueError, OSError):
            # Ignore permission errors when trying to set CPU limits
            pass

    def format_memory_limit() -> str:
        if memory_limit == float('inf'):
            return "infinite"
        if memory_limit > 1024 * 1024 * 1024:
            return f"{memory_limit / (1024 * 1024 * 1024):.2f} GB"
        elif memory_limit > 1024 * 1024:
            return f"{memory_limit / (1024 * 1024):.2f} MB"
        elif memory_limit > 1024:
            return f"{memory_limit / 1024:.2f} KB"
        else:
            return f"{memory_limit:.2f} B"

    try:
        return _explore_tile_shapes(job)
    except MemoryError as e:
        s = f"Job ran out of memory with memory limit {format_memory_limit()}"
        job.log_message(f"Tile shape exploration failed: {s}")
        raise RuntimeError(job.pretty_str()) from e
    except TimeoutError as e:
        s = f"Job timed out with time limit {job.time_limit:.2f} seconds"
        job.log_message(f"Tile shape exploration failed: {s}")
        raise RuntimeError(job.pretty_str()) from e

    finally:
        try:
            resource.setrlimit(resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        except (ValueError, OSError):
            # Ignore permission errors when trying to reset memory limits
            pass
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        except (ValueError, OSError):
            # Ignore permission errors when trying to reset CPU limits
            pass

# What I want to do:
# - For each pair of rank variables, check possible combinations of tile shapes
# - For each valid combination, add to a bloom filter
# - When we're combining one rank variable with (a set of other rank variables),
#   - For each combination of (next rank variable, existing rank variable)
#   


def generate_tile_shapes(
        pmapping: list[MappingNode],
        constraints: MappingConstraints,
        usage_df: dict[str, Callable],
        utilization_df: dict[str, Callable],
        specification: Specification,
        job: Job
):
    pmapping = pmapping.nodes
    workload = specification.workload

    initial_delta_choices = get_initial_delta_choices(pmapping[-1].einsum, workload)

    shape = get_rank_variable_bounds(workload, pmapping[-1].einsum)

    rank_var_to_tiling_segments = collect_tiling_segments(pmapping, shape, specification, initial_delta_choices)

    def update_mask(prev_mask, new_mask, cause: str, track_masked: bool = True):
        prev_sum = prev_mask.sum()
        new_mask = new_mask & prev_mask
        new_sum = new_mask.sum()
        if track_masked and prev_sum != 0 and prev_sum != new_sum:
            # print(f"Mask {cause}: {new_sum / prev_sum}")
            job.log_mask(cause, new_sum / prev_sum, prev_mask.shape[0])
        return new_mask

    def check_valid_tile_shape(combined_choices, is_symbols, other_rank_var_and_choices, indices, ranks, shape, track_masked=True):
        # print(f'\t Combined rank {rank_a} and {rank_b}: {choices_a.shape[0]} x {choices_b.shape[0]} -> {combined_choices.shape[0]}')
        if combined_choices.shape[0] == 0:
            return combined_choices

        n_rows = combined_choices.shape[0]
        n_loops = combined_choices.shape[1]

        # Insert ones
        combined_choices_with_ones = combined_choices
        combined_choices_with_largest = combined_choices
        complete_indices = indices.copy()

        for other_ranks, other_indices, other_is_symbol, other_choices in other_rank_var_and_choices:
            indices.extend(other_indices)
            is_symbols.extend(other_is_symbol)
            complete_indices.extend([i for i, s in zip(other_segments.indices, other_segments.is_symbol) if not s])

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
        mask = np.ones(corrected_choices.shape[0], dtype=np.bool)
        mask = update_mask(mask, constraints.check_tile_shape_constraints(corrected_choices, complete_indices), "tile shape constraints", track_masked=track_masked)

        # Check if capacity is overused
        for memory, usage_model in usage_df.items():
            usage = usage_model(*[
                corrected_choices[:,i] for i in range(corrected_choices.shape[1])
            ])
            mask = update_mask(mask, usage <= 1.0, f"{memory} usage", track_masked=track_masked)
            # if mask.sum() == 0:
            #     print(f'No valid memory usage for {rank_var}')

        # Compute utilization
        utilization = {}
        utilization2 = {}
        for component_dim, utilization_model in utilization_df.items():
            _, component, dim = component_dim.split('\0')
            utilization[(component, dim)] = utilization_model(*[
                corrected_choices[:,i] for i in range(corrected_choices.shape[1])
            ])
            utilization2[(component, dim)] = utilization_model(*[
                corrected_choices_2[:,i] for i in range(corrected_choices_2.shape[1])
            ])
            util = np.minimum(utilization[(component, dim)], utilization2[(component, dim)])
            mask = update_mask(mask, util <= 1.0, f"{component} {dim} utilization", track_masked=track_masked)
            util = util * mask
            mask = update_mask(mask, constraints.check_min_utilization_constraints(component, dim, util * mask, complete_indices), f"{component} {dim} minimum utilization", track_masked=track_masked)
            # if mask.sum() == 0:
            #     print(f'No valid utilization for {rank_var}')

        good_choices = combined_choices[mask,:]

        # # Check that we're less than the n_loops limit
        # TODO: Bring back loop limit
        # n_loops = np.zeros(good_choices.shape[0], dtype=np.int64)
        # for rank in ranks:
        #     tiling_segments = rank_var_to_tiling_segments[rank]
            
        #     # Previous size = the full rank variable size. Initialize a vector with that size 
        #     cur_size = np.zeros(good_choices.shape[0], dtype=np.int64) + shape[rank]
            
        #     for i in [indices.index(i) for i in tiling_segments.indices]:
        #         n_loops += cur_size != good_choices[:,i]
        #         cur_size = good_choices[:,i]
                
                
        max_loops = min(
            specification.mapper.ffm.max_loops,
            specification.mapper.ffm.max_loops_minus_ranks + len(ranks)
        )
        mask = np.ones(good_choices.shape[0], dtype=np.bool)
        mask = update_mask(mask, n_loops <= max_loops, "max loops", track_masked=track_masked)
        good_choices = good_choices[mask,:]

        return good_choices

    rank_var_and_choices: list[tuple[frozenset[str], list[int], list[bool], np.array]] = []
    rv2choices = {
        rank_var: make_shapes_for_one_rank(tiling_segments)
        for rank_var, tiling_segments in rank_var_to_tiling_segments.items()
    }

    nominal_n_mappings = 1

    for rank_var, tiling_segments in rank_var_to_tiling_segments.items():
        choices = rv2choices.pop(rank_var)
        nominal_n_mappings *= choices.shape[0]

        n_loops = choices.shape[1]

        good_choices = choices
        is_symbols = tiling_segments.is_symbol.copy()

        # Insert ones
        indices = tiling_segments.indices.copy()
        complete_indices = indices.copy()
        is_symbols = tiling_segments.is_symbol.copy()
        choices2 = choices.copy()
        for other_rank, other_segments in rank_var_to_tiling_segments.items():
            if rank_var == other_rank:
                continue
            indices.extend(other_segments.indices)
            is_symbols.extend(other_segments.is_symbol)
            complete_indices.extend([i for i, s in zip(other_segments.indices, other_segments.is_symbol) if not s])
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

        mask = constraints.check_tile_shape_constraints(corrected_choices, indices)
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
            mask = update_mask(mask, usage <= 1.0, f"{memory} usage")

        # if mask.sum() == 0:
        #     print(f'No valid memory usage for {rank_var}')

        utilization = {}
        utilization2 = {}
        for component_dim, utilization_model in utilization_df.items():
            _, component, dim = component_dim.split('\0')
            utilization[(component, dim)] = utilization_model(*[
                corrected_choices[:,i] for i in range(corrected_choices.shape[1])
            ])
            utilization2[(component, dim)] = utilization_model(*[
                corrected_choices2[:,i] for i in range(corrected_choices2.shape[1])
            ])
            util = np.minimum(utilization[(component, dim)], utilization2[(component, dim)])
            mask = update_mask(mask, util <= 1.0, f"{component} {dim} utilization")
            util = util * mask
            mask = update_mask(mask, constraints.check_min_utilization_constraints(component, dim, util, complete_indices), f"{component} {dim} minimum utilization")

        good_choices = choices[mask,:]

        if good_choices.shape[0] == 0:
            return good_choices, is_symbols, 0
        
        job.log_message(
            f"Rank {rank_var} has {good_choices.shape[0]} valid tile shape choices."
        )

        rank_var_and_choices.append((
            frozenset((rank_var,)),
            tiling_segments.indices.copy(),
            tiling_segments.is_symbol.copy(),
            good_choices[:,:n_loops]
        ))
        
        

    total_pmappings = 1
    for rv in rank_var_and_choices:
        total_pmappings *= rv[-1].shape[0]

    prev_rank_var_and_choices = rank_var_and_choices
    rank2prev_rank_var_and_choices = {next(iter(x[0])): x for x in prev_rank_var_and_choices}
    rank_var_and_choices = []
    for i, rank_var_and_choices_a in enumerate(prev_rank_var_and_choices):
        rank_a, index_a, is_symbol_a, choices_a = rank_var_and_choices_a
        for j, index in enumerate(index_a):
            rank_var_and_choices.append((
                rank_a,
                [index],
                [is_symbol_a[j]],
                np.unique(choices_a[:,j]).reshape(-1, 1)
            ))

    def get_combined_choices(rank_var_and_choices_a, rank_var_and_choices_b, other_rank_var_and_choices, shape, tile_shape=512, track_masked=True):
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
            
        job.create_new_mask_total(choices_a.shape[0] * choices_b.shape[0])
        job.log_message(f"Combining {choices_a.shape[0]} choices for {index_a} and {choices_b.shape[0]} choices for {index_b}")
        combined_choices = []
            
        # Now bin the choices of a by the choices for loops in rank_b
        bin_a = {}
        prev = rank2prev_rank_var_and_choices[next(iter(rank_b))]
        _, prev_index, _, prev_choices = prev

        a_index_prev = [i for i in index_a if i in prev_index]
        b_index_prev = [i for i in index_b if i in prev_index]
        prev_options = set([tuple(row) for row in prev_choices[:, [prev_index.index(i) for i in a_index_prev + b_index_prev]]])

        b_loops = choices_a[:, [index_a.index(i) for i in a_index_prev]]
        for i in range(choices_a.shape[0]):
            bin_a.setdefault(tuple(b_loops[i]), []).append(i)

        assert choices_b.shape[1] == 1
        for b_option in choices_b.flatten():
            valid_a_bins = []
            for k, v in bin_a.items():
                if tuple(k) + (b_option,) in prev_options:
                    valid_a_bins.append(v)
                    
            job.log_message(f"Found {len(valid_a_bins)} valid bins for {b_option}")

            valid_a_indices = list(itertools.chain.from_iterable([v for v in valid_a_bins]))
            cur_choices_a = choices_a[valid_a_indices]

            choices = np.concatenate(
                (
                    cur_choices_a,
                    np.repeat(np.array([b_option]),
                              repeats=cur_choices_a.shape[0],
                              axis=0).reshape(-1, 1)
                ),
                axis=1
            )

            good_choices = check_valid_tile_shape(choices, is_symbol_a + is_symbol_b, other_rank_var_and_choices, index_a + index_b, rank_a | rank_b, shape, track_masked=track_masked)
            combined_choices.append(good_choices)
            
        job.clear_mask_total()
            
        good_choices = np.concatenate(combined_choices, axis=0)
        good_choices = check_valid_tile_shape(good_choices, is_symbol_a + is_symbol_b, other_rank_var_and_choices, index_a + index_b, rank_a | rank_b, shape, track_masked=track_masked)
        return (
            new_rank,
            new_index,
            new_is_symbol,
            good_choices
        )

    # First, combine spatial loops
    # loops = [x for x in job.mapping.nodes if isinstance(x, (Temporal, Spatial))]
    # while True:
    #     def get_spatial():
    #         for i, rank_var_and_choices_a in enumerate(rank_var_and_choices):
    #             rank_a, index_a, is_symbol_a, choices_a = rank_var_and_choices_a
    #             spatial_a = any(isinstance(loops[i], Spatial) for i in index_a)
    #             if spatial_a:
    #                 return rank_var_and_choices.pop(i)
    #         return None

    #     spatial_a = get_spatial()
    #     spatial_b = get_spatial()
    #     if spatial_b is None:
    #         if spatial_a is not None:
    #             rank_var_and_choices.append(spatial_a)
    #         break
    #     rank_var_and_choices.insert(0, get_combined_choices(spatial_a, spatial_b, rank_var_and_choices, shape))

    # Start combining from the loop with the fewest choices
    _, fewest_index = min(
        ((x[-1].shape[0], i) for i, x in enumerate(rank_var_and_choices))
    )
    rank_var_and_choices.insert(0, rank_var_and_choices.pop(fewest_index))

    # Then, combine the loops that lead to the most reduction in the number of choices
    while len(rank_var_and_choices) > 1:
        best_reduction = 2
        best_index = None
        a = rank_var_and_choices.pop(0)
        choices_a = a[-1]
        for i, b in enumerate(rank_var_and_choices):
            choices_b = b[-1]

            # If we're going to have too many choices, skip
            if choices_a.shape[0] * choices_b.shape[0] > 100000:
                continue
            
            other_rank_var_and_choices = [x for k, x in enumerate(rank_var_and_choices) if k != i]
            combined = get_combined_choices(a, b, other_rank_var_and_choices, shape, track_masked=False)
            choices_combined = combined[-1]
            total_shape = max((choices_a.shape[0] * choices_b.shape[0]), 1)
            reduction = choices_combined.shape[0] / total_shape

            if reduction < best_reduction:
                best_reduction = reduction
                best_index = i

        if best_index is None:
            rank_var_and_choices.insert(0, a)
            break
        
        b = rank_var_and_choices.pop(best_index)
        rank_var_and_choices.insert(0, get_combined_choices(a, b, rank_var_and_choices, shape))

    # If we still have loops to combine, just combine them all
    while len(rank_var_and_choices) > 1:
        a, b = rank_var_and_choices.pop(0), rank_var_and_choices.pop(0)
        rank_var_and_choices.insert(0, get_combined_choices(a, b, rank_var_and_choices, shape))

    combined_choices = rank_var_and_choices[0][-1]
    is_symbol = rank_var_and_choices[0][2]
    indices = rank_var_and_choices[0][1]
    ranks = rank_var_and_choices[0][0]
    other_rank_var_and_choices = []  # All rank variables have been combined, so no others remain
    good_choices = check_valid_tile_shape(combined_choices, is_symbol, other_rank_var_and_choices, indices, ranks, shape)

    _, inverted_indices, is_symbol, choices = rank_var_and_choices[0]
    is_symbol = np.asarray(is_symbol)

    # Invert indices
    indices = invert_indices(inverted_indices)
    
    # print(f'Nominal n mappings: {nominal_n_mappings}')
    # print(f'Actual n mappings: {choices[:,indices].shape[0]}')
    # print(f'Ratio: {choices[:,indices].shape[0] / nominal_n_mappings}')
    # v = 1
    # for r in job.mask_ratios.values():
    #     v *= r
    # print(f'Product of ratios: {v}')
    # print(f'Product of ratios: {v * choices[:,indices].shape[0] / nominal_n_mappings}')


    return choices[:,indices], is_symbol[indices], total_pmappings

def invert_indices(inverted_indices):
    return np.argsort(inverted_indices)

def collect_tiling_segments(
    pmapping,
    rank_shape: dict,
    spec: "Specification",
    initial_delta_choices: dict[str, set[int]]={}
) -> dict[str, TilingSegment]:
    rank_var_to_tiling_segments = {}
    loop_idx = 0
    for node in pmapping:
        if isinstance(node, Temporal) or isinstance(node, Spatial):
            rank_var = node.rank_variable
            tile_shape = node.tile_shape
            tile_pattern = node.tile_pattern

            tiling_segment = rank_var_to_tiling_segments.setdefault(
                rank_var,
                TilingSegment(rank_shape[rank_var], spec.mapper.ffm.max_fused_loops_per_rank)
            )

            if tile_shape == 'symbol' or isinstance(tile_shape, sympy.Symbol):
                tiling_segment.add_symbol(loop_idx, node._fused)
            elif isinstance(tile_shape, int):
                tiling_segment.add_tile_shape(tile_shape, loop_idx, node._fused)
            elif isinstance(tile_pattern, Pattern):
                stride = tile_pattern.stride
                initial_tile_shape = tile_pattern.initial_tile_shape
                pattern_shape = tile_pattern.tile_shape
                if pattern_shape is not None:
                    raise ValueError('Recomputation not yet supported')
                assert stride is not None and initial_tile_shape is not None
                if isinstance(stride, int):
                    tiling_segment.add_tile_shape(stride, loop_idx, node._fused)
                elif isinstance(stride, sympy.Symbol):
                    tiling_segment.add_pattern(loop_idx,
                                               initial_delta_choices[rank_var], node._fused)
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

    # Max fused loops check
    n_loops = np.zeros(all_tile_shapes.shape[0], dtype=np.int64)
    for i in range(len(tiling_segments.is_fused)):
        if tiling_segments.is_fused[i]:
            prev = all_tile_shapes[:,i-1] if i > 0 else max_shape
            n_loops += all_tile_shapes[:,i] != prev
    all_tile_shapes = all_tile_shapes[n_loops <= tiling_segments.max_fused_loops,:]

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


def run_model(job: Job):
    pmapping = job.mapping
    spec = job.spec
    metrics = job.metrics
    track_memories = job.memories_track_all + job.memories_track_pmappings_only
    is_copy_op = job.is_copy_operation
    workload = spec.workload
    ert = spec.component_energy

    component_to_max_fanout = {}
    memory_to_size = {}
    for node in job.flattened_arch:
        if isinstance(node, architecture.TensorHolder):
            if isinstance(node, architecture.Memory):
                memory_to_size[node.name] = node.attributes.size
        component_to_max_fanout[node.name] = {s.name: s.fanout for s in node.spatial}

    df = {}

    reuse = analyze_reuse_and_add_reservations_to_mapping(job)
    overall_latency, comp_latency, mem_latency = get_latency(reuse, pmapping, workload, job.flattened_arch)
    actions = gather_actions(reuse, None, use_name=True)
    energy = compute_energy_from_actions(actions, ert, overall_latency)

    intermediate_tensors = workload.intermediate_tensor_names
    tensor_to_backing = {}
    for node in pmapping.nodes:
        if isinstance(node, TensorHolder):
            for tensor in node.tensors:
                if (
                    tensor not in tensor_to_backing
                    and tensor in intermediate_tensors
                ):
                    tensor_to_backing[tensor] = node.component

    total_occupancy = {}
    compute_unit = pmapping.nodes[-1].compute
    max_n_loops = 0
    
    for buffet, stats in reuse.buffet_stats.items():
        if buffet.level == compute_unit:
            continue

        occupancy = stats.max_occupancy
        
        if occupancy == 0:
            continue
        
        for tensor, backing in tensor_to_backing.items():
            if (is_copy_op or buffet.tensor == tensor) and buffet.level == backing:
                df[tensor2col(tensor)] = occupancy

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
                if memory in track_memories:
                    df[nameloop2col(memory, n_loop)] = running_total

    if metrics & Metrics.LATENCY:
        df[f'Total\0latency'] = overall_latency * spec.variables.global_cycle_period
        df[f'latency\0compute'] = comp_latency * spec.variables.global_cycle_period
        for component, latency in mem_latency.items():
            df[f'latency\0{component}'] = latency * spec.variables.global_cycle_period

    if metrics & Metrics.ENERGY:
        df[f'Total\0energy'] = sum(energy.values())
        for (component, action), energy in energy.items():
            df[f'energy\0{component}\0{action}'] = energy

    if metrics & Metrics.RESERVATIONS:
        for memory, occupancies in total_occupancy.items():
            df[f'reservations\0{memory}'] = sum(occupancies.values())

    per_memory_usage_df = {}
    for memory, occupancies in total_occupancy.items():
        per_memory_usage_df[memory] = sum(occupancies.values()) / memory_to_size[memory]

    utilization_df = {}
    for (component, einsum), per_dim_fanout in reuse.fanout.items():
        for dim, fanout in per_dim_fanout.items():
            utilization_df[f'utilization\0{component}\0{dim}'] = \
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
