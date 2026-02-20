from dataclasses import dataclass
import numpy as np

from accelforge.frontend.mapping import Mapping, Compute, Temporal, Spatial, Split, Nested
from accelforge.frontend.workload import RankVariable, Rank, Workload


class TemporalTrace:
    rank_variable: RankVariable


def trace_iterations(mapping: Mapping, spec):
    return _trace_iterations(mapping, spec.workload)


def _trace_iterations(node, workload: Workload):
    if isinstance(node, Nested):
        if isinstance(node.nodes[-1], (Nested, Split)):
            last_result = _trace_iterations(node.nodes[-1], workload)
        elif isinstance(node.nodes[-1], Compute):
            last_result = _trace_node(node.nodes[-1], workload, None)

        for child_node in reversed(node.nodes[:-1]):
            last_result = _trace_node(child_node, workload, last_result)
        return last_result
    elif isinstance(node, Split):
        last_result = {}
        for child_node in node.nodes:
            if isinstance(child_node, Compute):
                child_result = _trace_node(child_node, workload, None)
            elif isinstance(child_node, (Nested, Split)):
                child_result = _trace_iterations(child_node, workload)
            else:
                raise NotImplementedError()

            in_child = set(child_result.keys())
            in_last = set(last_result.keys())
            child_shape = None
            last_shape = None
            for rank_var in in_child & in_last:
                child_trace = child_result[rank_var]
                child_shape = child_trace.shape
                last_trace = last_result[rank_var]
                last_shape = last_trace.shape
                if rank_var in last_result:
                    last_result[rank_var] = np.concatenate([last_trace, child_trace])
                else:
                    last_result[rank_var] = child_trace
            for rank_var in in_last - in_child:
                last_trace = last_result[rank_var]
                nans = np.empty(child_shape)
                nans.fill(np.nan)
                last_result[rank_var] = np.concatenate([last_trace, nans])
            for rank_var in in_child - in_last:
                child_trace = child_result[rank_var]
                nans = np.empty(last_shape)
                nans.fill(np.nan)
                last_result[rank_var] = np.concatenate([nans, child_trace])
        return last_result


def _trace_node(node, workload: Workload, last_result):
    if isinstance(node, Spatial):
        raise NotImplementedError("Does not handle spatial for now.")
    elif isinstance(node, Temporal):
        rank_var = node.rank_variable
        tile_pattern = node.tile_pattern
        tile_shape = tile_pattern.tile_shape
        initial_tile_shape = tile_pattern.initial_tile_shape
        if initial_tile_shape is not None and initial_tile_shape != tile_shape:
            raise NotImplementedError("Does not handle imperfect for now")
        n_iterations = int(tile_pattern.calculated_n_iterations)
        last_shape = next(iter(last_result.values())).shape[0]
        next_result = {
            rank_var: np.tile(last_trace, n_iterations)
            for rank_var, last_trace in last_result.items()
        }
        next_result[rank_var] += np.repeat(
            np.arange(n_iterations)*tile_shape,
            last_shape
        )
        return next_result
    elif isinstance(node, Compute):
        einsum = workload.einsums[node.einsum]
        return {rank_var: np.ones((1,)) for rank_var in einsum.rank_variables}
    else:
        return last_result
