from collections.abc import Callable
from itertools import combinations, permutations
from fastfusion.exploration.constraints import PerEinsumDataflowConstraint, WILDCARD
from fastfusion.exploration.per_einsum_subspaces.subspaces.linear_mapping import LinearMapping


def make_spatial_fors(mapping: LinearMapping,
                      ranks,
                      max_factor,
                      min_loops=None,
                      unordered=False):
    original = mapping.copy()

    if unordered:
        combinatorial_func = combinations
    else:
        combinatorial_func = permutations

    if min_loops is None:
        min_loops = len(ranks)

    for r in range(min_loops, len(ranks) + 1):
        for ordered_ranks in combinatorial_func(ranks, r=r):
            mapping = original.copy()
            for r in ordered_ranks:
                mapping.add_spatial(
                    r, factor_constraint=f"<={max_factor}"
                )
            yield mapping


def make_temporal_fors(mapping: LinearMapping,
                       ranks,
                       dataflow_constraint: PerEinsumDataflowConstraint=None,
                       logfunc: Callable=None,
                       min_loops=None,
                       unordered=False):
    if dataflow_constraint is None:
        top_ranks = []
        other_ranks = ranks
    else:
        disallowed_ranks = dataflow_constraint.disallowed_ranks
        rank_order = dataflow_constraint.rank_order

        num_wildcards = sum(map(lambda x: x is WILDCARD, rank_order))
        top_ranks = []
        if num_wildcards == 0:
            top_ranks = rank_order
            other_ranks = set()
        elif num_wildcards == 1:
            top_ranks = rank_order[:-1]
            other_ranks = set(ranks) - set(top_ranks) - disallowed_ranks
        else:
            raise NotImplementedError("Constraint not implemented")

    if unordered:
        combinatorial_func = combinations
    else:
        combinatorial_func = permutations

    if min_loops is None:
        min_loops = len(ranks)

    original = mapping.copy()
    for i in range(len(top_ranks)+1):
        actual_min_loops = min(0, min_loops-i)
        for j in range(actual_min_loops, len(other_ranks) + 1):
            for ordered_ranks in combinatorial_func(other_ranks, r=j):
                mapping = original.copy()
                for r in top_ranks[:i]:
                    mapping.add_temporal(r)
                for r in ordered_ranks:
                    mapping.add_temporal(r)
                yield mapping


def make_temporal_fors_with_smallest_tile(original: LinearMapping,
                                          ranks,
                                          unordered=False):
    if not unordered:
        for ordered_ranks in permutations(ranks):
            mapping = original.copy()
            if len(ordered_ranks) != len(ranks):
                continue
            for r in ordered_ranks:
                mapping.add_temporal(r, tile_shape=1)
            yield mapping
    else:
        mapping = original.copy()
        for r in ranks:
            mapping.add_temporal(r, tile_shape=1)
        yield mapping


def make_temporal_fors_in_order(original: LinearMapping, ranks):
    for i in range(len(ranks)+1):
        mapping = original.copy()
        
        for r in ranks[:i]:
            mapping.add_temporal(r)
        yield mapping
