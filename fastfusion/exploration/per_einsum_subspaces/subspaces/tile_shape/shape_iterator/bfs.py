"""
High-level idea:
- Create tile shapes for each rank. E.g., S_A, S_B, S_C for tile shapes
  of ranks A, B, C.
- The full space of tile shapes is S_A x S_B x S_C.
- However, we can evaluate S_A x Id_B x Id_C where Id_B and Id_C are
  a space of just one tile shape combination where all tile shapes are
  one for ranks B and C (Id from "identity").
- For tile shapes in S_A that violate capacity constraints, those can be
  safely pruned.

Using the BfsIterator class:
- Get the currently surviving tile shapes for rank `i` by calling
  `get_shapes_for_ranks`.
- Do evaluation.
- Report prunable choices by calling `prune_shapes_for_ranks`.
"""
from itertools import chain

import numpy as np
from combinatorics.integer import integer_factorizations_to_n_parts

from fastfusion.exploration.per_einsum_subspaces.subspaces.tile_shape.shape_subspace import ShapeSubspace


class BfsIterator:
    def __init__(self, shape_subspace: ShapeSubspace):
        self.shape_subspace = shape_subspace
        self.shapes = shape_subspace.rank_shapes
        self.ranks = shape_subspace.ranks

        self.tile_constraints = shape_subspace.tile_constraints
        self.factor_constraints = shape_subspace.factor_constraints

        self.crossed_ranks_to_shapes: dict[tuple, np.array] = {
            self._generate_shapes_for_rank((r,))
            for r in self.ranks
        }

    def cross_ranks(self, list_of_ranks_to_cross: list[tuple]):
        for ranks in list_of_ranks_to_cross:
            self._assert_ranks_in_crossed_ranks(ranks)

        shapes_of_each_set_of_ranks = []
        for ranks in list_of_ranks_to_cross:
            shapes_of_each_set_of_ranks.extend(
                self.get_shapes_for_ranks(ranks)
            )
            del self.crossed_ranks_to_shapes[ranks]

        new_combined_ranks = tuple(chain(*list_of_ranks_to_cross))
        self.crossed_ranks_to_shapes[new_combined_ranks] = np.meshgrid(
            *shapes_of_each_set_of_ranks
        )

        return self.crossed_ranks_to_shapes[new_combined_ranks]

    def get_shapes_for_ranks(self, ranks: tuple):
        self._assert_ranks_in_crossed_ranks(ranks)
        return self.crossed_ranks_to_shapes[ranks]

    def prune_shapes_for_ranks(self, ranks: tuple, pruned_mask):
        self._assert_ranks_in_crossed_ranks(ranks)
        self.crossed_ranks_to_shapes[ranks] = tuple(
            tile_choices[pruned_mask,:]
            for tile_choices in self.crossed_ranks_to_shapes[ranks]
        )

    def _assert_ranks_in_crossed_ranks(self, ranks: tuple):
        if ranks not in self.crossed_ranks_to_shapes:
            raise KeyError(f'{ranks} not in any of crossed ranks.')

    def _generate_shapes_for_rank(self, rank):
        rank_shape = self.shapes[rank]
        n_loops = sum(r == rank for r in self.ranks)
        factor_choices = integer_factorizations_to_n_parts(rank_shape,
                                                           n_loops)

        factor_choices = np.asarray(factor_choices, dtype=np.int64)
        factor_choices = self._gather_and_apply_relevant_constraints(
            rank,
            self.factor_constraints,
            factor_choices
        )

        tile_choices = rank_shape / np.cumprod(factor_choices, axis=1)
        tile_choices = self._gather_and_apply_relevant_constraints(
            rank,
            self.tile_constraints,
            tile_choices
        )

        return tile_choices

    def _gather_and_apply_relevant_constraints(
        self,
        rank,
        constraints,
        choice_array
    ):
        relevant_constraints = [
            constraint
            for other_rank, constraint in zip(self.ranks, constraints)
            if other_rank == rank
        ]
        for i, constraint in enumerate(relevant_constraints):
            valid_mask = constraint(choice_array[:,i])
            choice_array = choice_array[valid_mask,:]
        return choice_array
