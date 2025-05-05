from .constraints import parse_tile_constraint, parse_factor_constraint


class ShapeSubspace:
    def __init__(self,
                 rank_shapes: dict[int, int],
                 ranks: list[int],
                 tile_constraints: list[list[str]]=None,
                 factor_constraints: list[list[str]]=None,
                 n_fusion_relevant_loops: int=None):
        self.rank_shapes = rank_shapes
        self.ranks = ranks
        
        self.position_to_last = {}
        self.n_fusion_relevant_loops = n_fusion_relevant_loops
        self._fill_position_to_last()
        # print(f'Made shape subspace with tile constraints {self.tile_constraints} and factor constraints {self.factor_constraints}')

        self.tile_constraints = [[] for _ in self.ranks]
        self._parse_tile_constraints(self, tile_constraints)

        self.factor_constraints = [[] for _ in self.ranks]
        self._parse_factor_constraints(self, factor_constraints)

    def _fill_position_to_last(self):
        self.rank_to_last = {}
        for i, r in enumerate(self.ranks):
            if r in self.rank_to_last:
                self.position_to_last[i] = self.rank_to_last[r]
            else:
                self.position_to_last[i] = None
            self.rank_to_last[r] = i

    def _parse_tile_constraints(self, tile_constraints):
        if tile_constraints is not None:
            for i, constraints in enumerate(tile_constraints):
                for constraint in constraints:
                    constraint, upward_prop_constraint = parse_tile_constraint(constraint)
                    self._add_tile_constraint(i, constraint, upward_prop_constraint)

    def _parse_factor_constraints(self, factor_constraints):
        if factor_constraints is not None:
            for i, constraints in enumerate(factor_constraints):
                for constraint in constraints:
                    constraint, upward_prop_constraint = parse_factor_constraint(constraint)
                    self._add_factor_constraint(i, constraint, upward_prop_constraint)

    def _add_tile_constraint(self, loop_idx, constraint, propagated_constraint):
        self.tile_constraints[loop_idx].append(constraint)
        if propagated_constraint is not None:
            last = self.position_to_last[loop_idx]
            while last is not None:
                self.tile_constraints[last].append(propagated_constraint)
                last = self.position_to_last[last]

    def _add_factor_constraint(self, loop_idx, constraint, propagated_constraint):
        self.factor_constraints[loop_idx].append(constraint)
        if propagated_constraint is not None:
            last = self.position_to_last[loop_idx]
            while last is not None:
                self.factor_constraints[last].append(propagated_constraint)
                last = self.position_to_last[last]