from dataclasses import dataclass


@dataclass
class MacArrayConstraint:
    array_shape_in_parallel_dimension: str
    array_shape_in_reduced_dimension: str

    weight_tensor: dict[str, str]
    parallel_rank: dict[str, str]
    reduced_rank: dict[str, str]


@dataclass
class PeArrayConstraint:
    array_shape: int


WILDCARD = '*'
SEPARATOR = '/'


@dataclass
class PerEinsumDataflowConstraint:
    disallowed_ranks: set
    rank_order: list

    @staticmethod
    def parse(pattern: list[str]):
        if SEPARATOR in pattern:
            separator_idx = pattern.index(SEPARATOR)
            disallowed_ranks = set(pattern[:separator_idx])
            rank_order = pattern[separator_idx+1:]
        else:
            separator_idx = 0
            disallowed_ranks = set()
            rank_order = pattern
        return PerEinsumDataflowConstraint(disallowed_ranks, rank_order)


@dataclass
class DataflowConstraint:
    # From EinsumId to PerEinsumDataflowConstraint
    einsum_to_constraint: dict[int, PerEinsumDataflowConstraint]

    @staticmethod
    def parse(pattern: dict[str, list[str]]):
        return DataflowConstraint({
            einsum_name : PerEinsumDataflowConstraint.parse(constraint)
            for einsum_name, constraint in pattern.items()
        })
