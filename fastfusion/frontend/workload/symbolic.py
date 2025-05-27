from dataclasses import dataclass
from typing import Any

import sympy

from .workload import TensorName, Einsum


def get_projection_expr(einsum: Einsum, tensor: TensorName):
    projection = einsum.tensor_accesses[tensor].projection
    return {
        rank_name: sympy.parsing.sympy_parser.parse_expr(proj_str)
        for rank_name, proj_str in projection.items()
    }


class Irrelevant:
    pass

@dataclass
class Relevant:
    rank: Any

@dataclass
class PartiallyRelevant:
    rank: Any


def get_rank_variable_relevancy(einsum: Einsum,
                                tensor: TensorName):
    relevancy = {}
    projection = einsum.tensor_accesses[tensor].projection
    for rank_variable in einsum.rank_variables:
        relevancy[rank_variable] = Irrelevant()
        for rank_name, projection_str in projection.items():
            projection_expr = sympy.parsing.sympy_parser.parse_expr(projection_str)
            is_simple = sympy.Add.make_args(projection_expr)
            is_relevant = sympy.symbols(f'{rank_variable}') in projection_expr.free_symbols

            if not is_relevant:
                continue

            if is_simple:
                relevancy[rank_variable] = Relevant(rank=rank_name)
            elif is_simple:
                relevancy[rank_variable] = PartiallyRelevant(rank=rank_name)

            break
    return relevancy