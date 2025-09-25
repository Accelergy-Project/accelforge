from dataclasses import dataclass
from typing import Any
from functools import reduce
from operator import mul

import sympy

from .workload import (
    TensorName,
    Einsum,
    EinsumName,
    Workload,
    RankName,
    RankVariableName,
)
from ._isl import get_rank_variable_bounds


def get_projection_expr(einsum: Einsum, tensor: TensorName) -> dict[str, sympy.Expr]:
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


def get_rank_variable_relevancy(einsum: Einsum, tensor: TensorName):
    relevancy = {}
    projection = einsum.tensor_accesses[tensor].projection
    for rank_variable in einsum.rank_variables:
        relevancy[rank_variable] = Irrelevant()
        for rank_name, projection_str in projection.items():
            projection_expr = sympy.parsing.sympy_parser.parse_expr(projection_str)
            is_simple = len(sympy.Add.make_args(projection_expr)) == 1
            is_relevant = (
                sympy.symbols(f"{rank_variable}") in projection_expr.free_symbols
            )

            if not is_relevant:
                continue

            if is_simple:
                relevancy[rank_variable] = Relevant(rank=rank_name)
            else:
                relevancy[rank_variable] = PartiallyRelevant(rank=rank_name)

            break
    return relevancy


def compute_dense_tile_occupancy(
    projection_expr: dict[str, sympy.Expr], rank_variable_shapes: dict
):
    substitutions = [
        (rank_variable, rank_variable_shape - 1)
        for rank_variable, rank_variable_shape in rank_variable_shapes.items()
    ]
    return reduce(
        mul,
        [index_expr.subs(substitutions) + 1 for index_expr in projection_expr.values()],
    )


def compute_rank_occupancy(projection_expr: sympy.Expr, rank_variable_shapes: dict):
    substitutions = [
        (rank_variable, rank_variable_shape - 1)
        for rank_variable, rank_variable_shape in rank_variable_shapes.items()
    ]
    return projection_expr.subs(substitutions) + 1


def get_stride_and_halo_of_einsum(
    einsum_name: str,
    workload: Workload,
    rank_variable_bounds: dict[RankVariableName, int] | None = None,
) -> dict[TensorName, dict[tuple[RankName, RankVariableName]], tuple[int, int]]:
    """
    Get stride and halo (initial delta) for an Einsum in workload.

    Returns dictionary mapping tensor to another dictionary mapping
    (rank, rank_var) to the stride and halo.
    """
    stride_and_halo = {}
    einsum = workload.einsums[einsum_name]
    if rank_variable_bounds is None:
        shape = get_rank_variable_bounds(workload, einsum_name)
    else:
        shape = rank_variable_bounds
    for tensor in einsum.tensor_names:
        stride_and_halo[tensor] = {}
        tensor_stride_and_halo = stride_and_halo[tensor]

        projection = get_projection_expr(einsum, tensor)
        tensor_accesses = einsum.tensor_accesses[tensor]
        for rank, rank_vars in tensor_accesses.rank2rank_variables.items():
            rank_projection = projection[rank]
            for rank_var in rank_vars:
                stride = rank_projection.coeff(rank_var)

                # Careful: in-place mutation of cons_shape
                original_shape = shape[rank_var]
                shape[rank_var] = 1
                halo = compute_rank_occupancy(rank_projection, shape) - 1
                shape[rank_var] = original_shape

                tensor_stride_and_halo[(rank, rank_var)] = (stride, halo)
    return stride_and_halo


def get_stride_and_halo(
    workload: Workload,
) -> dict[
    tuple[EinsumName, TensorName],
    dict[tuple[RankName, RankVariableName], tuple[int, int]],
]:
    """
    Get stride and halo (initial delta) for Einsums in workload.

    Returns dictionary mapping (Einsum, tensor) to another dictionary mapping
    (rank, rank_var) to the stride and halo.
    """
    stride_and_halo = {}
    for einsum in workload.einsums:
        stride_and_halo_of_einsum = get_stride_and_halo_of_einsum(einsum.name, workload)
        for tensor, ranks2stride_and_halo in stride_and_halo_of_einsum.items():
            stride_and_halo[(einsum.name, tensor)] = ranks2stride_and_halo
    return stride_and_halo
