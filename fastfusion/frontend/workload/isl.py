import islpy as isl

from .spec import Workload


def get_einsum_operation_space(workload: Workload, einsum_name: str) -> isl.Set:
    einsum_shape = workload.get_shape_isl_string(einsum_name)
    rank_variable_names = ','.join(workload.einsums[einsum_name].rank_variables)
    return isl.Set(f'{{ [{rank_variable_names}] : {einsum_shape} }}')


def get_dim_bounds(isl_set: isl.Set) -> [int]:
    bounds = []
    for i in range(isl_set.dim(isl.dim_type.set)):
        max_val = isl_set.dim_max_val(i)
        min_val = isl_set.dim_min_val(i)
        shape = max_val - min_val + 1  # max is inclusive
        bounds.append(shape.to_python())
    return bounds


def get_rank_variable_bounds(
    workload: Workload,
    einsum_name: str
) -> dict[str, int]:
    operation_space = get_einsum_operation_space(workload, einsum_name)
    dim_shapes = get_dim_bounds(operation_space)
    return {
        rank_var: shape
        for rank_var, shape in zip(workload.einsums[einsum_name].rank_variables,
                                   dim_shapes)
    }