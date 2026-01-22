import math
import islpy as isl

from fastfusion.frontend.renames import RankVariable
from fastfusion.frontend.workload import Workload, TensorName, Einsum, EinsumName


def get_einsum_operation_space(workload: Workload, einsum_name: str) -> isl.Set:
    """Return isl.Set of all operations in an einsum."""
    einsum_shape = workload.get_iteration_space_shape_isl_string(einsum_name)
    rank_variable_names = ",".join(
        map(str, workload.einsums[einsum_name].rank_variables)
    )
    try:
        return isl.Set(
            f"{{ {einsum_name}_operation[{rank_variable_names}] : {einsum_shape} }}"
        )
    except:
        raise Exception(f"Error creating isl.Set for {einsum_name}: {einsum_shape}")


def get_dim_bounds(isl_set: isl.Set) -> list[int]:
    bounds = []
    for i in range(isl_set.dim(isl.dim_type.set)):
        max_val = isl_set.dim_max_val(i)
        min_val = isl_set.dim_min_val(i)
        shape = max_val - min_val + 1  # max is inclusive
        try:
            bounds.append(shape.to_python())
        except:
            raise Exception(
                f"Shape is not an integer. Are all rank variables bounded? "
                f"Shape {shape} for rank variable {i} in {isl_set}"
            )
    return bounds


def get_rank_variable_bounds(
    workload: Workload, einsum_name: EinsumName
) -> dict[RankVariable, int]:
    """Return dictionary mapping rank variable name to bound."""
    operation_space = get_einsum_operation_space(workload, einsum_name)
    dim_shapes = get_dim_bounds(operation_space)
    return {
        rank_var: shape
        for rank_var, shape in zip(
            workload.einsums[einsum_name].rank_variables, dim_shapes
        )
    }


def get_projection_multi_aff(einsum: Einsum, tensor: TensorName) -> isl.MultiAff:
    """Return isl.MultiAff of projection from einsum to tensor."""
    rank_variables = einsum.rank_variables
    projection = einsum.tensor_accesses[tensor].projection

    rank_variables_str = ",".join(map(str, rank_variables))

    projection_str = ", ".join(
        f"{rank_name}={rank_projection}"
        for rank_name, rank_projection in projection.items()
    )

    return isl.MultiAff(
        f"{{ {einsum.name}_operation[{rank_variables_str}] -> "
        f"{tensor}[{projection_str}] }}"
    )


def get_projection_map(einsum: Einsum, tensor: TensorName) -> isl.Map:
    """Return isl.Map of projection from einsum to tensor."""
    return get_projection_multi_aff(einsum, tensor).as_map()


def get_tensor_data_space(workload: Workload, tensor: TensorName) -> isl.Set:
    """
    Get tensor data space based on the operation spaces of (for lack of
    a better term)'canonical' Einsums.

    Canonical Einsums (for this purpose) are all reader Einsums if the
    tensor is only ever read or all writer EInsums if the tensor is ever
    an output tensor.
    """
    writer_einsums = workload.einsums_with_tensor_as_output(tensor)
    if len(writer_einsums) == 0:
        reader_einsums = workload.einsums_with_tensor_as_input(tensor)
        canonical_einsums = reader_einsums
    else:
        canonical_einsums = writer_einsums

    tensor_data_space = None
    for einsum in canonical_einsums:
        operation_space = get_einsum_operation_space(workload, einsum.name)
        projection_map = get_projection_map(einsum, tensor)
        if tensor_data_space is None:
            tensor_data_space = operation_space.apply(projection_map)
        else:
            tensor_data_space = tensor_data_space.intersect(
                operation_space.apply(projection_map)
            )

    return tensor_data_space


def _card_box(data_space: isl.Set) -> int:
    dims = []
    for i in range(data_space.dim(isl.dim_type.set)):
        dim_min = data_space.dim_min(i)
        dim_max = data_space.dim_max(i)

        if dim_min.is_cst() and dim_max.is_cst():
            min_val = dim_min.as_aff().get_constant_val().to_python()
            max_val = dim_max.as_aff().get_constant_val().to_python()
        else:
            raise ValueError(f"Data space is not rectangular: {data_space}")

        dims.append(max_val - min_val + 1)

    return math.prod(dims)


ERRMSG = """ Non-box-shaped sets are not supported. This happens if ISL is installed without
Barvinok support. Please install ISL with Barvinok support, or use workloads with
rectangular data spaces and operation spaces. Non-rectangular spaces occur when the
workload contains complex expressions involving the rank variables, such as
multi-rank-variable inequalities.
Offending space: {space}.
"""


def get_tensor_size(workload: Workload, tensor: TensorName):
    """Get the size (num. of elements) of a tensor."""
    data_space = get_tensor_data_space(workload, tensor)
    if data_space.is_box():
        return _card_box(data_space)
    if not hasattr(data_space, "card"):
        raise RuntimeError(ERRMSG.format(space=str(data_space)))
    card_pwqp = isl.PwQPolynomial.card(data_space)
    return card_pwqp.eval(card_pwqp.domain().sample_point()).to_python()


def get_operation_space_size(workload: Workload, einsum_name: str):
    operation_space = get_einsum_operation_space(workload, einsum_name)
    if operation_space.is_box():
        return _card_box(operation_space)
    if not hasattr(operation_space, "card"):
        raise RuntimeError(ERRMSG.format(space=str(operation_space)))
    card_pwqp = isl.PwQPolynomial.card(operation_space)
    return card_pwqp.eval(card_pwqp.domain().sample_point()).to_python()
