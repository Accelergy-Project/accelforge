import islpy as isl

from .workload import Workload, TensorName, Einsum


def get_einsum_operation_space(workload: Workload, einsum_name: str) -> isl.Set:
    """Return isl.Set of all operations in an einsum."""
    einsum_shape = workload.get_shape_isl_string(einsum_name)
    rank_variable_names = ",".join(
        map(str, workload.einsums[einsum_name].rank_variables)
    )
    try:
        return isl.Set(f"{{ [{rank_variable_names}] : {einsum_shape} }}")
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


def get_rank_variable_bounds(workload: Workload, einsum_name: str) -> dict[str, int]:
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

    return isl.MultiAff(f"{{ [{rank_variables_str}] -> [{projection_str}] }}")


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
    writer_einsums = workload.einsums_that_write_tensor(tensor)
    if len(writer_einsums) == 0:
        reader_einsums = workload.einsums_that_read_tensor(tensor)
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


def get_tensor_size(workload: Workload, tensor: TensorName):
    """Get the size (num. of elements) of a tensor."""
    data_space = get_tensor_data_space(workload, tensor)
    card_pwqp = data_space.card()
    return card_pwqp.eval(card_pwqp.domain().sample_point()).to_python()
