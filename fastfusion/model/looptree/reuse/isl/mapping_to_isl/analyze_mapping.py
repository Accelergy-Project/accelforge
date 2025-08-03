"""
Flow of analysis:
-   From mapping, create the iteration space. The iteration space is the
    space of iterators in the mapping.
-   Create the relation from iteration space to operation space.
-   Create the relation from the iteration space to tensor space for each
    (buffer, tensor, einsum) tuple.
-   Run tile shape inference.

Adapted from: https://github.com/NVlabs/timeloop/blob/4cf6d4cd043bc2a5d2eb02afa9063d7117a4dc11/src/loop-analysis/mapping-to-isl/mapping-to-isl.cpp#L261
-   DataspaceId -> TensorName
-   LogicalBuffer -> Buffet
"""
from dataclasses import dataclass, field
from collections.abc import Sequence, Set
from collections.abc import Mapping as HashMap

import islpy as isl

from fastfusion.frontend.mapper import Mapping
from fastfusion.frontend.mapping import Nested, node_list
from fastfusion.frontend.workload.workload import Workload, RankVariableName, TensorName

import fastfusion.model.looptree.reuse.isl.isl_functions as isl_help
from fastfusion.model.looptree.reuse.isl.mapping_to_isl.types import *

@dataclass
class MappingInISL:
    occupancies: dict[BufferTensorEinsum, Occupancy] = field(default_factory=dict)
    buffer_to_skews: dict[BufferTensorEinsum, Skew] = field(default_factory=dict)
    compute_to_skew: dict[ComputeEinsum, Skew] = field(default_factory=dict)


def operations_to_tensor_from_einsum(
    workload: Workload
) -> HashMap[TensorName, isl.Map]:
    """
    Transforms operations in an einsum into their tensor equivalents.
    """
    tensor_name_to_operation_space_to_tensor: HashMap[TensorName, isl.Map]
    print(workload.shape)
    
    # Iterates through all the tensor names in the workload.
    tensor_name: TensorName
    for tensor_name in workload.shape:
        tensor_order: int = workload.shape.tensor_order[tensor_name]
        projection: str = workload.shape.projections[tensor_name]
        
        # Creates the space and affine representing the einsums.
        space: isl.Space = isl.Space.alloc(
            isl.DEFAULT_CONTEXT, 0, workload.shape.num_factorized_dims, tensor_order
        )
        multi_aff: isl.MultiAff = space.zero_multi_aff()

        # Iterates through all tensor dimensions and defines them in the affine.
        tensor_dim: int
        for tensor_dim in range(tensor_order):
            aff: isl.Aff = space.domain().zero_aff_on_domain()
            
            # Adds each term of the tensor on this dimension to the affine.
            for term in projection[tensor_dim]:
                coef_id: int = term[0]
                factorized_dim_id: int = term[1]

                # Affine only has non-unary coefficients on non-constant terms.
                if coef_id != workload.shape.num_coefficients:
                    aff = aff.set_coefficient_val(
                        isl.dim_type.in_, factorized_dim_id, workload.coefficients[coef_id]
                    )
                # Last term is a constant.
                else:
                    aff = aff.set_coefficient_val(
                        isl.dime_type.in_, factorized_dim_id, 1
                    )
            
            # Updates the multiaffine for this tensor.
            multi_aff = multi_aff.set_at(tensor_dim, aff)
        
        # Associates the name to the tensorified einsum operation.
        tensor_name_to_operation_space_to_tensor[tensor_name] = isl.Map.from_multi_aff(
            multi_aff
        )

    return tensor_name_to_operation_space_to_tensor


def tiling_from_mapping(nest: Nested) -> BranchTilings:
    """Given a mapping, compute the tilings for that mapping."""
    operation_space_dims: Set[int] = set()

    loops: node_list = nest.nodes
    for loop in loops:
        operation_space_dims.add(loop.dimension)
    
    tiling: isl.Map
    for dim in operation_space_dims:
        coefficents: Sequence[int] = []
        sizes: Sequence[int] = []
        residuals: Sequence[int] = []

        coefficient: int = 1
        for loop in loops:
            if dim != loop.dimension:
                continue
            coefficents.append(coefficient)
            sizes.append(loop.loop_bound)
            
            if loop.residual 
        


def occupancies_from_mapping(
    mapping: Mapping, workload: Workload
) -> HashMap[BufferTensorEinsum, Occupancy]:
    """
    Takes a mapping and a workload and calculates the data occupancies per storage
    unit per time step.
    """
    ops_to_tensor: HashMap[TensorName, isl.Map] = operations_to_tensor_from_einsum(workload)
    branch_tiling: BranchTilings = tiling_from_mapping(mapping).at(0)
    buf_skew: HashMap[Buffet, Skew] = logical_buf_skews_from_mapping(mapping, workload)

    result: HashMap[BufferTensorEinsum, Occupancy] = {}
    buf: Buffet
    skew: Skew
    for buf, skew in buf_skew.items():
        occupancy: isl.Map = skew.map.apply_range(
            isl_help.project_dim_in_after(
                branch_tiling.apply_range(ops_to_tensor.at(buf.tensor_name)),
                isl_help.dim(skew.map, isl.dim_type.out)   
            )
        )
        result[buf] = Occupancy(skew.dim_in_tags, occupancy)
    
    return occupancy


def 