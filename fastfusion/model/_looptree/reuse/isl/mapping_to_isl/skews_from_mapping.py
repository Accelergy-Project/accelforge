"""
Relevant Name Changes:
-   BufferID -> ComponentName
"""

from collections import defaultdict
from typing import Callable, List, Tuple
import islpy as isl

from fastfusion.frontend.mapping import (
    Mapping,
    MappingNode,
    # Iterations
    Loop,
    Spatial,
    Temporal,
    # Splits
    Pipeline,
    Sequential,
    # Logical hardware features
    Storage,
    Compute,
)
from fastfusion.frontend.workload import TensorName, Workload

from fastfusion.model._looptree.mapping_utilities import get_paths
from fastfusion.model._looptree.types import ComponentName
from fastfusion.model._looptree.reuse.isl.isl_functions import (
    dim_projector_mask,
    insert_equal_dims_map,
)
from fastfusion.model._looptree.reuse.isl.mapping_to_isl import DUMP_ISL_IR

from .types import (
    # Bookkeeping objects
    BufferTensorEinsum,
    ComputeEinsum,
    EinsumName,
    Skew,
    SkewsInfo,
    # Tags
    Tag,
    TemporalTag,
    SpatialTag,
    PipelineTag,
    SequentialTag,
)


def skews_from_mapping(mapping: Mapping, workload: Workload) -> SkewsInfo:
    """
    Given a mapping and workload, compute the skew relationships for buffers and
    computes.
    TODO: Fill this in with more accurate description.

    Parameters
    ----------
    mapping:
        The mapping being analyzed.
    workload:
        The workload being executed.

    Returns
    -------
    Skew information for buffer-tensor-einsum and compute-einsum combinations.
    """
    compute_einsum_to_skew: dict[ComputeEinsum, Skew] = defaultdict()
    buffer_tensor_einsum_to_skew: dict[BufferTensorEinsum, Skew] = defaultdict()

    for path in get_paths(mapping):
        leaf: Compute = path[-1]

        # Get the last storage node in path for a particular buffet.
        buffer_to_last_storage_node: dict[ComponentName, MappingNode] = {}
        buffer_node: List[Tuple[ComponentName, MappingNode]] = []
        all_buffer_tensors: List[Tuple[ComponentName, TensorName]] = []

        node: MappingNode
        for node in path:
            match node:
                case Storage():
                    buffer: ComponentName = node.component
                    buffer_to_last_storage_node[buffer] = node
                    buffer_node.append((buffer, node))
                    # TODO: Check this is correct
                    all_buffer_tensors.extend(
                        (buffer, tensor) for tensor in node.tensors
                    )
                case Compute():
                    compute: ComponentName = node.compute
                    buffer_to_last_storage_node[compute] = node
                    buffer_node.append((compute, node))

        node_to_current_buffer: dict[MappingNode, MappingNode] = {}
        buffer_idx: int = 0
        for node in path:
            _, cur_buf_last_node = buffer_node[buffer_idx]
            node_to_current_buffer[node] = cur_buf_last_node

            if node == cur_buf_last_node:
                buffer_idx += 1

        # Generate tags, map, and which dims (and tags) should be removed per buffer.
        tags: List[Tag] = []
        base_space: str = f"{leaf.compute}_spacetime"
        removal_map: isl.Map = (
            isl.Map.from_multi_aff(
                isl.MultiAff.identity_on_domain_space(
                    isl.Space.alloc(isl.DEFAULT_CONTEXT, 0, 0, 0).domain()
                )
            )
            .set_tuple_name(isl.dim_type.in_, base_space)
            .set_tuple_name(isl.dim_type.out, base_space)
        )

        buffer_storage_past: set[Tuple[ComponentName, TensorName]] = set()
        buffer_fully_complete: set[ComponentName] = set()
        buffer_to_dim_removal_mask: defaultdict[
            Tuple[ComponentName, TensorName], List[bool]
        ] = defaultdict(list)

        def add_tag(
            tag: Tag,
            mask_condition: Callable[[ComponentName, TensorName], bool] = (
                lambda b, t: b in buffer_fully_complete
            ),
        ) -> None:
            """
            Performs necessary modifications to removal_map and removal_mask to
            accommodate tagging.

            Parameters
            ----------
            tag:
                The tag to add.
            mask_condition:
                Boolean resolution for the removal mask.

            Postconditions
            --------------
            -   `tags` has another tag appended to it.
            -   `removal_map` has an input and output dimension added that are equal
                to each other.
            -   `removal_mask` has a new entry.
            """
            nonlocal tags
            tags.append(tag)
            nonlocal removal_map
            removal_map = insert_equal_dims_map(
                removal_map,
                removal_map.dim(isl.dim_type.in_),
                removal_map.dim(isl.dim_type.out),
                1,
            )
            if DUMP_ISL_IR:
                print(f"skew removal_map: {removal_map}")
                print(f"tag: {tag}")

            nonlocal all_buffer_tensors
            nonlocal buffer_to_dim_removal_mask
            for buffer_tensor in all_buffer_tensors:
                removal_mask = buffer_to_dim_removal_mask[buffer_tensor]
                removal_mask.append(mask_condition(*buffer_tensor))

        for node in path:
            match node:
                case Storage():
                    buffer_storage_past.update(
                        (node.component, tensor) for tensor in node.tensors
                    )
                    if node == buffer_to_last_storage_node[node.component]:
                        buffer_fully_complete.add(node.component)
                case Loop():
                    tag: Tag
                    if isinstance(node, Temporal):
                        tag: Tag = TemporalTag()
                    elif isinstance(node, Spatial):
                        tag: Tag = SpatialTag(0, node_to_current_buffer[node])
                    else:
                        raise ValueError(
                            f"Type {type(node)} is an iteration not in space or time."
                        )

                    # TODO: Verify logical equivalence to:
                    # https://github.com/NVlabs/timeloop/blob/32370826fdf1aa3c8deb0c93e6b2a2fc7cf053aa/src/loop-analysis/mapping-to-isl/fused-mapping-to-isl.cpp#L660-L671
                    add_tag(
                        tag,
                        lambda b, t: (
                            (b in buffer_fully_complete)
                            or (
                                (b, t) in buffer_storage_past
                                and isinstance(node, Temporal)
                            )
                        ),
                    )
                case Pipeline():
                    add_tag(PipelineTag())
                case Sequential():
                    add_tag(SequentialTag())

        for buffer_tensor in all_buffer_tensors:
            mask: List[bool] = buffer_to_dim_removal_mask[buffer_tensor]
            domain: isl.Set = removal_map.domain()
            projector: isl.Map = dim_projector_mask(domain.get_space(), mask)
            removal_projection: isl.Map = projector.apply_range(removal_map)
            # Attach tuple names per-buffer so downstream occupancy maps keep the spacetime label.
            space_name: str = f"{buffer_tensor[0]}_spacetime"
            removal_projection = removal_projection.set_tuple_name(
                isl.dim_type.in_, space_name
            ).set_tuple_name(isl.dim_type.out, space_name)

            buffer_tags: List[Tag] = [tag for i, tag in enumerate(tags) if not mask[i]]

            # TODO: This buffet structure makes no sense in this context:
            # https://github.com/NVlabs/timeloop/blob/32370826fdf1aa3c8deb0c93e6b2a2fc7cf053aa/src/loop-analysis/mapping-to-isl/fused-mapping-to-isl.cpp#L740-L743
            buffer_tensor_einsum_to_skew[BufferTensorEinsum(*buffer_tensor, leaf)] = (
                Skew(buffer_tags, removal_projection)
            )

        # TODO: Figure out what is actually:
        # https://github.com/NVlabs/timeloop/blob/32370826fdf1aa3c8deb0c93e6b2a2fc7cf053aa/src/loop-analysis/mapping-to-isl/fused-mapping-to-isl.cpp#L746
        compute_einsum_to_skew[ComputeEinsum(leaf.compute, leaf)] = Skew(
            tags, removal_map
        )
        einsum: EinsumName = leaf.einsum
        for tensor in workload.einsums[einsum].input_tensor_names:
            buffer_tensor_einsum_to_skew[
                BufferTensorEinsum(leaf.compute, tensor, leaf)
            ] = Skew(tags, removal_map)

        for tensor in workload.einsums[einsum].output_tensor_names:
            buffer_tensor_einsum_to_skew[
                BufferTensorEinsum(leaf.compute, tensor, leaf)
            ] = Skew(tags, removal_map)

    return SkewsInfo(buffer_tensor_einsum_to_skew, compute_einsum_to_skew)
