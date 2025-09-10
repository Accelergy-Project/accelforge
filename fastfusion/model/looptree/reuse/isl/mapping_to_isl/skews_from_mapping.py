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
    Iteration,
    Spatial,
    Temporal,
    # Splits
    Pipeline,
    Sequential,
    # Logical hardware features
    Storage,
    Compute,
)
from fastfusion.frontend.workload.workload import TensorName, Workload

from fastfusion.model.looptree.mapping_utilities import get_paths
from fastfusion.model.looptree.reuse import ComponentName
from fastfusion.model.looptree.reuse.isl.isl_functions import (
    dim_projector_mask,
    insert_equal_dims_map,
)

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
    Given a mapping ...
    TODO: Fill this in
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
                    all_buffer_tensors.extend((buffer, tensor) for tensor in node.tensors)
                case Compute():
                    compute: ComponentName = node.compute
                    buffer_to_last_storage_node[compute] = node
                    buffer_node.append((compute, node))

        node_to_current_buffer: dict[MappingNode, ComponentName] = {}
        buffer_idx: int = 0
        for node in path:
            cur_buf, cur_buf_last_node = buffer_node[buffer_idx]
            node_to_current_buffer[node] = cur_buf

            if node == cur_buf_last_node:
                buffer_idx += 1

        # Generate tags, map, and which dims (and tags) should be removed per buffer.
        tags: List[Tag] = []
        removal_map: isl.Map = isl.Map.from_multi_aff(
            isl.MultiAff.identity_on_domain_space(
                isl.Space.alloc(isl.DEFAULT_CONTEXT, 0, 0, 0).domain()
            )
        )
        buffer_storage_past: set[tuple[ComponentName, TensorName]] = set()
        buffer_fully_complete: set[ComponentName] = set()
        buffer_to_dim_removal_mask: defaultdict[
            Tuple[ComponentName, TensorName], List[bool]
        ] = defaultdict(list)

        def add_tag(
            tag: Tag,
            mask_condition: Callable[[ComponentName], bool] = (
                lambda buffer: buffer in buffer_fully_complete
            ),
        ) -> None:
            """
            Performs necessary modifications to removal_map and removal_mask to
            accomodate tagging.

            :param tag:             The tag to add.
            :param mask_condition:  Boolean resolution for the removal mask.

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

            nonlocal all_buffer_tensors
            nonlocal buffer_to_dim_removal_mask
            for buffer_tensor in all_buffer_tensors:
                removal_mask = buffer_to_dim_removal_mask[buffer_tensor]
                buffer = buffer_tensor[0]
                removal_mask.append(mask_condition(buffer))

        for node in path:
            match node:
                case Storage():
                    buffer_storage_past.update((node.component, tensor) for tensor in node.tensors)
                    if node == buffer_to_last_storage_node[node.component]:
                        buffer_fully_complete.add(node.component)
                case Iteration():
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
                        lambda buffer: (
                            (buffer in buffer_fully_complete)
                            or (
                                isinstance(node, Temporal)
                                and (buffer in buffer_storage_past)
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

            buffer_tags: List[Tag] = [tags[i] for i in range(len(tags)) if mask[i]]

            # TODO: This buffet structure makes no sense in this context:
            # https://github.com/NVlabs/timeloop/blob/32370826fdf1aa3c8deb0c93e6b2a2fc7cf053aa/src/loop-analysis/mapping-to-isl/fused-mapping-to-isl.cpp#L740-L743
            buffer_tensor_einsum_to_skew[
                BufferTensorEinsum(*buffer_tensor, leaf)
            ] = Skew(buffer_tags, removal_projection)

        # TODO: Figure out what is actually:
        # https://github.com/NVlabs/timeloop/blob/32370826fdf1aa3c8deb0c93e6b2a2fc7cf053aa/src/loop-analysis/mapping-to-isl/fused-mapping-to-isl.cpp#L746
        compute_einsum_to_skew[ComputeEinsum(leaf.compute, leaf)] = Skew(
            tags, removal_map
        )
        einsum: EinsumName = leaf.einsum
        for tensor in workload.tensors_read_by_einsum(einsum):
            buffer_tensor_einsum_to_skew[
                BufferTensorEinsum(leaf.compute, tensor, leaf)
            ] = Skew(tags, removal_map)

        for tensor in workload.tensors_written_by_einsum(einsum):
            buffer_tensor_einsum_to_skew[
                BufferTensorEinsum(leaf.compute, tensor, leaf)
            ] = Skew(tags, removal_map)

    return SkewsInfo(buffer_tensor_einsum_to_skew, compute_einsum_to_skew)


def skew_from_path(
    mapping_path, workload, accumulator: dict[BufferTensorEinsum, Skew]
) -> None:
    """
    Get compute and buffer skews in a path and collect them in accumulator.
    """
    einsum_name = mapping_path[-1]["einsum"]

    bte_and_idx = []
    all_tags = []
    cur_idx = 0
    for node in mapping_path:
        if node["type"] == "storage":
            bte = BufferTensorEinsum(node["target"], node["tensor"], einsum_name)
            bte_and_idx.append((bte, cur_idx))
        if node["type"] not in ["compute", "storage"]:
            all_tags.append(make_tag_from_node(node))
            cur_idx += 1

    # Make { [i0, i1, ..., iN] -> [0] } where N = len(einsum_name)-1
    iter_space_str = ", ".join(f"i{i}" for i in range(len(all_tags)))
    iter_space = isl.Space(f"{{ [{iter_space_str} ] }}")

    iteration_to_rank_variables = {
        rank_var: isl.PwAff(iter_space.zero_aff_on_domain())
        for rank_var in workload.EinsumOspaceDimensions()
    }
    for node in mapping_path:
        if node["type"] in ["spatial", "temporal"]:
            # Insert iteration variable to the left and update projection
            iter_to_rank_var = iteration_to_rank_variables[node["rank"]]
            if "tile_shape" in node:
                raise NotImplementedError()
            elif "factor" in node:
                raise NotImplementedError()
            else:
                raise NotImplementedError()
        elif node["type"] == "sequential":
            raise NotImplementedError()
        elif node["type"] == "pipeline":
            raise NotImplementedError()

    for bte, idx in bte_and_idx:
        # TODO
        accumulator[bte] = Skew(all_tags[:idx], skew_isl[:idx])


def make_tag_from_node(node):
    if node["type"] == "temporal":
        return TemporalTag()
    elif node["type"] == "spatial":
        return SpatialTag(
            node.get("spatial_dim", default=None), node.get("buffer", default=None)
        )
    elif node["type"] == "pipeline":
        return PipelineTag(
            node.get("spatial_dim", default=None), node.get("buffer", default=None)
        )
    elif node["type"] == "sequential":
        return SequentialTag()
    else:
        raise ValueError(f"Unsupported node type {node}")
