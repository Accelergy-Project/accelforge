"""
File for all the functions that conduct tiling analysis for the overall mapping
analysis.
"""

from collections import defaultdict, deque
from typing import List, Tuple

from pprint import pformat, pprint

import islpy as isl

from fastfusion.frontend.mapping import (
    # Types
    MappingNode,
    # Mapping objects
    Mapping,
    MappingNodeWithChildren,
    # Physical object types in Mappings.
    Compute,
    Storage,
    # Logical object types in Mappings.
    Iteration,
    Spatial,
    Temporal,
    Split,
)
from fastfusion.frontend.workload.workload import (
    # Workload class for all of FastFusion.
    Workload,
)
from fastfusion.frontend.workload._isl import (
    get_einsum_operation_space,
    get_projection_map,
)
from fastfusion.frontend.mapping import TensorName
from fastfusion.model.looptree.reuse.isl.isl_functions import (
    map_to_prior_coordinate,
)
from fastfusion.model.looptree.reuse.isl.mapping_to_isl import DUMP_ISL_IR
from fastfusion.model.looptree.reuse.isl.mapping_to_isl.types import (
    EinsumName,
    Tiling,
    BranchTiling,
)


def get_mapping_group_einsums(
    mapping: Mapping,
) -> defaultdict[MappingNode, set[EinsumName]]:
    """
    From a mapping, get the group of einsums for a given node.

    :param mapping: The mapping we are getting the grouped einsums for.
    :type mapping:  Mapping

    :return:    A dictionary relating a Node to a set of einsums.
    :rtype:     defaultdict[MappingNode, Set[EinsumName]]
    """
    # Each pair is a (current_node, last_non_branch_node)
    dfs_stack: deque[Tuple[MappingNode, MappingNode]] = deque()
    # Each pair is a (last_non_branch_node, set_of_children_nodes)
    child_stack: deque[Tuple[MappingNode, set[MappingNode]]] = deque()
    result: defaultdict[MappingNode, set[EinsumName]] = defaultdict(set)

    # Start DFS hierarchical search from the root.
    root: MappingNode = mapping
    dfs_stack.append((root, root))

    # Exhaustive DFS search.
    while dfs_stack:
        # Grabs latest node to search.
        node, last_non_branch = dfs_stack.pop()

        # Differentiates behavior by number of child nodes.
        match node:
            case MappingNodeWithChildren():
                match len(node.nodes):
                    # No children, log as a folded result.
                    case 0:
                        # Note:: Check necesary in case Distrobuffers elides
                        # computes into one large unit.
                        if isinstance(node, Compute):
                            result[last_non_branch].add(node.einsum)
                        else:
                            raise TypeError(
                                f"The following node should be of class "
                                f"Compute as it has no children:\n---\n{node}"
                            )
                    # Explore the children further.
                    case 1:
                        dfs_stack.append((node.nodes[0], last_non_branch))
                    # Log all branching children and explore all children.
                    case _:
                        children: set[MappingNode] = set(node.nodes)
                        child_stack.append((last_non_branch, children))
                        dfs_stack.extend((child, child) for child in children)
            # Assumed no children, log as a folded result.
            case Compute():
                result[last_non_branch].add(node.einsum)
            # TODO: I'm pretty sure these all had children at some point in Timeloop.
            case Spatial() | Temporal() | Storage():
                continue
            case _:
                raise AttributeError(
                    f"The following node of class {type(node)} has "
                    f"indeterminant number of children:\n---\n"
                    f"{node}"
                )

    # Push up einsums to parents.
    for node, children in reversed(child_stack):
        node_einsum_set: set[EinsumName] = result[node]
        for child in children:
            node_einsum_set.update(result[child])

    return result


def get_head_among_einsums(
    einsum_set: set[EinsumName], workload: Workload
) -> set[EinsumName]:
    """
    Gets the provider einsums that only consume data (i.e., sink einsums).

    :param einsum_set:  Set of einsums to consider.
    :param workload:    The workload context the einsums exist in.

    :type einsum_set:   Set[EinsumName]
    :type workload:     Workload

    :return:    The set of all head einsums.
    :rtype:     Set[EinsumName]
    """
    # Returns set of einsums that are not data producers.
    return {
        einsum
        for einsum in einsum_set
        if all(
            not any(
                consumer in einsum_set
                for consumer in workload.einsums_that_read_tensor(output_tensor)
            )
            for output_tensor in workload.tensors_written_by_einsum(einsum)
        )
    }


def add_new_tile_dim(old_tiling: Tiling, dim_idx: int, tile_size: int) -> Tiling:
    """
    Given a tiling, add a new dimension to the tiling.

    :param old_tiling:  The previous tiling the mapper proposed.
    :param dim_idx:     The index of the dimension being tiled.
    :param tile_size:   The size of the tiling on dim_idx.

    :type old_tiling:   Tiling
    :type dim_idx:      int
    :type tile_size:    int

    :return:    The new Tiling with tiled dimension at dim_idx.
    :rtype:     Tiling
    """

    # new_tiling has one extra dimension at the end compared to old_tiling.
    new_tiling = old_tiling.insert_dims(
        isl.dim_type.in_, old_tiling.dim(isl.dim_type.in_), 1
    )

    # Min and max of dim_idx. dimension being tiled as function of tiled dimensions.
    dim_min: isl.PwAff = new_tiling.dim_min(dim_idx)
    dim_max: isl.PwAff = new_tiling.dim_max(dim_idx)

    # Aff from tiled dimensions space to value of newest dim.
    new_dim_id: isl.Aff = isl.Aff.var_on_domain(
        dim_min.get_domain_space().to_local_space(),
        isl.dim_type.set,
        dim_min.dim(isl.dim_type.in_) - 1,
    )

    # Aff from tiled dimensions space to tile tile size constant.
    tile_size_aff: isl.Aff = isl.Aff.val_on_domain_space(
        dim_min.get_domain_space(), isl.Val.int_from_ui(isl.DEFAULT_CONTEXT, tile_size)
    )

    # PwAff from tiled dimension space to tile_size * newest_dim.
    tile_translate: isl.PwAff = isl.PwAff.from_aff(new_dim_id.mul(tile_size_aff))

    # What dim_min should be given new tiling.
    new_dim_min: isl.PwAff = dim_min.add(tile_translate)

    # What dim_max should be given new tiling.
    new_dim_max: isl.PwAff = new_dim_min.add(
        isl.PwAff.from_aff(tile_size_aff.add_constant_val(-1))
    )

    # TODO: Might be logically equivalent to new_dim_id:
    # https://github.com/NVlabs/timeloop/blob/32370826fdf1aa3c8deb0c93e6b2a2fc7cf053aa/src/loop-analysis/mapping-to-isl/tiling.cpp#L52-L59
    new_iter_id: isl.PwAff = isl.PwAff.from_aff(
        isl.Aff.var_on_domain(
            new_tiling.get_space().domain(),
            isl.dim_type.set,
            old_tiling.dim(isl.dim_type.in_),
        )
    )

    # The set of valid values of the new tiled dimensions.
    iter_set: isl.Set = new_tiling.domain()
    iter_set = iter_set.intersect(new_iter_id.le_set(dim_max.div(tile_size_aff).ceil()))
    iter_set = iter_set.intersect(new_dim_min.ge_set(dim_min))

    # The value of iter dims cannot exceed what was available before tiling.
    new_tiling = new_tiling.intersect_domain(iter_set)

    # The set of operations need to to follow the new tile bounds.
    identity: isl.PwAff = isl.PwAff.from_aff(
        isl.Aff.var_on_domain(new_tiling.get_space().range(), isl.dim_type.set, dim_idx)
    )
    new_tiling = new_tiling.intersect(new_dim_min.le_map(identity))
    new_tiling = new_tiling.intersect(new_dim_max.ge_map(identity))

    return new_tiling


def shared_input_based_tile_shape_inference(
    workload: Workload,
    tiling_info: defaultdict[EinsumName, Tiling],
    einsums: set[EinsumName],
    shared_input_tensor: TensorName,
    tiled_einsum: EinsumName,
) -> None:
    """
    Given a `tiled_einsum` in a `workload`, restrict the other `einsums`' execution
    in this tiling to one in which the data is shared with the `tiled_einsum`. This
    is because, when tiled, data is multicast so the other einsums being tiled together
    must shared data.

    :param workload:        The workload context the tiling is occuring in.
    :param tiling_info:     Relation of `EinsumName` and its viable tiling on hardware.
    :param einsums:         The set of all einsums.
    :param shared_input_tensor: The singular tensor `einsums` all read from.
    :param tiled_einsum:    The einsum being tiled.

    :type workload:     Workload
    :type tiling_info:  defaultdict[EinsumName, Tiling]
    :type einsums:      set[EinsumName]
    :type shared_input_tensor:  TensorName
    :type tiled_einsum: EinsumName

    :returns: None
    :rtype: None

    Postconditions:
    --------------
    `tiling_info` is updated such that each Tiling contains only compatible tilings
    with `tiled_einsum`.
    """
    # Gets the data tiled_einsum reads from shared_input_tensor
    tiled_einsum_read_accesses: isl.Map = get_projection_map(
        workload.einsums[tiled_einsum], shared_input_tensor
    )
    read_data: isl.Map = tiling_info[tiled_einsum].apply_range(
        tiled_einsum_read_accesses
    )

    # Goes through all other einsums and restrict their tilings to only the executable
    # operations after one of the einsums is tiled.
    for einsum in einsums:
        if einsum == tiled_einsum:
            continue

        read_accesses: isl.Map = get_projection_map(
            workload.einsums[einsum], shared_input_tensor
        )
        executable_operations: isl.Map = read_data.apply_range(read_accesses.reverse())
        executable_operations = executable_operations.intersect_range(
            get_einsum_operation_space(workload, einsum)
        )

        tiling_info[einsum] = tiling_info[einsum].intersect(executable_operations)


def consumer_based_tile_shape_inference(
    workload: Workload,
    tiling_info: defaultdict[EinsumName, Tiling],
    tensor_to_reuse_level: defaultdict[TensorName, int],
    einsums: set[EinsumName],
    tiled_einsum: EinsumName,
):
    """
    Given a `tiled_einsum` in a `workload`, restrict the other `einsums`' execution
    in this tiling to one in which the data is required for the tensors read by
    `tiled_einsum`. This is because, when tiled, data is multicast so the other
    einsums being tiled together must shared data.

    :param workload:        The workload context the tiling is occuring in.
    :param tiling_info:     Relation of `EinsumName` and its viable tiling on hardware.
    :param tensor_to_reuse_level:   A relation between a tensor and the amount of reuse occuring.
    :param einsums:         The set of all einsums.
    :param tiled_einsum:    The einsum being tiled.

    :type workload:     Workload
    :type tiling_info:  defaultdict[EinsumName, Tiling]
    :type tensor_to_reuse_level:    defaultdict[TensorName, int]
    :type einsums:      set[EinsumName]
    :type tiled_einsum: EinsumName

    :returns: None
    :rtype: None

    Postconditions:
    --------------
    `tiling_info` is updated such that each Tiling contains only compatible tilings
    with `tiled_einsum`.
    """
    # Goes recursively through tensor dependencies (read tensors) and tiles them.
    queue: deque[EinsumName] = deque([tiled_einsum])
    while queue:
        einsum: EinsumName = queue.popleft()
        tiling: Tiling = tiling_info[einsum]

        # For each tensor read by this einsum, tile that tensor's producers.
        for tensor in workload.tensors_read_by_einsum(einsum):
            producer_einsums: set[EinsumName] = {
                e.name for e in workload.einsums_that_write_tensor(tensor)
            }
            if len(producer_einsums) > 1:
                raise NotImplementedError(
                    "Tile shape inference cannot handle multiple einsums writing the same tensor."
                )

            # Not an intermediate tensor.
            if not producer_einsums:
                continue

            producer_einsums.intersection_update(einsums)
            # No producer einsum in this fusion set.
            if not producer_einsums:
                continue

            # Collates all the producer einsum read accesses.
            producer_einsum: EinsumName = next(iter(producer_einsums))
            read_accesses: isl.Map = get_projection_map(
                workload.einsums[producer_einsum], tensor
            )
            # Required data of the tiling as a mapping of read accesses.
            required_data: isl.Map = tiling.apply_range(read_accesses)

            # Calculates the data computed by the producer einsums.
            computed_data: isl.Map = required_data
            if tensor in tensor_to_reuse_level:
                reuse_level: int = tensor_to_reuse_level[tensor]
                shifter: isl.Map = map_to_prior_coordinate(
                    tiling.dim(isl.dim_type.in_), reuse_level
                )
                buffered_data: isl.Map = shifter.apply_range(required_data)
                computed_data = computed_data.subtract(buffered_data).coalesce()

            # Grabs the elements this tensor relies on from producer_einsums.
            producer_write_dependency: isl.Map = get_projection_map(
                workload.einsums[producer_einsum], tensor
            )
            # Gets the required operations to produce the current tensor.
            required_operations: isl.Map = computed_data.apply_range(
                producer_write_dependency.reverse()
            )
            required_operations = required_operations.intersect_range(
                get_einsum_operation_space(workload, producer_einsum)
            )

            # Mutations of the tilings of producer einsums.
            tiling_info[producer_einsum] = tiling_info[producer_einsum].intersect(
                required_operations
            )

            queue.append(producer_einsum)


def detect_shared_input_tensor(
    fused_set: set[EinsumName], workload: Workload
) -> List[TensorName]:
    """
    Given a set of fused einsums on a workload, detect the input tensor that they
    all are dependent on, if it exists.

    :param fused_set:   The set of fused einsums being analyzed.
    :param workload:    The workload context the einsums exist in.

    :type fused_set:    set[EinsumName]
    :type workload:     Workload

    :return:    The list of tensors shared by the inputs. Because we default to
                consumer-based analysis if there's more than 1 shared input among
                the tensors, we only return tuple sizes of {0, 1, 2}.
    :rtype:     List[EinsumName]
    """
    n_einsums: int = 0
    tensor_read_counts: defaultdict[TensorName, int] = defaultdict(lambda: 0)

    # Counts the number of times a tensor is read by an einsum.
    for einsum in fused_set:
        for tensor in workload.tensors_read_by_einsum(einsum):
            tensor_read_counts[tensor] += 1
        n_einsums += 1

    shared_input_tensors: List[TensorName] = []
    for tensor, count in tensor_read_counts.items():
        # Tensor is shared by all einsums.
        if count == n_einsums:
            shared_input_tensors.append(tensor)
            # Caller should resort to consumer-based fusing methods.
            if len(shared_input_tensors) > 1:
                return shared_input_tensors

    return shared_input_tensors


def tiling_from_mapping(mapping: Mapping, workload: Workload) -> BranchTiling:
    """
    Given a mapping and a workload generates a tiling.

    :param mapping: A mapping of data to hardware.
    :param workload:The problem being solved.

    :type mapping:  Mapping
    :type workload: Workload

    :return:    BranchTiling associating a node's ID with its tiling.
    :rtype:     BranchTiling
    """
    result: BranchTiling = BranchTiling()
    # Grabs the head einsums.
    mapping_groups: defaultdict[MappingNode, set[EinsumName]] = (
        get_mapping_group_einsums(mapping)
    )
    mapping_group_heads: defaultdict[MappingNode, set[EinsumName]] = defaultdict(
        set,
        {
            node: get_head_among_einsums(group, workload)
            for node, group in mapping_groups.items()
        },
    )

    tensor_to_reuse_level: defaultdict[TensorName, int] = defaultdict()
    dfs_stack: deque[MappingNode] = deque()

    # Maps last non-branch to tiling of each in the group.
    tiling_info: defaultdict[MappingNode, defaultdict[EinsumName, Tiling]] = (
        defaultdict(defaultdict)
    )

    # Initiates the DFS at the mapping root and appends its info.
    root: MappingNode = mapping
    dfs_stack.append(root)
    for einsum_name in workload.einsum_names:
        tiling: Tiling = isl.Map.from_range(
            get_einsum_operation_space(workload, einsum_name)
        )
        if DUMP_ISL_IR:
            print(f"Tiling: {tiling}")
        tiling_info[root][einsum_name] = tiling

    while dfs_stack:
        node = dfs_stack.pop()
        heads = mapping_group_heads[node]

        current_node: MappingNode = node
        is_tiling: bool = True

        while is_tiling:
            # Fuses current_node to one of the heads.
            match current_node:
                # For or Par-For loop handling.
                case Iteration():
                    if len(heads) != 1:
                        raise ValueError(
                            f"Cannot fuse tiled set with {len(heads)} heads.\n"
                            f"---\n"
                            f"mapping_group_heads={pformat(mapping_group_heads)}"
                        )

                    # Grabs rank_var to tile and the head to tile it from.
                    rank_var = current_node.rank_variable
                    head = next(iter(heads))

                    old_tiling: Tiling = tiling_info[node][head]
                    # set, not AbstractSet, iteration in python is the same.
                    # Downstreams of "heads" is also constaint.
                    isl_rank_idx: int = tuple(
                        workload.einsums[head].rank_variables
                    ).index(rank_var)

                    # Adds a new tile_dim to the old tiling.
                    if (
                        isinstance(current_node.tile_shape, int)
                        and current_node.tile_shape != 0
                    ):
                        new_tiling: Tiling = add_new_tile_dim(
                            old_tiling, isl_rank_idx, current_node.tile_shape
                        )
                    else:
                        raise NotImplementedError(
                            f"Tile size analysis not implemented for type {type(node)} "
                            f"with tile shape {current_node.tile_shape}"
                        )

                    # Saves the fused tiling.
                    tiling_info[node][head] = new_tiling

                    iteration_set: isl.Set = new_tiling.domain()
                    for einsum in mapping_groups[node]:
                        if einsum == head:
                            continue

                        tiling = tiling_info[node][einsum]
                        tiling = tiling.insert_dims(
                            isl.dim_type.in_, tiling.dim(isl.dim_type.in_), 1
                        )
                        tiling = tiling.intersect_domain(iteration_set)

                    # TODO: Verify this bodge: https://github.com/NVlabs/timeloop/blob/32370826fdf1aa3c8deb0c93e6b2a2fc7cf053aa/src/loop-analysis/mapping-to-isl/fused-mapping-to-isl.cpp#L406
                    is_tiling = False
                # Notes what reuse level the tensor is on.
                case Storage():
                    # See current_node is the highest level of Storage to determine reuse level.
                    # TODO: Check this is correct too.
                    for tensor in current_node.tensors:
                        # Check second term
                        if tensor not in tensor_to_reuse_level and current_node._must_keep_tensors:
                            random_einsum: EinsumName = next(iter(mapping_groups[node]))
                            tiling: Tiling = tiling_info[node][random_einsum]
                            tensor_to_reuse_level[tensor] = tiling.dim(
                                isl.dim_type.in_
                            )
                    # TODO: Check accuracy of not using nodes.
                    is_tiling = False
                # If we are at the Mapping root, just go to the actual Nodes.
                case Mapping():
                    # TODO: Check accuracy of not using flatten.
                    dfs_stack.append(mapping.nodes[0])
                    is_tiling = False
                # If we hit the compute node, we've finished tiling, end!
                case Compute():
                    result[current_node] = tiling_info[root][current_node.einsum]
                    is_tiling = False
                case Split():
                    fused_set: set[EinsumName] = mapping_groups[node]
                    if len(heads) != 1:
                        # There can't be a tiling, so no inference to be done.
                        break
                    shared_input_tensor: List[TensorName] = detect_shared_input_tensor(
                        fused_set, workload
                    )

                    random_head = next(iter(heads))
                    if len(shared_input_tensor) == 1:
                        shared_input_based_tile_shape_inference(
                            workload,
                            tiling_info[node],
                            fused_set,
                            shared_input_tensor[0],
                            random_head,
                        )
                    else:
                        consumer_based_tile_shape_inference(
                            workload,
                            tiling_info[node],
                            tensor_to_reuse_level,
                            fused_set,
                            random_head,
                        )

                    # Goes through each child node of the current node and propagate
                    # the tiling updates.
                    for idx, child in enumerate(current_node.nodes):
                        # Each child needs tilings for all Einsums in its group.
                        group: set[EinsumName] = mapping_groups[child]
                        tilings: defaultdict[EinsumName, Tiling] = defaultdict()

                        # For all einsums the child is involved in, update their tilings.
                        for einsum in group:
                            tiling: Tiling = tiling_info[node][einsum]
                            new_tiling: Tiling = tiling.add_dims(isl.dim_type.in_, 1)

                            tilings[einsum] = new_tiling.fix_input_si(
                                new_tiling.dim(isl.dim_type.in_) - 1, idx
                            )

                        # Update the tiling info for the child.
                        tiling_info[child] = tilings
                        # DFS tile on the child.
                        dfs_stack.append(child)

                    is_tiling = False
                case _:
                    raise NotImplementedError(
                        f"Type {type(node)} not handled.\n"
                        f"---\n"
                        f"node={node}"
                    )
                    is_tiling = False

    return result