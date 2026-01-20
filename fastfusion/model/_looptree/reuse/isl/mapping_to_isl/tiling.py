"""
File for all the functions that conduct tiling analysis for the overall mapping
analysis.
"""

from collections import defaultdict, deque
from typing import List, Tuple, Optional

from pprint import pformat

import islpy as isl

from fastfusion.frontend.mapping import (
    # Types
    MappingNode,
    # Mapping objects
    Mapping,
    MappingNodeWithChildren,
    Nested,
    # Physical object types in Mappings.
    Compute,
    Storage,
    # Logical object types in Mappings.
    Loop,
    Spatial,
    Temporal,
    Split,
)
from fastfusion.frontend.workload import (
    # Workload class for all of FastFusion.
    Workload,
)
from fastfusion.frontend._workload_isl._isl import (
    get_einsum_operation_space,
    get_projection_map,
)
from fastfusion.frontend.mapping import TensorName
from fastfusion.model._looptree.reuse.isl.isl_functions import (
    add_dims_preserve_name_map,
    insert_dims_preserve_name_map,
    map_to_prior_coordinate,
)
from fastfusion.model._looptree.reuse.isl.mapping_to_isl import DUMP_ISL_IR
from fastfusion.model._looptree.reuse.isl.mapping_to_isl.types import (
    EinsumName,
    Tiling,
    BranchTiling,
)


def get_mapping_group_einsums(
    mapping: Mapping,
) -> defaultdict[MappingNode, set[EinsumName]]:
    """
    From a mapping, get the group of einsums for a given node.

    Parameters
    ----------
    mapping:
        The mapping we are getting the grouped einsums for.

    Returns
    -------
    A dictionary relating a MappingNode to a set of einsums.
    """
    # Each pair is a (current_node, last_non_branch_node)
    dfs_stack: deque[Tuple[MappingNode, MappingNode]] = deque()
    # Each pair is a (last_non_branch_node, set_of_children_nodes)
    child_stack: deque[Tuple[MappingNode, set[MappingNode]]] = deque()
    result: defaultdict[MappingNode, set[EinsumName]] = defaultdict(set)

    # Start DFS hierarchical search from the root.
    dfs_stack.append((mapping, mapping))

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
            # These had children in Timeloop we had to add to the DFS, but because
            # of our extension of dfs_stack we can just skip this node.
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

    Parameters
    ----------
    einsum_set:
        Set of einsums to consider.
    workload:
        The workload context the einsums exist in.

    Returns
    -------
    The set of all head einsums.
    """
    # Returns set of einsums that are not data producers.
    return {
        einsum
        for einsum in einsum_set
        if all(
            not any(
                consumer.name in einsum_set
                for consumer in workload.einsums_with_tensor_as_input(output_tensor)
            )
            for output_tensor in workload.einsums[einsum].output_tensor_names
        )
    }


def add_new_tile_dim(
    old_tiling: Tiling, dim_idx: int, tile_size: int, rank_var: Optional[str] = None
) -> Tiling:
    """
    Given a tiling, add a new dimension to the tiling.

    Parameters
    ----------
    old_tiling:
        The previous tiling the mapper proposed.
    dim_idx:
        The index of the dimension being tiled.
    tile_size:
        The size of the tiling on dim_idx.
    rank_var:
        Rank variable name to assign to the new input dimension, if provided.

    Returns
    -------
    The new Tiling with tiled dimension at dim_idx.
    """

    # new_tiling has one extra dimension at the end compared to old_tiling.
    new_tiling = insert_dims_preserve_name_map(
        old_tiling, isl.dim_type.in_, old_tiling.dim(isl.dim_type.in_), 1
    )
    if rank_var:
        new_tiling = new_tiling.set_dim_name(
            isl.dim_type.in_, old_tiling.dim(isl.dim_type.in_), rank_var
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

    Parameters
    ----------
    workload:
        The workload context the tiling is occurring in.
    tiling_info:
        Relation of `EinsumName` and its viable tiling on hardware.
    einsums:
        The set of all einsums.
    shared_input_tensor:
        The singular tensor `einsums` all read from.
    tiled_einsum:
        The einsum being tiled.

    Returns
    -------
    None

    Postconditions
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

    Parameters
    ----------
    workload:
        The workload context the tiling is occurring in.
    tiling_info:
        Relation of `EinsumName` and its viable tiling on hardware.
    tensor_to_reuse_level:
        A relation between a tensor and the amount of reuse occurring.
    einsums:
        The set of all einsums.
    tiled_einsum:
        The einsum being tiled.

    Returns
    -------
    None

    Postconditions
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
        for tensor in workload.einsums[einsum].input_tensor_names:
            producer_einsums: set[EinsumName] = {
                e.name for e in workload.einsums[einsum].output_tensor_names
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

            # Collates all the consumer einsum read accesses.
            producer_einsum: EinsumName = next(iter(producer_einsums))
            read_accesses: isl.Map = get_projection_map(
                workload.einsums[einsum], tensor
            )
            # Required data of the tiling as a mapping of read accesses.
            required_data: isl.Map = tiling.apply_range(read_accesses)

            # Calculates the data computed by the producer einsums.
            computed_data: isl.Map = required_data
            if tensor in tensor_to_reuse_level:
                reuse_level: int = tensor_to_reuse_level[tensor]
                shifter: isl.Map = map_to_prior_coordinate(
                    tiling.dim(isl.dim_type.in_),
                    reuse_level,
                    tiling.get_tuple_name(isl.dim_type.in_),
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
            # TODO: Deal with fusing naming better (perhaps mix the names?)
            tiling_info[producer_einsum] = tiling_info[producer_einsum].intersect(
                required_operations.set_tuple_name(
                    isl.dim_type.in_,
                    tiling_info[producer_einsum].get_tuple_name(isl.dim_type.in_),
                )
            )

            queue.append(producer_einsum)


def detect_shared_input_tensor(
    fused_set: set[EinsumName], workload: Workload
) -> List[TensorName]:
    """
    Given a set of fused einsums on a workload, detect the input tensor that they
    all are dependent on, if it exists.

    Parameters
    ----------
    fused_set:
        The set of fused einsums being analyzed.
    workload:
        The workload context the einsums exist in.

    Returns
    -------
    The list of tensors shared by the inputs. Because we default to consumer-based
    analysis if there's more than 1 shared input among the tensors, we only return
    tuple sizes of {0, 1, 2}.
    """
    n_einsums: int = 0
    tensor_read_counts: defaultdict[TensorName, int] = defaultdict(lambda: 0)

    # Counts the number of times a tensor is read by an einsum.
    for einsum in fused_set:
        for tensor in workload.einsums[einsum].input_tensor_names:
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

    Parameters
    ----------
    mapping:
        A mapping of data to hardware.
    workload:
        The problem being solved.

    Returns
    -------
    BranchTiling associating a node's ID with its tiling.
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
    dfs_stack: deque[MappingNode] = deque([mapping])  # DFS starts at mapping root.

    # Maps last non-branch to tiling of each in the group.
    tiling_info: defaultdict[MappingNode, defaultdict[EinsumName, Tiling]] = (
        defaultdict(defaultdict)
    )

    # Appends info for the root.
    for einsum_name in workload.einsum_names:
        tiling_info[mapping][einsum_name] = isl.Map.from_range(
            get_einsum_operation_space(workload, einsum_name)
        ).set_tuple_name(isl.dim_type.in_, f"{einsum_name}_tiled_iteration")

    # Tracks rank_var specified to partitioned_rank_var index, as traversal
    # in tiling goes down the partition.
    rank_var_partitions: defaultdict[str, int] = defaultdict(lambda: 0)

    def _get_rank_var_partition(rank_var: str) -> str:
        """
        Given a rank_var, get the partition at the current point in execution
        and increment for the next retrieval.
        """
        nonlocal rank_var_partitions
        rank_var_partition: str = f"{rank_var}{rank_var_partitions[rank_var]}"
        rank_var_partitions[rank_var] += 1
        return rank_var_partition

    def _tile_branch(heads: set[EinsumName], fusing_node: MappingNode):
        """
        Given a set of `heads` to fuse at `fusing_node`, fuse as much as possible
        in this branch.

        Parameters
        ----------
        heads:
            The heads being fused.
        fusing_node:
            The node node in the mapping at which the fusing is happening.

        Preconditions
        -------------
        1. `dfs_stack`: initialized with tiles to proceed to explore.
        2. `tiling_info`: prima facie populated.
        3. `tensor_to_reuse_level`: initialized and unmutated from last time this
            function was run.

        Postconditions
        --------------
        1. `dfs_stack`: progressed to the next node to tile at.
        2. `tiling_info`: updated to include the fusing and tiling.
        3. `tensor_to_reuse_level`: populated if information has changed from tiling.
        """
        nonlocal dfs_stack
        nonlocal tiling_info
        nonlocal tensor_to_reuse_level

        current_node: MappingNode = fusing_node
        while True:
            # Fuses current_node to one of the heads.
            match current_node:
                # For or Par-For loop handling.
                case Loop():
                    if len(heads) != 1:
                        raise ValueError(
                            f"Cannot fuse tiled set with {len(heads)} heads.\n"
                        )

                    # Tiles `current_node.rank_variable` at `head`
                    head = next(iter(heads))
                    tiling: Tiling = tiling_info[fusing_node][head]
                    # Downstreams of "heads" is also constant as it is a set, not
                    # AbstractSet.
                    idx: int = tuple(workload.einsums[head].rank_variables).index(
                        current_node.rank_variable
                    )

                    # Adds a new tile_dim to the old tiling.
                    # TODO: Handle stride.
                    if (
                        isinstance(
                            _ := current_node.tile_pattern.initial_tile_shape, int
                        )
                        and (_ != 0)
                        and (_ == current_node.tile_pattern.tile_shape)
                    ):
                        tiling: Tiling = add_new_tile_dim(
                            tiling,
                            idx,
                            current_node.tile_pattern.initial_tile_shape,
                            _get_rank_var_partition(current_node.rank_variable),
                        )
                    else:
                        raise NotImplementedError(
                            f"Tile size analysis not implemented for type {type(fusing_node)} "
                            f"with tile shape {current_node.tile_pattern.initial_tile_shape}"
                        )

                    # Saves the fused tiling.
                    tiling_info[fusing_node][head] = tiling

                    # Adds the ranks to the tiling isl.Map.
                    iteration_set: isl.Set = tiling.domain()
                    for einsum in mapping_groups[fusing_node] - {head}:
                        tiling = tiling_info[fusing_node][einsum]
                        # Index variables for the branch.
                        tiling = insert_dims_preserve_name_map(
                            tiling, isl.dim_type.in_, tiling.dim(isl.dim_type.in_), 1
                        )
                        tiling = tiling.set_dim_name(
                            isl.dim_type.in_,
                            tiling.dim(isl.dim_type.in_) - 1,
                            _get_rank_var_partition(current_node.rank_variable),
                        )
                        # TODO: Figure out if this intersection is correct.
                        tiling = tiling.intersect_domain(
                            iteration_set.set_tuple_name(
                                tiling.get_tuple_name(isl.dim_type.in_)
                            )
                        )
                        tiling_info[fusing_node][einsum] = tiling

                    current_node = dfs_stack.pop()
                # Notes what reuse level the tensor is on.
                case Storage():
                    # See current_node is the highest level of Storage to determine reuse level.
                    for tensor in current_node.tensors:
                        # Check second term
                        if tensor not in tensor_to_reuse_level:
                            random_einsum: EinsumName = next(
                                iter(mapping_groups[fusing_node])
                            )
                            tiling: Tiling = tiling_info[fusing_node][random_einsum]
                            tensor_to_reuse_level[tensor] = tiling.dim(isl.dim_type.in_)

                    current_node = dfs_stack.pop()
                # If we are at the Mapping root, just go to the actual Nodes.
                case Mapping():
                    dfs_stack.extend(reversed(current_node.nodes))
                    current_node = dfs_stack.pop()
                # If we hit the compute node, we've finished tiling, end!
                case Compute():
                    result[current_node] = tiling_info[fusing_node][current_node.einsum]
                    return
                case Split():
                    fused_set: set[EinsumName] = mapping_groups[fusing_node]
                    if len(heads) != 1:
                        # There can't be a tiling, so no inference to be done.
                        break

                    random_head = next(iter(heads))
                    if len(_ := detect_shared_input_tensor(fused_set, workload)) == 1:
                        shared_input_based_tile_shape_inference(
                            workload,
                            tiling_info[fusing_node],
                            fused_set,
                            _[0],
                            random_head,
                        )
                    else:
                        consumer_based_tile_shape_inference(
                            workload,
                            tiling_info[fusing_node],
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
                            tiling: Tiling = tiling_info[fusing_node][einsum]
                            # Add dimension that iterates over branches.
                            new_tiling: Tiling = add_dims_preserve_name_map(
                                tiling, isl.dim_type.in_, 1
                            )

                            tilings[einsum] = new_tiling.fix_input_si(
                                new_tiling.dim(isl.dim_type.in_) - 1, idx
                            )

                        # Update the tiling info for the child.
                        tiling_info[child] = tilings
                        # DFS tile on the child.
                        dfs_stack.append(child)

                    return
                case Nested():
                    dfs_stack.extend(reversed(current_node.nodes))
                    current_node = dfs_stack.pop()
                case _:
                    raise NotImplementedError(
                        f"Type {type(fusing_node)} not handled.\n"
                        f"---\n"
                        f"node={pformat(fusing_node)}"
                    )

    while dfs_stack:
        fusing_node = dfs_stack.pop()
        if DUMP_ISL_IR:
            print(f"New Tiling Root: {pformat(fusing_node)}")
        _tile_branch(mapping_group_heads[fusing_node], fusing_node)

    return result
