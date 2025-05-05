from arch import Compute, Leaf, Storage
from .specification import Specification


def to_looptree(spec: Specification):
    # Architecture
    arch = spec.architecture

    nodes = arch.flatten()
    for i, n in enumerate(nodes[:-1]):
        assert isinstance(n, Storage), (
            f"All but the last node in the architecture must be storage nodes"
            f"Node {i} is a {type(n)}"
        )
    name2leaf = {n.name: n for n in nodes}
    memory_names = [n.name for n in nodes[:-1]]
    assert isinstance(
        nodes[-1], Compute
    ), "The last node in the architecture must be a compute node"

    workload = spec.workload
    looptree_workload = LooptreeWorkload()
    for einsum in workload.einsums:
        looptree_workload.add_einsum(einsum)
        looptree_workload.set_einsum_rank_variables(einsum, einsum.rank_variables)
        for tensor in einsum.tensors:
            looptree_workload.add_tensor(tensor)
            # TODO: Ensure that if there are duplicate tensors, then they have the same
            # ranks.
            looptree_workload.set_tensor_ranks(tensor, tensor.ranks)
            # TODO: Ensure that projection is well-formed
            looptree_workload.set_projection(
                einsum, tensor, tensor.projection, tensor.output
            )
        shape = list(einsum.shape)
        shape += [
            workload.shape[r] for r in einsum.rank_variables if r in workload.shape
        ]
        shape = " and ".join(shape)
        looptree_workload.set_einsum_shape(einsum, shape)

    # Constraints
    constraints = LooptreeConstraints()
    for constraint in spec.constraints:
        name = constraint.name
        if name not in memory_names:
            raise KeyError(
                f"Found constraint with name {name}, but there is no "
                f"matching node in the architecture. Architecture nodes: "
                f"{sorted(name2leaf.keys())}"
            )
        leaf: Leaf = name2leaf[name]
        has_x_fanout = leaf.spatial.mesh_X > 1
        has_y_fanout = leaf.spatial.mesh_Y > 1

        if constraint.spatial_X:
            assert has_x_fanout, f"{name} has a spatial_X constraint but no X fanout."
        if constraint.spatial_Y:
            assert has_y_fanout, f"{name} has a spatial_Y constraint but no Y fanout."
        if constraint.spatial:
            assert (
                has_x_fanout or has_y_fanout
            ), f"{name} has a spatial constraint but no X or Y fanout."
            assert not (has_x_fanout and has_y_fanout), (
                f"{name} has a spatial constraint and both X and Y fanout."
                f"Please specify spatial_X and spatial_Y constraints instead."
            )

        if constraint.tensors or constraint.temporal:
            constraint_type = "tensors" if constraint.tensors else "temporal"
            assert isinstance(leaf, Storage), (
                f"{name} has a tensors or temporal constraint, but is not a storage. "
                f"Ensure that it has a !Storage tag in the architecture."
            )

        # TODO: Constraints

    return looptree


# Each Einsum and tensor must have a dictionary so we can add keys if needed.
# Tensor dictionaries must be unique to each Einsum in case the properties of the tensor
# change throuought the cascade.
