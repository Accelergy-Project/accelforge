"""
ISL functions that encapsulate more commonly used workflows in looptree for the
sake of code concision.
"""

import logging
from typing import List

import islpy as isl


def project_dim_in_after(map_: isl.Map, start: int) -> isl.Map:
    """
    Projects out the input dims of idx [start, end] in map_.

    Parameters
    ----------
    map_:
        The map to project out dims from.
    start:
        The dim idx to start projecting dims out from.

    Returns
    -------
    map_ without the input dims [start:].
    """
    n_dim_in: int = map_.dim(isl.dim_type.in_)
    return (
        map_.project_out(isl.dim_type.in_, start, n_dim_in - start)
        if start <= n_dim_in
        else map_
    )


def dim_projector_range(space: isl.Space, start: int, n: int) -> isl.Map:
    """
    Given a space, create a map that projects out the dims [start: start+n).

    Parameters
    ----------
    space:
        The space to create the dim projector in.
    start:
        The index to start the projection.
    n:
        The number of dims from `start` to project out.

    Returns
    -------
    A `isl.Map` in `space` that projects out dims [start:start+n].
    """
    base_map: isl.Map = isl.Map.identity(isl.Space.map_from_set(space))
    # TODO: propagate tuple names from `space` onto `base_map` (e.g.,
    # `space.get_tuple_name(isl.dim_type.set)`) so the projector keeps the
    #  original set label on both domain and range.
    return isl.Map.project_out(base_map, isl.dim_type.in_, start, n)


def dim_projector_mask(space: isl.Space, mask: List[bool]) -> isl.Map:
    """
    Given a space, create a map that projects out the dims marked `True` in the mask.

    Parameters
    ----------
    space:
        The space the projector is created on.
    mask:
        The mask of the list of dims to be projected out.

    Returns
    -------
    A projection from space in[x_0, ..., x_n] -> out[x_1, ..., x_i, ... x_{n-1}]
    where `x_i ∉ out => mask[i]`.
    """
    projector: isl.Map = isl.Map.identity(isl.Space.map_from_set(space))
    # TODO: set tuple names on `projector` using the set name (e.g.,
    #  `space.get_tuple_name(isl.dim_type.set)`) so domain/range remain
    # attributable after masking.

    for i in range(len(mask) - 1, -1, -1):
        if mask[i]:
            projector = projector.project_out(isl.dim_type.in_, i, 1)

    return projector


def add_dims_preserve_name_map(
    map_: isl.Map, dim_type: isl.dim_type, n: int
) -> isl.Map:
    """
    Wrapper of `isl.Map.add_dims` that preserves the space name post
    addition.

    Parameters
    ----------
    map_:
        The map we're adding into.
    dim_type:
        The dimension tuple we're inserting into.
    n:
        The number of dimensions to insert.

    Returns
    -------
    Dimension-inserted maps with preservation.

    Postcondition
    -------------
    The entirety of dependencies with a given space name in the given context
    we're operating under.
    """
    name: str = map_.get_tuple_name(dim_type)
    map_ = map_.add_dims(dim_type, n)
    if name is None:
        logging.warning(f"unnamed space for {map_}", stack_info=True)
        return map_
    return map_.set_tuple_name(dim_type, name)


def insert_dims_preserve_name_map(
    map_: isl.Map, dim_type: isl.dim_type, pos: int, n: int
) -> isl.Map:
    """
    Wrapper of `isl.Map.insert_dims` that preserves the space name post
    insertion.

    Parameters
    ----------
    map_:
        The inserting map.
    dim_type:
        The dimension tuple we're inserting into.
    pos:
        The position to start inserting into, inclusive.
    n:
        The number of dimensions to insert.

    Returns
    -------
    Dimension-inserted maps with preservation.

    Postcondition
    -------------
    The entirety of dependencies with a given space name in the given context
    we're operating under.
    """
    name: str = map_.get_tuple_name(dim_type)
    map_ = map_.insert_dims(dim_type, pos, n)
    if name is None:
        logging.warning(f"unnamed space for {map_}", stack_info=True)
        return map_
    return map_.set_tuple_name(dim_type, name)


def insert_equal_dims_maff(
    maff: isl.MultiAff, in_pos: int, out_pos: int, n: int
) -> isl.MultiAff:
    """
    Given a multi affine, insert equal numbers of input and output dimensions and
    enforce equality between the values of the two dims.

    Parameters
    ----------
    maff:
        The multi affine base to insert dims into.
    in_pos:
        The index to start inserting input dimensions at in `maff`.
    out_pos:
        The index to start inserting output dimensions at in `maff`.
    n:
        The number of dimensions to insert.

    Returns
    -------
    A new maff which is equivalent to `maff` except it has `n` new input and
    output dimensions starting at `in_pos` and `out_pos` respectively.
    """
    # Inserts the `n` dimensions into a new maff base.
    maff = maff.insert_dims(isl.dim_type.in_, in_pos, n)
    maff = maff.insert_dims(isl.dim_type.out, out_pos, n)

    # Modifies each affine to create an equality relation between the input and output.
    for i in range(n):
        aff: isl.Aff = maff.get_at(out_pos + i)
        aff = aff.set_coefficient_val(isl.dim_type.in_, in_pos + i, 1)
        maff = maff.set_aff(out_pos + i, aff)

    return maff


def insert_equal_dims_map(map_: isl.Map, in_pos: int, out_pos: int, n: int) -> isl.Map:
    """
    Given a map, insert equal numbers of input and output dimensions and enforce
    equality between the values of the two dims.

    Parameters
    ----------
    map_:
        The map base to insert dims into.
    in_pos:
        The index to start inserting input dimensions at in `map_`.
    out_pos:
        The index to start inserting output dimensions at in `map_`.
    n:
        The number of dimensions to insert.

    Returns
    -------
    A new maff which is equivalent to `map_` except it has `n` new input and
    output dimensions starting at `in_pos` and `out_pos` respectively.
    """
    # Inserts the new input and output dimensions.
    map_ = insert_dims_preserve_name_map(map_, isl.dim_type.in_, in_pos, n)
    map_ = insert_dims_preserve_name_map(map_, isl.dim_type.out, out_pos, n)

    # Adds constraints for conservation between the new input and output dimensions
    # in the map.
    local_space: isl.LocalSpace = map_.get_space().to_local_space()
    for i in range(n):
        # out - in == 0 => out == in
        constraint: isl.Constraint = isl.Constraint.alloc_equality(local_space)
        constraint = constraint.set_coefficient_val(isl.dim_type.in_, in_pos + i, 1)
        constraint = constraint.set_coefficient_val(isl.dim_type.out, out_pos + i, -1)
        map_ = map_.add_constraint(constraint)

    return map_


def map_to_prior_coordinate(n_in_dims: int, shifted_idx: int, name: str) -> isl.Map:
    """
    Create a map that relates current time index vector to a previous index vector.
    It shifts the coordinate at shifted_idx back by 1.

    Goal: { [i0,...,i{n_in_dims-1}] ->
            [i0, ..., i{shifted_idx}-1, i{shifted_idx+1}, ..., i{n_in_dims-1}] }

    Parameters
    ----------
    n_in_dims:
        The number of input/output dims of the dataspace.
    shifted_idx:
        The coordinate being shifted.
    name:
        The name for the domain and range of the shifter.

    Returns
    -------
    A map relating a current index vector to a previous index to a previous
    index vector by shifting the coordinate at `shifted_idx` back by 1.


    Preconditions
    -------------
    -   0 <= shifted_idx <=n_in_dims-1
    """

    # Creates the space, map, and local_space the temporal reuse data map will exist
    # in.
    space: isl.Space = isl.Space.alloc(isl.DEFAULT_CONTEXT, 0, n_in_dims, n_in_dims)
    map_: isl.Map = isl.Map.empty(space)
    local_space: isl.LocalSpace = isl.LocalSpace.from_space(space)

    constraint: isl.Constraint
    # If there is any data replacement
    if shifted_idx > 0:
        # Create a temporary map.
        tmp_map: isl.Map = isl.Map.universe(space)
        # Model the conservation of data along each data dimension in that map.
        # out - in == 0 => out == in
        for i in range(shifted_idx - 1):
            constraint = isl.Constraint.alloc_equality(local_space)
            constraint = constraint.set_coefficient_val(isl.dim_type.out, i, 1)
            constraint = constraint.set_coefficient_val(isl.dim_type.in_, i, -1)
            tmp_map = tmp_map.add_constraint(constraint)

        # Sets constraints such that the pivot value is decremented.
        # out - in + 1 == 0 => out == in - 1
        constraint = isl.Constraint.alloc_equality(local_space)
        constraint = constraint.set_coefficient_val(
            isl.dim_type.out, shifted_idx - 1, 1
        )
        constraint = constraint.set_coefficient_val(
            isl.dim_type.in_, shifted_idx - 1, -1
        )
        constraint = constraint.set_constant_val(1)
        tmp_map = tmp_map.add_constraint(constraint)

        map_ = map_.union(tmp_map)

    # If we're pivoting any of the data, preserve the `shifted_idx` datapoints.
    if shifted_idx < n_in_dims:
        tmp_map: isl.Map = isl.Map.lex_gt(
            isl.Space.set_alloc(
                isl.DEFAULT_CONTEXT,
                map_.dim(isl.dim_type.param),
                n_in_dims - shifted_idx,
            )
        )
        tmp_map = insert_equal_dims_map(tmp_map, 0, 0, shifted_idx)
        map_ = map_.union(tmp_map)

    map_ = map_.set_tuple_name(isl.dim_type.in_, name).set_tuple_name(
        isl.dim_type.out, name
    )

    return map_


def map_to_shifted(domain_space: isl.Space, pos: int, shift: int) -> isl.Map:
    """
    Given a `domain_space`, return a map from a point in the `domain_space` to
    a point in that dimension `shift` ahead in the dimension at `pos`.

    Parameters
    ----------
    domain_space:
        The space on which to construct the shift on.
    pos:
        The dimension to construct the shift on.
    shift:
        The amount of shift

    Returns
    -------
    A mapping from `[x_0, x_1, ..., x_n] -> [x_0, ..., x_{pos} + shift, ..., x_n]`
    on `domain_space`.
    """
    maff: isl.MultiAff = domain_space.identity_multi_aff_on_domain()
    maff = maff.set_at(pos, maff.get_at(pos).set_constant_val(shift))
    return maff.as_map()


def reorder_projector(
    permutation: list[int], space: str, ctx: isl.Context = isl.DEFAULT_CONTEXT
) -> isl.Map:
    """
    A projection to reorder a space from [i_2, i_0, ..., i_n, ...] -> [i_0, i_1, ..., i_n]

    Parameters
    ----------
    permutation:
        A list where the elements correspond to indices of the space's dimensions
        and the ordering of the indices the reconstruction.
    space:
        The name of the space being permuted.

    Returns
    -------
    A map that reorders an arbitrary list of input dimensions into an enforced ordering.
    """
    # Constructs the permutation in the form of a string, because it's easier
    # and less operations than explicit construction via objects.
    pattern: str
    if len(permutation) == 0:
        pattern = "{ [] -> [] }"
    else:
        pattern = "{ [ "
        for i in range(len(permutation) - 1):
            dim_idx = permutation[i]
            pattern += f"i{dim_idx}, "
        pattern += f"i{permutation[-1]}] -> [ "

        for i in range(len(permutation) - 1):
            pattern += f"i{i}, "

        pattern += f"i{permutation[-1]} ] }}"

    # Creates the actual map.
    reorder_proj: isl.Map = isl.Map.read_from_str(ctx, pattern)
    reorder_proj = reorder_proj.set_tuple_name(isl.dim_type.in_, space).set_tuple_name(
        isl.dim_type.out, space
    )

    return reorder_proj
