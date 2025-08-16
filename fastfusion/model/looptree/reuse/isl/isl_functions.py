"""
ISL functions that encapsulate more commonly used workflows in looptree for the
sake of code concision.
"""

import islpy as isl

def dim(map: isl.Map, dim_type: isl.dim_type) -> int:
    return map.dim(dim_type)


def project_dim_in_after(map: isl.Map, start: int) -> isl.Map:
    """Projects out the dims"""
    n_dim_in: int = map.dim(isl.dim_type.in_)
    return map.project_out(isl.dim_type.in_, start, n_dim_in - start)


def insert_equal_dims_maff(maff: isl.MultiAff, in_pos: int, out_pos: int, n: int) -> isl.MultiAff:
    """
    Given a multi affine, insert equal numbers of input and output dimensions.

    :param maff:    The multi affine base to insert dims into.
    :param in_pos:  The index to start inserting input dimensions at in `maff`.
    :param out_pos: The index to start inserting output dimensions at in `maff`.
    :param n:       The number of dimensions to insert.

    :type maff:     isl.MultiAff
    :type in_pos:   int
    :type out_pos:  int
    :type n:        int

    :returns:       A new maff which is equivalent to `maff` except it has `n` new
                    input and output dimensions starting at `in_pos` and `out_pos`
                    respectively.
    :rtype:         isl.MultiAff
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
    Given a map, insert equal numbers of input and output dimensions.

    :param map_:    The map base to insert dims into.
    :param in_pos:  The index to start inserting input dimensions at in `map_`.
    :param out_pos: The index to start inserting output dimensions at in `map_`.
    :param n:       The number of dimensions to insert.

    :type maff:     isl.Map
    :type in_pos:   int
    :type out_pos:  int
    :type n:        int

    :returns:       A new maff which is equivalent to `map_` except it has `n` new
                    input and output dimensions starting at `in_pos` and `out_pos`
                    respectively.
    :rtype:         isl.Map
    """
    # Inserts the new input and output dimensions.
    map_ = map_.insert_dims(isl.dim_type.in_, in_pos, n)
    map_ = map_.insert_dims(isl.dim_type.out, out_pos, n)

    # Adds constraints for conservation between the new input and output dimensions
    # in the map.
    local_space: isl.LocalSpace = map_.get_space().to_local_space()
    for i in range(n):
        constraint: isl.Constraint = isl.Constraint.alloc_equality(local_space)
        constraint = constraint.set_coefficient_val(isl.dim_type.in_, in_pos + i, 1)
        constraint = constraint.set_coefficient_val(isl.dim_type.out, out_pos + i, -1)
        map_ = map_.add_constraint(constraint)
    
    return map_


def map_to_prior_data(n_in_dims: int, top: int) -> isl.Map:
    """
    Create a map that relates presence to new data presence.
    Goal: { [i0, ..., i{n_in_dims-1}] -> [i0, ..., i{top}-1, o{top+1}, ..., o{n_in_dims}] }

    :param n_in_dims:   The number of input/output dims of the dataspace.
    :param top:         The pivot point where input data got swapped out for new
                        data.

    :type n_in_dims:    int
    :type top:          int

    :returns:           


    Preconditions
    -------------
    -   0 <= top <=n_in_dims
    """

    # Creates the space, map, and local_space the temporal reuse data map will exist
    # in.
    space: isl.Space = isl.Space.alloc(isl.DEFAULT_CONTEXT, 0, n_in_dims, n_in_dims)
    map_: isl.Map = isl.Map.empty(space)
    local_space: isl.LocalSpace = isl.LocalSpace.from_space(space)

    constraint: isl.Constraint
    # If there is any data replacement
    if top > 0:
        # Create a temporary map.
        tmp_map: isl.Map = isl.Map.universe(space)
        # Model the conservation of data along each data dimension in that map.
        for i in range(top - 1):
            constraint = isl.Constraint.alloc_equality(local_space)
            constraint = constraint.set_coefficient_val(isl.dim_type.out, i, 1)
            constraint = constraint.set_coefficient_val(isl.dim_type.in_, i, -1)
            tmp_map = tmp_map.add_constraint(constraint)
        
        # 
        constraint = isl.Constraint.alloc_equality(local_space)
        constraint = constraint.set_coefficient_val(isl.dim_type.out, top-1, 1)
        constraint = constraint.set_coefficient_val(isl.dim_type.in_, top-1, -1)
        constraint = constraint.set_constant_val(1)
        tmp_map = tmp_map.add_constraint(constraint)

        map_ = map_.union(tmp_map)
    
    if top < n_in_dims:
        tmp_map: isl.Map = isl.Map.lex_gt(
            isl.Space.set_alloc(
                isl.DEFAULT_CONTEXT, map_.dim(isl.dim_type.param), n_in_dims - top
            )
        )
        tmp_map = insert_equal_dims_map(tmp_map, 0, 0, top)
        map_ = map_.union(tmp_map)
    
    return map_
