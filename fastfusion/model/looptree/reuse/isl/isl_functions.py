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