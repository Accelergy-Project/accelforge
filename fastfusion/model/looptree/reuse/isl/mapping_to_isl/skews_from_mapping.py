from functools import singledispatch

import islpy as isl

from .types import *


def skews_from_mapping(mapping, workload) -> dict[BufferTensorEinsum, Skew]:
    result: dict[BufferTensorEinsum, Skew] = {}

    for path in get_paths(mapping):
        SkewFromPath(path, workload, result)


def skew_from_path(
    mapping_path,
    workload,
    accumulator: dict[BufferTensorEinsum, Skew]
) -> None:
    """
    Get compute and buffer skews in a path and collect them in accumulator.
    """
    einsum_name = mapping_path[-1]['einsum']

    bte_and_idx = []
    all_tags = []
    cur_idx = 0
    for node in mapping_path:
        if node['type'] == 'storage':
            bte = BufferTensorEinsum(node['target'],
                                     node['tensor'],
                                     einsum_name)
            bte_and_idx.append((bte, cur_idx))
        if node['type'] not in ['compute', 'storage']:
            all_tags.append(make_tag_from_node(node))
            cur_idx += 1

    # Make { [i0, i1, ..., iN] -> [0] } where N = len(einsum_name)-1
    iter_space_str = ', '.join(f'i{i}' for i in range(len(all_tags)))
    iter_space = isl.Space(f'{{ [{iter_space_str} ] }}')

    iteration_to_rank_variables = {
        rank_var: isl.PwAff(iter_space.zero_aff_on_domain())
        for rank_var in workload.EinsumOspaceDimensions()
    }
    for node in mapping_path:
        if node['type'] in ['spatial', 'temporal']:
            # Insert iteration variable to the left and update projection
            iter_to_rank_var = iteration_to_rank_variables[node['rank']]
            if 'tile_shape' in node:
                raise NotImplementedError()
            elif 'factor' in node:
                raise NotImplementedError()
            else:
                raise NotImplementedError()
        elif node['type'] == 'sequential':
            raise NotImplementedError()
        elif node['type'] == 'pipeline':
            raise NotImplementedError()

    for bte, idx in bte_and_idx:
        # TODO
        accumulator[bte] = Skew(all_tags[:idx], skew_isl[:idx])



def make_tag_from_node(node):
    if node['type'] == 'temporal':
        return TemporalTag()
    elif node['type'] == 'spatial':
        return SpatialTag(
            node.get('spatial_dim', default=None),
            node.get('buffer', default=None)
        )
    elif node['type'] == 'pipeline':
        return PipelineTag(
            node.get('spatial_dim', default=None),
            node.get('buffer', default=None)
        )
    elif node['type'] == 'sequential':
        return SequentialTag()
    else:
        raise ValueError(f'Unsupported node type {node}')