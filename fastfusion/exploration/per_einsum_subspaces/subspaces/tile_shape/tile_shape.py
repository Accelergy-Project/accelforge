from collections import defaultdict
from functools import reduce
from operator import mul

from .shape_subspace import ShapeSubspace

from pytimeloop.looptree.des import LooptreeOutput


def explore_tile_shape(
    mapping,
    rank_shapes,
    compiled_result,
    max_capacity,
    max_fanout,
    tensors,
    only_count=False,
    prune=True
):
    ranks = []
    tile_constraints = []
    factor_constraints = []
    tensors_not_found = set(tensors)
    n_fusion_relevant_loops = None
    for node in mapping:
        if node["type"] in ["temporal", "spatial"] and "tile_shape" not in node:
            ranks.append(node["rank"])
            tile_constraint = []
            factor_constraint = []
            if "tile_constraint" in node:
                tile_constraint.append(node["tile_constraint"])
            if "factor_constraint" in node:
                factor_constraint.append(node["factor_constraint"])
            # if node["type"] == "temporal" and "factor_constraint" not in node and "tile_constraint" not in node:
            #     factor_constraint.append(">1")

            tile_constraints.append(tile_constraint)
            factor_constraints.append(factor_constraint)
        elif node["type"] == "storage" and tensors_not_found:
            tensors_not_found -= set(node["dspace"])
            if not tensors_not_found:
                n_fusion_relevant_loops = len(ranks)

    num_tile_shapes = 0
    num_valid_tile_shapes = 0

    shape_subspace = iter(ShapeSubspace(
            rank_shapes,
            ranks,
            tile_constraints=tile_constraints,
            factor_constraints=factor_constraints,
            n_fusion_relevant_loops=n_fusion_relevant_loops
    ))
    yield shape_subspace
    for shape in shape_subspace:
        num_tile_shapes += 1
        if only_count:
            continue

        result = LooptreeOutput()
        result.ops = call_with_arg(compiled_result.ops, shape)
        result.temporal_steps = call_with_arg(compiled_result.temporal_steps, shape)
        result.fanout = call_with_arg(compiled_result.fanout, shape)
        result.occupancy = call_with_arg(compiled_result.occupancy, shape)
        result.fills = call_with_arg(compiled_result.fills, shape)
        result.reads_to_parent = call_with_arg(compiled_result.reads_to_parent, shape)
        result.op_intensity = call_with_arg(compiled_result.op_intensity, shape)

        skip, valid = False, True

        total_capacity = defaultdict(lambda: 0)
        for (level, _), capacity in result.occupancy.items():
            total_capacity[level] += capacity
        for level, capacity in total_capacity.items():
            if level in max_capacity and capacity > max_capacity[level]:
                skip = True
                valid = False
                break

        if skip and prune:
            try:
                shape_subspace.skip_current_rank_iteration()
                continue
            except StopIteration:
                break

        for level, fanout in result.fanout.items():
            if level in max_fanout:
                valid = valid and reduce(mul, fanout, 1) <= reduce(mul, max_fanout[level], 1)

        if valid or not prune:
            num_valid_tile_shapes += valid
            yield shape, result, valid
            
    return num_tile_shapes, num_valid_tile_shapes


def call_with_arg(f, arg):
    if isinstance(next(iter(f.values())), tuple):
        return {k: (v[0], v[1](*arg)) for k, v in f.items()}
    else:
        return {k: v(*arg) for k, v in f.items()}

