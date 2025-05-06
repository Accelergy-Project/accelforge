import sympy


def lambdify(d, tile_shapes):
    if isinstance(next(iter(d.values())), tuple):
        return {
            k: (v[0], sympy.lambdify(tile_shapes, v[1]))
            for k, v in d.items()
        }
    else:
        return {
            k: sympy.lambdify(tile_shapes, v)
            for k, v in d.items()
        }


def compile_analysis_result(result, tile_shapes):
    lambdify_with_tile_shapes = lambda x: lambdify(x, tile_shapes)

    result.ops = lambdify_with_tile_shapes(result.ops)
    result.temporal_steps = lambdify_with_tile_shapes(result.temporal_steps)
    result.fanout = lambdify_with_tile_shapes(result.fanout)
    result.occupancy = lambdify_with_tile_shapes(result.occupancy)
    result.fills = lambdify_with_tile_shapes(result.fills)
    result.reads_to_parent = lambdify_with_tile_shapes(result.reads_to_parent)
    result.op_intensity = lambdify_with_tile_shapes(result.op_intensity)

    return result