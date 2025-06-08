from fastfusion.frontend.mapping import Iteration


def get_nondegenerate_loops_above_tensor(
    mapping,
    tile_shape: list[int],
    rank_variable_bounds: dict[RankVariableName, int],
    tensor2size: dict[str, int]
):
        storages = []
        null_loop_indices = set()

        for node in mapping:
            if isinstance(node, Iteration):

        for i, (t, l) in enumerate(zip(tile_shape, self.loops)):
            prev_size = rank_variable_bounds[l.rank_variable]
            if i > 0:
                prev_loop = next(
                    iter(
                        l2
                        for l2 in new_loops[i - 1 :: -1]
                        if l2.rank_variable == l.rank_variable
                    ),
                    None,
                )
                if prev_loop is not None:
                    prev_size = tile_shape[new_loops.index(prev_loop)]
            if prev_size == t:
                null_loop_indices.add(i)
            else:
                new_loops[i] = l.update(bound=t)

        new_loops = [l for i, l in enumerate(new_loops) if i not in null_loop_indices]

        storages = []
        for s in self.storage:
            above = s.above_loop_index
            above -= sum(above > i for i in null_loop_indices)
            storages.append(s.update(above_loop_index=above, size=tensor2size[s.name]))