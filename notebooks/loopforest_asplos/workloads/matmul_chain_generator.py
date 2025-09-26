from ruamel.yaml import YAML
yaml = YAML(typ='safe')


def make_matmul(einsum_id, M_shape, K_shape, N_shape):
    return {
        'shape': {
            'name': f'Matmul{einsum_id}',
            'dimensions': [ f'M{einsum_id}', f'N{einsum_id}', f'K{einsum_id}'],
            'data_spaces': [
                {
                    'name': f'Fmap{einsum_id}',
                    'dimensions': [ f'Fmap{einsum_id}_M', f'Fmap{einsum_id}_K' ],
                    'projection': f'[ M{einsum_id}, K{einsum_id} ]'
                },
                {
                    'name': f'Filter{einsum_id}',
                    'dimensions': [ f'Filter{einsum_id}_K', f'Filter{einsum_id}_N' ],
                    'projection': f'[ K{einsum_id}, N{einsum_id} ]'
                },
                {
                    'name': f'Fmap{einsum_id+1}',
                    'dimensions': [ f'Fmap{einsum_id+1}_M', f'Fmap{einsum_id+1}_K' ],
                    'projection': f'[ M{einsum_id}, N{einsum_id} ]'
                }
            ]
        },
        'instance': f'0 <= M{einsum_id} < {M_shape} and 0 <= N{einsum_id} < {N_shape} and 0 <= K{einsum_id} < {K_shape}'
    }


def make_chain_of_matmuls(n_matmuls, M_shape, reduced_rank_shapes):
    """
    Make a chain of `n_matmuls` matmuls. The M shape is the same
    for all matmuls. The K/N rank shapes are specified as a list
    `reduced_rank_shapes` with `n_matmuls+1` integer elements.
    """
    list_of_einsums = []
    for einsum_id in range(n_matmuls):
        K_shape = reduced_rank_shapes[einsum_id]
        N_shape = reduced_rank_shapes[einsum_id+1]
        list_of_einsums.append(make_matmul(einsum_id,
                                           M_shape,
                                           K_shape,
                                           N_shape))
    return {'problem': list_of_einsums}


def make_chain_of_matmuls_yaml(fname, *args, **kwargs):
    """
    Dumps the result of `make_chain_of_matmuls` to `fname` in YAML
    format. All arguments after `fname` are passed to
    `make_chain_of_matmuls`.
    """
    with open(fname, 'w') as f:
        yaml.dump(
            make_chain_of_matmuls(*args, **kwargs),
            f
        )