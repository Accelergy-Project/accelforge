from .subspaces import (
    infer_smallest_tile_shape,
    LinearMapping,
    make_storage,
    make_temporal_fors,
    make_temporal_fors_with_smallest_tile,
    make_spatial_fors
)
from pytimeloop.looptree.mapping_utilities import get_last_fused_loop_idx
from fastfusion.mapper.FFM.exploration.constraints import DataflowConstraint


"""
Mapping constraint:
DRAM
----
for loop over any rank
----
GlobalBuffer
----
par-for loop over H
par-for loop over M
for loop over any rank
----
LocalBuffer
----
par-for loop over M
par-for loop over C
for loop over any rank
----
Register
----
leftover for loop
"""

EINSUM_ID_TO_FULLY_PARALLEL_RANKS = {
    "I": {"BI", "MI", "DI"},
    "Q": set(),
    "K": set(),
    "V": set(),
    "QK": {"HQK", "BQK"},
    "AV": {"HAV", "BAV"},
    "Z": set(),
    "FFA": set(),
    "FFB": set(),
}

EINSUM_ID_TO_OUTPUT_PARALLEL_RANKS = {
    "I": set(),
    "Q": {"HQ", "EQ"},
    "K": {"HK", "EK"},
    "V": {"HV", "EV"},
    "QK": {"PQK"},
    "AV": {"FAV"},
    "Z": {"GZ"},
    "FFA": {"CFFA"},
    "FFB": {"CFFB"},
}

EINSUM_ID_TO_REDUCED_RANKS = {
    "I": set(),
    "Q": {"DQ"},
    "K": {"DK"},
    "V": {"DV"},
    "QK": {"EQK"},
    "AV": {"PAV"},
    "Z": {"HZ", "FZ"},
    "FFA": {"GFFA"},
    "FFB": {"JFFB"},
}

EINSUM_ID_TO_INPUT_OUTPUT_RANKS = {
    "I": set(),
    "Q": {"MQ", "BQ"},
    "K": {"MK", "BK"},
    "V": {"MV", "BV"},
    "QK": {"MQK"},
    "AV": {"MAV"},
    "Z": {"MZ", "BZ"},
    "FFA": {"MFFA", "BFFA"},
    "FFB": {"MFFB", "BFFB"}
}

EINSUM_ID_TO_WEIGHT_LIKE_TENSOR = {
    "I": "I_n_to_I",
    "Q": "W_n_to_Q",
    "K": "W_n_to_K",
    "V": "W_n_to_V",
    "QK": "K_K_to_QK",
    "AV": "V_V_to_AV",
    "Z": "W_n_to_Z",
    "FFA": "W_n_to_FFA",
    "FFB": "W_n_to_FFB",
}

EINSUM_ID_TO_INPUT_TENSOR = {
    "I": "I_n_to_I",
    "Q": "I_I_to_Q_K_V",
    "K": "I_I_to_Q_K_V",
    "V": "I_I_to_Q_K_V",
    "QK": "Q_Q_to_QK",
    "AV": "QK_QK_to_AV",
    "Z": "AV_AV_to_Z",
    "FFA": "Z_Z_to_FFA",
    "FFB": "FFA_FFA_to_FFB",
}

for i in range(1, 64):
    matmul_name = f"Matmul{i}"
    m, k, n = f"M{i}", f"K{i}", f"N{i}"
    EINSUM_ID_TO_FULLY_PARALLEL_RANKS[matmul_name] = set()
    EINSUM_ID_TO_OUTPUT_PARALLEL_RANKS[matmul_name] = {n}
    EINSUM_ID_TO_REDUCED_RANKS[matmul_name] = {k}
    EINSUM_ID_TO_INPUT_OUTPUT_RANKS[matmul_name] = {m}
    EINSUM_ID_TO_WEIGHT_LIKE_TENSOR[matmul_name] = f"Filter{i}"
    EINSUM_ID_TO_INPUT_TENSOR[matmul_name] = f"Fmap{i}"

for i in range(1, 64):
    matmul_name = f"MatmulB{i}"
    m, k, n = f"MB{i}", f"KB{i}", f"NB{i}"
    EINSUM_ID_TO_FULLY_PARALLEL_RANKS[matmul_name] = set()
    EINSUM_ID_TO_OUTPUT_PARALLEL_RANKS[matmul_name] = {n}
    EINSUM_ID_TO_REDUCED_RANKS[matmul_name] = {k}
    EINSUM_ID_TO_INPUT_OUTPUT_RANKS[matmul_name] = {m}
    EINSUM_ID_TO_WEIGHT_LIKE_TENSOR[matmul_name] = f"FilterB{i}"
    EINSUM_ID_TO_INPUT_TENSOR[matmul_name] = f"FmapB{i}"

def make_subspaces(tensors,
                   intermediate_tensors,
                   tensor_to_relevant_ranks,
                   einsum_id,
                   workload,
                   dataflow=None,
                   fuse=True,
                   ):
    """
    fully_parallel_ranks: in all tensors
    output_parallel_ranks: not in input
    reduced_ranks: not in output
    input_output_ranks: not in weight
    weight_like_tensor: tensor that will be stationary in systolic array
    """
    einsum_name = workload.einsum_id_to_name()[einsum_id]
    fully_parallel_ranks = {
        workload.dimension_name_to_id()[r]
        for r in EINSUM_ID_TO_FULLY_PARALLEL_RANKS[einsum_name]
    }
    output_parallel_ranks = {
        workload.dimension_name_to_id()[r]
        for r in EINSUM_ID_TO_OUTPUT_PARALLEL_RANKS[einsum_name]
    }
    reduced_ranks = {
        workload.dimension_name_to_id()[r]
        for r in EINSUM_ID_TO_REDUCED_RANKS[einsum_name]
    }
    input_output_ranks = {
        workload.dimension_name_to_id()[r]
        for r in EINSUM_ID_TO_INPUT_OUTPUT_RANKS[einsum_name]
    }
    weight_like_tensor = EINSUM_ID_TO_WEIGHT_LIKE_TENSOR[einsum_name]
    weight_like_tensor = workload.data_space_name_to_id()[weight_like_tensor]
    input_tensor = EINSUM_ID_TO_INPUT_TENSOR[einsum_name]
    input_tensor = workload.data_space_name_to_id()[input_tensor]
    output_tensor = next(iter(workload.tensors_written_by_einsum(einsum_id)))

    def off_chip_storage(mapping):
        if fuse:
            off_chip_must_retain = tensors - intermediate_tensors
            off_chip_can_retain = intermediate_tensors
        else:
            off_chip_must_retain = tensors
            off_chip_can_retain = set()
        yield from make_storage(
            mapping,
            level=0,
            must_retain_tensors=off_chip_must_retain,
            can_retain_tensors=off_chip_can_retain,
            tensor_to_relevant_ranks=tensor_to_relevant_ranks,
            explore_uneven=False,
            add_split_at_tensors=intermediate_tensors,
            return_retained_tensors=True,
            apply_lrp_after_loop_idx=None,
        )

    all_ranks = list(sorted(workload.einsum_ospace_dimensions(einsum_id)))
    assert fully_parallel_ranks | output_parallel_ranks | reduced_ranks | input_output_ranks == set(all_ranks)

    if dataflow == 'IS':
        bottom_ranks = output_parallel_ranks
        stationary_tensors = {input_tensor}
    elif dataflow == 'OS':
        bottom_ranks = reduced_ranks
        stationary_tensors = {output_tensor}
    elif dataflow == 'WS':
        bottom_ranks = input_output_ranks
        stationary_tensors = {weight_like_tensor}
    else:
        stationary_tensors = set()

    if dataflow is None:
        def fused_temporal_fors(mapping, unfused_tensors):
            for partial_mapping in make_temporal_fors(mapping,
                                                      all_ranks,
                                                      min_loops=0):
                yield partial_mapping, unfused_tensors, 0
    else:
        if 'IS' in dataflow:
            bottom_ranks = output_parallel_ranks
        elif 'OS' in dataflow:
            bottom_ranks = reduced_ranks
        elif 'WS' in dataflow:
            bottom_ranks = input_output_ranks

        def fused_temporal_fors(mapping, unfused_tensors):
            for pm in make_temporal_fors(mapping,
                                         set(all_ranks)-bottom_ranks,
                                         min_loops=0):
                for pm2 in make_storage(pm,
                                        level=1,
                                        must_retain_tensors=stationary_tensors,
                                        can_retain_tensors=set(),
                                        must_fully_reuse_tensors=set(),
                                        tensor_to_relevant_ranks=tensor_to_relevant_ranks,
                                        explore_uneven=True,
                                        add_split_at_tensors=stationary_tensors & unfused_tensors,
                                        must_have_terminal_storage=False,
                                        force_at_end=True):
                    stationary_index = len(pm2)
                    for pm3 in make_temporal_fors(pm2,
                                                bottom_ranks,
                                                min_loops=0):
                        yield pm3, unfused_tensors, stationary_index

    def glb_storage(mapping, unfused_tensors, stationary_index):
        glb_fused_tensors = intermediate_tensors - unfused_tensors
        last_fused_loop_idx = get_last_fused_loop_idx(mapping, intermediate_tensors)
        # last_fused_loop_idx = None
        if dataflow is not None and 'Unpinned' in dataflow:
            min_index, max_index = None, stationary_index
        elif dataflow is not None and 'Pinned' in dataflow:
            min_index, max_index = stationary_index, None
        else:
            min_index, max_index = None, None
        for partial_mapping in make_storage(mapping,
                                            level=1,
                                            must_retain_tensors=intermediate_tensors - stationary_tensors,
                                            can_retain_tensors=set(),
                                            must_fully_reuse_tensors=glb_fused_tensors,
                                            tensor_to_relevant_ranks=tensor_to_relevant_ranks,
                                            explore_uneven=True,
                                            add_split_at_tensors=glb_fused_tensors,
                                            must_have_terminal_storage=False,
                                            apply_lrp_after_loop_idx=last_fused_loop_idx,
                                            min_index=min_index,
                                            max_index=max_index):
            last_fused_loop_idx = get_last_fused_loop_idx(partial_mapping, intermediate_tensors)
            for pm2 in make_storage(partial_mapping,
                                    level=1,
                                    must_retain_tensors=set(),
                                    can_retain_tensors=tensors - intermediate_tensors - stationary_tensors,
                                    must_fully_reuse_tensors=set(),
                                    tensor_to_relevant_ranks=tensor_to_relevant_ranks,
                                    explore_uneven=True,
                                    add_split_at_tensors=set(),
                                    must_have_terminal_storage=True,
                                    apply_lrp_after_loop_idx=last_fused_loop_idx,
                                    min_index=min_index,
                                    max_index=max_index):

                success = True
                for i in range(last_fused_loop_idx + 1, len(pm2) - 1):
                    n1, n2 = pm2[i], pm2[i+1]
                    for ntype in ["temporal", "spatial"]:
                        if n1["type"] == ntype and n2["type"] == ntype:
                            if n1["rank"] <= n2["rank"]:
                                success = False
                                break
                if success:
                    yield pm2

    def core_spatial_fors(mapping):
        ranks = fully_parallel_ranks | output_parallel_ranks
        yield from make_spatial_fors(mapping, ranks, 4, unordered=True)

    def core_temporal_fors(mapping):
        for pm in make_temporal_fors_with_smallest_tile(mapping,
                                                        fully_parallel_ranks,
                                                        unordered=True):
            for pm2 in make_temporal_fors(pm, input_output_ranks,
                                          unordered=True):
                yield from make_temporal_fors(pm2,
                                              output_parallel_ranks,
                                              unordered=True)

    def llb_storage(mapping):
        yield from make_storage(mapping,
                                level=2,
                                must_retain_tensors={output_tensor},
                                can_retain_tensors=set(),
                                tensor_to_relevant_ranks=tensor_to_relevant_ranks,
                                explore_uneven=False)

    def pe_spatial_fors(mapping):
        for pm in make_spatial_fors(mapping, output_parallel_ranks, 128, unordered=True):
            yield from make_spatial_fors(pm, reduced_ranks, 128, unordered=True)

    def pe_temporal_fors(mapping):
        for pm in make_temporal_fors_with_smallest_tile(mapping,
                                                        output_parallel_ranks,
                                                        unordered=True):
            yield from make_temporal_fors_with_smallest_tile(pm,
                                                            reduced_ranks,
                                                            unordered=True)

    def register_storage(mapping):
        yield from make_storage(mapping,
                                level=3,
                                must_retain_tensors={weight_like_tensor},
                                can_retain_tensors=set(),
                                tensor_to_relevant_ranks=tensor_to_relevant_ranks,
                                explore_uneven=False)

    def mac_temporal_fors(mapping):
        yield from make_temporal_fors_with_smallest_tile(mapping,
                                                         input_output_ranks,
                                                         unordered=True)

    def mac(mapping):
        mapping.add_compute(einsum_id, 4)
        yield mapping

    return [
        lambda: [LinearMapping()],
        off_chip_storage,
        fused_temporal_fors,
        glb_storage,
        core_spatial_fors,
        core_temporal_fors,
        llb_storage,
        pe_spatial_fors,
        pe_temporal_fors,
        register_storage,
        mac_temporal_fors,
        mac
    ]