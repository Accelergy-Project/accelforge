from .subspaces import (
    infer_smallest_tile_shape,
    LinearMapping,
    make_storage,
    make_temporal_fors,
)
from pytimeloop.looptree.mapping_utilities import get_last_fused_loop_idx
from fastfusion.exploration.constraints import DataflowConstraint

def make_subspaces(tensors,
                   intermediate_tensors,
                   tensor_to_relevant_ranks,
                   einsum_id,
                   workload,
                   dataflow_constraint: DataflowConstraint=None):
    def off_chip_storage(mapping):
        off_chip_must_retain = tensors - intermediate_tensors
        off_chip_can_retain = intermediate_tensors
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

    def fused_temporal_fors(mapping, unfused_tensors):
        for partial_mapping in make_temporal_fors(mapping,
                                                  all_ranks,
                                                  dataflow_constraint=dataflow_constraint):
            # for partial_mapping in make_temporal_fors(mapping, all_ranks):
            # for partial_mapping in make_temporal_fors_with_smallest_tile(partial_mapping, all_ranks):
            # print(partial_mapping)
            yield partial_mapping, unfused_tensors


    def glb_storage(mapping, unfused_tensors):
        glb_fused_tensors = intermediate_tensors - unfused_tensors
        last_fused_loop_idx = get_last_fused_loop_idx(mapping, intermediate_tensors)
        # last_fused_loop_idx = None
        for partial_mapping in make_storage(mapping,
                                            level=1,
                                            must_retain_tensors=intermediate_tensors,
                                            can_retain_tensors=set(),
                                            must_fully_reuse_tensors=glb_fused_tensors,
                                            tensor_to_relevant_ranks=tensor_to_relevant_ranks,
                                            explore_uneven=True,
                                            add_split_at_tensors=glb_fused_tensors,
                                            must_have_terminal_storage=False,
                                            apply_lrp_after_loop_idx=last_fused_loop_idx):
            last_fused_loop_idx = get_last_fused_loop_idx(partial_mapping, intermediate_tensors)
            for pm2 in make_storage(partial_mapping,
                                    level=1,
                                    must_retain_tensors=tensors - intermediate_tensors,
                                    can_retain_tensors=set(),
                                    must_fully_reuse_tensors=set(),
                                    tensor_to_relevant_ranks=tensor_to_relevant_ranks,
                                    explore_uneven=True,
                                    add_split_at_tensors=set(),
                                    must_have_terminal_storage=True,
                                    apply_lrp_after_loop_idx=last_fused_loop_idx):
                prev = None
                success = True
                # last_glb_index = len(partial_mapping) - 1
                # for i, node in enumerate(partial_mapping):
                #     if node["type"] == "storage" and node["target"] == 1:
                #         last_glb_index = i
                prev = None
                # for node in partial_mapping[last_glb_index:]:
                #     if node["type"] != "temporal":
                #         continue
                #     if prev is not None:
                #         if prev["rank"] < node["rank"]:
                #             success = False
                #     prev = node
                for i in range(last_fused_loop_idx + 1, len(pm2) - 1):
                    n1, n2 = pm2[i], pm2[i+1]
                    for ntype in ["temporal", "spatial"]:
                        if n1["type"] == ntype and n2["type"] == ntype:
                            if n1["rank"] < n2["rank"]:
                                success = False
                                break
                if success:
                    yield pm2
        # for partial_mapping in make_storage(mapping,
        #                                     level=1,
        #                                     must_retain_tensors=intermediate_tensors,
        #                                     can_retain_tensors=set(),
        #                                     must_fully_reuse_tensors=glb_fused_tensors,
        #                                     tensor_to_relevant_ranks=tensor_to_relevant_ranks,
        #                                     explore_uneven=False,
        #                                     add_split_at_tensors=glb_fused_tensors,
        #                                     must_have_terminal_storage=False,
        #                                     apply_lrp_after_loop_idx=None):
        #     for pm2 in make_storage(partial_mapping,
        #                             level=1,
        #                             must_retain_tensors=tensors - intermediate_tensors,
        #                             can_retain_tensors=set(),
        #                             must_fully_reuse_tensors=set(),
        #                             tensor_to_relevant_ranks=tensor_to_relevant_ranks,
        #                             explore_uneven=False,
        #                             add_split_at_tensors=set(),
        #                             must_have_terminal_storage=True,
        #                             apply_lrp_after_loop_idx=None):
        #         success = True
        #         prev = None
        #         # last_fused_loop_idx = get_last_fused_loop_idx(partial_mapping, intermediate_tensors)
        #         for node in partial_mapping:#[last_fused_loop_idx:]:
        #             if node["type"] != "temporal":
        #                 continue
        #             if prev is not None:
        #                 if prev["rank"] < node["rank"]:
        #                     success = False
        #             prev = node
        #         if success:
        #             yield pm2
        #         yield pm2


    def tile_shape_optimization(mapping):
        for partial_mapping in infer_smallest_tile_shape(mapping,
                                                         workload,
                                                         einsum_id,
                                                         tensor_to_relevant_ranks,
                                                         hw_level=1):
            yield partial_mapping

    def mac(mapping):
        mapping.add_compute(einsum_id, 2)
        yield mapping

    return [
        lambda: [LinearMapping()],
        off_chip_storage,
        fused_temporal_fors,
        glb_storage,
        tile_shape_optimization,
        mac
    ]