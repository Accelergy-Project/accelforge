from collections import defaultdict
from functools import reduce
from operator import or_

from fastfusion.fastmodel import compile_mapping
from fastfusion.exploration.constraints import *
from fastfusion.util.logging import log_worker
from .per_einsum_subspaces.subspaces import (
    LinearMapping,
    make_temporal_fors,
    make_spatial_fors,
    make_storage,
    explore_tile_shape
)

from fastfusion.exploration.process_results import Metrics, process_result

from pytimeloop.looptree.equivalent_ranks import EquivalentGroups
from pytimeloop.looptree.mapping_utilities import get_intermediate_tensors

from bindings.looptree import LooptreeWorkload, LooptreeWorkloadDependencyAnalyzer


@log_worker(f"{__name__}:_mapper_place_fusion_level")
def mapper_place_fusion_level(
    config,
    pe_array_constraint: PeArrayConstraint,
    mac_array_constraint: MacArrayConstraint,
    spec,
    explore_glb_uneven,
    explore_pe_uneven,
    einsum_id,
    energy_dict,
    partial_mapping,
    log_queue=None,
    verbose_stream=None,
    metrics=Metrics.all_metrics(),
):
    # if log_queue is not None:
    #     log_queue.info(f"[{einsum_id}] Exploring mapspace of Einsum {einsum_id}")
    logfunc = lambda msg: None # log_queue.debug(f"[{einsum_id}] " + msg)

    workload = LooptreeWorkload.parse_cfg(config.root["problem"])
    analyzer = LooptreeWorkloadDependencyAnalyzer(workload)
    equivalent_groups = EquivalentGroups.from_workload(workload, analyzer)

    einsum_id_to_name = workload.einsum_id_to_name()
    rank_name_to_id   = workload.dimension_name_to_id()
    tensor_name_to_id = workload.data_space_name_to_id()

    mac_parallel_shape = mac_array_constraint.array_shape_in_parallel_dimension
    mac_reduced_shape = mac_array_constraint.array_shape_in_reduced_dimension

    einsum_name_to_parallel_rank_name = mac_array_constraint.parallel_rank
    einsum_name_to_reduced_rank_name = mac_array_constraint.reduced_rank

    bindings, max_fanout, max_capacity = get_hardware_levels(spec.architecture)

    data = defaultdict(list)
    tensors = workload.tensors_read_by_einsum(einsum_id) \
            | workload.tensors_written_by_einsum(einsum_id)
    intermediate_tensors = tensors & get_intermediate_tensors(workload)

    einsum_name = einsum_id_to_name[einsum_id]
    mac_parallel_rank_name = einsum_name_to_parallel_rank_name[einsum_name]
    mac_parallel_rank_name = rank_name_to_id[mac_parallel_rank_name]
    mac_reduced_rank_name = einsum_name_to_reduced_rank_name[einsum_name]
    mac_reduced_rank_name = rank_name_to_id[mac_reduced_rank_name]

    weight_tensor_name = mac_array_constraint.weight_tensor[einsum_name]
    weight_tensor_name = tensor_name_to_id[weight_tensor_name]
    weight_ranks = analyzer.einsum_dims_relevant_to_tensor(einsum_id, weight_tensor_name)
    other_weight_ranks = weight_ranks - {mac_parallel_rank_name, mac_reduced_rank_name}
    all_ranks = workload.einsum_ospace_dimensions(einsum_id)
    non_weight_ranks = set(all_ranks) - weight_ranks

    tensor_to_relevant_ranks = {
        tensor: analyzer.einsum_dims_relevant_to_tensor(einsum_id, tensor)
        for tensor in tensors
    }

    einsum_shape = {
        rank_name: workload.get_rank_shape(rank_name)[1] + 1 for rank_name in all_ranks
    }
    count = 0

    for partial_mapping in make_temporal_fors(  # PE temporal
        partial_mapping, all_ranks
    ):
        # No bypassing at PE level. Can relax to explore more mappings
        pe_must_retain = tensors
        pe_can_retain = set()
        for partial_mapping in make_storage(  # PE storage
            partial_mapping,
            level=2,  # PE level
            must_retain_tensors=pe_must_retain,
            can_retain_tensors=pe_can_retain,
            tensor_to_relevant_ranks=tensor_to_relevant_ranks,
            explore_uneven=explore_pe_uneven,
        ):
            found_storage = set()
            fail = False
            for i, p in enumerate(partial_mapping):
                if p["type"] == "storage":
                    for t in set(p["dspace"]) - found_storage:
                        for p2 in partial_mapping[:i]:
                            if p2["type"] in ["temporal", "spatial"] and p2["rank"] not in tensor_to_relevant_ranks[t]:
                                fail = True
                                break
                    found_storage |= set(p["dspace"])
                if len(found_storage) < len(tensors) or i == 0:
                    continue
                prev = partial_mapping[i - 1]
                for t in ["spatial"]: # "temporal", TEMPORAL DOESN"T WORK. WEIRD INTERACTIONS WITH LOOP RELEVANCE PRINCIPLE
                    if p["type"] == t and prev["type"] == t and p["rank"] < prev["rank"]:
                        fail = True
                        # print(f'Skipping partial mapping: {partial_mapping}')
            if fail:
                continue
            for partial_mapping in make_mac_level_loops(
                partial_mapping,
                einsum_id,
                mac_parallel_rank_name,
                mac_parallel_shape,
                mac_reduced_rank_name,
                mac_reduced_shape,
                non_weight_ranks,
                other_weight_ranks,
            ):
                _, compiled_results = compile_mapping(
                    partial_mapping, workload, analyzer
                )
                # print(f'Einsum {einsum_id} partial mapping: {partial_mapping}')
                tile_shape_explorer = explore_tile_shape(
                    partial_mapping,
                    einsum_shape,
                    compiled_results,
                    max_capacity,
                    max_fanout,
                    tensors=tensors,
                )
                # HACKY: Pop out the subspace object as the first in the iterator
                shape_subspace = next(tile_shape_explorer)

                for shape, res in tile_shape_explorer:
                    count += 1
                    # print(f'Running partial mapping: {partial_mapping} with shape: {shape}')
                    process_result(
                        res,
                        shape,
                        data,
                        einsum_id,
                        intermediate_tensors,
                        partial_mapping,
                        bindings,
                        workload,
                        energy_dict,
                        equivalent_groups,
                        logfunc=logfunc,
                        explore_fusion_uneven=explore_glb_uneven,
                        einsum_shape=einsum_shape,
                        metrics=metrics,
                        einsum_id_to_name=einsum_id_to_name,
                        rank_name_to_name={v: k for k, v in rank_name_to_id.items()},
                        tensor_name_to_name={v: k for k, v in tensor_name_to_id.items()},
                    )
                    if count % 1e4 == 0:
                        print(f"Einsum {einsum_id} #{count}, fulltiling: {fulltiling}")
                    # shape_subspace.register_result(is_pareto, results)
    # assert False
    return einsum_id, data, count


@log_worker(f"{__name__}:_get_top_loop_jobs")
def get_top_loop_jobs(
    config,
    pe_array_constraint: PeArrayConstraint,
    mac_array_constraint: MacArrayConstraint,
    spec,
    explore_glb_uneven,
    explore_pe_uneven,
    einsums_to_explore,
    energy_dict,
    log_queue=None,
    verbose_stream=None,
    metrics=Metrics.all_metrics(),
):
    args = []
    for einsum_id in einsums_to_explore:
        if log_queue is not None:
            log_queue.info(f"[{einsum_id}] Exploring mapspace of Einsum {einsum_id}")
            logfunc = lambda msg: log_queue.debug(f"[{einsum_id}] " + msg)
        else:
            logfunc = lambda msg: None  # do nothing

        workload = LooptreeWorkload.parse_cfg(config.root["problem"])
        analyzer = LooptreeWorkloadDependencyAnalyzer(workload)

        data = {}
        data[einsum_id] = defaultdict(lambda: defaultdict(lambda: list()))
        tensors = workload.tensors_read_by_einsum(einsum_id) \
                | workload.tensors_written_by_einsum(einsum_id)
        intermediate_tensors = tensors & get_intermediate_tensors(workload)
        all_ranks = workload.einsum_ospace_dimensions(einsum_id)


        tensor_to_relevant_ranks = {
            tensor: analyzer.einsum_dims_relevant_to_tensor(einsum_id, tensor)
            for tensor in tensors
        }

        top_level_ranks = reduce(
            or_, (tensor_to_relevant_ranks[t] for t in intermediate_tensors), set()
        )

        mapping = LinearMapping()
        logfunc(f"Allowed top-level loop ranks: {top_level_ranks}")

        off_chip_must_retain = tensors - intermediate_tensors
        off_chip_can_retain = intermediate_tensors
        for partial_mapping in make_storage(  # Off-chip level
            mapping,
            level=0,
            must_retain_tensors=off_chip_must_retain,
            can_retain_tensors=off_chip_can_retain,
            tensor_to_relevant_ranks=tensor_to_relevant_ranks,
            explore_uneven=False,
            add_split_at_tensors=intermediate_tensors,
        ):
            for partial_mapping in make_temporal_fors(  # GLB temporal
                partial_mapping,
                top_level_ranks,
            ):
                glb_must_retain = set(intermediate_tensors)
                glb_can_retain = set()
                for partial_mapping in make_storage(  # GLB level
                    partial_mapping,
                    level=1,
                    must_retain_tensors=glb_must_retain,
                    can_retain_tensors=glb_can_retain,
                    must_fully_reuse_tensors=set(intermediate_tensors),
                    tensor_to_relevant_ranks=tensor_to_relevant_ranks,
                    explore_uneven=explore_glb_uneven,
                    add_split_at_tensors=intermediate_tensors,
                    must_have_terminal_storage=True,  # GLB only opt.
                    logfunc=None
                ):                   
                    for partial_mapping in make_spatial_fors(  # PE spatial
                        partial_mapping,
                        all_ranks,
                        max_factor=pe_array_constraint.array_shape
                    ):
                        # found_storage = set()
                        # fail = False
                        # for i, p in enumerate(partial_mapping):
                        #     if p["type"] == "storage":
                        #         found_storage |= set(p["dspace"])
                        #     if len(found_storage) < len(tensors) or i == 0:
                        #         continue
                        #     prev = partial_mapping[i - 1]
                        #     for t in ["temporal", "spatial"]:
                        #         if p["type"] == t and prev["type"] == t and p["rank"] < prev["rank"]:
                        #             fail = True
                        #             print(f'Skipping partial mapping: {partial_mapping}')
                        # if fail:
                        #     continue
                        print(f'Partial mapping: {partial_mapping}')
                        args.append(dict(
                            config=config,
                            pe_array_constraint=pe_array_constraint,
                            mac_array_constraint=mac_array_constraint,
                            spec=spec,
                            explore_glb_uneven=explore_glb_uneven,
                            explore_pe_uneven=explore_pe_uneven,
                            einsum_id=einsum_id,
                            energy_dict=energy_dict,
                            partial_mapping=partial_mapping,
                            log_queue=log_queue,
                            verbose_stream=verbose_stream,
                            metrics=metrics,
                        ))
    return args


def make_mac_level_loops(
    mapping,
    einsum_id,
    parallel_rank,
    parallel_rank_shape,
    reduced_rank,
    reduced_rank_shape,
    non_weight_ranks,
    other_weight_ranks,
):
    mapping = mapping.copy()
    for rank in other_weight_ranks:
        mapping.add_temporal(rank, 1)
    mapping.add_temporal(parallel_rank, parallel_rank_shape)
    mapping.add_temporal(reduced_rank, reduced_rank_shape)
    for rank in non_weight_ranks:
        mapping.add_temporal(rank, 1)
    mapping.add_spatial(parallel_rank, 1)
    mapping.add_spatial(reduced_rank, 1)
    mapping.add_compute(einsum_id, 3)
    yield mapping


def get_hardware_levels(arch):
    bindings = {}
    fanout = {}
    max_capacity = {}
    words_per_read = {}
    for node in arch["nodes"]:
        bindings_id = len(bindings)
        bindings[bindings_id] = node["name"]
        fanout[bindings_id] = (node.spatial.meshX, node.spatial.meshY)
        attribute = node.attributes
        if "width" in attribute and "depth" in attribute:
            width = attribute.width
            depth = attribute.depth
            datawidth = attribute.datawidth
            if all(x is not None for x in (width, depth, datawidth)):
                max_capacity[bindings_id] = (
                    attribute.width * attribute.depth / attribute.datawidth
                )
            if all(x is not None for x in (width, datawidth)):
                words_per_read[bindings_id] = width // datawidth
    return bindings, fanout, max_capacity, words_per_read


def count(it):
    count = 0
    for _ in it:
        count += 1
    return count
