from copy import deepcopy
from collections import defaultdict
import itertools
import time

from joblib import delayed

from combinatorics.dependent_product import dependent_product
from combinatorics.splitter import split_dependent_product
import pandas as pd

from pytimeloop.fastfusion.fastmodel import compile_mapping
from pytimeloop.fastfusion.mapper.constraints import *
from pytimeloop.fastfusion.mapper.per_einsum_mapper import (
    explore_tile_shape,
    process_result,
    get_hardware_levels,
)
from pytimeloop.fastfusion.mapper.per_einsum_subspaces.snowcat import make_subspaces
from pytimeloop.fastfusion.mapper.per_einsum_subspaces.snowcat_ffmt import (
    make_ffmt_subspaces,
)
from pytimeloop.fastfusion.mapper.per_einsum_subspaces.four_level_arch import (
    make_subspaces as make_four_level_subspaces,
)
from pytimeloop.fastfusion.pareto import MAPPING, Pareto, makepareto
from pytimeloop.fastfusion.sim import SIM, Loop, Tiling, TensorStorage
from pytimeloop.fastfusion.util import fzs, parallel
from pytimeloop.looptree.equivalent_ranks import EquivalentGroups
from pytimeloop.looptree.mapping_utilities import get_intermediate_tensors
from pytimeloop.fastfusion.mapper.process_results import Metrics, process_result

from bindings.looptree import LooptreeWorkload, LooptreeWorkloadDependencyAnalyzer
from combinatorics.integer import integer_factorizations_to_n_parts


import functools
import re


@functools.lru_cache(maxsize=None)
def parse_constraint(constraint_str: str):
    parser = re.compile("^(>=|>|<|<=|==)\s*(\d+)")
    match = parser.match(constraint_str)
    if match is None:
        raise RuntimeError(f"Cannot parse constraint {constraint_str}")
    comparison, limit = match.groups()
    limit = int(limit)
    # if limit == 128:
    #     comparison = "=="
    if comparison == ">=":
        return lambda x: x >= limit  # , lambda x: x >= limit
    elif comparison == ">":
        return lambda x: x > limit  # , lambda x: x > limit
    elif comparison == "<=":
        return lambda x: x <= limit  # , None
    elif comparison == "<":
        return lambda x: x < limit  # , None
    elif comparison == "==":
        return lambda x: x == limit  # , lambda x: x >= limit
    else:
        raise RuntimeError(f"Unknown comparison operator {comparison}")


def join_string_columns(df, cols, str_base):
    # https://stackoverflow.com/questions/39291499/how-to-concatenate-multiple-column-values-into-a-single-column-in-pandas-datafra
    # Response from 3UqU57GnaX
    columns_strings = [f"df['{c}']" for c in cols]
    return eval("+','+".join(columns_strings) + f"+','+'{str_base}'")


def per_worker_exploration(
    workload,
    task_spaces,
    einsum_shape,
    max_capacity,
    max_fanout,
    tensors,
    einsum_id,
    intermediate_tensors,
    bindings,
    energy_dict,
    bandwidth_dict,
    equivalent_groups,
    explore_glb_uneven,
    metrics,
    einsum_id_to_name,
    rank_id_to_name,
    tensor_id_to_name,
    rank_name_to_shared_name,
    input_tensors,
    output_tensors,
    tag_with,
    tensor_to_relevant_ranks,
    task_space_args,
    prune,
):
    # analyzer = LooptreeWorkloadDependencyAnalyzer(workload)
    local_task_spaces = deepcopy(task_spaces)
    local_task_spaces[0] = lambda: task_spaces[0](*task_space_args)
    result = {}
    n_mappings = 0

    t0 = time.time()
    for partial_mapping in dependent_product(local_task_spaces):
        # _, compiled_results = compile_mapping(
        #     partial_mapping, workload, analyzer
        # )

        all_loops = []
        all_storages = []
        all_dspaces = set()
        n_fused_loops = []
        backing_storage = []
        for i, pm in enumerate(partial_mapping):
            if pm["type"] == "spatial" or pm["type"] == "temporal":
                pm["loop index"] = len(all_loops)
                pm["index"] = i
                all_loops.append(pm)
            elif pm["type"] == "storage":
                pm["storage index"] = len(all_storages)
                pm["index"] = i
                all_storages.append(pm)
                dspaces = set(pm["dspace"])
                backed_tensors = dspaces - all_dspaces
                if backed_tensors:
                    all_dspaces |= dspaces
                    n_fused_loops = len(all_loops)

                for t in backed_tensors:
                    backing_storage.append(
                        TensorStorage(
                            tensor_id_to_name[t],
                            len(all_loops),
                            bindings[pm["target"]],
                            0,  # TODO: fill out tile size here
                        )
                    )

        backing_storage = fzs(backing_storage)

        rank2loops = {}
        for loop in all_loops:
            rank2loops.setdefault(loop["rank"], []).append(loop)

        all_dfs = {}

        # Create a dataframe with the factors for each rank
        for rank_id, loops in rank2loops.items():
            rank_size = einsum_shape[rank_id]
            rank = rank_id_to_name[rank_id]
            choices = list(integer_factorizations_to_n_parts(rank_size, len(loops)))

            for i, loop in enumerate(loops):
                loop["column name"] = f"rank{rank} loop {i}"

            # Check if choices satisfy constraints here
            satisfies_constrains = [True] * len(choices)
            for i, choice in enumerate(choices):
                for c, loop in enumerate(loops):
                    if "factor_constraint" in loop:
                        factor_constraint = parse_constraint(loop["factor_constraint"])
                        if not factor_constraint(choice[c]):
                            satisfies_constrains[i] = False
                            break
                    # TODO: Tile constraint
            choices = [
                choice for i, choice in enumerate(choices) if satisfies_constrains[i]
            ]

            df = pd.DataFrame(choices, columns=[l["column name"] for l in loops])
            for s in all_storages:
                colname = f"storage {s['target']} rank{rank} repeats"
                df[colname] = 1
                colname = f"storage {s['target']} rank{rank} tile size"
                df[colname] = 1

            for l in loops:
                storages_below = [s for s in all_storages if s["index"] > l["index"]]
                for s in storages_below:
                    colname = f"storage {s['target']} rank{rank} repeats"
                    df.loc[:, colname] *= df[l["column name"]]

            for s in all_storages:
                colfrom = f"storage {s['target']} rank{rank} repeats"
                colto = f"storage {s['target']} rank{rank} tile size"
                df.loc[:, colto] = rank_size // df[colfrom]

            all_dfs[rank_id] = df

        # Put together a dataframe to populate
        df = pd.DataFrame(
            columns=[
                f"storage {s['target']} tensor{tensor_id_to_name[dspace]} tile size"
                for s in all_storages
                for dspace in s["dspace"]
            ]
            + [
                f"storage {s['target']} tensor{tensor_id_to_name[dspace]} repeats"
                for s in all_storages
                for dspace in s["dspace"]
            ]
        )
        df.loc[0] = 1

        # For each rank, merge the dataframes. We update the tile size and repeats
        # as early as possible because the dataframe will grow with each rank
        # we add
        for rank_id, df2 in all_dfs.items():
            df = df.merge(df2, how="cross")
            rank = rank_id_to_name[rank_id]
            for l in rank2loops[rank_id]:
                for s in all_storages:
                    for dspace in s["dspace"]:
                        tensor_name = tensor_id_to_name[dspace]
                        if rank not in tensor_to_relevant_ranks[tensor_name]:
                            continue
                        for target in ["tile size", "repeats"]:
                            colfrom = f"storage {s['target']} rank{rank} {target}"
                            colto = (
                                f"storage {s['target']} tensor{tensor_name} {target}"
                            )
                            df.loc[:, colto] *= df[colfrom]

        if len(df) == 0:
            continue
        n_mappings += len(df)

        # Trim to only the columns we care about
        # TODO: Preserve fused loops
        # TODO: Record everything in some way so we can recover the mappings

        # Memory usage. We do this earlier so we can clear invalid mappings
        for s in all_storages:
            colto = f"{s['target']} usage"
            if colto not in df.columns:
                df[colto] = 0
            for d in s["dspace"]:
                tensor_name = tensor_id_to_name[d]
                colfrom = f"storage {s['target']} tensor{tensor_name} tile size"
                df[colto] += df[colfrom]

        for s in all_storages:
            if s["target"] in max_capacity:
                colname = f"{s['target']} usage"
                df = df[df[colname] <= max_capacity[s["target"]]]

        if len(df) == 0:
            continue

        # Now we need the usage and total accesses for each column
        cols = ["energy", "latency"]
        for s in all_storages:
            cols.append(f"storage {s['target']} reads")
            cols.append(f"storage {s['target']} writes")
        df[cols] = 0
        df["energy"] = df["energy"].astype("float64")
        df["latency"] = df["latency"].astype("float64")

        # Need a real calculation that takes into account reads/write datasoaces
        # and mutlicast. Also need to take into account reads to the compute
        # node. I'm just doing a quick and dirty here.

        for i, s in enumerate(all_storages):
            for dspace in s["dspace"]:
                tensor_name = tensor_id_to_name[dspace]
                tile_size = f"storage {s['target']} tensor{tensor_name} tile size"
                repeats = f"storage {s['target']} tensor{tensor_name} repeats"
                for parent in reversed(all_storages[:i]):
                    if dspace in parent["dspace"]:
                        transfers = df[tile_size] * df[repeats]
                        df[f"storage {parent['target']} reads"] += transfers
                        df[f"storage {s['target']} writes"] += transfers
                        break

        for i, s in enumerate(all_storages):
            memory_name = bindings[s["target"]]
            for action in ["reads", "writes"]:
                energy = energy_dict.get((memory_name, action[:-1]), 0)
                if energy != 0:
                    df.loc[:, "energy"] += (
                        df[f"storage {s['target']} {action}"].astype("float64") * energy
                    )

        # TODO: Could we pareto prune here?

        # Pack everything into a column so that we can recover later
        # TODO: We may want to move this later so we don't have to do any recording until we've
        #       picked Pareto-optimal mappings. Depends on whether we're memory limited by the
        #       dataframe size or not.

        def loop2str(loop):
            looptype = "S" if loop["type"] == "spatial" else "T"
            return f"{looptype}{loop['rank']}-"

        storage_str = ",".join(
            [
                f"{s['target']}|" + "-".join([str(d) for d in sorted(s["dspace"])])
                for s in all_storages
            ]
        )

        for i, l in enumerate(all_loops):
            df[f"loop{i}"] = loop2str(l) + df[l["column name"]].astype(str)

        # Pack into one string
        df[MAPPING] = join_string_columns(
            df, [f"loop{i}" for i in range(len(all_loops))], storage_str
        )

        keepcols = [
            c
            for c in df.columns
            if "rank" not in c and "loop" not in c and "repeats" not in c
        ]
        keepcols = [
            MAPPING,
            "energy",
            "latency",
            "usage",
        ]
        fused_loop_cols = [l["column name"] for l in all_loops[:n_fused_loops]]
        keepcols = [c for c in df.columns if any(k in c for k in keepcols)]
        df = df[keepcols + fused_loop_cols]

        grouped = df.groupby(fused_loop_cols) if fused_loop_cols else [((), df)]

        # Partition by fused loops
        total_added = 0
        for i, (key, group) in enumerate(grouped):
            loops = tuple(
                Loop(
                    rank_names=fzs((rank_id_to_name[l["rank"]],)),
                    bound=int(b),
                    is_spatial=l["type"] == "spatial",
                )
                for l, b in zip(all_loops, key)
            )
            group = group.drop(columns=fused_loop_cols)
            tiling = Tiling(loops=loops, storage=backing_storage)
            if tiling in result:
                result[tiling] = makepareto(
                    pd.concat([result[tiling], group]).fillna(0)
                )
            else:
                result[tiling] = group
            total_added += len(group)

    t1 = time.time()
    pareto_optimal_mappings = sum(len(v) for v in result.values())
    print(
        f"Found {pareto_optimal_mappings}/{n_mappings}. Mappings per second: {n_mappings/(t1-t0)}"
    )

    return einsum_id, result, n_mappings

    #     tile_shape_explorer = explore_tile_shape(
    #         partial_mapping,
    #         einsum_shape,
    #         compiled_results,
    #         max_capacity,# if prune else {},
    #         max_fanout, # This should be {} if we don't prune BUT we don't have mechanisms
    #         # to catch invalid fanout later, so we need to keep it here to
    #         # prevent invalid-fanout mappings from propagating
    #         tensors=tensors,
    #         prune=prune,
    #     )
    #     # HACKY: Pop out the subspace object as the first in the iterator
    #     shape_subspace = next(tile_shape_explorer)
    #     for shape, res, valid in tile_shape_explorer:
    #         # assert len(intermediate_tensors) <= 1
    #         n_mappings += 1
    #         process_result(
    #             res,
    #             shape,
    #             result,
    #             einsum_id,
    #             intermediate_tensors,
    #             partial_mapping,
    #             bindings,
    #             workload,
    #             energy_dict,
    #             bandwidth_dict,
    #             equivalent_groups,
    #             explore_fusion_uneven=explore_glb_uneven,
    #             einsum_shape=einsum_shape,
    #             metrics=metrics,
    #             einsum_id_to_name=einsum_id_to_name,
    #             rank_id_to_name=rank_id_to_name,
    #             tensor_id_to_name=tensor_id_to_name,
    #             rank_name_to_shared_name=rank_name_to_shared_name,
    #             input_tensors=input_tensors,
    #             output_tensors=output_tensors,
    #             tag_with=tag_with,
    #             tensor_to_relevant_ranks=tensor_to_relevant_ranks,
    #             copy_einsums={"I"},
    #             prune=prune,
    #             valid=valid,
    #         )

    # if prune:
    #     return einsum_id, {k: makepareto(pd.DataFrame(v).fillna(0)) for k, v in result.items()}, n_mappings
    # return einsum_id, {k: pd.DataFrame(v).fillna(0) for k, v in result.items()}, n_mappings


def _per_einsum_mapper_snowcat(
    config,
    bindings,
    max_fanout,
    max_capacity,
    explore_glb_uneven,
    einsum_id,
    energy_dict,
    bandwidth_dict,
    ffmt=False,
    ffmt_refetch_weights=True,
    dataflow_constraint=None,
    metrics=Metrics.all_metrics(),
    tag_with: tuple[callable] = (),
    four_level=False,
    prune=True,
    dataflow=None,
    fuse=True,
):
    workload = LooptreeWorkload.parse_cfg(config.root["problem"])
    analyzer = LooptreeWorkloadDependencyAnalyzer(workload)
    equivalent_groups = EquivalentGroups.from_workload(workload, analyzer)

    einsum_id_to_name = workload.einsum_id_to_name()
    rank_name_to_id = workload.dimension_name_to_id()
    tensor_name_to_id = workload.data_space_name_to_id()

    tensors = workload.tensors_read_by_einsum(
        einsum_id
    ) | workload.tensors_written_by_einsum(einsum_id)
    intermediate_tensors = tensors & get_intermediate_tensors(workload)
    all_ranks = workload.einsum_ospace_dimensions(einsum_id)

    all_ranks = workload.einsum_ospace_dimensions(einsum_id)

    tensor_to_relevant_ranks = {
        tensor: analyzer.einsum_dims_relevant_to_tensor(einsum_id, tensor)
        for tensor in tensors
    }

    einsum_shape = {
        rank_name: workload.get_rank_shape(rank_name)[1] + 1 for rank_name in all_ranks
    }

    if not ffmt and not four_level:
        subspaces = make_subspaces(
            tensors,
            intermediate_tensors,
            tensor_to_relevant_ranks,
            einsum_id,
            workload,
            dataflow_constraint[einsum_id],
        )
    elif four_level:
        subspaces = make_four_level_subspaces(
            tensors,
            intermediate_tensors,
            tensor_to_relevant_ranks,
            einsum_id,
            workload,
            dataflow=dataflow,
            fuse=fuse,
        )
    else:
        subspaces = make_ffmt_subspaces(
            tensors,
            intermediate_tensors,
            tensor_to_relevant_ranks,
            einsum_id,
            workload,
            refetch_weights=ffmt_refetch_weights,
        )

    n_jobs = 1024
    parallelized_spaces, task_spaces = split_dependent_product(
        n_split_min=n_jobs, spaces=subspaces
    )

    partial_mappings = list(dependent_product(parallelized_spaces))
    partial_mappings = [x if isinstance(x, tuple) else (x,) for x in partial_mappings]
    rank_id_to_name = {v: k for k, v in rank_name_to_id.items()}
    tensor_id_to_name = {v: k for k, v in tensor_name_to_id.items()}
    input_tensors = set(
        tensor_id_to_name[t] for t in workload.tensors_read_by_einsum(einsum_id)
    )
    output_tensors = set(
        tensor_id_to_name[t] for t in workload.tensors_written_by_einsum(einsum_id)
    )
    rank_name_to_shared_name = {
        rank_id_to_name[k]: v for k, v in equivalent_groups.rank_to_group_id.items()
    }
    tensor_to_relevant_ranks = {
        tensor_id_to_name[t]: {rank_id_to_name[r] for r in v}
        for t, v in tensor_to_relevant_ranks.items()
    }

    # successful_partial_mappings = []
    # for p in partial_mappings:
    #     partial_mapping = p[0]
    #     found_storage = set()
    #     fail = False
    #     for i, p in enumerate(partial_mapping):
    #         if p["type"] == "storage":
    #             for t in set(p["dspace"]) - found_storage:
    #                 for p2 in partial_mapping[:i]:
    #                     if p2["type"] in ["temporal", "spatial"] and p2["rank"] not in tensor_to_relevant_ranks[t]:
    #                         fail = True
    #                         break
    #             found_storage |= set(p["dspace"])
    #         if len(found_storage) < len(tensors) or i == 0:
    #             continue
    #         prev = partial_mapping[i - 1]
    #         for t in ["spatial"]: # "temporal", TEMPORAL DOESN"T WORK. WEIRD INTERACTIONS WITH LOOP RELEVANCE PRINCIPLEz
    #         if not fail:
    #             successful_partial_mappings.append(p)
    # partial_mappings = successful_partial_mappings

    # Remove things that won't be used later or in per_worker_exploration

    # # for pm in partial_mappings:
    # #     per_worker_exploration(*pm)
    # data[einsum_id] = defaultdict(list)
    # for res in parallel(
    #     [delayed(per_worker_exploration)(*pm) for pm in partial_mappings],
    #     return_as_generator=True,
    #     pbar=f"Generating data for Einsum {einsum_id}. {i+1}/{len(einsums_to_explore)}",
    # ):
    #     for k, v in res.items():
    #         data[einsum_id][k[0]] += v

    worker_args = dict(
        workload=workload,
        task_spaces=task_spaces,
        einsum_shape=einsum_shape,
        max_capacity=max_capacity,
        max_fanout=max_fanout,
        tensors=tensors,
        einsum_id=einsum_id,
        intermediate_tensors=intermediate_tensors,
        bindings=bindings,
        energy_dict=energy_dict,
        bandwidth_dict=bandwidth_dict,
        equivalent_groups=equivalent_groups,
        explore_glb_uneven=explore_glb_uneven,
        metrics=metrics,
        einsum_id_to_name=einsum_id_to_name,
        rank_id_to_name=rank_id_to_name,
        tensor_id_to_name=tensor_id_to_name,
        rank_name_to_shared_name=rank_name_to_shared_name,
        input_tensors=input_tensors,
        output_tensors=output_tensors,
        tag_with=tag_with,
        tensor_to_relevant_ranks=tensor_to_relevant_ranks,
        prune=prune,
    )

    return [
        delayed(per_worker_exploration)(task_space_args=pm, **worker_args)
        for pm in partial_mappings
    ]


def per_einsum_mapper_snowcat(
    config,
    spec,
    explore_glb_uneven,
    einsums_to_explore,
    energy_dict,
    ffmt=False,
    ffmt_refetch_weights=True,
    dataflow_constraint=None,
    metrics=Metrics.all_metrics(),
    tag_with: tuple[callable] = (),
    four_level=False,
    prune=True,
    dataflow=None,
    fuse=True,
):
    bindings, max_fanout, max_capacity, words_per_read = get_hardware_levels(
        spec.architecture
    )
    energy_dict = deepcopy(energy_dict)
    bandwidth_dict = words_per_read
    words_per_read = {bindings[k]: v for k, v in words_per_read.items()}
    for k, v in energy_dict.items():
        if k[0] in words_per_read:
            energy_dict[k] /= words_per_read[k[0]]

    jobs = list(
        j
        for einsum_id in einsums_to_explore
        for j in _per_einsum_mapper_snowcat(
            config,
            bindings,
            max_fanout,
            max_capacity,
            explore_glb_uneven,
            einsum_id,
            energy_dict,
            bandwidth_dict,
            ffmt=ffmt,
            ffmt_refetch_weights=ffmt_refetch_weights,
            dataflow_constraint=dataflow_constraint,
            metrics=metrics,
            tag_with=tag_with,
            four_level=four_level,
            prune=prune,
            dataflow=dataflow,
            fuse=fuse,
        )
    )
    data = {einsum_id: defaultdict(list) for einsum_id in einsums_to_explore}

    n_mappings = 0
    for einsum_id, result, n_mappings_this_job in parallel(
        jobs, pbar="Generating Single-Einsum Mappings", return_as="generator"
    ):
        d = data[einsum_id]
        n_mappings += n_mappings_this_job
        for k, v in result.items():
            d[k].append(v)

    def makesim(einsum_id, tiling, data):
        return einsum_id, SIM(tiling, makepareto(data))

    def makepareto(data):
        return Pareto(
            pd.concat(data).fillna(0), skip_pareto=len(data) == 1 or not prune
        )

    jobs = []
    for einsum_id, tilings in data.items():
        for tiling, dfs in tilings.items():
            jobs.append(delayed(makesim)(einsum_id, tiling, dfs))

    # Free memory by:
    # - Deleting data dict, replace with new fresh dict
    # - Delete jobs after running
    data = defaultdict(list)
    for einsum_id, sim in parallel(
        jobs, pbar="Generating SIMs", return_as="generator", delete_job_after=True
    ):
        data[einsum_id].append(sim)

    return dict(data), n_mappings
