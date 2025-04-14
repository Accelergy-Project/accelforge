from collections import defaultdict
from copy import deepcopy
import logging.handlers
from pathlib import Path
import logging

from pytimeloop.fastfusion.util import parallel

logger = logging.getLogger(__name__)

from ruamel.yaml import YAML
from joblib import delayed

yaml = YAML(typ="safe")

from bindings.looptree import LooptreeWorkload, LooptreeWorkloadDependencyAnalyzer

from pytimeloop.looptree.equivalent_ranks import EquivalentGroups

from pytimeloop.fastfusion.mapper.constraints import *
from pytimeloop.fastfusion.layerdeduplication import is_equivalent
from pytimeloop.fastfusion.mapper.logging import make_queue_and_listener
from pytimeloop.fastfusion.mapper.per_einsum_mapper import (
    get_top_loop_jobs,
    mapper_place_fusion_level,
)
from pytimeloop.fastfusion.sim import Tiling, Loop, TensorStorage
from pytimeloop.fastfusion.pareto import (
    LOGSTRING,
    MAPPING,
    STATS,
    DICT_COLUMNS,
    TENSORS,
)
from pytimeloop.fastfusion.mapper.process_results import Metrics

from pytimeloop.timeloopfe.v4 import Ert
from pytimeloop.timeloopfe.common.backend_calls import call_accelergy_verbose


def mapper(
    config,
    pe_array_constraint: PeArrayConstraint,
    mac_array_constraint: MacArrayConstraint,
    explore_glb_uneven,
    explore_pe_uneven,
    spec,
    tmp_path,
    verbose_stream=None,
    metrics=Metrics.all_metrics(),
):
    logger.info(f"Calling mapper for {spec}")

    # log_queue, log_queue_listener = make_queue_and_listener()
    log_queue, log_queue_listener = None, None

    workload = LooptreeWorkload.parse_cfg(config.root["problem"])
    analyzer = LooptreeWorkloadDependencyAnalyzer(workload)
    equivalent_groups = EquivalentGroups.from_workload(workload, analyzer)

    einsum_name_to_id = workload.einsum_name_to_id()

    if isinstance(tmp_path, Path):
        tmp_path = str(tmp_path)
    call_accelergy_verbose(spec, tmp_path)
    ert_dict = yaml.load(Path(tmp_path) / "ERT.yaml")
    ert = Ert(ert_dict["ERT"])
    energy_dict = ert.to_dict()

    grouped_similar_einsums = convert_rank_to_group_renaming(
        detect_similar_einsums(workload, analyzer), equivalent_groups
    )
    logger.info(
        f"Found {len(grouped_similar_einsums)} unique Einsums\n"
        + f"\tConverter: {grouped_similar_einsums}"
    )

    args = get_top_loop_jobs(
        einsums_to_explore=list(grouped_similar_einsums.keys()),
        config=config,
        pe_array_constraint=pe_array_constraint,
        mac_array_constraint=mac_array_constraint,
        explore_glb_uneven=explore_glb_uneven,
        explore_pe_uneven=explore_pe_uneven,
        spec=spec,
        energy_dict=energy_dict,
        log_queue=log_queue,
        verbose_stream=verbose_stream,
        metrics=metrics,
    )

    print(f"Number of jobs: {len(args)}")
    n_workers = 64
    logger.debug(f"Starting {n_workers} workers")
    if log_queue_listener is not None:
        log_queue_listener.start()

    # for a in args:
    #     mapper_place_fusion_level(**a)

    result = parallel(delayed(mapper_place_fusion_level)(**a) for a in args)
    data = {einsum_id: defaultdict(list) for einsum_id in grouped_similar_einsums}
    total = 0
    for einsum_id, mappings, count in result:
        for k, v in mappings.items():
            data[einsum_id][k].extend(v)
        total += count
    print(f"Total number of mappings: {total}")
    if log_queue_listener is not None:
        log_queue_listener.stop()
    logger.info(f"Mapper finished for {spec}")

    generated_data = {}
    logger.info(f"Generating data for non-unique Einsums")
    einsum_id_to_name = workload.einsum_id_to_name()
    dimension_name_to_id = workload.dimension_name_to_id()
    data_space_name_to_id = workload.data_space_name_to_id()
    dimension_id_to_name = {v: k for k, v in dimension_name_to_id.items()}
    data_space_id_to_name = {v: k for k, v in data_space_name_to_id.items()}
    for from_einsum, others in grouped_similar_einsums.items():
        for to_einsum, (rank_renaming, tensor_renaming) in others.items():
            logger.info(
                f"Generating data for {to_einsum}. "
                + f"Rank renaming={rank_renaming}. "
                + f"Tensor renaming={tensor_renaming}"
            )
            rank_renaming = {
                dimension_id_to_name[int(k)]: dimension_id_to_name[int(v)]
                for k, v in rank_renaming.items()
            }
            tensor_renaming = {
                data_space_id_to_name[int(k)]: data_space_id_to_name[int(v)]
                for k, v in tensor_renaming.items()
            }
            logger.info(
                f"Generating data for {to_einsum}. "
                + f"Rank renaming={rank_renaming}. "
                + f"Tensor renaming={tensor_renaming}"
            )
            generated_data[to_einsum] = generate_data(
                einsum_id_to_name[from_einsum],
                einsum_id_to_name[to_einsum],
                data[from_einsum],
                rank_renaming,
                tensor_renaming,
            )

    for einsum, mapping in generated_data.items():
        data[einsum] = mapping

    logger.info(f"Final set of Einsums: {set(data.keys())}")

    # data has to come out in sorted Einsum-id order
    data = {k: v for k, v in sorted(data.items(), key=lambda item: item[0])}
    data = {einsum_id_to_name[k]: v for k, v in data.items()}

    return data


def generate_data(
    from_einsum: int, to_einsum: int, data, rank_renaming, tensor_renaming
):
    return {
        _convert_tiling(tiling, rank_renaming, tensor_renaming): _convert_stats(
            from_einsum, to_einsum, stats, rank_renaming, tensor_renaming
        )
        for tiling, stats in data.items()
    }


def _convert_tiling(tiling: Tiling, rank_renaming, tensor_renaming):
    return tiling.rename(rank_renaming, tensor_renaming)


def _convert_stats(
    from_einsum: int, to_einsum: int, stats, rank_renaming, tensor_renaming
):
    stats = deepcopy(stats)
    for s in stats:
        for d in DICT_COLUMNS:
            if d in s:
                s[d][to_einsum] = s[d].pop(from_einsum)
        if MAPPING in s:
            s[MAPPING][to_einsum] = s[MAPPING][to_einsum].rename(
                rank_renaming, tensor_renaming
            )
        if TENSORS in s:
            s[TENSORS][to_einsum] = [
                t.rename(rank_renaming, tensor_renaming) for t in s[TENSORS][to_einsum]
            ]
    return stats


def detect_similar_einsums(workload, analyzer, return_all_as_unique=False):
    if return_all_as_unique:
        return {ref: {} for ref in workload.einsum_id_to_name()}

    ref_to_to_einsums = {}
    for einsum in workload.einsum_id_to_name():
        found = False
        for from_einsum in ref_to_to_einsums:
            rank_renaming, tensor_renaming = is_equivalent(
                from_einsum, einsum, workload, analyzer
            )
            if rank_renaming is not None:
                ref_to_to_einsums[from_einsum][einsum] = (
                    rank_renaming,
                    tensor_renaming,
                )
                found = True
                break
        if not found:
            ref_to_to_einsums[einsum] = {}
    return ref_to_to_einsums


def convert_rank_to_group_renaming(ref_to_to_einsums, equiv_ranks):
    return {
        ref: {
            other: (_convert_rank_renaming(rank_renaming, equiv_ranks), tensor_renaming)
            for other, (rank_renaming, tensor_renaming) in others.items()
        }
        for ref, others in ref_to_to_einsums.items()
    }


def _convert_rank_renaming(rank_renaming, equiv_ranks):
    # The Tiling class uses string ids
    return {
        str(equiv_ranks.rank_to_group_id[r1]): str(equiv_ranks.rank_to_group_id[r2])
        for r1, r2 in rank_renaming.items()
    }
