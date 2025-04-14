from collections import defaultdict
from copy import deepcopy
import logging.handlers
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)
from ruamel.yaml import YAML

yaml = YAML(typ="safe")

from bindings.looptree import LooptreeWorkload, LooptreeWorkloadDependencyAnalyzer

from pytimeloop.looptree.equivalent_ranks import (
    EquivalentGroups,
    PairwiseEquivalentRanks,
)

from pytimeloop.fastfusion.mapper.constraints import *
from pytimeloop.fastfusion.layerdeduplication import is_equivalent
from pytimeloop.fastfusion.mapper.logging import make_queue_and_listener
from pytimeloop.fastfusion.mapper.per_einsum_mapper_snowcat import (
    per_einsum_mapper_snowcat,
    get_hardware_levels,
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
from pytimeloop.fastfusion.sim import SIM


def mapper(
    config,
    explore_glb_uneven,
    spec,
    tmp_path,
    ffmt: bool = False,
    ffmt_refetch_weights: bool = True,
    metrics=Metrics.all_metrics(),
    tag_with: tuple[callable] = (),
    four_level=False,
    prune=True,
    dataflow=None,
    fuse=True,
):
    logger.info(f"Calling mapper for {spec}")

    # log_queue, log_queue_listener = make_queue_and_listener()

    workload = LooptreeWorkload.parse_cfg(config.root["problem"])
    analyzer = LooptreeWorkloadDependencyAnalyzer(workload)
    equivalent_groups = EquivalentGroups.from_workload(workload, analyzer)

    # if "mapping_constraints" in spec:
    #     dataflow_constraint = DataflowConstraint.parse(
    #         spec["mapping_constraints"],
    #         workload
    #     )
    # else:
    #     dataflow_constraint = DataflowConstraint.default(workload)

    if isinstance(tmp_path, Path):
        tmp_path = str(tmp_path)
        print(f"Calling Accelergy in {tmp_path}")

    # try:
    #     call_accelergy_verbose(spec, tmp_path, extra_args=[f" > {tmp_path}/accelergy.log 2>&1"])
    # except Exception as e:
    #     logger.error(f"Error running Accelergy: {e}. Log in {tmp_path}/accelergy.log")
    #     raise
    ert_dict = yaml.load(Path(tmp_path) / "ERT.yaml")
    ert = Ert(ert_dict["ERT"])
    energy_dict = ert.to_dict()

    if not ffmt:
        separated_einsums = None
    else:
        separated_einsums = get_ffmt_separated_einsums(workload)
    if not tag_with:
        grouped_similar_einsums = convert_rank_to_group_renaming(
            detect_similar_einsums(workload, analyzer, separated_einsums),
            equivalent_groups,
        )
    else:
        grouped_similar_einsums = {
            einsum: {} for einsum in workload.einsum_id_to_name()
        }
    logger.info(
        f"Found {len(grouped_similar_einsums)} unique Einsums\n"
        + f"\tConverter: {grouped_similar_einsums}"
    )

    data, n_mappings = per_einsum_mapper_snowcat(
        einsums_to_explore=list(grouped_similar_einsums.keys()),
        config=config,
        explore_glb_uneven=explore_glb_uneven,
        spec=spec,
        energy_dict=energy_dict,
        ffmt=ffmt,
        ffmt_refetch_weights=ffmt_refetch_weights,
        # dataflow_constraint=dataflow_constraint,
        metrics=metrics,
        tag_with=tag_with,
        four_level=four_level,
        prune=prune,
        dataflow=dataflow,
        fuse=fuse,
    )

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

    equiv_ranks = PairwiseEquivalentRanks(workload)
    equiv_ranks_dict = defaultdict(set)
    rank_ids = set().union(
        *(
            set(workload.einsum_ospace_dimensions(einsum_id))
            for einsum_id in workload.einsum_id_to_name()
        )
    )
    for rank_id in rank_ids:
        rank_name = dimension_id_to_name[rank_id]
        try:
            equiv_ranks_dict[rank_name] = set(
                dimension_id_to_name[x] for x in equiv_ranks[rank_id]
            )
        except IndexError:
            equiv_ranks_dict[rank_name] = set()

    einsum2ranks = {}
    for einsum_id in einsum_id_to_name:
        einsum2ranks[einsum_id_to_name[einsum_id]] = set(
            dimension_id_to_name[x]
            for x in workload.einsum_ospace_dimensions(einsum_id)
        )

    bindings, max_fanout, max_capacity, _ = get_hardware_levels(spec.architecture)

    return (
        data,
        equiv_ranks_dict,
        einsum2ranks,
        bindings,
        max_fanout,
        max_capacity,
        n_mappings,
    )


def generate_data(
    from_einsum: int, to_einsum: int, data, rank_renaming, tensor_renaming
):
    def convert(sim):
        return SIM(
            _convert_tiling(sim.tiling, rank_renaming, tensor_renaming),
            _convert_stats(
                from_einsum, to_einsum, sim.mapping, rank_renaming, tensor_renaming
            ),
        )

    return [convert(sim) for sim in data]

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
    data = stats.data
    for c in data.columns:
        if c in DICT_COLUMNS:
            data[c] = data[c].apply(lambda x: {to_einsum: x[from_einsum]})
        if c == MAPPING:
            data[c] = data[c].apply(
                lambda x: {
                    to_einsum: x[to_einsum].rename(rank_renaming, tensor_renaming)
                }
            )
        if c == TENSORS:
            data[c] = data[c].apply(
                lambda x: {
                    to_einsum: [
                        t.rename(rank_renaming, tensor_renaming) for t in x[to_einsum]
                    ]
                }
            )
    return stats


def detect_similar_einsums(workload, analyzer, separated_einsums=None):
    if separated_einsums is None:
        separated_einsums = [{i for i in workload.einsum_id_to_name()}]

    total_ref_to_einsums = {}
    for einsum_group in separated_einsums:
        ref_to_to_einsums = {}
        for einsum in einsum_group:
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
        total_ref_to_einsums.update(ref_to_to_einsums)
    return total_ref_to_einsums


def convert_rank_to_group_renaming(ref_to_to_einsums, equiv_ranks):
    return {
        ref: {
            other: (_convert_rank_renaming(rank_renaming, equiv_ranks), tensor_renaming)
            for other, (rank_renaming, tensor_renaming) in others.items()
        }
        for ref, others in ref_to_to_einsums.items()
    }


def get_ffmt_separated_einsums(workload):
    einsum_id_to_name = workload.einsum_id_to_name()
    if len(einsum_id_to_name) == 1:
        return [{0}]
    elif len(einsum_id_to_name) == 2:
        return [{0}, {1}]
    elif len(einsum_id_to_name) == 3:
        return [{0}, {1}, {2}]

    first_einsum = {0}
    second_einsum = {1}
    last_einsum = {max(workload.einsum_id_to_name().keys())}
    other_einsums = (
        set(workload.einsum_id_to_name().keys())
        - first_einsum
        - second_einsum
        - last_einsum
    )
    return [first_einsum, second_einsum, other_einsums, last_einsum]


def _convert_rank_renaming(rank_renaming, equiv_ranks):
    # The Tiling class uses string ids
    return {str(r1): str(r2) for r1, r2 in rank_renaming.items()}
