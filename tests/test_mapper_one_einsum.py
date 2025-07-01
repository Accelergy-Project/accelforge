from pathlib import Path
import unittest

from fastfusion.frontend import Specification, Workload

from fastfusion.mapper.FFM.exploration import metrics
from fastfusion.mapper.FFM.exploration.tile_shape_exploration.tile_shape_exploration import get_initial_delta_choices
from fastfusion.mapper.FFM.exploration.mapper_multi_einsum import get_sims
from fastfusion.mapper.FFM.exploration.mapping_filter_tags import get_one_split_tag
from fastfusion.mapper.FFM.exploration.mapping_filter_tags.onesplit import ONE_SPLIT
from fastfusion.mapper.FFM.tags import Tags, TagMatch
from fastfusion.mapper.FFM.pareto import nameloop2col

from .simcache import make_sim_pickle_cache


PARENT_DIR = Path(__file__).parent


class TestExploration(unittest.TestCase):
    def test_mha(self):
        spec = Specification.from_yaml(
            PARENT_DIR / "four_level.arch.yaml",
            PARENT_DIR / "mha.workload.yaml",
            PARENT_DIR / "mha.renames.yaml",
        )
        spec.estimate_energy_area()

        workload = spec.workload

        einsum_name = "K"
        einsum = workload.einsums[einsum_name]
        rank_variables = einsum.rank_variables

        sims, decompress_data = get_sims(spec, einsum_names=[einsum_name])

    def test_mha_full(self):
        config_names = [
            "snowcat.arch",
            "mha_full.workload",
            "mha.renames"
        ]
        paths = [PARENT_DIR / f"{config_name}.yaml" for config_name in config_names]
        spec = Specification.from_yaml(*paths)
        spec.estimate_energy_area()

        sim_cache = make_sim_pickle_cache(config_names)
        print(sim_cache.fname)

        sims, decompress_data = sim_cache.set(get_sims(spec))
        for per_einsum_sims in sims.values():
            for sim in per_einsum_sims:
                for resource, levels in sim.mappings.right_reservations.items():
                    for level in levels:
                        self.assertTrue(
                            nameloop2col(resource, level) in sim.mappings.data.columns,
                            f"{resource} at {level} not in {sim.mappings.data.columns}. Compatibility: {sim.compatibility}"
                        )

    def test_mha_with_tags(self):
        spec = Specification.from_yaml(
            PARENT_DIR / "four_level.arch.yaml",
            PARENT_DIR / "mha.workload.yaml",
            PARENT_DIR / "mha.renames.yaml",
        )
        spec.estimate_energy_area()

        workload = spec.workload

        einsum_name = "K"
        einsum = workload.einsums[einsum_name]
        rank_variables = einsum.rank_variables

        def tagger(pmapping):
            return get_one_split_tag(pmapping)

        sims, decompress_data = get_sims(spec, einsum_names=["Q"], tagger=tagger)

    def test_conv_with_snowcat(self):
        spec = Specification.from_yaml(
            PARENT_DIR / "snowcat.arch.yaml",
            PARENT_DIR / "mobilenet_long.workload.yaml",
        )
        spec.estimate_energy_area()

        sims, decompress_data = get_sims(spec,
                                         einsum_names=['Dwise0'],
                                         metrics=metrics.Metrics.ENERGY)


class TestInitialDeltaGeneration(unittest.TestCase):
    def test_mobilenet_long(self):
        workload = Workload.from_yaml(Path(__file__).parent / 'mobilenet_long.workload.yaml')
        choices = get_initial_delta_choices('Dwise0', workload)