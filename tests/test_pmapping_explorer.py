from pathlib import Path
import unittest

from fastfusion.frontend import Specification, Workload

from fastfusion.mapper import Metrics
from fastfusion.mapper.FFM import make_pmappings
from fastfusion.mapper.FFM._make_pmappings.mapper_multi_einsum import get_sims
from fastfusion.mapper.FFM._pmapping_group import nameloop2col

from .simcache import make_sim_pickle_cache


PARENT_DIR = Path(__file__).parent


class TestPmappingExploration(unittest.TestCase):
    def test_mha(self):
        spec = Specification.from_yaml(
            PARENT_DIR / "four_level.arch.yaml",
            PARENT_DIR / "mha.workload.yaml",
            PARENT_DIR / "mha.renames.yaml",
        )
        spec.mapper.ffm.metrics = Metrics.ENERGY | Metrics.LATENCY
        pmappings = make_pmappings(spec, ['Q'])

    def test_mha_full(self):
        config_names = [
            "snowcat.arch",
            "mha_full.workload",
            "mha.renames"
        ]
        paths = [PARENT_DIR / f"{config_name}.yaml" for config_name in config_names]
        spec = Specification.from_yaml(*paths)

        sim_cache = make_sim_pickle_cache(config_names)

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

        sims, decompress_data = get_sims(spec, einsum_names=["Q"])

    def test_conv_with_snowcat(self):
        spec = Specification.from_yaml(
            PARENT_DIR / "snowcat.arch.yaml",
            PARENT_DIR / "mobilenet_long.workload.yaml",
        )
        config_names = [
            "snowcat.arch",
            "mobilenet_long.workload",
        ]
        paths = [PARENT_DIR / f"{config_name}.yaml" for config_name in config_names]
        spec = Specification.from_yaml(*paths)

        sim_cache = make_sim_pickle_cache(config_names)

        sims, decompress_data = sim_cache.set(get_sims(spec))
        for per_einsum_sims in sims.values():
            for sim in per_einsum_sims:
                print(sim.compatibility)


class TestInitialDeltaGeneration(unittest.TestCase):
    def test_mobilenet_long(self):
        workload = Workload.from_yaml(Path(__file__).parent / 'mobilenet_long.workload.yaml')
        choices = get_initial_delta_choices('Dwise0', workload)