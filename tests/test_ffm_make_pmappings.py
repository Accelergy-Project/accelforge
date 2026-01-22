from pathlib import Path
import unittest

from fastfusion.frontend import Spec, Workload

from fastfusion.mapper import Metrics
from fastfusion.mapper.FFM import make_pmappings
from fastfusion.mapper.FFM._make_pmappings.make_pmappings import make_pmappings
from fastfusion.mapper.FFM._join_pmappings.pmapping_dataframe import nameloop2col

from pmappingcache import make_pmapping_pickle_cache


PARENT_DIR = Path(__file__).parent


class TestPmappingExploration(unittest.TestCase):
    def test_mha(self):
        spec = Spec.from_yaml(
            PARENT_DIR / "four_level.arch.yaml",
            PARENT_DIR / "mha.workload.yaml",
            PARENT_DIR / "mha.renames.yaml",
        )
        spec.mapper.ffm.metrics = Metrics.ENERGY | Metrics.LATENCY
        pmappings = make_pmappings(spec, ["Q"])

    def test_mha_full(self):
        config_names = ["snowcat.arch", "mha_full.workload", "mha.renames"]
        paths = [PARENT_DIR / f"{config_name}.yaml" for config_name in config_names]
        spec = Spec.from_yaml(*paths)

        pmapping_cache = make_pmapping_pickle_cache(config_names)

        pmapping_groups, decompress_data = pmapping_cache.set(make_pmappings(spec))
        for per_einsum_pmappings in pmapping_groups.values():
            for pmapping_group in per_einsum_pmappings:
                for (
                    resource,
                    levels,
                ) in pmapping_group.mappings.right_reservations.items():
                    for level in levels:
                        self.assertTrue(
                            nameloop2col(resource, level)
                            in pmapping_group.mappings.data.columns,
                            f"{resource} at {level} not in {pmapping_group.mappings.data.columns}. Compatibility: {pmapping_group.compatibility}",
                        )

    def test_mha_with_tags(self):
        spec = Spec.from_yaml(
            PARENT_DIR / "four_level.arch.yaml",
            PARENT_DIR / "mha.workload.yaml",
            PARENT_DIR / "mha.renames.yaml",
        )

        pmapping_groups, decompress_data = make_pmappings(spec, einsum_names=["Q"])

    def test_conv_with_snowcat(self):
        spec = Spec.from_yaml(
            PARENT_DIR / "snowcat.arch.yaml",
            PARENT_DIR / "mobilenet_long.workload.yaml",
        )
        config_names = [
            "snowcat.arch",
            "mobilenet_long.workload",
        ]
        paths = [PARENT_DIR / f"{config_name}.yaml" for config_name in config_names]
        spec = Spec.from_yaml(*paths)

        pmapping_cache = make_pmapping_pickle_cache(config_names)

        pmapping_groups, decompress_data = pmapping_cache.set(make_pmappings(spec))
        for per_einsum_pmappings in pmapping_groups.values():
            for pmapping_group in per_einsum_pmappings:
                print(pmapping_group.compatibility)


class TestInitialDeltaGeneration(unittest.TestCase):
    def test_mobilenet_long(self):
        workload = Workload.from_yaml(
            Path(__file__).parent / "mobilenet_long.workload.yaml"
        )
        choices = get_initial_delta_choices("Dwise0", workload)
