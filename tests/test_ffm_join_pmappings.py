from pathlib import Path
import pickle
import unittest

from accelforge.frontend import Spec
from accelforge.mapper.FFM._make_pmappings.make_pmappings import make_pmappings
from accelforge.mapper.FFM._join_pmappings.compatibility_util import (
    join_compatibilities,
    pmappings2untiled_compatibilities,
)
from accelforge.mapper.FFM._join_pmappings.join_pmappings import join_pmappings

from pmappingcache import make_pmapping_pickle_cache


PARENT_DIR = Path(__file__).parent


class TestPreJoin(unittest.TestCase):
    def test_mha_full(self):
        config_names = ["snowcat.arch", "mha_full.workload", "mha.renames"]
        paths = [PARENT_DIR / f"{config_name}.yaml" for config_name in config_names]
        spec = Spec.from_yaml(*paths)
        flattened_arch = spec._get_flattened_architecture()

        pmapping_cache = make_pmapping_pickle_cache(config_names)
        pmapping_groups, decompress_data = pmapping_cache.get(
            lambda: make_pmappings(spec, flattened_arch)
        )

        untiled_compats = pmappings2untiled_compatibilities(pmapping_groups)
        einsum2important_compats = join_compatibilities(untiled_compats, spec)

        for einsum, compats in untiled_compats.items():
            before_size = sum(
                len(pm.mappings.data.index) for pm in pmapping_groups[einsum]
            )
            after_size = sum(
                len(pm.mappings.data.index) for pm in einsum2pruned_pmappings[einsum]
            )

            print(
                f"{einsum} has {len(einsum2important_compats[einsum])}"
                f"/{len(compats)} compatibilities left and "
                f"{after_size}/{before_size} pmappings left."
            )


class TestJoin(unittest.TestCase):
    def test_mha(self):
        spec = Spec.from_yaml(
            PARENT_DIR / "four_level.arch.yaml",
            PARENT_DIR / "mha.workload.yaml",
            PARENT_DIR / "mha.renames.yaml",
        )

        flattened_arch = spec._get_flattened_architecture()
        pmapping_groups, decompress_data = make_pmappings(spec, flattened_arch)
        mappings = join_pmappings(
            pmapping_groups, spec, flattened_arch, drop_valid_reservations=False
        )

    def test_mha_full(self):
        config_names = ["snowcat.arch", "mha_full.workload", "mha.renames"]
        paths = [PARENT_DIR / f"{config_name}.yaml" for config_name in config_names]
        spec = Spec.from_yaml(*paths)
        flattened_arch = spec._get_flattened_architecture()

        pmapping_cache = make_pmapping_pickle_cache(config_names)
        pmapping_groups, decompress_data = pmapping_cache.get(
            lambda: make_pmappings(spec, flattened_arch)
        )

        mappings = join_pmappings(pmapping_groups, spec, flattened_arch)

    def test_mha_full_with_prejoin_pruning(self):
        config_names = ["snowcat.arch", "mha_full.workload", "mha.renames"]
        paths = [PARENT_DIR / f"{config_name}.yaml" for config_name in config_names]
        spec = Spec.from_yaml(*paths)
        flattened_arch = spec._get_flattened_architecture()

        pmapping_cache = make_pmapping_pickle_cache(config_names)
        pmapping_groups, decompress_data = pmapping_cache.get(
            lambda: make_pmappings(spec, flattened_arch)
        )

        untiled_compats = pmappings2untiled_compatibilities(pmapping_groups)
        einsum2important_compats = join_compatibilities(untiled_compats, spec)

        mappings = join_pmappings(einsum2pruned_pmappings, spec, flattened_arch)

    def test_mobilenet(self):
        config_names = [
            "snowcat.arch",
            "mobilenet_long.workload",
        ]
        paths = [PARENT_DIR / f"{config_name}.yaml" for config_name in config_names]
        spec = Spec.from_yaml(*paths)
        flattened_arch = spec._get_flattened_architecture()

        pmapping_cache = make_pmapping_pickle_cache(config_names)
        pmapping_groups, decompress_data = pmapping_cache.get(
            lambda: make_pmappings(spec)
        )

        mappings = join_pmappings(pmapping_groups, spec, flattened_arch)
