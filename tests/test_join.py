from pathlib import Path
import pickle
import unittest

from fastfusion.frontend import Specification
from fastfusion.mapper.FFM._make_pmappings.mapper_multi_einsum import get_sims
from fastfusion.mapper.FFM._join_pmappings.compatibility_util import join_compatibilities, sims2untiled_compats, remove_unimportant_sims
from fastfusion.mapper.FFM._join_pmappings.join_pmappings import join_sims

from simcache import make_sim_pickle_cache


PARENT_DIR = Path(__file__).parent


class TestPreJoin(unittest.TestCase):
    def test_mha_full(self):
        config_names = [
            "snowcat.arch",
            "mha_full.workload",
            "mha.renames"
        ]
        paths = [PARENT_DIR / f"{config_name}.yaml" for config_name in config_names]
        spec = Specification.from_yaml(*paths)
        flattened_arch = spec.get_flattened_architecture()

        sim_cache = make_sim_pickle_cache(config_names)
        sims, decompress_data = sim_cache.get(lambda: get_sims(spec, flattened_arch))

        untiled_compats = sims2untiled_compats(sims)
        einsum2important_compats = join_compatibilities(untiled_compats, spec)
        einsum2pruned_sims = remove_unimportant_sims(sims, einsum2important_compats)

        for einsum, compats in untiled_compats.items():
            before_size = sum(len(sim.mappings.data.index) for sim in sims[einsum])
            after_size = sum(len(sim.mappings.data.index) for sim in einsum2pruned_sims[einsum])

            print(f"{einsum} has {len(einsum2important_compats[einsum])}"
                  f"/{len(compats)} compatibilities left and "
                  f"{after_size}/{before_size} pmappings left.")


class TestJoin(unittest.TestCase):
    def test_mha(self):
        spec = Specification.from_yaml(
            PARENT_DIR / "four_level.arch.yaml",
            PARENT_DIR / "mha.workload.yaml",
            PARENT_DIR / "mha.renames.yaml"
        )

        flattened_arch = spec.get_flattened_architecture()
        sims, decompress_data = get_sims(spec, flattened_arch)
        mappings = join_sims(sims, spec, flattened_arch, drop_valid_reservations=False)

    def test_mha_full(self):
        config_names = [
            "snowcat.arch",
            "mha_full.workload",
            "mha.renames"
        ]
        paths = [PARENT_DIR / f"{config_name}.yaml" for config_name in config_names]
        spec = Specification.from_yaml(*paths)
        flattened_arch = spec.get_flattened_architecture()

        sim_cache = make_sim_pickle_cache(config_names)
        sims, decompress_data = sim_cache.get(lambda: get_sims(spec, flattened_arch))

        mappings = join_sims(sims, spec, flattened_arch)

    def test_mha_full_with_prejoin_pruning(self):
        config_names = [
            "snowcat.arch",
            "mha_full.workload",
            "mha.renames"
        ]
        paths = [PARENT_DIR / f"{config_name}.yaml" for config_name in config_names]
        spec = Specification.from_yaml(*paths)
        flattened_arch = spec.get_flattened_architecture()

        sim_cache = make_sim_pickle_cache(config_names)
        sims, decompress_data = sim_cache.get(lambda: get_sims(spec, flattened_arch))

        untiled_compats = sims2untiled_compats(sims)
        einsum2important_compats = join_compatibilities(untiled_compats, spec)
        einsum2pruned_sims = remove_unimportant_sims(sims, einsum2important_compats)

        mappings = join_sims(einsum2pruned_sims, spec, flattened_arch)

    def test_mobilenet(self):
        config_names = [
            "snowcat.arch",
            "mobilenet_long.workload",
        ]
        paths = [PARENT_DIR / f"{config_name}.yaml" for config_name in config_names]
        spec = Specification.from_yaml(*paths)
        flattened_arch = spec.get_flattened_architecture()

        sim_cache = make_sim_pickle_cache(config_names)
        sims, decompress_data = sim_cache.get(lambda: get_sims(spec))

        mappings = join_sims(sims, spec, flattened_arch)
