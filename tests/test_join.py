from pathlib import Path
import pickle
import unittest

from fastfusion.frontend import Specification
from fastfusion.mapper.FFM.exploration.mapper_multi_einsum import get_sims
from fastfusion.mapper.FFM.joining.simexplore import join_sims

from .simcache import make_sim_pickle_cache


PARENT_DIR = Path(__file__).parent


class TestJoin(unittest.TestCase):
    def test_mha(self):
        spec = Specification.from_yaml(
            PARENT_DIR / "four_level.arch.yaml",
            PARENT_DIR / "mha.workload.yaml",
            PARENT_DIR / "mha.renames.yaml"
        )
        spec.estimate_energy_area()

        flattened_arch = spec.get_flattened_architecture()
        sims, decompress_data = get_sims(spec, flattened_arch)
        mappings = join_sims(sims, spec, flattened_arch, drop_valid_reservations=False)
        mappings.decompress(decompress_data)

    def test_mha_full(self):
        config_names = [
            "snowcat.arch",
            "mha_full.workload",
            "mha.renames"
        ]
        paths = [PARENT_DIR / f"{config_name}.yaml" for config_name in config_names]
        spec = Specification.from_yaml(*paths)
        spec.estimate_energy_area()
        flattened_arch = spec.get_flattened_architecture()

        sim_cache = make_sim_pickle_cache(config_names)
        sims, decompress_data = sim_cache.get(lambda: get_sims(spec, flattened_arch))

        mappings = join_sims(sims, spec, flattened_arch, drop_valid_reservations=False)
        mappings.decompress(decompress_data)

        print(mappings.data.sort_values('metric_Energy'))

    def test_mobilenet(self):
        spec = Specification.from_yaml(
            PARENT_DIR / "snowcat.arch.yaml",
            PARENT_DIR / "mobilenet_long.workload.yaml",
        )
        spec.estimate_energy_area()

        flattened_architecture = spec.get_flattened_architecture()

        sims, decompress_data = get_sims(spec, flattened_architecture, except_from_imperfect={'q0', 'r0', 's0', 'q1', 'r1', 's2', 'q2'})
        mappings = join_sims(sims, spec, flattened_architecture, drop_valid_reservations=False)
        mappings.decompress(decompress_data)