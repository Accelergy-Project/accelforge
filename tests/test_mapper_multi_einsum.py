from pathlib import Path
import unittest

from fastfusion.frontend import Specification
from fastfusion.mapper.FFM.exploration.mapper_multi_einsum import get_sims


PARENT_DIR = Path(__file__).parent


class TestMultiEinsumMapper(unittest.TestCase):
    def test_mha(self):
        spec = Specification.from_yaml(
            PARENT_DIR / "four_level.arch.yaml",
            PARENT_DIR / "mha.workload.yaml",
            PARENT_DIR / "mha.renames.yaml",
        )
        spec.estimate_energy_area()

        workload = spec.workload
        flattened_architecture = spec.get_flattened_architecture()

        sims, decompress_data = get_sims(spec, flattened_architecture)

    def test_mobilenet(self):
        spec = Specification.from_yaml(
            PARENT_DIR / "snowcat.arch.yaml",
            PARENT_DIR / "mobilenet.workload.yaml",
        )
        spec.estimate_energy_area()

        workload = spec.workload
        flattened_architecture = spec.get_flattened_architecture()

        sims, decompress_data = get_sims(spec, flattened_architecture)
        for sim in sims['PwiseA0']:
            print(sim.compatibility)