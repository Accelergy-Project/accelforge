from pathlib import Path
import unittest

from fastfusion.frontend import Specification
from fastfusion.mapper.FFM.exploration.mapper_multi_einsum import get_sims
from fastfusion.mapper.FFM.joining.simexplore import join_sims


PARENT_DIR = Path(__file__).parent


class TestJoin(unittest.TestCase):
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