import unittest
from pathlib import Path

import time

from fastfusion.frontend.mapping import Mapping
from fastfusion.mapper.FFM.exploration.tile_shape_exploration import *


class TestTileShapeExploration(unittest.TestCase):
    def test_pmapping(self):
        PARENT_DIR = Path(__file__).parent
        specification = Specification.from_yaml(
            PARENT_DIR / 'conv.workload.yaml',
            PARENT_DIR / 'four_level.arch.yaml'
        )

        mapping = Mapping.from_yaml(PARENT_DIR / 'conv_sym.mapping.yaml')

        result = explore_tile_shapes(mapping, [], specification)
        self.assertTrue('Latency' in result)
        self.assertTrue('MainMemory_Occupancy' in result)
        self.assertTrue('LocalBuffer_Occupancy' in result)
