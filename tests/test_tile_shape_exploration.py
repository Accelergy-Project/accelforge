import unittest
from pathlib import Path

from fastfusion.frontend.mapping import Mapping
from fastfusion.frontend.workload import Workload
from fastfusion.mapper.FFM.exploration.tile_shape_exploration import *


class TestTileShapeExploration(unittest.TestCase):
    def test_pmapping(self):
        workload = Workload.from_yaml(Path(__file__).parent / 'conv.workload.yaml')
        mapping = Mapping.from_yaml(Path(__file__).parent / 'conv_sym.mapping.yaml')
        explore_tile_shapes(mapping,
                            workload,
                            [])