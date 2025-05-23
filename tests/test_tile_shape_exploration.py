import unittest
from pathlib import Path

from fastfusion.frontend.mapping import Mapping
from fastfusion.mapper.FFM.exploration.tile_shape_exploration import *


class TestTileShapeExploration(unittest.TestCase):
    def test_pmapping(self):
        mapping = Mapping.from_yaml(Path(__file__).parent / 'conv_sym.mapping.yaml')
        explore_tile_shapes(mapping,
                            {'p': 10,
                             'r': 3,
                             'c': 2,
                             'm': 4},
                             [])