import unittest
from pathlib import Path

import time

from fastfusion.frontend.specification import Specification, Mapping
from fastfusion.mapper.FFM.exploration.contraints.constraints import MappingConstraints
from fastfusion.mapper.FFM.exploration.tile_shape_exploration import *
from fastfusion.mapper.FFM.exploration.metrics import Metrics


class TestTileShapeExploration(unittest.TestCase):
    def test_pmapping(self):
        PARENT_DIR = Path(__file__).parent
        specification = Specification.from_yaml(
            PARENT_DIR / 'conv.workload.yaml',
            PARENT_DIR / 'four_level.arch.yaml'
        )
        specification.estimate_energy_area()

        mapping = Mapping.from_yaml(PARENT_DIR / 'conv_sym.mapping.yaml')

        result = explore_tile_shapes(mapping,
                                     MappingConstraints(),
                                     specification,
                                     specification.get_flattened_architecture(),
                                     Metrics.LATENCY)
        data, total_pmappings = result
        self.assertTrue('metric_Latency' in data.columns)

if __name__ == '__main__':
    unittest.main()
