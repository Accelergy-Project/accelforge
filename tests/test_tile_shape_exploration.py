import unittest
from pathlib import Path

import time

from fastfusion.frontend.specification import Specification, Mapping
from fastfusion.mapper.FFM._make_pmappings.contraints.constraints import MappingConstraints
from fastfusion.mapper.FFM._make_pmappings.mapper_one_einsum.mapper_job import Job
from fastfusion.mapper.FFM._make_pmappings.mapper_one_einsum import *
from fastfusion.frontend.mapper.metrics import Metrics
from fastfusion.frontend import arch


class TestTileShapeExploration(unittest.TestCase):
    def test_pmapping(self):
        PARENT_DIR = Path(__file__).parent
        specification = Specification.from_yaml(
            PARENT_DIR / 'conv.workload.yaml',
            PARENT_DIR / 'four_level.arch.yaml'
        )
        specification.calculate_component_energy_area(area=False)

        mapping = Mapping.from_yaml(PARENT_DIR / 'conv_sym.mapping.yaml')
        
        flattened_arch = specification.get_flattened_architecture()
        memories_track_all = [m.name for m in flattened_arch if isinstance(m, arch.Memory)]
        memories_track_pmappings_only = []
        
        job = Job(
            mapping=mapping,
            constraints=MappingConstraints(),
            spec=specification,
            metrics=Metrics.LATENCY,
            job_id=0,
            rank_variable_bounds={},
            memories_track_all=memories_track_all,
            memories_track_pmappings_only=memories_track_pmappings_only,
        )

        result = explore_tile_shapes(job)
        data, total_pmappings = result
        self.assertTrue('Total_latency' in data.columns)

if __name__ == '__main__':
    unittest.main()
