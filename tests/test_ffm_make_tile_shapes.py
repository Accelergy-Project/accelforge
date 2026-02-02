import unittest
from pathlib import Path

import time

from accelforge.frontend.spec import Spec, Mapping
from accelforge.mapper.FFM._make_pmappings.contraints.constraints import (
    MappingConstraints,
)
from accelforge.mapper.FFM._make_pmappings.pmapper_job import Job
from accelforge.mapper.FFM._make_pmappings.make_pmappings import make_tile_shapes
from accelforge.frontend.mapper.metrics import Metrics
from accelforge.frontend import arch


class TestTileShapeExploration(unittest.TestCase):
    def test_pmapping(self):
        PARENT_DIR = Path(__file__).parent
        spec = Spec.from_yaml(
            PARENT_DIR / "conv.workload.yaml", PARENT_DIR / "four_level.arch.yaml"
        )
        spec = spec.calculate_component_area_energy_latency_leak(area=False)

        mapping = Mapping.from_yaml(PARENT_DIR / "conv_sym.mapping.yaml")

        flattened_arch = spec._get_flattened_architecture()
        memories_track_all = [
            m.name for m in flattened_arch if isinstance(m, arch.Memory)
        ]
        memories_track_pmappings_only = []

        job = Job(
            mapping=mapping,
            constraints=MappingConstraints(),
            spec=spec,
            metrics=Metrics.LATENCY,
            job_id=0,
            rank_variable_bounds={},
            memories_track_all=memories_track_all,
            memories_track_pmappings_only=memories_track_pmappings_only,
        )

        result, _ = make_tile_shapes(job)
        data, n_total_pmappings = result
        self.assertTrue("Total_latency" in data.columns)


if __name__ == "__main__":
    unittest.main()
