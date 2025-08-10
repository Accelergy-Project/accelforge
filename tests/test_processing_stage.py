from math import isclose
import unittest
from pathlib import Path

from fastfusion.frontend import Specification
from fastfusion.frontend.architecture import Architecture
from fastfusion.frontend.mapping import Mapping
from fastfusion.frontend.workload import Workload

from fastfusion.mapper.metrics import Metrics
from fastfusion.model.looptree.accesses import isl_buffer_accesses_from_buffet_actions, Accesses
from fastfusion.model.looptree.energy import gather_actions
from fastfusion.model.looptree.latency import get_latency
from fastfusion.model.looptree.reuse.summarized.symbolic import BuffetStats, analyze_reuse, Compute, Buffet
from fastfusion.mapper.FFM import make_pmappings, join_pmappings


PARENT_DIR = Path(__file__).parent


class TestProcessingStage(unittest.TestCase):
    def test_processing_stage(self):
        spec = Specification.from_yaml([
            Path(__file__).parent / 'processing_stage.arch.yaml',
            Path(__file__).parent / 'matmul.workload.yaml'
        ])
        spec.mapper.ffm.metrics = Metrics.ENERGY
        spec.calculate_component_energy_area()
        pmappings = make_pmappings(spec)
        mappings = join_pmappings(spec, pmappings)

        energy = mappings.data.iloc[0]['Total_energy']
        self.assertAlmostEqual(energy, 100 * (128 * 64 * 2 + 128 * 128))

if __name__ == '__main__':
    unittest.main()