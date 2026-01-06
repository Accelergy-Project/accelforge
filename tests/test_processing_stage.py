from math import isclose
import unittest
from pathlib import Path

from fastfusion.frontend import Spec
from fastfusion.frontend.arch import Arch
from fastfusion.frontend.mapping import Mapping
from fastfusion.frontend.workload import Workload

from fastfusion.frontend.mapper.metrics import Metrics
from fastfusion.model._looptree.accesses import (
    isl_buffer_accesses_from_buffet_actions,
    Accesses,
)
from fastfusion.model._looptree.energy import gather_actions
from fastfusion.model._looptree.latency import get_latency
from fastfusion.model._looptree.types import Buffet
from fastfusion.model._looptree.reuse.symbolic import (
    BuffetStats,
    analyze_reuse_and_add_reservations_to_mapping,
    Compute,
)
from fastfusion.mapper.FFM import make_pmappings, join_pmappings


PARENT_DIR = Path(__file__).parent


class TestProcessingStage(unittest.TestCase):
    def test_processing_stage(self):
        spec = Spec.from_yaml(
            [
                Path(__file__).parent / "processing_stage.arch.yaml",
                Path(__file__).parent / "matmul.workload.yaml",
            ]
        )
        spec.mapper.ffm.metrics = Metrics.ENERGY
        pmappings = make_pmappings(spec)
        mappings = join_pmappings(spec, pmappings)

        energy = mappings.data.iloc[0]["Total_energy"]
        self.assertAlmostEqual(energy, 100 * (128 * 64 * 2 + 128 * 128))


if __name__ == "__main__":
    unittest.main()
