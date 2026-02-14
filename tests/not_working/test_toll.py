from math import isclose
import unittest
from pathlib import Path

from accelforge.frontend import Spec
from accelforge.frontend.arch import Arch
from accelforge.frontend.mapping import Mapping
from accelforge.frontend.workload import Workload

from accelforge.frontend.mapper.metrics import Metrics
from accelforge.model._looptree.accesses import (
    isl_buffer_accesses_from_buffet_actions,
    Accesses,
)
from accelforge.model._looptree.energy import gather_actions
from accelforge.model._looptree.latency import get_latency
from accelforge.model._looptree.types import Buffet
from accelforge.model._looptree.reuse.symbolic import (
    BuffetStats,
    analyze_reuse_and_add_reservations_to_mapping,
    Compute,
)
from accelforge.mapper.FFM import make_pmappings, join_pmappings


PARENT_DIR = Path(__file__).parent


class TestProcessingStage(unittest.TestCase):
    def test_toll(self):
        spec = Spec.from_yaml(
            [
                Path(__file__).parent / "toll.arch.yaml",
                Path(__file__).parent / "matmul.workload.yaml",
            ]
        )
        spec.mapper.metrics = Metrics.ENERGY
        pmappings = make_pmappings(spec)
        mappings = join_pmappings(pmappings, metrics=Metrics.ENERGY)

        energy = mappings.data.iloc[0]["Total_energy"]
        self.assertAlmostEqual(energy, 100 * (128 * 64 * 2 + 128 * 128))


if __name__ == "__main__":
    unittest.main()
