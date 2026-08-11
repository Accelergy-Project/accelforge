"""
Tests for the FullyConnectedMulticastModel.

The model costs each delivery that crosses the fabric at one hop and self-
deliveries at zero, so an 8-GPU one-hot all-to-all costs 56 hops (64 chunks - 8
self-deliveries) -- a third of the HypercubeMulticastModel's 168 on the same
input. See accelforge/model/_looptree/reuse/isl/distributed/README.md.
"""

import unittest
from pathlib import Path

from accelforge.model._looptree.reuse.isl.distributed.distributed_buffers import (
    FullyConnectedMulticastModel,
)
from .helpers import run_hops_gamut


class TestFullyConnectedMulticastModel(unittest.TestCase):
    """
    Tests the FullyConnectedMulticastModel with a series of premade test cases.
    """

    TEST_CASES_FILE: Path = (
        Path(__file__).parent / "fully_connected" / "test_cases.yaml"
    )

    def test_gamut(self):
        """
        Tests the entire gamut of test cases we have specified in the yaml.
        """
        run_hops_gamut(
            FullyConnectedMulticastModel, self.TEST_CASES_FILE, "fully_connected_hops"
        )
