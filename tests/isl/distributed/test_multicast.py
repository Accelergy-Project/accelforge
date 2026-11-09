"""
File ported from here:
https://github.com/rengzhengcodes/timeloop/blob/distributed-multicast-dev/src/unit-test/multicast/test-multicast.cpp
"""

import unittest
from pathlib import Path

from accelforge.model._looptree.reuse.isl.distributed.distributed_buffers import (
    HypercubeMulticastModel,
)
from .helpers import run_hops_gamut


class TestHypercubeMulticastModel(unittest.TestCase):
    """
    Tests the HypercubeMulticastModel with a series of premade test cases.
    """

    TEST_CASES_FILE: Path = Path(__file__).parent / "multicast" / "test_cases.yaml"

    def test_gamut(self):
        """
        Tests the entire gamut of test cases we have specified in the yaml.
        """
        run_hops_gamut(HypercubeMulticastModel, self.TEST_CASES_FILE, "hypercube_hops")
