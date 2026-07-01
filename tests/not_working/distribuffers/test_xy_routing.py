"""
Tests for the XYRoutingMulticastModel (dimension-order routing on a 2-D mesh).

XY routing sends each packet along X then Y, so a multicast tree is an X-segment
along the source row plus an independent Y-segment down each destination column.
The expected hop counts here are HAND-DERIVED (the repo has no XY oracle -- the
yaml's extent_DOR_hops is a free-routing floor, not the XY cost); each case's
geometry is documented in fully_connected/../xy_routing/test_cases.yaml so the
numbers can be re-checked. See
accelforge/model/_looptree/reuse/isl/distributed/README.md.
"""

import unittest
from pathlib import Path

from accelforge.model._looptree.reuse.isl.distributed.distributed_buffers import (
    XYRoutingMulticastModel,
)
from .helpers import run_hops_gamut


class TestXYRoutingMulticastModel(unittest.TestCase):
    """
    Tests the XYRoutingMulticastModel with a series of premade test cases.
    """

    TEST_CASES_FILE: Path = Path(__file__).parent / "xy_routing" / "test_cases.yaml"

    def test_gamut(self):
        """
        Tests the entire gamut of test cases we have specified in the yaml.
        """
        run_hops_gamut(XYRoutingMulticastModel, self.TEST_CASES_FILE, "xy_routing_hops")
