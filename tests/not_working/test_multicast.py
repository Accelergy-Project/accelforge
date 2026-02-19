"""
File ported from here:
https://github.com/rengzhengcodes/timeloop/blob/distributed-multicast-dev/src/unit-test/multicast/test-multicast.cpp
"""

import unittest
from pathlib import Path

import islpy as isl

from accelforge.model._looptree.reuse.isl.mapping_to_isl.types import (
    # Data movement descriptors.
    Fill,
    Occupancy,
    # Tags
    Tag,
    SpatialTag,
    TemporalTag,
)
from accelforge.model._looptree.reuse.isl.distributed.distributed_buffers import (
    HypercubeMulticastModel,
)
from accelforge.model._looptree.reuse.isl.spatial import TransferInfo
from .util import load_solutions


def construct_spacetime(dims: list) -> list[Tag]:
    """
    Given a list of dimension tags as strings, convert them into the proper `Tag`
    objects.

    Parameters
    ----------
    dims:
        The list of dim tags as strings.

    Returns
    -------
    list[Tag] where list[i] is the tag corresponding to dims[i].
    """
    spacetime: list[Tag] = []
    for dim in dims:
        if dim["type"] == "Temporal":
            spacetime.append(TemporalTag())
        elif dim["type"] == "Spatial":
            spacetime.append(SpatialTag(dim["spatial_dim"], dim["target"]))

    return spacetime


class TestHypercubeMulticastModel(unittest.TestCase):
    """
    Tests the HypercubeMulticastModel with a series of premade test cases.
    """

    TEST_CASES_FILE: str = Path(__file__).parent / "multicast" / "test_cases.yaml"
    testcases: dict = load_solutions(TEST_CASES_FILE)

    def test_gamut(self):
        """
        Tests the entire gamut of test cases we have specified in the yaml.
        """
        for test in self.testcases:
            # Reads test case parameters and constructs the necessary objects.
            dim_tags: list[Tag] = construct_spacetime(test["dims"])
            fill: Fill = Fill(dim_tags, test["fill"])
            occ: Occupancy = Occupancy(dim_tags, test["occ"])
            dist_fn: isl.Map = test["dist_fn"]
            multicast_model: HypercubeMulticastModel = HypercubeMulticastModel(dist_fn)

            # Applies the model.
            info: TransferInfo = multicast_model.apply(0, fill, occ)
            # Checks the results.
            sum_extract: int = info.hops.eval(
                isl.Point.zero(info.hops.domain().get_space())
            )

            # The block is used for debugging test cases not yet implemented.
            if test["expected"]["hypercube_hops"] is None:
                print("~~~Test case in progress:~~~")
                print(f"Fill: {fill}")
                print(f"Occ: {occ}")
                print(f"Dist Fn: {dist_fn}")
                print(f"Returned: {sum_extract}")
            else:
                assert sum_extract == test["expected"]["hypercube_hops"]
