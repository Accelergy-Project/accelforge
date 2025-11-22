"""
File ported from here:
https://github.com/rengzhengcodes/timeloop/blob/distributed-multicast-dev/src/unit-test/multicast/test-multicast.cpp
"""

import unittest
from math import isclose
from pathlib import Path

import islpy as isl

from fastfusion.model.looptree.reuse.isl.mapping_to_isl.types import (
    # Data movement descriptors.
    Fill,
    Occupancy,
    # Tags
    Tag,
    SpatialTag,
    TemporalTag
)
from fastfusion.model.looptree.reuse.isl.distributed_buffers import (
    HypercubeMulticastModel
)
from fastfusion.model.looptree.reuse.isl.spatial import TransferInfo
from ..util import load_solutions

def construct_spacetime(dims: list) -> list[Tag]:
    spacetime: list[Tag] = []
    for dim in dims:
        if dim["type"] == "Temporal":
            spacetime.append(TemporalTag())
        elif dim["type"] == "Spatial":
            spacetime.append(
                SpatialTag(dim["spatial_dim"], dim["target"])
            )
    
    return spacetime


# class TestSimpleMulticastModel(unittest.TestCase):
#     """
#     Tests the distributed simple multicast model.
#     """

#     def test_0(self):
#         fill: Fill = Fill(
#             TemporalTag(), SpatialTag(0, None), SpatialTag(1, None),
#             isl.Map.read_from_str(
#                 isl.DEFAULT_CONTEXT,
#                 "{ spacetime[t, x, y] -> data[t + x + y] : "
#                 "0 <= x < 2 and 0 <= y < 2 and 0 <= t < 4 }"
#             )
#         )
#         occ: Occupancy = Occupancy(
#             fill.tags, fill.map_
#         )

#         multicast_model: SimpleMulticastModel = SimpleMulticastModel(False)
#         info = multicast_model.apply(0, fill, occ)

#         assert info.fulfilled_map.map_ == isl.Map.read_from_str(
#             isl.DEFAULT_CONTEXT,
#             "{ spacetime[t, x, y] -> data[t + x + y] : "
#             "0 <= t < 4 and 0 <= x < 2 and 0 <= y < 2 }"
#         )
#         assert info.parent_reads.map_ == isl.Map.read_from_str(
#             isl.DEFAULT_CONTEXT,
#             "{ spacetime[t] -> data[d] : 0 <= t < 4 and t <= d < t + 3 }"
#         )
#         assert len(info.compat_access_stats) == 1

#         for mutlicast_scatter, stats in info.compat_access_stats:
#             multicast, scatter = mutlicast_scatter

#             assert multicast == 1
#             assert scatter == 1
#             assert stats.accesses == 12
#             assert stats.hops == 1
        
#         multicast_model: SimpleMulticastModel = SimpleMulticastModel(True)
#         info = multicast_model.Apply(0, fill, occ)

#         assert info.fulfilled_fill.map_ == isl.Map.read_from_str(
#             isl.DEFAULT_CONTEXT,
#             "{ spacetime[t, x, y] -> data[t + x + y] : "
#             "0 <= t < 4 and 0 <= x < 2 and 0 <= y < 2 }"
#         )
#         assert info.parent_reads.map_ == isl.Map.read_from_str(
#             isl.DEFAULT_CONTEXT,
#             "{ spacetime[t] -> data[d] : 0 <= t < 4 and t <= d < t + 3 }"
#         )
#         assert len(info.compat_access_stats) == 1
#         for multicast_scatter, stats in info.compat_access_stats:
#             multicast, scatter = multicast_scatter

#             assert multicast == 1
#             assert scatter == 1
#             assert stats.accesses == 12
#             assert isclose(
#                 stats.hops, 3.667,
#                 abs_tol=0.001
#             )

#     def test_spatial_PC(self):
#         fill: Fill = Fill(
#             [TemporalTag(), SpatialTag(0, None), SpatialTag(1, None)],
#             isl.Map.read_from_str(
#                 isl.DEFAULT_CONTEXT,
#                 "{ spacetime[t, x, y] -> data[d, y] : "
#                 "0 <= x < 4 and 0 <= y < 2 and 0 <= t < 4 and x <= d < x+2 }"
#             )
#         )
#         occ: Occupancy = Occupancy(fill.tags, fill.map_)

#         multicast_model: SimpleMulticastModel = SimpleMulticastModel(True)

#         info = multicast_model.apply(0, fill, occ)

#         assert info.fulfilled_fill.map_ == isl.Map.read_from_str(
#             isl.DEFAULT_CONTEXT,
#             "{ spacetime[t, x, y] -> data[d, y] : "
#             "0 <= x < 4 and 0 <= y < 2 and 0 <= t < 4 and x <= d < x+2 }"
#         )
#         assert info.parent_reads.map_ == isl.Map.read_from_str(
#             isl.DEFAULT_CONTEXT,
#             "{ spacetime[t] -> data[d, y] : 0 <= y < 2 and 0 <= t < 4 and 0 <= d < 5 }"
#         )
#         assert len(info.compat_access_stats) == 1

#         for multicast_scatter, stats in info.compat_access_stats:
#             multicast, scatter = multicast_scatter
#             assert multicast == 1
#             assert scatter == 1
#             assert stats.accesses == 40
#             assert isclose(stats.hops, 5.2, abs_tol=0.001)


class TestHypercubeMulticastModel(unittest.TestCase):
    TEST_CASES_FILE: str = Path(__file__).parent / "multicast" / "test_cases.yaml"
    testcases: dict = load_solutions(TEST_CASES_FILE)

    def test_gamut(self):
        for test in self.testcases:
            # Reads test case parameters and constructs the necessary objects.
            dim_tags: list[Tags] = construct_spacetime(test["dims"])
            fill: Fill = Fill(dim_tags, test["fill"])
            occ: Occupancy = Occupancy(dim_tags, test["occ"])
            dist_fn: isl.Map = test["dist_fn"]
            multicast_model: HypercubeMulticastModel = HypercubeMulticastModel()

            # Applies the model.
            info: TransferInfo = multicast_model.apply(fill, occ, dist_fn)
            # Checks the results.
            sum_extract: int = info.hops.eval(
                isl.Point.zero(info.hops.domain().get_space())
            )

            # The block is used for debugging test cases not yet implemented.
            if test["expected"]["hypercube_hops"] is None:
                print("~~~Test case in progress:~~~")
                print(f"Fill: {fill}")
                print(f"Occ: {occ}")
                print(f"Dist Fn: {dist}")
                print(f"Returned: {sum_extract}")
            else:
                assert sum_extract == test["expected"]["hypercube_hops"]
