"""
Tests the temporal reuse analysis in ISL.
"""

import unittest

import islpy as isl

from fastfusion.model._looptree.reuse.isl.mapping_to_isl.types import (
    Occupancy,
    SpatialTag,
    TemporalTag,
)
from fastfusion.model._looptree.reuse.isl.temporal import (
    TemporalReuse,
    analyze_temporal_reuse,
)


class TestTemporalReuseAnalysis(unittest.TestCase):
    """
    Tests the temporal reuse analysis in ISL.
    """

    def test_multiple_loop_reuse_sandbox(self):
        """
        Tests a very basic temporal reuse analysis case without needing the
        `occupancies_from_mapping` function.
        """
        # Loads in the CONV1D Config
        occ: Occupancy = Occupancy(
            [TemporalTag(), SpatialTag(0, None), TemporalTag()],
            isl.Map.read_from_str(
                isl.DEFAULT_CONTEXT,
                (
                    "{ generic_iteration[t1, x, t0] -> tensor[d] : "
                    "t0 <= d < t0+3 and 0 <= t1 < 2 and 0 <= x < 2 and 0 <= t0 < 2 }"
                ),
            ).coalesce(),
        )

        result: TemporalReuse = analyze_temporal_reuse(occ, True, True)
        soln: isl.Map = isl.Map.read_from_str(
            isl.DEFAULT_CONTEXT,
            "{ generic_iteration[t1, x, t0] -> tensor[d] : "
            "0 <= x < 2 and "
            "(((t1 = 0) and (t0 = 0) and (0 <= d < 3)) or "
            " ((t1 = 0) and (t0 = 1) and (d = 3)) or "
            " ((t1 = 1) and (t0 = 0) and (d = 0)) or "
            " ((t1 = 1) and (t0 = 1) and (d = 3))"
            ")}",
        ).coalesce()
        fill: isl.Map = result.fill.map_
        eff_occ: isl.Map = result.effective_occupancy.map_
        print(eff_occ)
        assert soln.is_equal(fill), f"Expected:\n{soln}\nReceived:\n{fill}"
