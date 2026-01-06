"""
Tests the spatial ISL models.
"""

import unittest

import islpy as isl

from fastfusion.frontend.mapping import MappingNode
from fastfusion.model._looptree.reuse.isl.mapping_to_isl.types import (
    Fill,
    Occupancy,
    SpatialTag,
    TemporalTag,
)
from fastfusion.model._looptree.reuse.isl.spatial import (
    SimpleLinkTransferModel,
    TransferInfo,
)


class TestSimpleLinkTransferModel(unittest.TestCase):
    """
    Test the simple link transfer model. Adapted from the original test at the
    following link:
    https://github.com/NVlabs/timeloop/blob/master/src/unit-test/test-simple-link-transfer.cpp
    """

    def test_simple_link_transfer_model_sandbox(self):
        """Independent sanity check of `SimpleLinkTransferModel`."""
        buffer: MappingNode = MappingNode()
        fill: Fill = Fill(
            [TemporalTag(), SpatialTag(0, buffer), SpatialTag(1, buffer)],  # type: ignore
            isl.Map.read_from_str(
                isl.DEFAULT_CONTEXT,
                "{ spacetime[t, x, y] -> data[t+x+y] : 0 <= x < 2 and 0 <= y < 2 and 0 <= t < 2}",
            ),
        )
        occ: Occupancy = Occupancy(fill.tags, fill.map_)
        link_transfer_model: SimpleLinkTransferModel = SimpleLinkTransferModel()
        info: TransferInfo = link_transfer_model.apply(buffer, fill, occ)

        assert info.fulfilled_fill.map_ == (
            fulfilled_soln := isl.Map.read_from_str(
                isl.DEFAULT_CONTEXT,
                "{ spacetime[t = 1, x, y = 0] -> data[1 + x] : 0 <= x <= 1; "
                "  spacetime[t = 1, x = 0, y] -> data[1 + y] : 0 <= y <= 1 }",
            )
        ), (
            "`fulfilled_fill` and solution mismatch\n"
            "--------------------------------------\n"
            f"solution:\n{fulfilled_soln}\n"
            f"received:\n{info.fulfilled_fill.map_}\n"
        )

        assert info.unfulfilled_fill.map_ == (
            unfulfilled_soln := isl.Map.read_from_str(
                isl.DEFAULT_CONTEXT,
                "{ spacetime[t = 0, x, y] -> data[x + y] : 0 <= x <= 1 and 0 <= y <= 1; "
                "  spacetime[t = 1, x = 1, y = 1] -> data[3] }",
            )
        ), (
            "`unfulfilled_fill` and solution mismatch\n"
            "----------------------------------------\n"
            f"solution:\n{unfulfilled_soln}\n"
            f"received:\n{info.unfulfilled_fill.map_}\n"
        )
