"""
Tests that a routing the distance function does not define is an error.

The topology is a star with the switch as a node: `noc[0]` is the switch and
`noc[1..4]` are GPUs, with `hops[1]` only between the switch and a GPU. A GPU
cannot fill from another GPU directly (`InvalidRoutingError`); the same
transfer staged through the switch (GPU -> switch, then switch -> GPU) is
legal and loads one link per stage.
"""

import unittest

import islpy as isl

from accelforge.model._looptree.reuse.isl.distributed.distributed_buffers import (
    StarMulticastModel,
)
from accelforge.model._looptree.reuse.isl.distributed.mesh_casts import (
    InvalidRoutingError,
    identify_mesh_casts,
)
from accelforge.model._looptree.reuse.isl.mapping_to_isl.types import (
    Fill,
    Occupancy,
    SpatialTag,
)

_TAGS = [SpatialTag(0, 0)]


def _map(text: str) -> isl.Map:
    return isl.Map.read_from_str(isl.DEFAULT_CONTEXT, text)


# { [dst -> src] -> hops[1] } only when exactly one end is the switch noc[0].
SWITCH_DIST: isl.Map = _map(
    "{ [noc[d] -> noc[s]] -> hops[1] : s = 0 and 1 <= d <= 4; "
    "[noc[d] -> noc[s]] -> hops[1] : d = 0 and 1 <= s <= 4 }"
)


class TestInvalidRouting(unittest.TestCase):
    """A fill matched to a source over an undefined `[dst -> src]` pair raises."""

    def test_gpu_to_gpu_raises(self):
        """GPU 1 holds the datum and GPU 2 needs it: no direct route exists."""
        occ = _map("{ noc[1] -> data[0] }")
        fill = _map("{ noc[2] -> data[0] }")
        with self.assertRaisesRegex(InvalidRoutingError, "routing is invalid"):
            identify_mesh_casts(occ, fill, SWITCH_DIST)
        with self.assertRaises(InvalidRoutingError):
            StarMulticastModel(SWITCH_DIST).apply(
                0, Fill(_TAGS, fill), Occupancy(_TAGS, occ)
            )

    def test_one_undefined_candidate_raises(self):
        """A legal source does not excuse an illegal one for the same datum."""
        occ = _map("{ noc[0] -> data[0]; noc[1] -> data[0] }")
        fill = _map("{ noc[2] -> data[0] }")
        with self.assertRaises(InvalidRoutingError):
            identify_mesh_casts(occ, fill, SWITCH_DIST)

    def test_self_delivery_needs_hops0(self):
        """A node filling a datum it holds is applied to `[n -> n]`, undefined here."""
        occ = _map("{ noc[1] -> data[0] }")
        with self.assertRaises(InvalidRoutingError):
            identify_mesh_casts(occ, occ, SWITCH_DIST)

    def test_no_source_is_unfulfilled_not_invalid(self):
        """A datum nobody holds is no routing at all: unfulfilled, no error."""
        occ = _map("{ noc[1] -> data[0] }")
        fill = _map("{ noc[2] -> data[1] }")
        info = StarMulticastModel(SWITCH_DIST).apply(
            0, Fill(_TAGS, fill), Occupancy(_TAGS, occ)
        )
        self.assertTrue(info.unfulfilled_fill.map_.is_equal(fill))

    def test_other_time_is_not_a_route(self):
        """On `noc[t, x]`, a source at another time step is no routing; one at the same time is."""
        dist = _map(
            "{ [noc[t, xd] -> noc[t, xs]] -> hops[1] : 0 <= t < 2 and 0 <= xd, xs < 2 and xd != xs }"
        )
        temporal = [True, False]
        # The datum sits at noc[1, 0] and is needed at noc[0, 0]: another time, not an error.
        identify_mesh_casts(
            _map("{ noc[1, 0] -> data[0] }"),
            _map("{ noc[0, 0] -> data[0] }"),
            dist,
            temporal,
        )
        # Same time, but [noc[0, 0] -> noc[0, 0]] has no hops: invalid.
        with self.assertRaises(InvalidRoutingError):
            identify_mesh_casts(
                _map("{ noc[0, 0] -> data[0] }"),
                _map("{ noc[0, 0] -> data[0] }"),
                dist,
                temporal,
            )
        # Without the mask every pair is checked, so the other-time source is invalid too.
        with self.assertRaises(InvalidRoutingError):
            identify_mesh_casts(
                _map("{ noc[1, 0] -> data[0] }"), _map("{ noc[0, 0] -> data[0] }"), dist
            )

    def test_via_switch_is_legal(self):
        """GPU 1 -> switch, then switch -> GPUs 2 and 3: one link load per hop."""
        model = StarMulticastModel(SWITCH_DIST)
        up = model.apply(
            0,
            Fill(_TAGS, _map("{ noc[0] -> data[0] }")),
            Occupancy(_TAGS, _map("{ noc[1] -> data[0] }")),
        )
        down = model.apply(
            0,
            Fill(_TAGS, _map("{ noc[2] -> data[0]; noc[3] -> data[0] }")),
            Occupancy(_TAGS, _map("{ noc[0] -> data[0] }")),
        )
        for info in (up, down):
            self.assertTrue(info.unfulfilled_fill.map_.is_empty())
        self.assertEqual(int(up.edge_pressure.eval_edge("spoke_out", [1])), 1)
        self.assertEqual(int(down.edge_pressure.eval_edge("spoke_in", [2])), 1)
        self.assertEqual(int(down.edge_pressure.eval_edge("spoke_in", [3])), 1)
        self.assertEqual(int(down.edge_pressure.eval_edge("spoke_in", [4])), 0)


if __name__ == "__main__":
    unittest.main()
