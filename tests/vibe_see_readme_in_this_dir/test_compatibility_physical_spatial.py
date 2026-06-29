"""
Tests for recording spatial-for loops below a physically-distributed storage in
``Compatibility`` (see ``TensorReservation.physical_spatial_loops``).

Whether a storage is distributed is read from the reservation node's
``_component_object`` (set when the reservation is created), so these tests stub that
object directly on the reservation node rather than passing an architecture.

    pytest tests/test_compatibility_physical_spatial.py -v
    python -m unittest tests.test_compatibility_physical_spatial
"""

import types
import unittest

import accelforge as af
from accelforge.frontend.spec import Spec
from accelforge.frontend.mapping import Reservation, Spatial
from accelforge.mapper.FFM._join_pmappings.compatibility import Compatibility, Loop


class _StubStorage:
    """A storage that is distributed along a fixed set of physical dimensions."""

    def __init__(self, distributed, physical_dims):
        self._distributed = distributed
        self._physical_dims = set(physical_dims)

    def _is_distributed(self):
        return self._distributed

    def _has_physical_dim(self, dim_name):
        return dim_name in self._physical_dims


def _matmul_einsum():
    spec = Spec.from_yaml(
        af.examples.arches.simple,
        af.examples.workloads.matmuls,
        jinja_parse_data={"N_EINSUMS": 2, "M": 64, "KN": 64},
    )
    name = list(spec.workload.einsum_names)[0]
    return spec.workload.einsums[name]


def _mapping(nodes):
    return types.SimpleNamespace(nodes=list(nodes))


class TestPhysicalSpatialLoops(unittest.TestCase):
    def setUp(self):
        self.einsum = _matmul_einsum()
        # T1 is indexed by rank variables m -> M and n1 -> N1; n0 -> N0 does NOT index it.
        self.res = Reservation(purposes=["T1"], resource="DistBuf")
        self.sp_m = Spatial(rank_variable="m", name="X", component="DistBuf", tile_shape=4)
        self.sp_n1 = Spatial(rank_variable="n1", name="Y", component="DistBuf", tile_shape=4)
        self.sp_n0 = Spatial(rank_variable="n0", name="X", component="DistBuf", tile_shape=4)
        self.mapping = _mapping([self.res, self.sp_m, self.sp_n1, self.sp_n0])

    def _from(self, storage):
        # The reservation carries the arch component it reserves; that object is what
        # from_mapping consults to decide whether the storage is distributed.
        self.res._component_object = storage
        return Compatibility.from_mapping(self.mapping, {"T1"}, self.einsum)

    def _phys(self, compat):
        (t,) = compat.tensors
        return t.physical_spatial_loops

    def test_no_component_object_records_nothing(self):
        self.assertEqual(self._phys(self._from(None)), ())

    def test_non_distributed_records_nothing(self):
        self.assertEqual(self._phys(self._from(_StubStorage(False, {"X", "Y"}))), ())

    def test_dim_filter(self):
        # Only X is a physical dim: only the 'm' loop (over X, indexing T1) is recorded.
        phys = self._phys(self._from(_StubStorage(True, {"X"})))
        self.assertEqual(len(phys), 1)
        loop = phys[0]
        self.assertTrue(loop.is_spatial)
        self.assertEqual(str(loop.rank_name), "M")
        self.assertEqual(loop.spatial_dim, "X")
        # Concrete fanout is preserved through clearing.
        self.assertEqual(loop.tile_pattern.tile_shape, 4)

    def test_excludes_loop_not_indexing_tensor(self):
        # X and Y physical: 'm'->X and 'n1'->Y recorded; 'n0' (over X) doesn't index T1.
        phys = self._phys(self._from(_StubStorage(True, {"X", "Y"})))
        self.assertEqual(len(phys), 2)
        self.assertEqual({str(l.rank_name) for l in phys}, {"M", "N1"})
        self.assertEqual({l.spatial_dim for l in phys}, {"X", "Y"})

    def test_stops_at_next_storage_level(self):
        # A lower reservation ends the region; spatial loops below it are not recorded.
        lower = Reservation(purposes=["T1"], resource="DistBuf")
        self.res._component_object = _StubStorage(True, {"X"})
        mapping = _mapping([self.res, lower, self.sp_m])
        compat = Compatibility.from_mapping(mapping, {"T1"}, self.einsum)
        # Backing reservation (first) has the loop below the lower one excluded.
        backing = compat.get_reservation_of_tensor("T1")
        self.assertEqual(backing.physical_spatial_loops, ())

    def test_distribution_difference_breaks_equality(self):
        a = self._from(_StubStorage(True, {"X"}))
        b = self._from(_StubStorage(True, {"X", "Y"}))
        c = self._from(_StubStorage(True, {"X"}))
        self.assertEqual(a, c)
        self.assertEqual(hash(a), hash(c))
        self.assertNotEqual(a, b)
        # Structural (cleared) compatibilities also differ.
        self.assertNotEqual(
            a.clear_symbolic_tile_patterns(), b.clear_symbolic_tile_patterns()
        )

    def test_regular_loops_unaffected(self):
        # A reservation with above-storage loops keeps physical_spatial_loops empty
        # when the storage is not distributed, and n_loops/above_loop_index are unchanged.
        compat = self._from(_StubStorage(False, set()))
        (t,) = compat.tensors
        self.assertEqual(t.above_loop_index, len(t.loops))
        self.assertEqual(t.physical_spatial_loops, ())


if __name__ == "__main__":
    unittest.main()
