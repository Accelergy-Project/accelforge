from unittest import TestCase

from accelforge.frontend.arch.components import TopologySpec
from accelforge.frontend._workload_isl._symbolic import (
    Irrelevant,
    PartiallyRelevant,
    Relevant,
)
from accelforge.model._looptree.reuse.symbolic._network import (
    AllToAllTopologyModel,
    MeshTopologyModel,
    get_topology_model,
)


class _NoDistribution:
    """Stand-in source component that is not physically distributed."""

    def _get_physical_fanout_along(self, dim_name, default=1):
        return 1


class _Distributed:
    """Stand-in source component physically distributed along a dimension."""

    def __init__(self, fanout, stride):
        self.fanout = fanout
        self.stride = stride

    def _get_physical_fanout_along(self, dim_name, default=1):
        return self.fanout

    def _get_physical_stride_along(self, dim_name):
        return self.stride


class TestMeshTopologyModel(TestCase):
    """Unit tests for the mesh cost model in isolation."""

    def _cost(self, relevancy, *, n, stride, volume=10, src=None):
        return MeshTopologyModel().per_loop_transfer_cost(
            relevancy,
            shape_repeats=n,
            last_fanout=stride,
            volume=volume,
            src_component=src if src is not None else _NoDistribution(),
            dim_name="X",
        )

    def test_registry_resolves_model(self):
        self.assertIsInstance(get_topology_model(TopologySpec.MESH), MeshTopologyModel)
        self.assertIsInstance(get_topology_model("mesh"), MeshTopologyModel)

    def test_multicast(self):
        # Irrelevant: one value flows down the line, dropped at each of the
        # (n - 1) downstream nodes. Each link carries it at most once.
        n, stride, volume = 4, 2, 10
        cost = self._cost(Irrelevant(), n=n, stride=stride, volume=volume)
        self.assertEqual(cost.total_cost, (n - 1) * stride * volume)
        self.assertEqual(cost.max_hops, n * stride)
        self.assertEqual(cost.max_traffic, volume)

    def test_unicast(self):
        # Relevant (not distributed): each destination needs its own data
        # delivered i*stride hops away, so the total is quadratic and the link
        # nearest the source carries traffic for all (n - 1) downstream nodes.
        n, stride, volume = 4, 2, 10
        cost = self._cost(Relevant("n0"), n=n, stride=stride, volume=volume)
        self.assertEqual(cost.total_cost, sum(range(n)) * stride * volume)
        self.assertEqual(cost.max_hops, n * stride)
        self.assertEqual(cost.max_traffic, (n - 1) * volume)

    def test_unicast_distributed_binds_locally(self):
        # When the source is physically distributed, data binds as locally as
        # possible, reducing hops relative to the non-distributed unicast.
        n, stride, volume = 4, 1, 10
        src = _Distributed(fanout=2, stride=4)
        cost = self._cost(Relevant("n0"), n=n, stride=stride, volume=volume, src=src)

        # physical_stride / last_fanout = 4, capped at shape_repeats = 4
        n_dsts_per_physical = 4
        n_activated_physical = 1  # n*stride / physical_stride = 4/4
        self.assertEqual(
            cost.total_cost,
            n_activated_physical * sum(range(n_dsts_per_physical)) * stride * volume,
        )
        self.assertEqual(cost.max_hops, (n_dsts_per_physical - 1) * stride)
        self.assertEqual(cost.max_traffic, (n_dsts_per_physical - 1) * volume)

    def test_partially_relevant_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self._cost(PartiallyRelevant("n0"), n=4, stride=2)


class TestAllToAllTopologyModel(TestCase):
    """Unit tests for the all-to-all (switch) cost model in isolation."""

    def _cost(self, relevancy, n, *, volume=10, last_fanout=99):
        # last_fanout is deliberately large and arbitrary: an all-to-all switch
        # must ignore physical stride entirely.
        return AllToAllTopologyModel().per_loop_transfer_cost(
            relevancy,
            shape_repeats=n,
            last_fanout=last_fanout,
            volume=volume,
            src_component=_NoDistribution(),
            dim_name="X",
        )

    def test_registry_resolves_model(self):
        # Resolves both by enum and by the StrEnum value (the form that survives
        # the arch evaluation pipeline).
        self.assertIsInstance(
            get_topology_model(TopologySpec.ALL_TO_ALL), AllToAllTopologyModel
        )
        self.assertIsInstance(get_topology_model("all_to_all"), AllToAllTopologyModel)

    def test_multicast(self):
        n, volume = 5, 10
        cost = self._cost(Irrelevant(), n, volume=volume)
        # Linear in destinations, one switch hop, shared link traffic.
        self.assertEqual(cost.total_cost, (n - 1) * volume)
        self.assertEqual(cost.max_hops, AllToAllTopologyModel.HOPS_PER_TRANSFER)
        self.assertEqual(cost.max_traffic, volume)

    def test_unicast(self):
        n, volume = 5, 10
        cost = self._cost(Relevant("n0"), n, volume=volume)
        # Same (linear) total cost as multicast and constant hops, but the
        # source's uplink to the switch carries every distinct message.
        self.assertEqual(cost.total_cost, (n - 1) * volume)
        self.assertEqual(cost.max_hops, AllToAllTopologyModel.HOPS_PER_TRANSFER)
        self.assertEqual(cost.max_traffic, (n - 1) * volume)

    def test_independent_of_stride(self):
        # Stride (last_fanout) must not affect any component of the cost.
        a = self._cost(Relevant("n0"), 5, last_fanout=1)
        b = self._cost(Relevant("n0"), 5, last_fanout=1000)
        self.assertEqual(
            (a.total_cost, a.max_hops, a.max_traffic),
            (b.total_cost, b.max_hops, b.max_traffic),
        )

    def test_linear_unlike_mesh_quadratic(self):
        # Against an identical mesh scenario, all-to-all unicast is linear while
        # the mesh is quadratic, and all-to-all hops are constant (< distance).
        n, volume, stride = 6, 1, 1
        kwargs = dict(
            shape_repeats=n,
            last_fanout=stride,
            volume=volume,
            src_component=_NoDistribution(),
            dim_name="X",
        )
        a2a = AllToAllTopologyModel().per_loop_transfer_cost(Relevant("n0"), **kwargs)
        mesh = MeshTopologyModel().per_loop_transfer_cost(Relevant("n0"), **kwargs)

        self.assertEqual(a2a.total_cost, (n - 1) * volume)
        self.assertEqual(mesh.total_cost, sum(range(n)) * stride * volume)
        self.assertLess(a2a.total_cost, mesh.total_cost)
        self.assertLess(a2a.max_hops, mesh.max_hops)

    def test_accumulate_max_hops_persists(self):
        # overall_max_hops accumulates across calls for a given network.
        model = AllToAllTopologyModel()
        h = AllToAllTopologyModel.HOPS_PER_TRANSFER
        self.assertEqual(model.accumulate_max_hops("net", h), h)
        self.assertEqual(model.accumulate_max_hops("net", h), 2 * h)
