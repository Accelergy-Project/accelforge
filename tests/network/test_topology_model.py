from unittest import TestCase

from accelforge.frontend.arch.components import TopologySpec
from accelforge.frontend.renames import RANK_DUPLICATE
from accelforge.frontend._workload_isl._symbolic import (
    Irrelevant,
    PartiallyRelevant,
    Relevant,
)
from accelforge.model._looptree.reuse.symbolic._binding import (
    binding_is_local,
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
            RANK_DUPLICATE,
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
        self.assertEqual(cost.max_hops, (n - 1) * stride)
        self.assertEqual(cost.max_traffic, volume)

    def test_unicast(self):
        # Relevant (not distributed): each destination needs its own data
        # delivered i*stride hops away, so the total is quadratic and the link
        # nearest the source carries traffic for all (n - 1) downstream nodes.
        n, stride, volume = 4, 2, 10
        cost = self._cost(Relevant("n0"), n=n, stride=stride, volume=volume)
        self.assertEqual(cost.total_cost, sum(range(n)) * stride * volume)
        self.assertEqual(cost.max_hops, (n - 1) * stride)
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
        self.assertAlmostEqual(
            cost.total_cost,
            n_activated_physical * sum(range(n_dsts_per_physical)) * stride * volume,
        )
        self.assertAlmostEqual(cost.max_hops, (n_dsts_per_physical - 1) * stride)
        self.assertAlmostEqual(cost.max_traffic, (n_dsts_per_physical - 1) * volume)

    def test_partially_relevant_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self._cost(PartiallyRelevant("n0"), n=4, stride=2)


class TestAllToAllTopologyModel(TestCase):
    """Unit tests for the all-to-all (switch) cost model in isolation."""

    def _cost(self, relevancy, n, *, volume=10, last_fanout=99):
        # last_fanout is deliberately large and arbitrary: an all-to-all switch
        # must ignore physical stride entirely.
        return AllToAllTopologyModel().per_loop_transfer_cost(
            "m0", # stand-in rank different from the dst rank n0 (see below)
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
        a2a = AllToAllTopologyModel().per_loop_transfer_cost("m0", Relevant("n0"), **kwargs)
        mesh = MeshTopologyModel().per_loop_transfer_cost("m0", Relevant("n0"), **kwargs)

        self.assertEqual(a2a.total_cost, (n - 1) * volume)
        self.assertEqual(mesh.total_cost, sum(range(n)) * stride * volume)
        self.assertLess(a2a.total_cost, mesh.total_cost)
        self.assertLess(a2a.max_hops, mesh.max_hops)

    def test_accumulate_max_hops_persists(self):
        # overall_max_hops accumulates across calls for a given network.
        model = AllToAllTopologyModel()
        h = AllToAllTopologyModel.HOPS_PER_TRANSFER
        self.assertEqual(model.accumulate_max_hops("net", "X", h), h)
        self.assertEqual(model.accumulate_max_hops("net", "X", h), 2 * h)

    def test_the_dimensions_accumulate_apart(self):
        # A network whose links differ by direction charges each dimension its
        # own action, so hops along Y must not swell X's running total.
        model = AllToAllTopologyModel()
        h = AllToAllTopologyModel.HOPS_PER_TRANSFER
        model.accumulate_max_hops("net", "X", h)
        self.assertEqual(model.accumulate_max_hops("net", "Y", h), h)
        self.assertEqual(model.accumulate_max_hops("net", "X", h), 2 * h)


class TestBindingIsLocal(TestCase):
    """Whether a destination can read what it needs from the source instance
    next to it, which is what the transfer models charge for."""

    def _local(self, bound, relevancy):
        return binding_is_local(bound, relevancy)

    def test_matching_binding_is_local(self):
        self.assertTrue(self._local("N0", Relevant("N0")))
        self.assertTrue(self._local(RANK_DUPLICATE, Irrelevant()))

    def test_absent_binding_is_local(self):
        # A mapping with no binding is modelled as the locally-optimal one.
        self.assertTrue(self._local(None, Relevant("N0")))

    def test_replicated_source_is_local(self):
        # Every instance holds the whole tile, so it holds any slice of it.
        self.assertTrue(self._local(RANK_DUPLICATE, Relevant("N0")))

    def test_partitioned_along_the_wrong_rank_is_not_local(self):
        self.assertFalse(self._local("M", Relevant("N0")))


class TestMeshScatteredDelivery(TestCase):
    """Delivering distinct data to each destination from a source partitioned
    along a rank other than the one the destinations are spread over. Each
    destination's tile then straddles every instance, so it is collected from all
    of them instead of read from the one next door."""

    def _cost(self, *, n=4, last_fanout=1, volume=32, fanout=4, stride=1, local=False):
        if local:
            src_rank = RANK_DUPLICATE
        else:
            src_rank = "m0" # stand in rank different than n0 (see below)
        return MeshTopologyModel().per_loop_transfer_cost(
            src_rank,
            Relevant("n0"),
            shape_repeats=n,
            last_fanout=last_fanout,
            volume=volume,
            src_component=_Distributed(fanout=fanout, stride=stride),
            dim_name="X",
        )

    def test_one_instance_per_destination(self):
        # n instances a unit apart, one per destination: each destination pulls
        # volume/n from each, so the total is the sum of |i - j| over all pairs
        # times a chunk -- (n^2 - 1)/3 * volume, the closed form in analysis.ipynb.
        n, volume = 4, 32
        cost = self._cost(n=n, volume=volume)
        self.assertAlmostEqual(cost.total_cost, (n**2 - 1) * volume / 3)
        self.assertEqual(cost.max_hops, n - 1)
        self.assertAlmostEqual(cost.max_traffic, n * volume / 4)

    def test_matches_brute_force_hops(self):
        # sum over (instance, destination) pairs of distance * (volume / n).
        for n in range(1, 9):
            volume = 24
            expected = sum(abs(i - j) * volume / n for i in range(n) for j in range(n))
            self.assertAlmostEqual(
                self._cost(n=n, volume=volume, fanout=n).total_cost,
                expected,
                msg=f"n={n}",
            )

    def test_costs_more_than_a_local_binding(self):
        local = self._cost(local=True)
        scattered = self._cost(local=False)
        self.assertGreater(scattered.total_cost, local.total_cost)
        self.assertGreater(scattered.max_hops, local.max_hops)

    def test_one_instance_degrades_to_plain_unicast(self):
        # With a single instance there is nothing to scatter over: whatever the
        # mapping claims the binding is, the delivery is the ordinary unicast.
        n, last_fanout, volume = 4, 2, 10
        scattered = self._cost(
            n=n,
            last_fanout=last_fanout,
            volume=volume,
            fanout=2,
            stride=n * last_fanout,
        )
        self.assertAlmostEqual(
            scattered.total_cost, sum(range(n)) * last_fanout * volume
        )

    def test_scales_with_distance_between_instances(self):
        near = self._cost(n=4, last_fanout=1, fanout=4, stride=1)
        far = self._cost(n=8, last_fanout=2, fanout=4, stride=2)
        self.assertGreater(far.total_cost, near.total_cost)

    def test_irrelevant_loops_are_binding_sensitive_too(self):
        # All destinations want the whole tile, and here every one of them sits
        # with an instance. Replicated, each reads its copy without moving
        # anything. Partitioned, each of the n instances has to multicast the 1/n
        # share it holds the length of the line -- which comes to the same wire as
        # one multicast of the whole tile, so the cost of partitioning here is a
        # whole broadcast. (analysis.ipynb makes the same point the other way
        # round: on a mesh an all-gather and a broadcast are the same length.)
        n, last_fanout, volume = 4, 1, 32
        kwargs = dict(
            shape_repeats=n,
            last_fanout=last_fanout,
            volume=volume,
            src_component=_Distributed(fanout=n, stride=last_fanout),
            dim_name="X",
        )
        local = MeshTopologyModel().per_loop_transfer_cost(
            RANK_DUPLICATE, Irrelevant(), **kwargs,
        )
        # m0 is stand-in rank
        scattered = MeshTopologyModel().per_loop_transfer_cost(
            "m0", Irrelevant(), **kwargs,
        )
        self.assertEqual(local.total_cost, 0)
        self.assertEqual(scattered.total_cost, (n - 1) * last_fanout * volume)


def _switch_crossing_volume(n, last_fanout, n_physical, stride, volume, local):
    """Brute-force ground truth for a switch: enumerate every (destination,
    source instance) pair and count what has to cross. A unit crosses unless the
    destination sits exactly on the instance holding it, and distance never
    matters, so this is the whole cost model."""
    destinations = [j * last_fanout for j in range(n)]
    instances = {i * stride for i in range(n_physical)}
    crossing = 0.0
    for position in destinations:
        if local:
            # The destination's whole tile is in the one instance holding it,
            # free only if that instance is right here.
            crossing += 0 if position in instances else volume
        else:
            # A 1/n_physical share from every instance; the local one is free.
            for instance in instances:
                crossing += 0 if position == instance else volume / n_physical
    return crossing


class TestAllToAllHonoursCoLocation(TestCase):
    """A switch charges one traversal per unit that has to cross it, so what a
    destination can read from the instance it sits with is free -- and how much
    that is depends on the binding."""

    def _cost(self, relevancy, *, n, last_fanout, fanout, stride, volume, local):
        if local:
            src_rank = RANK_DUPLICATE
        else:
            src_rank = "m0"
        return AllToAllTopologyModel().per_loop_transfer_cost(
            src_rank,
            relevancy,
            shape_repeats=n,
            last_fanout=last_fanout,
            volume=volume,
            src_component=_Distributed(fanout=fanout, stride=stride),
            dim_name="X",
        )

    def test_one_instance_per_destination(self):
        # Four PEs, one buffer instance each, each PE wanting 4 values of a 4x4
        # tensor. Bound along the rank the loop spreads over, every PE reads its
        # row from its own instance and the switch is never touched. Bound along
        # the other rank, each PE's row has one element in each instance, so it
        # pulls 3 of its 4 across: 12 element-transfers in all.
        common = dict(n=4, last_fanout=1, fanout=4, stride=1, volume=4)
        self.assertEqual(self._cost(Relevant("N0"), **common, local=True).total_cost, 0)
        self.assertEqual(
            self._cost(Relevant("N0"), **common, local=False).total_cost, 12
        )

    def test_nothing_crossing_costs_no_latency_either(self):
        cost = self._cost(
            Relevant("N0"), n=4, last_fanout=1, fanout=4, stride=1, volume=4, local=True
        )
        self.assertEqual(cost.max_hops, 0)
        self.assertEqual(cost.max_traffic, 0)

    def test_undistributed_source_is_unaffected_by_the_binding(self):
        # One instance holds everything, so there is no other rank to bind along
        # and the cost is the familiar "n - 1 destinations, one traversal each".
        n, volume = 5, 10
        for local in (True, False):
            if local:
                src_rank = RANK_DUPLICATE
            else:
                src_rank = "m0"
            cost = AllToAllTopologyModel().per_loop_transfer_cost(
                src_rank,
                Relevant("n0"),
                shape_repeats=n,
                last_fanout=1,
                volume=volume,
                src_component=_NoDistribution(),
                dim_name="X",
            )
            self.assertEqual(cost.total_cost, (n - 1) * volume)
            self.assertEqual(cost.max_hops, AllToAllTopologyModel.HOPS_PER_TRANSFER)

    def test_matches_brute_force_over_array_geometries(self):
        volume = 12.0
        checked = 0
        for n in (1, 2, 4, 8):
            for last_fanout in (1, 2, 4):
                for stride in (1, 2, 4, 8):
                    n_physical = max(n * last_fanout // stride, 1)
                    # The instances have to span the fanout for the geometry to
                    # be one the analyzer could hand us.
                    if n_physical * stride != max(n * last_fanout, stride):
                        continue
                    for local in (True, False):
                        for relevancy in (Relevant("N0"), Irrelevant()):
                            cost = self._cost(
                                relevancy,
                                n=n,
                                last_fanout=last_fanout,
                                fanout=n_physical,
                                stride=stride,
                                volume=volume,
                                local=local,
                            )
                            self.assertAlmostEqual(
                                cost.total_cost,
                                _switch_crossing_volume(
                                    n, last_fanout, n_physical, stride, volume, local
                                ),
                                msg=f"n={n} fanout={last_fanout} P={n_physical} "
                                f"stride={stride} local={local}",
                            )
                            checked += 1
        self.assertGreater(checked, 40, "the sweep degenerated")


def _mesh_hops(n, last_fanout, n_physical, stride, volume, local, irrelevant):
    """Brute-force ground truth for the mesh: place the destinations and the
    source instances on the line and add up the links every delivery crosses.

    An instance serves the sub-array beneath it, so the destinations whose
    coordinates fall within its stride are the ones it sits among.
    """
    destinations = [j * last_fanout for j in range(n)]
    instances = [i * stride for i in range(n_physical)]
    chunk = volume / n_physical

    def span(positions):
        # A multicast streams from end to end of the range it covers, crossing
        # each link in it exactly once.
        return max(positions) - min(positions)

    if not local:
        if irrelevant:
            # Every destination wants the whole tile, and each instance holds a
            # share of it that all of them need.
            return sum(span(destinations + [q]) * chunk for q in instances)
        # Every (destination, instance) pair exchanges one chunk.
        return sum(abs(p - q) * chunk for p in destinations for q in instances)

    served = {q: [] for q in instances}
    for p in destinations:
        served[instances[min(p // stride, n_physical - 1)]].append(p)
    if irrelevant:
        return sum(span(mine + [q]) * volume for q, mine in served.items() if mine)
    return sum(abs(p - q) * volume for q, mine in served.items() for p in mine)


class TestMeshAgainstBruteForce(TestCase):
    """The mesh cost model against an enumeration of the actual traffic, swept
    over array geometries.

    It is exact wherever the destinations and the source instances share a
    spacing, which is the usual case. Where the source is distributed at a
    different granularity from the one the loop fans out at, the closed forms
    thin one evenly-spaced grid onto the other and lose a few percent at the
    boundary -- always by under-counting, and only when the data is scattered.
    """

    GEOMETRIES = [
        (n, last_fanout, stride)
        for n in (1, 2, 4, 8)
        for last_fanout in (1, 2, 4)
        for stride in (1, 2, 4, 8)
        # The instances have to span the fanout for the geometry to be one the
        # analyzer could hand us.
        if max(n * last_fanout // stride, 1) * stride == max(n * last_fanout, stride)
    ]

    def _check(self, n, last_fanout, stride, local, irrelevant):
        n_physical = max(n * last_fanout // stride, 1)
        volume = 12.0
        if local:
            src_rank = RANK_DUPLICATE
        else:
            # stand in rank
            src_rank = "m0"
        got = MeshTopologyModel().per_loop_transfer_cost(
            src_rank,
            Irrelevant() if irrelevant else Relevant("N0"),
            shape_repeats=n,
            last_fanout=last_fanout,
            volume=volume,
            src_component=_Distributed(fanout=n_physical, stride=stride),
            dim_name="X",
        )
        want = _mesh_hops(n, last_fanout, n_physical, stride, volume, local, irrelevant)
        return got.total_cost, want

    def test_local_delivery_is_exact(self):
        # Every destination reads from the instance it sits among, so the line
        # breaks into independent groups and the closed forms are the truth.
        for n, last_fanout, stride in self.GEOMETRIES:
            for irrelevant in (True, False):
                got, want = self._check(n, last_fanout, stride, True, irrelevant)
                self.assertAlmostEqual(
                    got,
                    want,
                    msg=f"n={n} fanout={last_fanout} stride={stride} "
                    f"irrelevant={irrelevant}",
                )

    def test_scattered_delivery_is_exact_on_a_matched_grid(self):
        checked = 0
        for n, last_fanout, stride in self.GEOMETRIES:
            if last_fanout != stride:
                continue
            for irrelevant in (True, False):
                got, want = self._check(n, last_fanout, stride, False, irrelevant)
                self.assertAlmostEqual(
                    got,
                    want,
                    msg=f"n={n} spacing={stride} irrelevant={irrelevant}",
                )
                checked += 1
        self.assertGreater(checked, 10, "the sweep degenerated")

    def test_scattered_delivery_never_over_counts(self):
        # Measured worst case: 4.5% low, or 17% for a single-iteration loop,
        # where the one destination sits at the end of the line rather than
        # spread along it.
        for n, last_fanout, stride in self.GEOMETRIES:
            for irrelevant in (True, False):
                got, want = self._check(n, last_fanout, stride, False, irrelevant)
                where = (
                    f"n={n} fanout={last_fanout} stride={stride} "
                    f"irrelevant={irrelevant}"
                )
                self.assertLessEqual(got, want + 1e-9, msg=where)
                self.assertGreaterEqual(got, want * (0.83 - 1e-9), msg=where)


class TestAllToAllSpatialReduction(TestCase):
    """A spatial loop over a rank an output does not project onto leaves one
    partial copy of the tile at each destination, and they have to be summed.

    This is the one place an output inverts an input's arithmetic: the
    replicating binding that lets every destination read an input from next door
    for free is, for an output, the promise that every instance ends up holding
    the finished tile -- an all-reduce.
    """

    def _cost(self, src_rank, relevancy, *, n=5, volume=10, src=None, output=True):
        return AllToAllTopologyModel().per_loop_transfer_cost(
            src_rank,
            relevancy,
            shape_repeats=n,
            last_fanout=1,
            volume=volume,
            src_component=src if src is not None else _NoDistribution(),
            dim_name="X",
            is_reduced_output=output,
        )

    def test_replicated_output_is_an_all_reduce(self):
        n, volume = 5, 10
        cost = self._cost(RANK_DUPLICATE, Irrelevant(), n=n, volume=volume)
        # Reduce-scatter then all-gather: two passes of "all but one partial
        # crosses", two traversals deep, one tile per link in each pass.
        self.assertEqual(cost.total_cost, 2 * (n - 1) * volume)
        self.assertEqual(cost.max_hops, 2 * AllToAllTopologyModel.HOPS_PER_TRANSFER)
        self.assertEqual(cost.max_traffic, 2 * volume)

    def test_replication_is_the_expensive_binding_for_an_output(self):
        # The bug this guards: read as an input, a replicated source with an
        # instance beside every destination is next door to everyone and costs
        # nothing. The identical call for an output is an all-reduce.
        n, volume = 5, 10
        common = dict(
            n=n, volume=volume, src=_Distributed(fanout=n, stride=1)
        )
        as_input = self._cost(RANK_DUPLICATE, Irrelevant(), output=False, **common)
        as_output = self._cost(RANK_DUPLICATE, Irrelevant(), **common)
        self.assertEqual(as_input.total_cost, 0)
        self.assertEqual(as_output.total_cost, 2 * (n - 1) * volume)

    def test_undistributed_source_funnels_every_partial_into_one_link(self):
        # One instance forms the sum, so its link carries all n - 1 partials and
        # is squarely the bottleneck -- nothing combines them in flight.
        n, volume = 5, 10
        cost = self._cost("m0", Irrelevant(), n=n, volume=volume)
        self.assertEqual(cost.total_cost, (n - 1) * volume)
        self.assertEqual(cost.max_hops, AllToAllTopologyModel.HOPS_PER_TRANSFER)
        self.assertEqual(cost.max_traffic, (n - 1) * volume)

    def test_owners_share_the_reduction_traffic(self):
        # Partitioned over four instances, each owns a quarter of the tile and
        # takes a quarter of each crossing partial.
        n, volume = 4, 12
        cost = self._cost(
            "m0", Irrelevant(), n=n, volume=volume, src=_Distributed(fanout=4, stride=1)
        )
        self.assertEqual(cost.total_cost, (n - 1) * volume)
        self.assertEqual(cost.max_traffic, (n - 1) * volume / 4)

    def test_all_reduce_is_twice_the_plain_reduce(self):
        duplicated = self._cost(RANK_DUPLICATE, Irrelevant())
        partitioned = self._cost("m0", Irrelevant())
        self.assertEqual(duplicated.total_cost, 2 * partitioned.total_cost)
        self.assertEqual(duplicated.max_hops, 2 * partitioned.max_hops)

    def test_relevant_loop_over_an_output_is_still_a_unicast(self):
        # A loop over a rank the output *does* project onto hands each
        # destination a distinct slice to write back. Nothing is reduced, and it
        # costs what delivering those slices costs.
        for src_rank in (RANK_DUPLICATE, "m0"):
            self.assertEqual(
                self._cost(src_rank, Relevant("n0")),
                self._cost(src_rank, Relevant("n0"), output=False),
            )

    def test_single_iteration_loop_reduces_nothing(self):
        # One destination, one partial, already the answer.
        for src_rank in (RANK_DUPLICATE, "m0"):
            cost = self._cost(src_rank, Irrelevant(), n=1)
            self.assertEqual(cost.total_cost, 0)
            self.assertEqual(cost.max_hops, 0)
            self.assertEqual(cost.max_traffic, 0)

    def test_partially_relevant_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self._cost(RANK_DUPLICATE, PartiallyRelevant("n0"))


class TestMeshSpatialReduction(TestCase):
    """On a mesh a reduction is the multicast played backwards.

    A reducing node adds its own partial to what arrives before forwarding, so
    one tile's worth crosses each link and the line costs its span -- not the
    quadratic sum it would cost to carry every partial separately to a sink.
    """

    N, DELTA, VOLUME = 5, 2, 10
    SPAN = (N - 1) * DELTA

    def _cost(self, src_rank, relevancy, *, src=None, output=True, n=None):
        return MeshTopologyModel().per_loop_transfer_cost(
            src_rank,
            relevancy,
            shape_repeats=self.N if n is None else n,
            last_fanout=self.DELTA,
            volume=self.VOLUME,
            src_component=src if src is not None else _NoDistribution(),
            dim_name="X",
            is_reduced_output=output,
        )

    def _distributed(self):
        # One instance beside every destination, each holding a whole copy: the
        # binding that hands an input to its neighbour for nothing.
        return _Distributed(fanout=self.N, stride=self.DELTA)

    def test_reducing_costs_what_the_multicast_costs(self):
        # Same links, same count, opposite direction.
        reduce = self._cost("m0", Irrelevant())
        multicast = self._cost("m0", Irrelevant(), output=False)
        self.assertEqual(reduce.total_cost, multicast.total_cost)
        self.assertEqual(reduce.max_hops, multicast.max_hops)
        self.assertEqual(reduce.total_cost, self.SPAN * self.VOLUME)

    def test_replication_is_the_expensive_binding_for_an_output(self):
        # As an input, every destination reads a whole copy from the instance it
        # sits with and nothing moves. As an output, those copies are partial
        # sums and every instance is promised the finished tile.
        src = self._distributed()
        self.assertEqual(
            self._cost(RANK_DUPLICATE, Irrelevant(), src=src, output=False).total_cost,
            0,
        )
        self.assertEqual(
            self._cost(RANK_DUPLICATE, Irrelevant(), src=src).total_cost,
            2 * self.SPAN * self.VOLUME,
        )

    def test_the_all_reduce_spans_the_whole_line_twice(self):
        # Reduced in, then carried back out. The line does not break into the
        # independent groups a replicated multicast enjoys: a partial in one
        # group is still needed by the instances in all the others.
        cost = self._cost(RANK_DUPLICATE, Irrelevant(), src=self._distributed())
        self.assertEqual(cost.max_hops, 2 * self.SPAN)
        self.assertEqual(cost.total_cost, 2 * self._cost("m0", Irrelevant()).total_cost)

    def test_combining_keeps_one_tile_on_each_link(self):
        # However many destinations there are, a link carries the running sum
        # once. A switch, which cannot add anything in flight, instead funnels
        # every partial onto the owner's link.
        for n in (2, 5, 50):
            with self.subTest(n=n):
                self.assertEqual(
                    self._cost("m0", Irrelevant(), n=n).max_traffic, self.VOLUME
                )
        self.assertEqual(
            self._cost(
                RANK_DUPLICATE, Irrelevant(), src=self._distributed()
            ).max_traffic,
            2 * self.VOLUME,
        )

    def test_the_memories_owe_a_tile_for_every_partial_that_moves(self):
        # Distance is the network's business; what a memory reads out and writes
        # back in is a count of tiles, and the same count on either topology.
        reduce = self._cost("m0", Irrelevant())
        self.assertEqual(reduce.reduction_tiles, self.N - 1)
        self.assertEqual(reduce.max_reduction_tiles_per_unit, self.N - 1)

        all_reduce = self._cost(
            RANK_DUPLICATE, Irrelevant(), src=self._distributed()
        )
        self.assertEqual(all_reduce.reduction_tiles, 2 * (self.N - 1))
        self.assertEqual(
            all_reduce.max_reduction_tiles_per_unit, 2 * (self.N - 1) / self.N
        )

    def test_a_single_destination_reduces_nothing(self):
        for src_rank in (RANK_DUPLICATE, "m0"):
            cost = self._cost(src_rank, Irrelevant(), n=1)
            self.assertEqual(cost.total_cost, 0)
            self.assertEqual(cost.max_hops, 0)
            self.assertEqual(cost.max_traffic, 0)
            self.assertEqual(cost.reduction_tiles, 0)

    def test_relevant_loop_over_an_output_is_still_a_unicast(self):
        # A rank the output does project onto gives each destination a distinct
        # slice to write back. Nothing is combined.
        for src_rank in (RANK_DUPLICATE, "m0"):
            self.assertEqual(
                self._cost(src_rank, Relevant("n0")),
                self._cost(src_rank, Relevant("n0"), output=False),
            )
