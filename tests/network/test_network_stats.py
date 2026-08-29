"""
How a network's stats compose as the analysis walks out of the mapping.

Three quantities travel together and they do not compose alike:

- `total_hops` is everything that ever crosses the network, and feeds energy.
- `max_hops` is the longest single route, a per-transfer pipeline fill cost.
- `max_traffic` is what the busiest link carries per dimension, summed over the
  whole run, and is what the bandwidth term of a network's latency divides by its
  throughput.

Repetition in time multiplies the first and the last and leaves the middle alone;
repetition in space leaves the last alone, because parallel instances have links
of their own.
"""

from types import SimpleNamespace
from unittest import TestCase

from accelforge.frontend._workload_isl._symbolic import Irrelevant, Relevant
from accelforge.frontend.arch.components import TopologySpec
from accelforge.model._looptree.reuse.symbolic._network import NetworkAnalyzer
from accelforge.model._looptree.reuse.symbolic._stats import (
    NetworkStats,
    SymbolicAnalysisOutput,
)
from accelforge.model._looptree.types import Buffet, Network


class TestNetworkStatsRepeat(TestCase):
    """`repeat` is the temporal case: the same node run n times over."""

    def _stats(self):
        return NetworkStats(total_hops=5, max_hops=3, max_traffic={"X": 7})

    def test_hops_and_traffic_accumulate(self):
        repeated = self._stats().repeat(4)
        self.assertEqual(repeated.total_hops, 20)
        self.assertEqual(repeated.max_traffic, {"X": 28})

    def test_the_longest_route_does_not_grow(self):
        # Running a transfer again does not make it travel further.
        self.assertEqual(self._stats().repeat(4).max_hops, 3)

    def test_repeating_once_changes_nothing(self):
        once = self._stats().repeat(1)
        self.assertEqual(once.total_hops, 5)
        self.assertEqual(once.max_traffic, {"X": 7})

    def test_the_original_is_left_alone(self):
        # `repeat` shallow-copies, so scaling max_traffic in place would corrupt
        # the child stats the caller is still holding.
        original = self._stats()
        original.repeat(4)
        self.assertEqual(original.total_hops, 5)
        self.assertEqual(original.max_traffic, {"X": 7})


class TestNetworkStatsAccumulateInSeries(TestCase):
    """`accumulate_in_series` folds in something that happens afterwards on the
    same links -- another tile shape of the same loop, say."""

    def test_hops_and_traffic_add_up(self):
        first = NetworkStats(total_hops=5, max_hops=2, max_traffic={"X": 7})
        first.accumulate_in_series(
            NetworkStats(total_hops=3, max_hops=1, max_traffic={"X": 4})
        )
        self.assertEqual(first.total_hops, 8)
        self.assertEqual(first.max_traffic, {"X": 11})

    def test_the_longest_route_is_the_longer_of_the_two(self):
        first = NetworkStats(total_hops=0, max_hops=2, max_traffic={})
        first.accumulate_in_series(NetworkStats(total_hops=0, max_hops=5))
        self.assertEqual(first.max_hops, 5)

    def test_a_dimension_only_the_other_side_touches_is_kept(self):
        first = NetworkStats(max_traffic={"X": 7})
        first.accumulate_in_series(NetworkStats(max_traffic={"Y": 4}))
        self.assertEqual(first.max_traffic, {"X": 7, "Y": 4})

    def test_accumulating_onto_a_blank_gives_back_the_other(self):
        blank = NetworkStats()
        blank.accumulate_in_series(
            NetworkStats(total_hops=5, max_hops=2, max_traffic={"X": 7})
        )
        self.assertEqual(blank.total_hops, 5)
        self.assertEqual(blank.max_hops, 2)
        self.assertEqual(blank.max_traffic, {"X": 7})


class _Component:
    """Stand-in for a flattened-arch node: an undistributed source, and a switch."""

    topology = TopologySpec.ALL_TO_ALL

    def _get_physical_fanout_along(self, dim_name, default=1):
        return 1


class _Arch:
    def __init__(self, above: bool):
        self._above = above

    def __getitem__(self, name):
        return _Component()

    def is_above(self, a, b):
        return self._above


class _Analyzer(NetworkAnalyzer):
    """What a destination asks for is the workload's business, not the
    accumulation's, so both sizes are pinned here. They differ because a network
    counts its own actions while a memory counts values."""

    VOLUME = 10
    OCCUPANCY = 2

    def _get_data_volume(self, network, child_shape):
        return self.VOLUME

    def _get_tile_occupancy(self, network, child_shape):
        return self.OCCUPANCY


NETWORK = Network(
    "A",
    "MatmulA",
    Buffet("A", "MatmulA", "DistributedBuffer"),
    Buffet("A", "MatmulA", "Register"),
    component="NoC",
)


class TestSpatialAccumulationKeepsEveryDimension(TestCase):
    """A spatial loop costs its own dimension's links, and everything the levels
    below it put on the links of *other* dimensions carries through untouched --
    the instances of this loop run side by side, each with its own links.

    Dropping those other dimensions leaves the busiest link looking empty, and a
    network with nothing to be bandwidth-bound by.
    """

    def _accumulate(self, child_traffic, *, shape_repeats=4):
        node = SimpleNamespace(name="X", component="Array", rank_variable="m")
        info = SimpleNamespace(
            job=SimpleNamespace(flattened_arch=_Arch(above=False)),
            tensor_to_relevancy={"A": {"m": Irrelevant()}, "I": {"m": Relevant("M")}},
            bindings=SimpleNamespace(rank_by_loop={}),
            workload=SimpleNamespace(
                einsums={"MatmulA": SimpleNamespace(output_tensor_names=["A"])}
            ),
            is_copy_operation=None,
        )
        child = SymbolicAnalysisOutput()
        child.network_stats[NETWORK] = NetworkStats(
            total_hops=5, max_hops=1, max_traffic=dict(child_traffic)
        )

        accumulated = {}
        _Analyzer(accumulated, info, "MatmulA", node).accumulate_child_result(
            child, shape_repeats, {}
        )
        return accumulated[NETWORK]

    def test_a_dimension_below_survives(self):
        # The child's traffic sits on some inner array dimension's links, which
        # this loop neither shares nor relieves.
        stats = self._accumulate({"P": 7})
        self.assertEqual(stats.max_traffic.get("P"), 7)

    def test_this_loops_own_dimension_is_charged(self):
        # Reducing four partials of 10, one of which is already home.
        stats = self._accumulate({"P": 7})
        self.assertEqual(stats.max_traffic.get("X"), 30)

    def test_the_child_adds_to_this_loops_own_dimension(self):
        # Traffic the child already put on *this* dimension shares these links.
        stats = self._accumulate({"X": 7})
        self.assertEqual(stats.max_traffic.get("X"), 37)

    def test_hops_still_accumulate(self):
        stats = self._accumulate({"P": 7})
        self.assertEqual(stats.total_hops, 30 + 5 * 4)
        self.assertEqual(stats.max_hops, 2)

    def test_the_memories_owe_for_the_combining(self):
        # Four partials reduced into the one instance holding the tensor: three
        # tiles of 2 values are read out of a peer and written back in, and with
        # a single owner that instance does all of it itself.
        stats = self._accumulate({"P": 7})
        self.assertEqual(stats.total_reduction_values, 3 * _Analyzer.OCCUPANCY)
        self.assertEqual(
            stats.max_per_unit_reduction_values, 3 * _Analyzer.OCCUPANCY
        )
