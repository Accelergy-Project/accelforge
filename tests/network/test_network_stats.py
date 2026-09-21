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

All three are kept per spatial dimension, because a network's links need not be
alike in every direction: a dimension with a `hop_<dim>` action of its own is
charged its own energy, latency and throughput.
"""

from types import SimpleNamespace
from typing import override
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
        return NetworkStats(
            total_hops={"X": 5, "Y": 2},
            max_hops={"X": 3, "Y": 1},
            max_traffic={"X": 7},
        )

    def test_hops_and_traffic_accumulate(self):
        repeated = self._stats().repeat(4)
        self.assertEqual(repeated.total_hops, {"X": 20, "Y": 8})
        self.assertEqual(repeated.max_traffic, {"X": 28})

    def test_the_longest_route_does_not_grow(self):
        # Running a transfer again does not make it travel further.
        self.assertEqual(self._stats().repeat(4).max_hops, {"X": 3, "Y": 1})

    def test_repeating_once_changes_nothing(self):
        once = self._stats().repeat(1)
        self.assertEqual(once.total_hops, {"X": 5, "Y": 2})
        self.assertEqual(once.max_traffic, {"X": 7})

    def test_the_original_is_left_alone(self):
        # `repeat` shallow-copies, so scaling the per-dimension dicts in place
        # would corrupt the child stats the caller is still holding.
        original = self._stats()
        original.repeat(4)
        self.assertEqual(original.total_hops, {"X": 5, "Y": 2})
        self.assertEqual(original.max_hops, {"X": 3, "Y": 1})
        self.assertEqual(original.max_traffic, {"X": 7})


class TestNetworkStatsAccumulateInSeries(TestCase):
    """`accumulate_in_series` folds in something that happens afterwards on the
    same links -- another tile shape of the same loop, say."""

    def test_hops_and_traffic_add_up(self):
        first = NetworkStats(
            total_hops={"X": 5}, max_hops={"X": 2}, max_traffic={"X": 7}
        )
        first.accumulate_in_series(
            NetworkStats(total_hops={"X": 3}, max_hops={"X": 1}, max_traffic={"X": 4})
        )
        self.assertEqual(first.total_hops, {"X": 8})
        self.assertEqual(first.max_traffic, {"X": 11})

    def test_the_longest_route_is_the_longer_of_the_two(self):
        first = NetworkStats(max_hops={"X": 2}, max_traffic={})
        first.accumulate_in_series(NetworkStats(max_hops={"X": 5}))
        self.assertEqual(first.max_hops, {"X": 5})

    def test_the_dimensions_are_maxed_apart_from_one_another(self):
        # Y's longer route does not make X's look longer: the two are charged
        # different actions, at different latencies.
        first = NetworkStats(max_hops={"X": 2, "Y": 1})
        first.accumulate_in_series(NetworkStats(max_hops={"X": 1, "Y": 5}))
        self.assertEqual(first.max_hops, {"X": 2, "Y": 5})

    def test_a_dimension_only_the_other_side_touches_is_kept(self):
        first = NetworkStats(max_traffic={"X": 7}, total_hops={"X": 5})
        first.accumulate_in_series(
            NetworkStats(max_traffic={"Y": 4}, total_hops={"Y": 3})
        )
        self.assertEqual(first.max_traffic, {"X": 7, "Y": 4})
        self.assertEqual(first.total_hops, {"X": 5, "Y": 3})

    def test_accumulating_onto_a_blank_gives_back_the_other(self):
        blank = NetworkStats()
        blank.accumulate_in_series(
            NetworkStats(total_hops={"X": 5}, max_hops={"X": 2}, max_traffic={"X": 7})
        )
        self.assertEqual(blank.total_hops, {"X": 5})
        self.assertEqual(blank.max_hops, {"X": 2})
        self.assertEqual(blank.max_traffic, {"X": 7})

    def test_a_dimension_nothing_has_crossed_reads_as_zero(self):
        self.assertEqual(NetworkStats().total_hops["X"], 0)
        self.assertEqual(NetworkStats().max_hops["X"], 0)


class _Component:
    """Stand-in for a flattened-arch node: an undistributed source, and a switch
    that is the interconnect of the array named ``Array``."""

    topology = TopologySpec.ALL_TO_ALL
    _physical_spatial_component = "Array"

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


CHILD_DIM = "P"
"""The inner array dimension the levels below the loop under test fanned out
over. Kept distinct from that loop's own ``X`` so the two do not blur together."""

NETWORK = Network(
    "A",
    "MatmulA",
    Buffet("A", "MatmulA", "DistributedBuffer"),
    Buffet("A", "MatmulA", "Register"),
    component="NoC",
)


class _Accumulates:
    """Runs one spatial loop's worth of accumulation over a stub arch in which
    ``NoC`` is the interconnect of the array named ``Array``."""

    def _accumulate(self, child_traffic, *, shape_repeats=4, loop_component="Array"):
        node = SimpleNamespace(name="X", component=loop_component, rank_variable="m")
        info = SimpleNamespace(
            job=SimpleNamespace(flattened_arch=_Arch(above=False)),
            tensor_to_relevancy={"A": {"m": Irrelevant()}, "I": {"m": Relevant("M")}},
            bindings=SimpleNamespace(rank_by_loop={}, replicating_loops={}),
            # No binding loop iteration counts recorded, so nothing replicates
            # and no dimension's share of the tile comes off. `child.fanout` is
            # empty here anyway, so there is no other dimension to divide by.
            precomputed_iterations={},
            workload=SimpleNamespace(
                einsums={"MatmulA": SimpleNamespace(output_tensor_names=["A"])}
            ),
            is_copy_operation=None,
        )
        child = SymbolicAnalysisOutput()
        child.network_stats[NETWORK] = NetworkStats(
            # The levels below this loop fanned out over some inner dimension, so
            # what they contributed sits on that dimension's links, not this
            # loop's.
            total_hops={CHILD_DIM: 5},
            max_hops={CHILD_DIM: 1},
            max_traffic=dict(child_traffic),
        )

        accumulated = {}
        _Analyzer(accumulated, info, "MatmulA", node).accumulate_child_result(
            child, shape_repeats, {}
        )
        return accumulated[NETWORK]


class TestSpatialAccumulationKeepsEveryDimension(_Accumulates, TestCase):
    """A spatial loop costs its own dimension's links, and everything the levels
    below it put on the links of *other* dimensions carries through untouched --
    the instances of this loop run side by side, each with its own links.

    Dropping those other dimensions leaves the busiest link looking empty, and a
    network with nothing to be bandwidth-bound by.
    """

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
        self.assertEqual(stats.total_hops, {"X": 30, CHILD_DIM: 5 * 4})
        self.assertEqual(stats.max_hops, {"X": 1, CHILD_DIM: 1})

    def test_the_longest_route_is_split_between_the_dimensions(self):
        # One hop of X and one of P: two links in all, but each charged to the
        # action of the dimension it belongs to, so a network whose X and Y links
        # run at different rates is costed at the right one for each.
        stats = self._accumulate({"P": 7})
        self.assertEqual(sum(stats.max_hops.values()), 2)

    def test_this_loops_hops_do_not_land_on_the_childs_dimension(self):
        # The 30 this loop costs is X's traffic to carry; charging it to P would
        # have it paid at the wrong rate, and twice over once P is charged too.
        stats = self._accumulate({"P": 7})
        self.assertEqual(stats.total_hops[CHILD_DIM], 5 * 4)

    def test_the_memories_owe_for_the_combining(self):
        # Four partials reduced into the one instance holding the tensor: three
        # tiles of 2 values are read out of a peer and written back in, and with
        # a single owner that instance does all of it itself.
        stats = self._accumulate({"P": 7})
        self.assertEqual(stats.total_reduction_values, 3 * _Analyzer.OCCUPANCY)
        self.assertEqual(
            stats.max_per_unit_reduction_values, 3 * _Analyzer.OCCUPANCY
        )


class TestOnlyTheArraysOwnDimensionsAreCharged(_Accumulates, TestCase):
    """A `!Network` inside an `!Array` is that array's interconnect. Its links
    run between array units, so only the array's own spatial dimensions put
    traffic on it -- a fanout declared by a component inside a unit (a PE grid
    under the array's buffer, say) is private wiring that never reaches it, even
    though the network sits above it in the flattened hierarchy.

    `_Component` puts the network in the array named ``Array``, so a loop of any
    other component is one of those inner fanouts.
    """

    def _accumulate_inner(self, child_traffic, *, shape_repeats=4):
        return self._accumulate(
            child_traffic,
            shape_repeats=shape_repeats,
            loop_component="ProcessingElement",
        )

    def test_an_inner_fanout_puts_nothing_on_the_array_network(self):
        # The 30 of `test_this_loops_own_dimension_is_charged` is what this same
        # loop would cost were it one of the array's own.
        stats = self._accumulate_inner({"P": 7})
        self.assertIsNone(stats.max_traffic.get("X"))
        self.assertEqual(stats.total_reduction_values, 0)

    def test_the_child_still_carries_up(self):
        stats = self._accumulate_inner({"P": 7})
        self.assertEqual(stats.max_traffic.get("P"), 7)
        self.assertEqual(stats.total_hops, {CHILD_DIM: 5 * 4})
        self.assertEqual(stats.max_hops, {CHILD_DIM: 1})
        # Nothing of this loop's own reaches the array's network.
        self.assertEqual(stats.total_hops["X"], 0)

    def test_a_network_outside_any_array_still_carries_every_fanout(self):
        # Nothing scopes it, so it keeps the older behaviour of costing whatever
        # fans out below it -- what `hierarchical.yaml`'s PeArray relies on.
        try:
            _Component._physical_spatial_component = None
            stats = self._accumulate_inner({"P": 7})
        finally:
            _Component._physical_spatial_component = "Array"
        self.assertEqual(stats.max_traffic.get("X"), 30)


class _DistributedComponent(_Component):
    """A source memory with one instance at every coordinate of a 4(X) x 2(Y)
    array, the coordinates numbered with Y innermost -- so a step along X skips
    two of them and a step along Y skips one."""

    FANOUT = {"X": 4, "Y": 2}
    STRIDE = {"X": 2, "Y": 1}

    def _has_physical_dim(self, dim_name):
        return dim_name in self.FANOUT

    @override
    def _get_physical_fanout_along(self, dim_name, default=1):
        return self.FANOUT.get(dim_name, default)

    def _get_physical_stride_along(self, dim_name):
        return self.STRIDE[dim_name]


class _DistributedArch(_Arch):
    """The source is distributed; the network is still the plain switch."""

    def __getitem__(self, name):
        return _Component() if name == "NoC" else _DistributedComponent()


# The Y binding loop, which `_accumulate_across` may make a replicating one.
_Y_BINDING_LOOP = 1


class _AccumulatesAcrossTwoDimensions:
    """One spatial loop on X over a source distributed in both X and Y, with a
    spatial loop on Y already accounted for below it.

    The reduction is the same one `_Accumulates` costs -- four partials of
    `VOLUME`, over two instances along X -- so the figures here are that one's
    divided by whatever share of the tile the Y loop has taken away.
    """

    # Floats because the figures reach the test as sympy Floats, which compare
    # equal to a Python float and not to an int.
    UNDIVIDED_TRAFFIC = 15.0
    """3 partials of VOLUME crossing, over the 2 instances along X."""
    UNDIVIDED_REDUCTION_VALUES = 3.0
    """1.5 tiles of OCCUPANCY, the share of the combining one X owner does."""

    def _accumulate_across(self, child_fanout, *, replicated_along_y=None):
        node = SimpleNamespace(name="X", component="Array", rank_variable="m")
        replicating_loops = {}
        precomputed_iterations = {}
        if replicated_along_y is not None:
            replicating_loops[("DistributedBuffer", "A")] = [(_Y_BINDING_LOOP, "Y")]
            precomputed_iterations[_Y_BINDING_LOOP] = replicated_along_y
        info = SimpleNamespace(
            job=SimpleNamespace(flattened_arch=_DistributedArch(above=False)),
            tensor_to_relevancy={"A": {"m": Irrelevant()}},
            bindings=SimpleNamespace(
                rank_by_loop={}, replicating_loops=replicating_loops
            ),
            precomputed_iterations=precomputed_iterations,
            workload=SimpleNamespace(
                einsums={"MatmulA": SimpleNamespace(output_tensor_names=["A"])}
            ),
            is_copy_operation=None,
        )
        child = SymbolicAnalysisOutput()
        child.network_stats[NETWORK] = NetworkStats(
            total_hops={CHILD_DIM: 5}, max_hops={CHILD_DIM: 1}
        )
        child.fanout[("Array", "MatmulA")] = dict(child_fanout)

        accumulated = {}
        _Analyzer(accumulated, info, "MatmulA", node).accumulate_child_result(
            child, 4, {}
        )
        return accumulated[NETWORK]


class TestTheLinesBesideThisOneCarryTheirOwnShare(
    _AccumulatesAcrossTwoDimensions, TestCase
):
    """A link is one instance's worth of this dimension, so the busiest one
    carries a slice of the tile the topology model costed and not the whole of
    it: the Y loop below has spread that tile over the Y instances, and each of
    them reduces its own slice along a line of X links of its own.

    The topology model divides by the instances along X and has no way to know
    about Y, and `child_shape` -- which is what fixes the tile it is costing --
    has not been cut by the Y loop yet, because that loop is *below* this one.
    Left undivided, the model charges one line for traffic that FANOUT_Y lines
    carry between them, and the bandwidth term of the network's latency does not
    move at all as the array grows along Y.
    """

    def test_with_nothing_below_the_whole_tile_crosses(self):
        # The baseline: no second dimension, so nothing comes off.
        stats = self._accumulate_across({})
        self.assertEqual(stats.max_traffic.get("X"), self.UNDIVIDED_TRAFFIC)

    def test_the_other_dimension_takes_its_share_of_the_traffic(self):
        # Two Y instances share the tile, so each X line carries half of it.
        stats = self._accumulate_across({"Y": 2})
        self.assertEqual(stats.max_traffic.get("X"), self.UNDIVIDED_TRAFFIC / 2)

    def test_the_reduction_share_is_divided_the_same_way(self):
        # The two per-instance figures are measured against the same slice.
        stats = self._accumulate_across({"Y": 2})
        self.assertEqual(
            stats.max_per_unit_reduction_values,
            self.UNDIVIDED_REDUCTION_VALUES / 2,
        )

    def test_a_tile_replicated_along_the_other_dimension_is_not_divided(self):
        # Each Y instance holds a whole copy rather than a slice, so every X line
        # really does carry the whole tile.
        stats = self._accumulate_across({"Y": 2}, replicated_along_y=2)
        self.assertEqual(stats.max_traffic.get("X"), self.UNDIVIDED_TRAFFIC)

    def test_wiring_inside_an_array_unit_is_not_a_dimension_to_divide_by(self):
        # A PE grid under the source is not somewhere the source has instances,
        # so it spreads nothing.
        stats = self._accumulate_across({"reuse_input": 2})
        self.assertEqual(stats.max_traffic.get("X"), self.UNDIVIDED_TRAFFIC)

    def test_the_total_is_not_divided(self):
        # Summed over the lines the slices add back up to the whole tile, so the
        # hop count -- and the energy it drives -- was right as it stood.
        for child_fanout in ({}, {"Y": 2}):
            with self.subTest(child_fanout=child_fanout):
                stats = self._accumulate_across(child_fanout)
                self.assertEqual(stats.total_hops, {"X": 30, CHILD_DIM: 5 * 4})
                self.assertEqual(
                    stats.total_reduction_values, 3 * _Analyzer.OCCUPANCY
                )
