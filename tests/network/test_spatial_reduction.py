"""
End-to-end tests for the traffic a spatial loop over a reduced rank creates.

The fixture is a 1-D array of four coordinates whose DistributedBuffer has one
physical instance at each, so the single spatial loop below it *is* its binding.
MatmulA is ``A[m, nA] = sum over nI of I[m, nI] * WA[nI, nA]``, so a loop over
``nI`` is irrelevant to the output A and a loop over ``m`` is irrelevant to the
input WA. Both make the model infer a replicating binding, and there the
resemblance stops:

- WA replicated is the free case. Every destination reads the whole tile from the
  instance it sits with and the switch is never touched.
- A replicated is a spatial reduction. Each of the four destinations holds a
  partial copy of the same tile, none of them is the answer, and the binding
  promises every instance the finished tile -- so the partials are all-reduced.
"""

from pathlib import Path
from unittest import TestCase

import accelforge as af
from accelforge.frontend.mapping import iter_binding_blocks
from accelforge.frontend.renames import RANK_DUPLICATE
from accelforge.model._looptree.reuse.symbolic._network import NetworkAnalyzer

INPUT_FILES_DIR = Path(__file__).parent / "input_files" / "networked"

M = 4
NA = 4
NI = 4
FANOUT = 4
BITS_PER_VALUE = 8

HOPS = "MatmulA<SEP>action<SEP>NoC<SEP>{tensor}<SEP>hop"

# No loop above the DistributedBuffer tiles either tensor, so a destination asks
# for the whole thing.
A_VOLUME = M * NA * BITS_PER_VALUE
WA_VOLUME = NI * NA * BITS_PER_VALUE


class SpatialReductionTestCase(TestCase):
    TOPOLOGY = "all_to_all"

    def _spec(
        self,
        tensor,
        rank,
        m=M,
        outer_tile=None,
        fanout=FANOUT,
        spatial_tile=1,
        hop_throughput="inf",
        mem_throughput="inf",
    ):
        return af.Spec.from_yaml(
            af.examples.workloads.basic.matmuls,
            INPUT_FILES_DIR / "reduction.yaml",
            INPUT_FILES_DIR / "one_matmul_to_reduction.yaml",
            jinja_parse_data={
                "N_EINSUMS": 1,
                "M": m,
                "KN": NI,
                "FANOUT": fanout,
                "DIST_TENSOR": tensor,
                "SPATIAL_RANK": rank,
                "SPATIAL_TILE": spatial_tile,
                "TOPOLOGY": self.TOPOLOGY,
                # One trip through everything below unless a test asks otherwise.
                "OUTER_TILE": m if outer_tile is None else outer_tile,
                "HOP_THROUGHPUT": hop_throughput,
                "MEM_THROUGHPUT": mem_throughput,
            },
        )

    def evaluate(self, tensor, rank, binding=None):
        """Hops the NoC charges for `tensor`, held in the DistributedBuffer over a
        spatial loop on `rank`. `binding` gives the rank the binding loop binds;
        None uses the locally-optimal binding the model infers for itself."""
        result = self._spec(tensor, rank).evaluate_mapping()
        if binding is not None:
            mapping = result.mapping(0)
            rebound = False
            for block in iter_binding_blocks(mapping.nodes):
                if block.component != "DistributedBuffer":
                    continue
                self.assertEqual(len(block.binding), 1)
                block.binding[0].rank_variable = binding
                rebound = True
            self.assertTrue(rebound, "the fixture lost its distributed storage")
            spec = self._spec(tensor, rank)
            spec.mapping = mapping
            result = spec.evaluate_mapping()
        return result.data[HOPS.format(tensor=tensor)].iloc[0]

    def inferred_binding(self, tensor, rank):
        mapping = self._spec(tensor, rank).evaluate_mapping().mapping(0)
        for block in iter_binding_blocks(mapping.nodes):
            if block.component == "DistributedBuffer":
                return [loop.rank_variable for loop in block.binding]
        self.fail("the fixture lost its distributed storage")


class TestReplicatedInputStaysFree(SpatialReductionTestCase):
    """The case the reduction must not disturb."""

    def test_inferred_binding_replicates(self):
        self.assertEqual(self.inferred_binding("WA", "m"), [RANK_DUPLICATE])

    def test_no_traffic(self):
        # Every instance holds the whole of WA, so each destination reads it from
        # next door and nothing crosses the switch.
        self.assertEqual(self.evaluate("WA", "m"), 0)


class TestReducedOutputIsAllReduced(SpatialReductionTestCase):
    """A loop over the reduced rank leaves one partial per destination."""

    def test_inferred_binding_replicates(self):
        # Same inference as the input above -- which is exactly why the cost model
        # cannot read it the same way.
        self.assertEqual(self.inferred_binding("A", "nI"), [RANK_DUPLICATE])

    def test_all_reduce_is_charged(self):
        # Reduce-scatter then all-gather: two passes in which all but one of the
        # FANOUT partials crosses the switch.
        self.assertEqual(self.evaluate("A", "nI"), 2 * (FANOUT - 1) * A_VOLUME)

    def test_reducing_is_not_free(self):
        # The regression this whole change exists for: a replicated output used to
        # be read as "everyone already has it" and cost nothing at all.
        self.assertGreater(self.evaluate("A", "nI"), 0)

    def test_partitioned_output_reduces_once(self):
        # Bound along a rank it does project onto, only the instances owning each
        # piece need the sum, so it is formed once and does not travel back out.
        self.assertEqual(
            self.evaluate("A", "nI", binding="M"), (FANOUT - 1) * A_VOLUME
        )

    def test_replicating_costs_twice_what_partitioning_does(self):
        self.assertEqual(
            self.evaluate("A", "nI", binding=RANK_DUPLICATE),
            2 * self.evaluate("A", "nI", binding="M"),
        )


LATENCY = "MatmulA<SEP>latency<SEP>NoC"


class TestNetworkLatencyCountsTheWholeRun(SpatialReductionTestCase):
    """`max_link_traffic` is the busiest link's traffic over the *whole run*, so
    dividing it by the link throughput gives the time the run spends
    bandwidth-bound -- the same kind of quantity a memory's default
    `total_latency` reports. Reported per transfer instead, as it once was, the
    number says the same thing however long the run goes on.
    """

    HOP_THROUGHPUT = 256

    def _latency(self, m):
        # One trip per value of m, so every tensor on the NoC -- the reduced
        # output and both inputs alike -- crosses it m times over.
        return (
            self._spec(
                "A", "nI", m=m, outer_tile=1, hop_throughput=self.HOP_THROUGHPUT
            )
            .evaluate_mapping()
            .data[LATENCY]
            .iloc[0]
        )

    def test_latency_is_linear_in_the_number_of_trips(self):
        one = self._latency(1)
        self.assertGreater(one, 0)
        for m in (2, 4):
            with self.subTest(m=m):
                self.assertAlmostEqual(self._latency(m), m * one, places=5)

    def test_hop_latency_is_not_multiplied_by_the_trips(self):
        # The other half of the split: hops on the longest route is a per-transfer
        # fill cost, so it must not grow with the run. With an infinitely fast
        # link the whole latency is that term, and it is the same at any size.
        def fill(m):
            return (
                self._spec("A", "nI", m=m, outer_tile=1)
                .evaluate_mapping()
                .data[LATENCY]
                .iloc[0]
            )

        self.assertEqual(fill(1), fill(4))


class TestTotalTrafficIsConserved(SpatialReductionTestCase):
    """Cutting the run into more trips carrying proportionally less each cannot
    change what crosses the network in total.
    """

    EXPECTED_HOPS = 2 * (FANOUT - 1) * A_VOLUME

    def _hops(self, outer_tile):
        return (
            self._spec("A", "nI", outer_tile=outer_tile)
            .evaluate_mapping()
            .data[HOPS.format(tensor="A")]
            .iloc[0]
        )

    def test_hops_are_the_same_however_the_outer_loop_is_tiled(self):
        # Total data reduced is fixed, so cutting the run into more trips carrying
        # proportionally less each cannot change what crosses the network.
        for outer_tile in (1, 2, M):
            with self.subTest(outer_tile=outer_tile):
                self.assertEqual(self._hops(outer_tile), self.EXPECTED_HOPS)


READ = "MatmulA<SEP>action<SEP>DistributedBuffer<SEP>{tensor}<SEP>read"
WRITE = "MatmulA<SEP>action<SEP>DistributedBuffer<SEP>{tensor}<SEP>write"
BUFFER_LATENCY = "MatmulA<SEP>latency<SEP>DistributedBuffer"


class TestTheMergePointPaysForCombining(SpatialReductionTestCase):
    """Carrying the partials across the network is only half of a reduction. The
    memories at either end have to read each one out and write it back in, and
    until they are charged for that the instances are holding partial sums that
    nothing ever adds up.

    Fanout is what is varied here: with a single instance there is nothing to
    combine, which gives an honest baseline for the same mapping.
    """

    MEM_THROUGHPUT = 64

    def _row(self, fanout):
        # NI values spread over `fanout` instances, so one instance's slice grows
        # as the array shrinks and the total work stays put.
        return (
            self._spec(
                "A",
                "nI",
                fanout=fanout,
                spatial_tile=NI // fanout,
                mem_throughput=self.MEM_THROUGHPUT,
            )
            .evaluate_mapping()
            .data.iloc[0]
        )

    def test_one_instance_has_nothing_to_combine(self):
        alone = self._row(1)
        self.assertEqual(alone[HOPS.format(tensor="A")], 0)
        # Just the write-back of the one copy of A, read out again on its way up.
        self.assertEqual(alone[READ.format(tensor="A")], A_VOLUME)
        self.assertEqual(alone[WRITE.format(tensor="A")], A_VOLUME)

    def test_every_bit_carried_is_read_out_and_written_in(self):
        # One hop carries one bit across this switch, and each of those bits
        # leaves one instance and lands in another.
        alone = self._row(1)
        for fanout in (2, FANOUT):
            with self.subTest(fanout=fanout):
                row = self._row(fanout)
                hops = row[HOPS.format(tensor="A")]
                self.assertGreater(hops, 0)
                self.assertEqual(
                    row[READ.format(tensor="A")] - alone[READ.format(tensor="A")],
                    hops,
                )
                self.assertEqual(
                    row[WRITE.format(tensor="A")] - alone[WRITE.format(tensor="A")],
                    hops,
                )

    def test_the_merge_point_spends_time_on_it(self):
        # The buffer's latency is its whole traffic spread over the instances
        # doing it -- an all-reduce shares the combining out evenly. Charge the
        # reduction to the totals but not to the per-instance figures and this
        # comes out at the write-back alone.
        for fanout in (1, 2, FANOUT):
            with self.subTest(fanout=fanout):
                row = self._row(fanout)
                traffic = row[READ.format(tensor="A")] + row[WRITE.format(tensor="A")]
                self.assertAlmostEqual(
                    row[BUFFER_LATENCY],
                    traffic / (fanout * self.MEM_THROUGHPUT),
                    places=5,
                )

    def test_a_replicated_input_is_not_charged(self):
        # The same inferred binding on an input: every instance reads its copy
        # from next door, and there is nothing to combine afterwards.
        row = (
            self._spec("WA", "m", mem_throughput=self.MEM_THROUGHPUT)
            .evaluate_mapping()
            .data.iloc[0]
        )
        self.assertEqual(row[HOPS.format(tensor="WA")], 0)
        self.assertEqual(row[READ.format(tensor="WA")], WA_VOLUME)
        self.assertEqual(row[WRITE.format(tensor="WA")], WA_VOLUME)


class TestReducedOutputIsAllReducedOnAMesh(TestReducedOutputIsAllReduced):
    """The same story on a mesh. The hop counts happen to agree here because the
    destinations sit one array coordinate apart, so a switch traversal and a link
    crossing cost the same; what is being checked is that the topology is asked at
    all, and answers.
    """

    TOPOLOGY = "mesh"


class TestTheMergePointPaysForCombiningOnAMesh(TestTheMergePointPaysForCombining):
    """What a memory reads out and writes back in is a count of tiles, so it does
    not depend on how far they travelled -- the merge point owes the same on
    either topology."""

    TOPOLOGY = "mesh"


class TestReplicatedInputStaysFreeOnAMesh(TestReplicatedInputStaysFree):
    TOPOLOGY = "mesh"


FANOUT_X = 4


class Reduction2DTestCase(TestCase):
    """The same reduction on a 2-D array.

    A destination of a spatial loop on X is a whole *column* of the array, and
    the tile it asks for is spread over the instances along Y below it. Those
    instances reduce their own slices along X side by side and at the same time,
    so what any one of them combines is its slice -- and the topology model,
    which divides by the instances along its own dimension only, has no way to
    know that. Charge one instance for the whole column's tile and the
    per-instance figure comes out `FANOUT_Y` times too big; it then exceeds what
    any instance could do under any schedule, and the latency it drives does not
    move at all as the array grows along Y.
    """

    TOPOLOGY = "all_to_all"
    MEM_THROUGHPUT = 64
    HOP_THROUGHPUT = "inf"
    SPATIAL_RANK_Y = "m"

    def _spec(self, fanout_y, fanout_x=FANOUT_X):
        return af.Spec.from_yaml(
            af.examples.workloads.basic.matmuls,
            INPUT_FILES_DIR / "reduction_2d.yaml",
            INPUT_FILES_DIR / "one_matmul_to_reduction_2d.yaml",
            jinja_parse_data={
                "N_EINSUMS": 1,
                "M": M,
                "KN": NI,
                "FANOUT_X": fanout_x,
                "FANOUT_Y": fanout_y,
                # The iterations have to fill the fanout in both dimensions. With
                # the reduced rank on both loops they share it out between them.
                "SPATIAL_TILE_X": NI // fanout_x,
                "SPATIAL_TILE_Y": (
                    M // fanout_y
                    if self.SPATIAL_RANK_Y == "m"
                    else NI // (fanout_x * fanout_y)
                ),
                "SPATIAL_RANK_Y": self.SPATIAL_RANK_Y,
                "TOPOLOGY": self.TOPOLOGY,
                "HOP_THROUGHPUT": self.HOP_THROUGHPUT,
                "MEM_THROUGHPUT": self.MEM_THROUGHPUT,
            },
        )

    def _row(self, fanout_y, fanout_x=FANOUT_X):
        return self._spec(fanout_y, fanout_x).evaluate_mapping().data.iloc[0]

    def _per_unit_traffic(self, row):
        """What one instance reads out and writes back in, recovered from the
        time the buffer spends on it at one instance's throughput."""
        return row[BUFFER_LATENCY] * self.MEM_THROUGHPUT

    def _total_traffic(self, row):
        return row[READ.format(tensor="A")] + row[WRITE.format(tensor="A")]


class TestOneInstanceCombinesOnlyItsOwnSlice(Reduction2DTestCase):
    """The Y loop is over ``m``, which A projects onto, so A is partitioned along
    Y and each Y instance holds ``M / FANOUT_Y`` of the tile the X line
    combines."""

    def test_the_other_dimension_does_not_change_what_there_is_to_combine(self):
        # The baseline the per-instance figures are checked against: sharding A
        # across Y does not change the total, because the X line still combines
        # FANOUT_X partials of the whole tile however that tile is spread.
        alone = self._row(1)
        for fanout_y in (2, 4):
            with self.subTest(fanout_y=fanout_y):
                row = self._row(fanout_y)
                for column in (
                    HOPS.format(tensor="A"),
                    READ.format(tensor="A"),
                    WRITE.format(tensor="A"),
                ):
                    self.assertEqual(row[column], alone[column])

    def test_all_reduce_latency_is_the_traffic_spread_over_every_instance(self):
        # An all-reduce shares the combining out evenly, so the whole traffic
        # divided by the whole array is what any one instance spends time on --
        # the 1-D invariant of TestTheMergePointPaysForCombining, over an array
        # with a second dimension to divide by as well.
        for fanout_y in (1, 2, 4):
            with self.subTest(fanout_y=fanout_y):
                row = self._row(fanout_y)
                self.assertAlmostEqual(
                    row[BUFFER_LATENCY],
                    self._total_traffic(row)
                    / (FANOUT_X * fanout_y * self.MEM_THROUGHPUT),
                    places=5,
                )

    def test_latency_falls_as_the_other_dimension_spreads_the_tile(self):
        # The regression stated as the thing that used to go wrong: with the
        # total work fixed, twice the instances is half the time each. Charge one
        # instance for the whole column's tile and this stays put instead.
        alone = self._row(1)[BUFFER_LATENCY]
        for fanout_y in (2, 4):
            with self.subTest(fanout_y=fanout_y):
                self.assertAlmostEqual(
                    self._row(fanout_y)[BUFFER_LATENCY], alone / fanout_y, places=5
                )

    def test_no_instance_combines_more_than_the_partials_it_receives(self):
        # An upper bound that holds whatever the schedule: an instance owns
        # A / (FANOUT_X * FANOUT_Y) of the tile and cannot be handed more than
        # one partial of it per other destination on its line, twice over for the
        # all-gather, plus its own write-back. Anything above this is work no
        # instance is doing.
        for fanout_y in (1, 2, 4):
            with self.subTest(fanout_y=fanout_y):
                row = self._row(fanout_y)
                owned = A_VOLUME / (FANOUT_X * fanout_y)
                self.assertLessEqual(
                    self._per_unit_traffic(row),
                    2 * (2 * (FANOUT_X - 1) + 1) * owned,
                )


class TestOneInstanceCombinesOnlyItsOwnSliceOnAMesh(
    TestOneInstanceCombinesOnlyItsOwnSlice
):
    """What a memory reads out and writes back in is a count of tiles, so it does
    not depend on how far they travelled -- the merge point owes the same on
    either topology."""

    TOPOLOGY = "mesh"


class TestTheBusiestLinkCarriesOnlyItsOwnLine(Reduction2DTestCase):
    """The network side of the same story.

    A link is one instance's worth of its dimension, so what the busiest X link
    carries is one Y instance's slice of the tile -- the FANOUT_Y lines of X
    links run side by side, each all-reducing a slice of its own. The topology
    model divides by the instances along X only, and the tile it is costing comes
    from `child_shape`, which the Y loop has not cut because that loop is nested
    *below* this one. Charge one line for what all of them carry between them and
    the bandwidth term of the network's latency does not move as the array grows
    along Y -- which is exactly the direction it should move fastest in.

    Nested the other way round -- the Y loop outer -- `child_shape` is already
    cut and there is nothing to divide out. The two nests are the same machine, so
    they have to come out the same, and this is the one that needed the work.
    """

    HOP_THROUGHPUT = 64

    def _latency(self, fanout_y):
        # The fixture's hop costs no time, so the whole of this is the bandwidth
        # term: the busiest link's traffic over the link's throughput.
        return self._row(fanout_y)[LATENCY]

    def test_latency_falls_as_the_other_dimension_spreads_the_tile(self):
        alone = self._latency(1)
        for fanout_y in (2, 4):
            with self.subTest(fanout_y=fanout_y):
                self.assertLess(self._latency(fanout_y), alone)

    def test_the_reduced_output_takes_its_share_off_the_busiest_link(self):
        # A is all-reduced along X, so it puts two passes of the tile on the
        # busiest link, and FANOUT_Y lines share that between them. Nothing else
        # on the NoC depends on FANOUT_Y, so the whole of the fall is A's.
        alone = self._latency(1)
        for fanout_y in (2, 4):
            with self.subTest(fanout_y=fanout_y):
                self.assertAlmostEqual(
                    self._latency(fanout_y),
                    alone - 2 * A_VOLUME * (1 - 1 / fanout_y) / self.HOP_THROUGHPUT,
                    places=5,
                )

    def test_the_hops_it_pays_for_are_unchanged(self):
        # The lines carry a slice each and there are that many of them, so the
        # total -- and the energy it drives -- was right before any of this.
        alone = self._row(1)[HOPS.format(tensor="A")]
        for fanout_y in (2, 4):
            with self.subTest(fanout_y=fanout_y):
                self.assertEqual(self._row(fanout_y)[HOPS.format(tensor="A")], alone)


class TestTheBusiestLinkCarriesOnlyItsOwnLineOnAMesh(
    TestTheBusiestLinkCarriesOnlyItsOwnLine
):
    """A mesh line reduces as it goes, so one tile's worth crosses each of its
    links -- a different number from the switch's, spread the same way."""

    TOPOLOGY = "mesh"


class TestReducingAlongBothDimensions(Reduction2DTestCase):
    """The reduced rank on *both* array dimensions: the case the division must
    not be applied to.

    A is replicated in both -- every instance holds a whole partial copy of the
    tile, not a slice of one -- so neither dimension spreads the tile for the
    other and there is nothing to divide out.
    """

    SPATIAL_RANK_Y = "nI"
    FANOUT = 2
    N_REDUCED_DIMENSIONS = 2

    def _row(self, fanout_y=FANOUT, fanout_x=FANOUT):
        return super()._row(fanout_y, fanout_x)

    def test_a_tile_replicated_along_the_other_dimension_is_not_divided_out(self):
        # Each dimension all-reduces: reduce-scatter then all-gather, and with
        # every instance holding a whole copy each pass reads out and writes in
        # the whole tile. So one instance combines at least the tile twice per
        # pass, in each dimension. Divide by an instance count that replicates
        # rather than partitions and it comes out under that.
        self.assertGreaterEqual(
            self._per_unit_traffic(self._row()),
            2 * self.N_REDUCED_DIMENSIONS * A_VOLUME,
        )

    def test_one_instance_never_owes_more_than_the_whole_array_does(self):
        # A per-instance figure above the total is work nothing is doing.
        row = self._row()
        self.assertLessEqual(self._per_unit_traffic(row), self._total_traffic(row))


class TestReducingAlongBothDimensionsOnAMesh(TestReducingAlongBothDimensions):
    TOPOLOGY = "mesh"


class TestTheSpreadBelowIsNeverMoreThanTheArrayHolds(TestCase):
    """A count of instances sharing a tile is at least one. Below one, the
    division it is used for becomes a multiplication, and an instance is charged
    more than the topology model said any instance combines.

    The nest that can produce one: the reduced rank on two X loops with the Y
    loop between them. A is replicated along X in both of them, so the binding
    reports their product as the replication along X -- but only the inner one is
    below the Y loop and spreads anything for it. Read the product as the spread
    below and the Y loop divides its tile by four where two instances share it.
    `max_loops_per_spatial_dimension` above one is all it takes to map this.
    """

    TOPOLOGY = "all_to_all"
    MEM_THROUGHPUT = 64
    FANOUT_X = 4
    FANOUT_Y = 2
    # nI over three loops: 2 iterations each, the two X loops and the Y between.
    KN = 8

    def _spec(self):
        return af.Spec.from_yaml(
            af.examples.workloads.basic.matmuls,
            INPUT_FILES_DIR / "reduction_2d.yaml",
            INPUT_FILES_DIR / "one_matmul_to_reduction_2d_nested.yaml",
            jinja_parse_data={
                "N_EINSUMS": 1,
                "M": M,
                "KN": self.KN,
                "FANOUT_X": self.FANOUT_X,
                "FANOUT_Y": self.FANOUT_Y,
                "TILE_X_OUTER": self.KN // 2,
                "TILE_Y": self.KN // 4,
                "TILE_X_INNER": self.KN // 8,
                "TOPOLOGY": self.TOPOLOGY,
                "HOP_THROUGHPUT": "inf",
                "MEM_THROUGHPUT": self.MEM_THROUGHPUT,
            },
        )

    def _sharers_reported(self):
        """Every value `_sharers_along_other_dims` returns over one evaluation of
        the fixture, with the spatial dimension it was asked about."""
        original = NetworkAnalyzer._sharers_along_other_dims
        seen = []

        def recording(analyzer, *args, **kwargs):
            sharers = original(analyzer, *args, **kwargs)
            seen.append((analyzer.node.name, sharers))
            return sharers

        NetworkAnalyzer._sharers_along_other_dims = recording
        try:
            self._spec().evaluate_mapping()
        finally:
            NetworkAnalyzer._sharers_along_other_dims = original
        return seen

    def test_never_reports_fewer_than_one_sharer(self):
        seen = self._sharers_reported()
        self.assertTrue(seen, "the fixture stopped reducing spatially")
        for dim, sharers in seen:
            with self.subTest(dim=dim, sharers=sharers):
                self.assertGreaterEqual(sharers, 1)

    def test_one_instance_never_owes_more_than_the_whole_array_does(self):
        row = self._spec().evaluate_mapping().data.iloc[0]
        traffic = row[READ.format(tensor="A")] + row[WRITE.format(tensor="A")]
        self.assertLessEqual(row[BUFFER_LATENCY] * self.MEM_THROUGHPUT, traffic)


class TestTheSpreadBelowIsNeverMoreThanTheArrayHoldsOnAMesh(
    TestTheSpreadBelowIsNeverMoreThanTheArrayHolds
):
    TOPOLOGY = "mesh"
