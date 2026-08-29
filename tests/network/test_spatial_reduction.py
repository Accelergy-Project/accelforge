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
