"""
End-to-end tests for delivering data from a source whose binding is not the
locally-optimal one.

The fixture is a 2x2 array whose DistributedBuffer has one physical instance per
array coordinate, so the two spatial loops below it *are* its binding, one per
array dimension, with two instances a single hop apart along each. It holds W0,
which is indexed by n0 and n1, and the loops below it iterate m on X and n0 on Y.
The locally-optimal binding therefore replicates W0 along X (m does not index it)
and partitions it along N0 on Y.

Bind W0 along some other rank and a destination's tile no longer sits in the
instance next to it: it straddles both instances, and each has to ship its share
over. On a mesh that turns a free local read into an all-to-all delivery costing
`(n^2 - 1) * volume / 3` hops for `n` instances -- the closed form in
``analysis.ipynb``.
"""

from pathlib import Path
from unittest import TestCase

import accelforge as af
from accelforge.frontend.mapping import iter_binding_blocks
from accelforge.frontend.renames import RANK_DUPLICATE

INPUT_FILES_DIR = Path(__file__).parent / "input_files" / "networked"

M = 4
KN = 4
M_TILE = 2
N_TILE = 2
BITS_PER_VALUE = 8

# Two instances of the DistributedBuffer along each array dimension, one hop
# apart, and two iterations of the spatial loop that binds each.
N_INSTANCES = 2
SPACING = 1

W0_HOPS = "Matmul0<SEP>action<SEP>NoC<SEP>W0<SEP>hop"
W0_USAGE = "Matmul0<SEP>usage<SEP>memory<SEP>DistributedBuffer<SEP>W0"

# W0 is N0 x N1, and no loop above the DistributedBuffer tiles either, so every
# instance's share of it comes only from the binding.
W0_ACTIONS = KN * KN * BITS_PER_VALUE


class BindingCostTestCase(TestCase):
    buffer_size = "inf"
    topology = "mesh"

    def _spec(self):
        return af.Spec.from_yaml(
            af.examples.workloads.basic.matmuls,
            INPUT_FILES_DIR / "binding.yaml",
            INPUT_FILES_DIR / "one_matmul_to_binding.yaml",
            jinja_parse_data={
                "N_EINSUMS": 1,
                "M": M,
                "KN": KN,
                "M_TILE": M_TILE,
                "N_TILE": N_TILE,
                "BUFFER_SIZE": self.buffer_size,
                "TOPOLOGY": self.topology,
            },
        )

    def evaluate(self, binding=None):
        """Evaluate the fixture. `binding` gives the ranks that W0's binding
        loops bind, [X, Y]; None uses the locally-optimal binding the model
        infers for itself."""
        result = self._spec().evaluate_mapping()
        if binding is None:
            return result.data

        # Re-evaluate the mapping the first pass produced, with its binding
        # rewritten. `evaluate_mapping` keeps a binding that is already there.
        mapping = result.mapping(0)
        rebound = False
        for storage, _ in iter_binding_blocks(mapping.nodes):
            if storage.component != "DistributedBuffer":
                continue
            self.assertEqual(len(storage.binding), len(binding))
            for loop, rank in zip(storage.binding, binding):
                loop.rank_variable = rank
            rebound = True
        self.assertTrue(rebound, "the fixture lost its distributed storage")

        spec = self._spec()
        spec.mapping = mapping
        return spec.evaluate_mapping().data

    def default_binding(self):
        mapping = self._spec().evaluate_mapping().mapping(0)
        for storage, _ in iter_binding_blocks(mapping.nodes):
            if storage.component == "DistributedBuffer":
                return [loop.rank_variable for loop in storage.binding]
        self.fail("the fixture lost its distributed storage")


class DeliveryTests:
    """Shared body for both topologies. Each supplies what its own closed form
    says a scattered dimension costs; the shape of the answer is the same."""

    # What one destination reads: W0 is N0 x N1, the X loop is above any tiling
    # of it, and the Y loop tiles N0.
    X_VOLUME = KN * KN * BITS_PER_VALUE
    Y_VOLUME = N_TILE * KN * BITS_PER_VALUE

    scattered_on_x: float
    scattered_on_y: float

    def test_default_binding_is_locally_optimal(self):
        # Replicated along X, whose loop (m) does not index W0; partitioned along
        # N0 on Y, which its loop does index.
        self.assertEqual(self.default_binding(), [RANK_DUPLICATE, "N0"])

    def test_locally_optimal_binding_never_leaves_an_instance(self):
        # Every PE sits with an instance that holds everything it asks for, so
        # nothing goes onto the network at all.
        self.assertEqual(self.evaluate()[W0_HOPS].iloc[0], 0)

    def test_locally_optimal_binding_costs_what_inferring_it_does(self):
        # Spelling the locally-optimal binding out explicitly must change nothing.
        self.assertEqual(
            self.evaluate([RANK_DUPLICATE, "N0"])[W0_HOPS].iloc[0],
            self.evaluate()[W0_HOPS].iloc[0],
        )

    def test_replicated_source_delivers_locally_wherever_it_is_needed(self):
        # An instance that holds the whole tile holds any slice of it, so
        # replicating along Y as well still serves every destination from next
        # door, however the destinations want the data partitioned.
        self.assertEqual(self.evaluate([RANK_DUPLICATE, RANK_DUPLICATE])[W0_HOPS].iloc[0], 0)

    def test_scattering_the_relevant_dimension(self):
        # Binding N1 to Y partitions W0 along a rank other than the one its
        # destinations are spread over, so no destination's tile is next door any
        # more and each is collected from both instances.
        self.assertEqual(
            self.evaluate([RANK_DUPLICATE, "N1"])[W0_HOPS].iloc[0], self.scattered_on_y
        )

    def test_scattering_the_irrelevant_dimension(self):
        # The X loop iterates m, which does not index W0, so every PE along X
        # wants the whole tile. Replicated, each reads it locally; partitioned,
        # each has to fetch the half it does not hold.
        self.assertEqual(
            self.evaluate(["N0", "N0"])[W0_HOPS].iloc[0], self.scattered_on_x
        )

    def test_both_dimensions_add_up(self):
        self.assertEqual(
            self.evaluate(["N1", "N1"])[W0_HOPS].iloc[0],
            self.scattered_on_x + self.scattered_on_y,
        )

    def test_rebinding_only_charges_the_tensor_it_moves(self):
        # T0 and T1 are not held in the distributed buffer, so rebinding W0 must
        # leave their traffic alone.
        base, rebound = self.evaluate(), self.evaluate(["N1", "N1"])
        for tensor in ("T0", "T1"):
            col = f"Matmul0<SEP>action<SEP>NoC<SEP>{tensor}<SEP>hop"
            self.assertEqual(base[col].iloc[0], rebound[col].iloc[0])


class TestScatteredDeliveryOnAMesh(DeliveryTests, BindingCostTestCase):
    """A mesh pays for the links each delivery crosses."""

    topology = "mesh"

    # Each instance multicasts the half it holds across the line: n multicasts of
    # a 1/n share, which is one broadcast's worth of wire.
    scattered_on_x = (N_INSTANCES - 1) * SPACING * DeliveryTests.X_VOLUME
    # An all-to-all delivery, (n^2 - 1)/3 chunks' worth of links, once per
    # iteration of the X loop above.
    scattered_on_y = (
        (M / M_TILE) * (N_INSTANCES**2 - 1) * SPACING * DeliveryTests.Y_VOLUME / 3
    )


class TestScatteredDeliveryOnASwitch(DeliveryTests, BindingCostTestCase):
    """A switch charges one traversal per unit that has to cross it, whatever the
    distance. With only two instances a hop apart these come to the same numbers
    as the mesh; the topologies part company on wider arrays."""

    topology = "all_to_all"

    # Of every n units delivered, the one share a PE already holds stays put.
    scattered_on_x = (N_INSTANCES - 1) * DeliveryTests.X_VOLUME
    scattered_on_y = (M / M_TILE) * (N_INSTANCES - 1) * DeliveryTests.Y_VOLUME


class TestBindingOccupancy(BindingCostTestCase):
    """A replicated binding puts a whole copy of the tile in every instance,
    where a partitioned one gives each instance a share."""

    # One whole copy of W0 exactly fills the buffer, so usage reads off as the
    # fraction of the tile that one instance holds.
    buffer_size = W0_ACTIONS

    def test_partitioned_along_one_dimension(self):
        # The locally-optimal binding splits W0 two ways along Y and replicates
        # it along X.
        self.assertEqual(self.evaluate()[W0_USAGE].iloc[0], 1 / 2)

    def test_partitioned_along_both_dimensions(self):
        self.assertEqual(self.evaluate(["N1", "N0"])[W0_USAGE].iloc[0], 1 / 4)

    def test_replicated_along_both_dimensions(self):
        # Every instance holds the whole tensor: the cheapest binding to move
        # data for is the most expensive one to store.
        self.assertEqual(self.evaluate([RANK_DUPLICATE, RANK_DUPLICATE])[W0_USAGE].iloc[0], 1)
