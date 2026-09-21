"""
End-to-end tests for a network whose links are not alike in every direction.

A network may name an action per array dimension -- `hop_X`, `hop_Y` -- instead
of a single `hop` shared by all of them. Each then carries its own energy,
latency and throughput, and the model books a hop to the action of the dimension
it was along. A dimension the network names no action for falls back to the
shared `hop`, which is what every network naming only that one does for every
dimension, so nothing about the older specs moves.

The fixture is the 2-D reduction array of `test_spatial_reduction`: MatmulA is
``A[m, nA] = sum over nI of I[m, nI] * WA[nI, nA]``, held in a DistributedBuffer
with one instance per array coordinate. With the reduced rank ``nI`` on both
spatial loops, A is all-reduced along X and again along Y, so both dimensions
carry traffic and the two can be told apart by what they cost.
"""

from pathlib import Path
from unittest import TestCase

import accelforge as af

INPUT_FILES_DIR = Path(__file__).parent / "input_files" / "networked"

M = 4
# nI over both spatial loops: FANOUT_X iterations of the outer, FANOUT_Y of the
# inner.
NI = 8
FANOUT_X = 4
FANOUT_Y = 2

HOP = "MatmulA<SEP>action<SEP>NoC<SEP>{tensor}<SEP>{action}"
ENERGY = "MatmulA<SEP>energy<SEP>NoC<SEP>{tensor}<SEP>{action}"
NOC_LATENCY = "MatmulA<SEP>component_latency<SEP>NoC"
TOTAL_LATENCY = "Total<SEP>latency"

TENSORS = ("A", "I", "WA")


class AnisotropicTestCase(TestCase):
    """One evaluation of the 2-D reduction fixture, on the isotropic NoC that
    names a single ``hop`` or on the anisotropic one that names an action per
    dimension."""

    TOPOLOGY = "mesh"
    SPATIAL_RANK_Y = "nI"

    def _row(self, *, anisotropic=True, **hop_costs):
        arch = "reduction_2d_anisotropic.yaml" if anisotropic else "reduction_2d.yaml"
        jinja = {
            "N_EINSUMS": 1,
            "M": M,
            "KN": NI,
            "FANOUT_X": FANOUT_X,
            "FANOUT_Y": FANOUT_Y,
            # The iterations have to fill the fanout in both dimensions; with the
            # reduced rank on both loops they share it out between them.
            "SPATIAL_TILE_X": NI // FANOUT_X,
            "SPATIAL_TILE_Y": (
                M // FANOUT_Y
                if self.SPATIAL_RANK_Y == "m"
                else NI // (FANOUT_X * FANOUT_Y)
            ),
            "SPATIAL_RANK_Y": self.SPATIAL_RANK_Y,
            "TOPOLOGY": self.TOPOLOGY,
            "MEM_THROUGHPUT": "inf",
            **hop_costs,
        }
        if not anisotropic:
            # The isotropic fixture knows only the one hop action, so X's costs
            # are the costs of every dimension.
            jinja["HOP_THROUGHPUT"] = jinja.pop("X_HOP_THROUGHPUT", "inf")
            jinja.pop("X_HOP_ENERGY", None)
            jinja.pop("X_HOP_LATENCY", None)
        spec = af.Spec.from_yaml(
            af.examples.workloads.basic.matmuls,
            INPUT_FILES_DIR / arch,
            INPUT_FILES_DIR / "one_matmul_to_reduction_2d.yaml",
            jinja_parse_data=jinja,
        )
        return spec.evaluate_mapping().data.iloc[0]

    def _hops(self, row, action, tensor="A"):
        column = HOP.format(tensor=tensor, action=action)
        self.assertIn(column, row.index, f"the NoC never charged {action}")
        return row[column]

    def _energy(self, row, action, tensor="A"):
        # An action nothing ever performed contributes no energy and so no
        # column of its own.
        return row.get(ENERGY.format(tensor=tensor, action=action), 0)


class TestTheDimensionsAreCountedApart(AnisotropicTestCase):
    """The hops are the same movement either way; naming an action per dimension
    only changes which action each is booked to."""

    def test_each_dimension_gets_its_own_column(self):
        row = self._row()
        for tensor in TENSORS:
            with self.subTest(tensor=tensor):
                self.assertIn(HOP.format(tensor=tensor, action="hop_X"), row.index)
                self.assertIn(HOP.format(tensor=tensor, action="hop_Y"), row.index)

    def test_the_shared_action_is_gone(self):
        self.assertNotIn(HOP.format(tensor="A", action="hop"), self._row().index)

    def test_both_dimensions_carry_something(self):
        # Otherwise the tests below would pass on a model that lumped every hop
        # onto one dimension.
        row = self._row()
        self.assertGreater(self._hops(row, "hop_X"), 0)
        self.assertGreater(self._hops(row, "hop_Y"), 0)

    def test_the_two_add_up_to_what_one_action_counted(self):
        # Nothing about the movement changed, so the same hops are there; they
        # are just told apart by the dimension they were along.
        split = self._row()
        together = self._row(anisotropic=False)
        for tensor in TENSORS:
            with self.subTest(tensor=tensor):
                self.assertAlmostEqual(
                    self._hops(split, "hop_X", tensor)
                    + self._hops(split, "hop_Y", tensor),
                    together[HOP.format(tensor=tensor, action="hop")],
                    places=5,
                )


class TestTheHopsLandOnTheDimensionTheyRanAlong(AnisotropicTestCase):
    """The Y loop over ``m``, a rank A projects onto, partitions A along Y
    instead of reducing it: every Y instance holds its own slice and reads it
    from where it sits. So A crosses X links and no Y ones, and a model that
    booked the reduction to the wrong dimension would charge it at Y's rate."""

    SPATIAL_RANK_Y = "m"

    def test_the_reduction_is_all_on_its_own_dimension(self):
        row = self._row()
        self.assertGreater(self._hops(row, "hop_X"), 0)
        self.assertEqual(self._hops(row, "hop_Y"), 0)

    def test_a_dimension_that_carries_nothing_costs_nothing(self):
        self.assertEqual(self._energy(self._row(Y_HOP_ENERGY=1000), "hop_Y"), 0)


class TestEachDimensionIsChargedItsOwnEnergy(AnisotropicTestCase):
    def test_energy_follows_the_dimensions_own_action(self):
        # Doubling what a step along X costs doubles the energy of the hops along
        # X, and the count of them does not move.
        cheap = self._row(X_HOP_ENERGY=1)
        dear = self._row(X_HOP_ENERGY=2)
        self.assertGreater(self._energy(cheap, "hop_X"), 0)
        self.assertEqual(self._hops(dear, "hop_X"), self._hops(cheap, "hop_X"))
        self.assertAlmostEqual(
            self._energy(dear, "hop_X"), 2 * self._energy(cheap, "hop_X"), places=5
        )

    def test_the_other_dimensions_energy_is_untouched(self):
        # The two actions are priced independently: what a step along X costs
        # says nothing about what a step along Y costs.
        dear_x = self._row(X_HOP_ENERGY=2)
        cheap_x = self._row(X_HOP_ENERGY=1)
        for tensor in TENSORS:
            with self.subTest(tensor=tensor):
                self.assertAlmostEqual(
                    self._energy(dear_x, "hop_Y", tensor),
                    self._energy(cheap_x, "hop_Y", tensor),
                    places=5,
                )


class TestEachDimensionIsBoundByItsOwnThroughput(AnisotropicTestCase):
    """The bandwidth term of a network's latency is its busiest link over that
    link's throughput. With the dimensions on actions of their own, each is
    divided by its own."""

    def test_a_slower_dimension_takes_longer(self):
        fast = self._row(X_HOP_THROUGHPUT=64, Y_HOP_THROUGHPUT=64)
        slow = self._row(X_HOP_THROUGHPUT=32, Y_HOP_THROUGHPUT=64)
        self.assertGreater(slow[NOC_LATENCY], fast[NOC_LATENCY])

    def test_the_other_dimension_is_slowed_independently(self):
        fast = self._row(X_HOP_THROUGHPUT=64, Y_HOP_THROUGHPUT=64)
        slow_y = self._row(X_HOP_THROUGHPUT=64, Y_HOP_THROUGHPUT=32)
        self.assertGreater(slow_y[NOC_LATENCY], fast[NOC_LATENCY])

    def test_matching_throughputs_give_back_the_shared_action(self):
        # Two dimension-specific actions at one throughput are the single `hop`
        # at that throughput: the split is bookkeeping, not a different model.
        self.assertAlmostEqual(
            self._row(X_HOP_THROUGHPUT=64, Y_HOP_THROUGHPUT=64)[NOC_LATENCY],
            self._row(anisotropic=False, X_HOP_THROUGHPUT=64)[NOC_LATENCY],
            places=4,
        )

    def test_the_longest_route_is_charged_per_dimension(self):
        # The wind-up term: the longest route's hops, each at the per-hop latency
        # of the dimension it is along.
        quick = self._row(X_HOP_LATENCY=0)
        slow = self._row(X_HOP_LATENCY=1)
        self.assertGreater(slow[TOTAL_LATENCY], quick[TOTAL_LATENCY])

    def test_the_wind_up_of_each_dimension_adds_to_the_route(self):
        # A route crossing X and then Y waits out both, so the two per-hop
        # latencies add rather than one of them winning.
        neither = self._row(X_HOP_LATENCY=0, Y_HOP_LATENCY=0)[TOTAL_LATENCY]
        x_only = self._row(X_HOP_LATENCY=1, Y_HOP_LATENCY=0)[TOTAL_LATENCY]
        y_only = self._row(X_HOP_LATENCY=0, Y_HOP_LATENCY=1)[TOTAL_LATENCY]
        both = self._row(X_HOP_LATENCY=1, Y_HOP_LATENCY=1)[TOTAL_LATENCY]
        self.assertAlmostEqual(
            both - neither, (x_only - neither) + (y_only - neither), places=4
        )


class TestADimensionWithNoActionOfItsOwn(AnisotropicTestCase):
    """`X_HOP_ACTION: hop` leaves the network naming `hop` and `hop_Y`, so X has
    no action of its own and falls back to the shared one -- which is what every
    dimension of a network naming only `hop` does."""

    def test_the_fallback_carries_the_dimensions_hops(self):
        row = self._row(X_HOP_ACTION="hop")
        self.assertGreater(self._hops(row, "hop"), 0)
        self.assertNotIn(HOP.format(tensor="A", action="hop_X"), row.index)

    def test_it_counts_what_the_named_action_counted(self):
        named = self._row()
        fallback = self._row(X_HOP_ACTION="hop")
        for tensor in TENSORS:
            with self.subTest(tensor=tensor):
                self.assertAlmostEqual(
                    self._hops(fallback, "hop", tensor),
                    self._hops(named, "hop_X", tensor),
                    places=5,
                )


class TestADimensionWithNowhereToBookItsHops(AnisotropicTestCase):
    """`X_HOP_ACTION: hop_Z` names an action for a dimension the array does not
    have, leaving X to fall back to a `hop` the network does not declare either.
    Nothing can be charged, and the error has to say so rather than blame an
    action the spec never mentions."""

    def test_the_error_says_what_to_declare(self):
        with self.assertRaises(ValueError) as caught:
            self._row(X_HOP_ACTION="hop_Z")
        self.assertIn("hop_<dim>", str(caught.exception))


class TestTheDimensionsAreCountedApartOnASwitch(TestTheDimensionsAreCountedApart):
    """Nothing here is the mesh's doing -- the switch charges one hop per
    transfer, and that hop still belongs to a dimension."""

    TOPOLOGY = "all_to_all"
