"""
Tests for adapting one binding of a tensor into another.

Two Einsums that agree on everything about a tensor they share but the ranks its
binding loops bind leave it laid out two different ways across the physical instances
of the memory that holds it. They can still be fused, by shuffling the data among
those instances in between -- and the mapper is charged what the shuffle costs.

The closed forms the cost tests check against are the ones the network model already
uses, read for the case where the source and the destinations are the same memory's
instances:

* A dimension along which the two bindings name the same rank costs nothing: the data
  is already where it has to be.
* A dimension along which the *left* side replicated costs nothing either: an instance
  holding a whole copy holds any slice of it.
* A dimension along which the left side partitioned and the right side wants the whole
  tile is an all-gather, which on a line costs one broadcast's worth of wire.
* A dimension along which the two partition along different ranks is an all-to-all.
"""

import unittest
from pathlib import Path

import pandas as pd

import accelforge as af
from accelforge.frontend.mapping import TilePattern
from accelforge.frontend.mapping.mapping import (
    Binding,
    Compute,
    MappingNodeWithChildren,
    Nested,
    _binding_str,
)
from accelforge.frontend.renames import RANK_DUPLICATE, Rank
from accelforge.frontend.spec import Spec
from accelforge.mapper.FFM._join_pmappings.binding_adapter import BindingAdapters
from accelforge.mapper.FFM._join_pmappings.compatibility import (
    BINDING_RANK_ANY,
    Compatibility,
    Loop,
    TensorReservation,
)
from accelforge.util._frozenset import fzs

INPUT_FILES = Path(__file__).parent / "input_files"
ARCH = INPUT_FILES / "binding_adapter.arch.yaml"
MAPPING = INPUT_FILES / "mapping" / "fused_matmuls_to_binding_adapter.yaml"

MEMORY = "SharedBuffer"
"""The only physically-distributed memory of the fixture, 2x2 with its instances one
array coordinate apart along each dimension."""
MEMORY_SIZE = 1048576
N_INSTANCES = 2
DUP = RANK_DUPLICATE

# The fixture's workload is a chain of two 8x8 matmuls over 8-bit values, so the
# tensor the two Einsums share is this many bits in all.
M = 8
KN = 8
BITS_PER_VALUE = 8
SHARED_TENSOR_BITS = M * KN * BITS_PER_VALUE

TILES_COLUMN = "fused_loop<SEP>Matmul0<SEP>n_iterations<SEP>0"
BINDING_COLUMN = "binding<SEP>Matmul0<SEP>n_iterations<SEP>1"
OCCUPANCY_COLUMN = "tensor<SEP>T1"


def _spec(**jinja) -> Spec:
    settings = {
        "N_EINSUMS": 2,
        "M": M,
        "KN": KN,
        "M_TILE": M // 2,
        "N_TILE": KN // 2,
        "MAY_ADAPT": "All",
        "HOP_LATENCY": 1,
    }
    settings.update(jinja)
    return Spec.from_yaml(
        ARCH, af.examples.workloads.basic.matmuls_any_einsums, jinja_parse_data=settings
    )


def _evaluate(**jinja) -> pd.DataFrame:
    """The fixture's mapping, evaluated -- which runs the join, and the adapter with
    it."""
    spec = _spec(**jinja)
    spec.mapping = Spec.from_yaml(
        MAPPING,
        jinja_parse_data={
            "M_TILE": M // 2,
            "N_TILE": KN // 2,
            **{k: v for k, v in jinja.items() if k.startswith("MATMUL")},
        },
    ).mapping
    return spec.evaluate_mapping().data


def _adapters(**jinja) -> BindingAdapters:
    """The adapters of the fixture, as the joiner builds them."""
    spec = _spec(**jinja)._spec_eval_expressions(einsum_name="Matmul0")
    return BindingAdapters(
        spec, spec._get_flattened_architecture(einsum_name="Matmul0")
    )


def _compatibility(
    binding: list[tuple[str, str]],
    tensor: str = "T1",
    resource: str = MEMORY,
    n_iterations=N_INSTANCES,
    fused_loops: tuple = (),
) -> Compatibility:
    """
    A Compatibility whose only reservation is of `tensor` in `resource`, bound as
    `binding` gives it (as ``(spatial dimension, rank)``) below `fused_loops`.
    """
    loops = tuple(
        Loop(
            rank_name=Rank("M"),
            tile_pattern=TilePattern(calculated_n_iterations=n),
            is_spatial=False,
        )
        for n in fused_loops
    )
    return Compatibility(
        tensors=fzs(
            [
                TensorReservation(
                    loops=loops,
                    name=tensor,
                    resource_name=resource,
                    physical_spatial_loops=tuple(
                        Loop(
                            rank_name=Rank(rank),
                            tile_pattern=TilePattern(calculated_n_iterations=n_iterations),
                            is_spatial=True,
                            spatial_dim=dimension,
                        )
                        for dimension, rank in binding
                    ),
                )
            ]
        ),
        reservation_indices=fzs([len(loops)]),
    )


def _binding_of(compatibility: Compatibility) -> list[tuple[str, str]]:
    return [
        (loop.spatial_dim, loop.rank_name)
        for tensor in compatibility.tensors
        for loop in tensor.physical_spatial_loops
    ]


class TestAdaptableCompatibilities(unittest.TestCase):
    """
    Setting the bindings aside is what lets the joiner find the two sides that differ
    only in them; adopting the other side's is what it does once it has paid to move
    the data.
    """

    def test_bindings_that_differ_only_in_rank_compare_equal_once_cleared(self):
        a = _compatibility([("X", DUP), ("Y", "N1")])
        b = _compatibility([("X", DUP), ("Y", "N2")])
        self.assertNotEqual(a, b)
        self.assertEqual(a.clear_bound_ranks(), b.clear_bound_ranks())

    def test_clearing_keeps_the_dimensions(self):
        # A binding over other dimensions is a different binding, not one that differs
        # only in the rank it names, so setting the ranks aside must not merge them.
        a = _compatibility([("X", "N1"), ("Y", DUP)])
        b = _compatibility([("Y", "N1"), ("X", DUP)])
        self.assertNotEqual(a.clear_bound_ranks(), b.clear_bound_ranks())

    def test_clearing_keeps_the_iteration_counts(self):
        a = _compatibility([("X", "N1")], n_iterations=2)
        b = _compatibility([("X", "N1")], n_iterations=4)
        self.assertNotEqual(a.clear_bound_ranks(), b.clear_bound_ranks())

    def test_clearing_leaves_no_real_rank_behind(self):
        cleared = _compatibility([("X", DUP), ("Y", "N1")]).clear_bound_ranks()
        self.assertEqual(
            [rank for _, rank in _binding_of(cleared)],
            [BINDING_RANK_ANY, BINDING_RANK_ANY],
        )

    def test_adopting_takes_the_other_sides_ranks(self):
        left = _compatibility([("X", DUP), ("Y", "N1")])
        right = _compatibility([("X", "M"), ("Y", "N2")])
        self.assertEqual(
            _binding_of(left.update_binding_to_match(right)), [("X", "M"), ("Y", "N2")]
        )

    def test_adopting_leaves_a_tensor_the_other_side_does_not_hold_alone(self):
        left = _compatibility([("X", "N1")], tensor="T1")
        right = _compatibility([("X", DUP)], tensor="T2")
        self.assertEqual(_binding_of(left.update_binding_to_match(right)), [("X", "N1")])

    def test_bindings_of_different_lengths_do_not_line_up(self):
        # Two groups that dropped a different set of degenerate binding loops leave no
        # way to tell which loop of one answers which of the other.
        left = next(iter(_compatibility([("X", DUP), ("Y", "N1")]).tensors))
        right = next(iter(_compatibility([("Y", "N1")]).tensors))
        self.assertFalse(left.binding_spatial_dim_matches(right))


class TestAdapterCost(unittest.TestCase):
    """
    What one adaptation costs, checked against the closed forms of the module
    docstring. Hop energy is 1 in the fixture, so an energy reads off as a hop count.
    """

    def cost(
        self,
        left: list[tuple[str, str]],
        right: list[tuple[str, str]],
        rows: dict | None = None,
        fused_loops: tuple = (),
        n_iterations=N_INSTANCES,
        **jinja,
    ):
        """
        ``(energy, latency)`` per row of adapting `left` into `right`, or None if the
        two cannot be adapted at all.

        The occupancy handed over is the *left* side's -- what one instance holds
        before the data is moved -- since that is the side the adapter is applied to
        and so the only one whose columns it has. What it converts that into is
        checked in :class:`TestRebindingFactor`.
        """
        adaptations = _adapters(**jinja).adaptations(
            _compatibility(left, fused_loops=fused_loops, n_iterations=n_iterations),
            _compatibility(right, fused_loops=fused_loops, n_iterations=n_iterations),
            {OCCUPANCY_COLUMN, TILES_COLUMN, BINDING_COLUMN},
        )
        if adaptations is None:
            return None
        partitioned = sum(1 for _, rank in left if rank != DUP)
        occupancy = SHARED_TENSOR_BITS / N_INSTANCES**partitioned
        n_rows = max((len(v) for v in (rows or {}).values()), default=1)
        columns = {OCCUPANCY_COLUMN: [occupancy / MEMORY_SIZE] * n_rows}
        columns.update(rows or {})
        df = pd.DataFrame(columns)
        energy = pd.Series(0.0, index=df.index)
        latency = pd.Series(0.0, index=df.index)
        for adaptation in adaptations:
            adaptation_energy, adaptation_latency = adaptation.cost(df)
            energy = energy + adaptation_energy
            latency = latency + adaptation_latency
        return energy, latency

    def hops(self, *args, **kwargs) -> float:
        energy, _ = self.cost(*args, **kwargs)
        return energy.iloc[0]

    def test_agreeing_bindings_need_no_adapter_at_all(self):
        binding = [("X", DUP), ("Y", "N1")]
        self.assertEqual(
            _adapters().adaptations(
                _compatibility(binding), _compatibility(binding), {OCCUPANCY_COLUMN}
            ),
            [],
        )

    def test_a_replicated_source_is_already_wherever_it_is_wanted(self):
        # Every instance holds a whole copy, so it holds any slice of it too.
        self.assertEqual(
            self.hops([("X", DUP), ("Y", DUP)], [("X", DUP), ("Y", "N1")]), 0
        )

    def test_gathering_a_partitioned_source_into_a_replicated_one(self):
        # Each of the two instances along Y holds half of what both of them now want,
        # so a whole tile's worth crosses in each of the two X columns.
        self.assertEqual(
            self.hops([("X", DUP), ("Y", "N1")], [("X", DUP), ("Y", DUP)]),
            N_INSTANCES * SHARED_TENSOR_BITS,
        )

    def test_partitioning_along_another_rank_is_an_all_to_all(self):
        # Every instance's tile straddles both instances, so of the two halves each
        # holds, one has to move: a whole tile's worth over the two X columns.
        self.assertEqual(
            self.hops([("X", DUP), ("Y", "N1")], [("X", DUP), ("Y", "N2")]),
            SHARED_TENSOR_BITS,
        )

    def test_moving_a_partition_from_one_dimension_to_another(self):
        # Partitioned along X and replicated down Y, into the other way round.
        self.assertEqual(
            self.hops([("X", "N1"), ("Y", DUP)], [("X", DUP), ("Y", "N1")]),
            SHARED_TENSOR_BITS,
        )

    def test_a_dimension_that_already_agrees_is_free(self):
        # X binds M on both sides, so only the disagreement on Y is charged.
        self.assertEqual(
            self.hops([("X", "M"), ("Y", "N1")], [("X", "M"), ("Y", DUP)]),
            SHARED_TENSOR_BITS,
        )

    def test_both_dimensions_add_up(self):
        # Swapping which dimension partitions which rank moves a quarter tile between
        # every pair of instances; the two dimensions cost a half tile each.
        self.assertEqual(
            self.hops([("X", "N1"), ("Y", "N2")], [("X", "N2"), ("Y", "N1")]),
            SHARED_TENSOR_BITS,
        )

    def test_a_mesh_agrees_with_a_switch_on_two_instances_one_hop_apart(self):
        # They part company on wider arrays; here every route is a single link either
        # way, so the two topologies have to give the same answer.
        for left, right in [
            ([("X", DUP), ("Y", "N1")], [("X", DUP), ("Y", DUP)]),
            ([("X", DUP), ("Y", "N1")], [("X", DUP), ("Y", "N2")]),
            ([("X", "N1"), ("Y", "N2")], [("X", "N2"), ("Y", "N1")]),
        ]:
            with self.subTest(left=left, right=right):
                self.assertEqual(
                    self.hops(left, right, TOPOLOGY="mesh"),
                    self.hops(left, right, TOPOLOGY="all_to_all"),
                )

    def test_the_adaptation_happens_once_per_tile(self):
        # The producing side refills the buffer once per iteration of the fused loops
        # above it, and every refill has to be moved again.
        once = self.hops([("X", DUP), ("Y", "N1")], [("X", DUP), ("Y", "N2")])
        thrice = self.hops(
            [("X", DUP), ("Y", "N1")],
            [("X", DUP), ("Y", "N2")],
            rows={TILES_COLUMN: [3]},
            fused_loops=(TILES_COLUMN,),
        )
        self.assertEqual(thrice, 3 * once)

    def test_rows_are_charged_by_their_own_iteration_counts(self):
        # How far the data has to move depends on how many instances take part, which
        # is a per-row number, so two rows of one group can cost different amounts.
        _, latency = self.cost(
            [("X", DUP), ("Y", "N1")],
            [("X", DUP), ("Y", "N2")],
            rows={BINDING_COLUMN: [1, 2]},
            n_iterations=BINDING_COLUMN,
        )
        # A binding loop of one iteration neither partitions nor replicates anything,
        # so there is nothing to move and no hop on the critical path.
        self.assertEqual(list(latency), [0.0, 1.0])

    def test_latency_is_the_networks_own_formula(self):
        # Latency-bound: the longest route is one switch traversal.
        _, latency = self.cost(
            [("X", DUP), ("Y", "N1")], [("X", DUP), ("Y", "N2")], HOP_LATENCY=3
        )
        self.assertEqual(latency.iloc[0], 3)

    def test_a_bandwidth_bound_network_charges_its_busiest_link(self):
        # Bandwidth-bound: no per-hop latency at all, so what is left is the traffic
        # over the most congested link divided by what the link can carry. An instance
        # holds half the tile and hands over half of that, so the busiest uplink
        # carries a quarter of the tile.
        busiest_link = SHARED_TENSOR_BITS / 4
        _, latency = self.cost(
            [("X", DUP), ("Y", "N1")],
            [("X", DUP), ("Y", "N2")],
            HOP_LATENCY=0,
            HOP_THROUGHPUT=busiest_link / 4,
        )
        self.assertAlmostEqual(latency.iloc[0], 4, places=4)

    def test_a_tensor_the_memory_may_not_adapt_is_not_adaptable(self):
        self.assertIsNone(
            self.cost(
                [("X", DUP), ("Y", "N1")],
                [("X", DUP), ("Y", "N2")],
                MAY_ADAPT="Nothing",
            )
        )

    def test_an_infinite_memory_cannot_be_adapted(self):
        # The mapper carries an occupancy as a fraction of the memory holding it, and
        # no fraction of an infinite memory says how many bits there are to move.
        self.assertIsNone(
            self.cost(
                [("X", DUP), ("Y", "N1")],
                [("X", DUP), ("Y", "N2")],
                SHARED_SIZE="float('inf')",
            )
        )

    def test_a_memory_that_is_not_the_distributed_one_is_not_adaptable(self):
        adaptations = _adapters().adaptations(
            _compatibility([("X", DUP)], resource="GlobalBuffer"),
            _compatibility([("X", "N1")], resource="GlobalBuffer"),
            {OCCUPANCY_COLUMN},
        )
        self.assertIsNone(adaptations)

    def test_nothing_is_adaptable_when_no_memory_allows_it(self):
        self.assertFalse(_adapters(MAY_ADAPT="Nothing"))

    def test_only_the_einsums_that_touch_a_tensor_get_a_say(self):
        """
        The constraints are evaluated per-Einsum, so a set expression like ``All``
        resolves to the tensors of the Einsum it was evaluated against. An Einsum that
        has never heard of the tensor being adapted is silent about it, not opposed --
        otherwise no tensor of a chain longer than two could ever be adapted.
        """
        einsums = ["Matmul0", "Matmul1", "Matmul2"]
        flattened, spec = [], None
        for einsum in einsums:
            evaluated = Spec.from_yaml(
                ARCH,
                af.examples.workloads.basic.matmuls_any_einsums,
                jinja_parse_data={
                    "N_EINSUMS": len(einsums),
                    "M": M,
                    "KN": KN,
                    "MAY_ADAPT": "All",
                },
            )._spec_eval_expressions(einsum_name=einsum)
            spec = spec or evaluated
            flattened += list(
                evaluated._get_flattened_architecture(einsum_name=einsum)
            )

        target = BindingAdapters(spec, flattened).targets[MEMORY]
        # T1 is shared by the first two Einsums and unknown to the third.
        self.assertTrue(target.allows("T1"))
        # A tensor no Einsum has heard of is nobody's to allow.
        self.assertFalse(target.allows("NoSuchTensor"))

    def test_a_memory_with_no_network_below_it_is_an_error(self):
        # Adapting means moving data, and there has to be something to charge for it.
        with self.assertRaises(ValueError):
            _adapters(NO_NETWORK=True)


class TestRebindingFactor(unittest.TestCase):
    """
    The adapter is applied to the left side of a join on its own, so the occupancy it
    reads is the left side's -- but the volume the network model is written in terms
    of is what one instance ends up holding, once the data has been moved. What turns
    one into the other is how many instances share the tile on either side, which the
    two bindings say by themselves.
    """

    def factor(self, left, right, n_iterations=N_INSTANCES) -> float:
        (adaptation,) = _adapters().adaptations(
            _compatibility(left, n_iterations=n_iterations),
            _compatibility(right, n_iterations=n_iterations),
        )
        return adaptation._rebinding_factor(tuple(adaptation._iterations))

    def occupancy(self, binding) -> float:
        """What one instance holds under `binding`, which is what the model's own
        occupancy column says: the tile, split as many ways as the binding
        partitions it."""
        partitioned = sum(1 for _, rank in binding if rank != DUP)
        return SHARED_TENSOR_BITS / N_INSTANCES**partitioned

    def check(self, left, right):
        self.assertAlmostEqual(
            self.occupancy(left) * self.factor(left, right),
            self.occupancy(right),
        )

    def test_partitioning_what_was_replicated_leaves_each_instance_less(self):
        self.check([("X", DUP), ("Y", DUP)], [("X", DUP), ("Y", "N1")])

    def test_replicating_what_was_partitioned_leaves_each_instance_more(self):
        self.check([("X", DUP), ("Y", "N1")], [("X", DUP), ("Y", DUP)])

    def test_moving_a_partition_between_dimensions_changes_nothing(self):
        self.check([("X", "N1"), ("Y", DUP)], [("X", DUP), ("Y", "N1")])

    def test_partitioning_along_another_rank_changes_nothing(self):
        self.check([("X", DUP), ("Y", "N1")], [("X", DUP), ("Y", "N2")])

    def test_both_dimensions_multiply(self):
        self.check([("X", DUP), ("Y", DUP)], [("X", "N1"), ("Y", "N2")])


class TestMakingAdapters(unittest.TestCase):
    """
    An adapter is made from the two sides' compatibilities alone, before any pmapping
    is looked at, and says what it leads out of and what it leads into.
    """

    def adapter(self, left, right):
        return _adapters().adapter(_compatibility(left), _compatibility(right))

    def test_an_adapter_leads_into_the_binding_it_was_given(self):
        adapter = self.adapter([("X", DUP), ("Y", "N1")], [("X", "M"), ("Y", "N2")])
        self.assertEqual(
            _binding_of(adapter.adapt(adapter.source)), [("X", "M"), ("Y", "N2")]
        )

    def test_adapting_a_compatibility_leaves_its_tile_patterns_alone(self):
        # The target is one of the joiner's bucket keys, whose tile patterns it has
        # already cleared, so nothing but the bindings may be taken from it.
        left = _compatibility(
            [("X", DUP), ("Y", "N1")], fused_loops=(TILES_COLUMN,)
        )
        adapter = _adapters().adapter(
            left,
            _compatibility(
                [("X", DUP), ("Y", "N2")]
            ).clear_tile_patterns_and_reservation_indices(),
        )
        self.assertEqual(adapter.adapt(left).symbols(), left.symbols())

    def test_two_sides_that_agree_call_for_no_adapter(self):
        binding = [("X", DUP), ("Y", "N1")]
        self.assertIsNone(self.adapter(binding, binding))

    def test_a_binding_that_may_not_be_adapted_calls_for_no_adapter(self):
        adapters = _adapters(MAY_ADAPT="Nothing")
        self.assertIsNone(
            adapters.adapter(
                _compatibility([("X", DUP), ("Y", "N1")]),
                _compatibility([("X", DUP), ("Y", "N2")]),
            )
        )

    def test_an_adapter_says_which_columns_it_reads(self):
        adapter = self.adapter([("X", DUP), ("Y", "N1")], [("X", DUP), ("Y", "N2")])
        self.assertIn(OCCUPANCY_COLUMN, adapter.required_columns())
        self.assertTrue(adapter.can_charge({OCCUPANCY_COLUMN}))
        self.assertFalse(adapter.can_charge(set()))

    def test_an_adapter_names_the_tensors_it_moves(self):
        adapter = self.adapter([("X", DUP), ("Y", "N1")], [("X", DUP), ("Y", "N2")])
        self.assertEqual(adapter.tensors, ["T1"])


class TestAdapterInTheJoiner(unittest.TestCase):
    """
    The joiner end to end, on a mapping whose two Einsums are spelled out as binding
    the tensor they share two different ways: Matmul0 leaves T1 partitioned along N1
    across Y, and Matmul1 wants a whole copy of it in every instance along Y.

    Every action of the fixture but a network hop is free, so the total energy is a hop
    count -- and the difference between it and the hops the two Einsums are separately
    charged for is exactly what the adapter cost.
    """

    evaluate = staticmethod(_evaluate)

    @staticmethod
    def einsum_hops(data) -> float:
        """The hops the two Einsums are charged for on their own, which the adapter's
        are deliberately not part of."""
        return sum(
            data[c].iloc[0] for c in data.columns if "action<SEP>NoC" in c
        )

    @staticmethod
    def einsum_latency(data) -> float:
        """The latency the two Einsums are charged for on their own: each one's
        longest component, summed, which is what the joiner adds up."""
        total = 0.0
        for einsum in ("Matmul0", "Matmul1"):
            prefix = f"{einsum}<SEP>latency<SEP>"
            total += max(
                data[c].iloc[0] for c in data.columns if c.startswith(prefix)
            )
        return total

    def adapter_energy(self, **jinja) -> float:
        data = self.evaluate(**jinja)
        hop_energy = jinja.get("HOP_ENERGY", 1)
        return data["Total<SEP>energy"].iloc[0] - self.einsum_hops(data) * hop_energy

    def adapter_latency(self, **jinja) -> float:
        data = self.evaluate(**jinja)
        return data["Total<SEP>latency"].iloc[0] - self.einsum_latency(data)

    def test_disagreeing_bindings_do_not_fuse_without_an_adapter(self):
        with self.assertRaises(ValueError):
            self.evaluate(MAY_ADAPT="Nothing")

    def test_disagreeing_bindings_fuse_with_an_adapter(self):
        self.assertTrue(len(self.evaluate()))

    def test_the_adapter_is_charged_on_top_of_what_the_einsums_move(self):
        # Matmul0 leaves T1 split two ways along Y and Matmul1 wants every instance
        # along Y to hold all of it, so a whole tile's worth crosses in each of the two
        # X columns -- less the half each instance already holds, which stays put.
        self.assertEqual(self.adapter_energy(), SHARED_TENSOR_BITS)

    def test_agreeing_bindings_are_charged_nothing_extra(self):
        self.assertEqual(self.adapter_energy(MATMUL1_Y="N1"), 0)

    def test_the_adapter_energy_scales_with_the_hop_energy(self):
        self.assertEqual(
            self.adapter_energy(HOP_ENERGY=3), 3 * SHARED_TENSOR_BITS
        )

    def test_the_adapter_lengthens_the_critical_path(self):
        # One switch traversal, on top of everything the two Einsums do themselves.
        self.assertEqual(self.adapter_latency(), 1)

    def test_agreeing_bindings_lengthen_nothing(self):
        self.assertEqual(self.adapter_latency(MATMUL1_Y="N1"), 0)


class TestAdapterInTheVisualization(unittest.TestCase):
    """
    A binding is per-Einsum, so the picture of a fused mapping has to be too. Where the
    two sides agree, the tensor they share is drawn once, above the split. Where an
    adapter stands between them they are drawn one per branch, each with the binding of
    the Einsum that asked for it -- neither one may speak for the other.
    """

    def mapping(self, **jinja):
        return _evaluate(**jinja)["Total<SEP>mapping"].iloc[0]()

    @staticmethod
    def bindings_by_einsum(mapping) -> dict[str, list[str]]:
        """The binding of every bound (memory, tensor) pair of the mapping, keyed by
        the Einsum whose branch it sits in -- or by None for the ones above the split,
        which belong to every Einsum at once."""
        result = {}

        def walk(node, einsum):
            for child in node.nodes:
                if isinstance(child, Binding):
                    result.setdefault(einsum, []).extend(
                        _binding_str(loops) for _, _, loops in child.entries()
                    )
                if isinstance(child, Nested):
                    computes = child.get_nodes_of_type(Compute)
                    walk(child, computes[0].einsum if computes else einsum)
                elif isinstance(child, MappingNodeWithChildren):
                    walk(child, einsum)

        walk(mapping, None)
        return result

    def test_each_einsum_shows_the_binding_it_asked_for(self):
        self.assertEqual(
            self.bindings_by_einsum(self.mapping()),
            {"Matmul0": ["X:M, Y:N1"], "Matmul1": ["X:M, Y:*"]},
        )

    def test_no_binding_speaks_for_both_einsums_when_they_disagree(self):
        self.assertNotIn(None, self.bindings_by_einsum(self.mapping()))

    def test_agreeing_bindings_are_drawn_once_above_the_split(self):
        self.assertEqual(
            self.bindings_by_einsum(self.mapping(MATMUL1_Y="N1")),
            {None: ["X:M, Y:N1"]},
        )

    def test_the_picture_renders(self):
        self.assertIn("<svg", str(self.mapping().render()))


if __name__ == "__main__":
    unittest.main()
