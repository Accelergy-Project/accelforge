"""
Tests for the order of the spatial loops that the mapper inserts.
"""

import unittest
from pathlib import Path

import accelforge as af
from accelforge.frontend.mapping import Spatial
from accelforge.frontend.renames import TensorName
from accelforge.frontend.spec import Spec
from accelforge.mapper import Metrics
from accelforge.mapper.FFM._make_pmappings.make_pmappings import get_jobs
from accelforge.mapper.FFM._make_pmappings.make_pmapping_templates.make_loops import (
    permitted_tensor_orders,
    spatial_loop_orders,
)
from accelforge.util._setexpressions import InvertibleSet

ARCH = Path(__file__).parent / "input_files" / "distributed_binding.arch.yaml"
# The same arch, but with the order of the spatial loops over the array's X dimension
# left open for a test to constrain through `X_TENSOR_ORDER_OPTIONS`.
CONSTRAINABLE_ARCH = (
    Path(__file__).parent / "input_files" / "spatial_tensor_order.arch.yaml"
)
BATCHED_MATMUL = Path(__file__).parent / "input_files" / "batched_matmul.workload.yaml"

# A matmul: no rank variable is relevant to all three tensors.
MATMUL_RELEVANCY = {
    "I": {"m", "nI"},
    "WA": {"nI", "nA"},
    "A": {"m", "nA"},
}


def _tensors(*names):
    """A set of tensors of the matmul, as an evaluated set expression would give it."""
    return InvertibleSet(
        instance=frozenset(names),
        full_space=frozenset(MATMUL_RELEVANCY),
        space_type=TensorName,
    )


def _jobs(workload, explore_loop_orders=True, arch=ARCH, **jinja_parse_data):
    spec = Spec.from_yaml(arch, workload, jinja_parse_data=jinja_parse_data)
    spec.mapper.explore_loop_orders = explore_loop_orders
    spec = spec._spec_eval_expressions(eval_arch=False, eval_non_arch=True)
    einsum2jobs = get_jobs(
        spec,
        Metrics.ENERGY,
        spec.workload.einsum_names,
        fail_if_no_pmappings_for_einsum=True,
        print_progress=False,
    )
    return [
        job
        for compatibility2jobs in einsum2jobs.values()
        for job_list in compatibility2jobs.values()
        for job in job_list
    ]


def _blocks_by_dimension(mapping):
    """One `((component, spatial dimension), loop order)` per block of spatial loops,
    a block being a run of back-to-back spatial loops over the same dimension."""
    blocks, block, block_key = [], [], None
    for node in mapping.nodes:
        key = (node.component, node.name) if isinstance(node, Spatial) else None
        if key is None or key != block_key:
            if block:
                blocks.append((block_key, tuple(block)))
            block, block_key = [], key
        if key is not None:
            block.append(node.rank_variable)
    if block:
        blocks.append((block_key, tuple(block)))
    return blocks


def _spatial_blocks(mapping):
    """The rank variables of each block of spatial loops."""
    return [list(block) for _, block in _blocks_by_dimension(mapping)]


class TestSpatialLoopOrders(unittest.TestCase):
    """Tests of the loop orders generated for one block of spatial loops."""

    def test_one_order_per_tensor_that_the_outermost_loops_tile(self):
        # One order per tensor that the outermost loops may tile: the loops that tile
        # it go first, then the ones that tile the next tensor and reuse this one.
        orders = list(
            spatial_loop_orders({"m", "nI", "nA"}, MATMUL_RELEVANCY, [], True)
        )
        self.assertCountEqual(
            orders,
            [
                ("m", "nI", "nA"),  # I first: `nA` reuses `I` and tiles `WA` or `A`
                ("m", "nA", "nI"),  # A first: `nI` reuses `A` and tiles `I` or `WA`
                ("nA", "nI", "m"),  # WA first: `m` reuses `WA` and tiles `I` or `A`
            ],
        )

    def test_rank_variable_relevant_to_all_tensors_is_always_highest(self):
        relevancy = {t: rvs | {"b"} for t, rvs in MATMUL_RELEVANCY.items()}
        orders = list(
            spatial_loop_orders({"b", "m", "nI", "nA"}, relevancy, [], True)
        )
        self.assertTrue(orders)
        for order in orders:
            self.assertEqual(order[0], "b", f"`b` is not highest in {order}.")

    def test_rank_variable_irrelevant_to_all_tensors_is_always_lowest(self):
        orders = list(
            spatial_loop_orders({"m", "nI", "nA", "unused"}, MATMUL_RELEVANCY, [], True)
        )
        self.assertTrue(orders)
        for order in orders:
            self.assertEqual(order[-1], "unused", f"`unused` is not lowest in {order}.")

    def test_only_one_order_if_not_exploring(self):
        orders = list(
            spatial_loop_orders({"m", "nI", "nA"}, MATMUL_RELEVANCY, [], False)
        )
        self.assertEqual(len(orders), 1)
        self.assertIn(
            orders[0],
            list(spatial_loop_orders({"m", "nI", "nA"}, MATMUL_RELEVANCY, [], True)),
        )


class TestSpatialLoopOrderExploration(unittest.TestCase):
    """Tests of the spatial loop orders of the generated pmapping templates."""

    def test_rank_variable_relevant_to_all_tensors_is_highest(self):
        # `b` is the only rank variable of the batched matmul that is relevant to
        # every tensor, so it must be the first loop of every block it appears in.
        checked = 0
        for job in _jobs(BATCHED_MATMUL):
            for block in _spatial_blocks(job.mapping):
                if "b" not in block:
                    continue
                self.assertEqual(
                    block[0],
                    "b",
                    f"`b` is relevant to all tensors, so it should be the highest "
                    f"spatial loop of the block, but the block is {block}.",
                )
                checked += 1
        self.assertGreater(checked, 0, "No spatial loop of `b` was generated.")

    def test_orders_are_explored_only_if_exploring(self):
        def orders_of(explore_loop_orders):
            return {
                tuple(block)
                for job in _jobs(
                    af.examples.workloads.basic.matmuls,
                    explore_loop_orders=explore_loop_orders,
                    N_EINSUMS=2,
                    M=8,
                    KN=8,
                )
                for block in _spatial_blocks(job.mapping)
            }

        explored = orders_of(True)
        self.assertGreater(len(explored), 1)
        # Each Einsum of the chain has three tensors, so each has three orders.
        self.assertEqual(len(explored), 6)

        not_explored = orders_of(False)
        self.assertEqual(len(not_explored), 2)
        self.assertTrue(not_explored <= explored)


class TestTensorOrderOptions(unittest.TestCase):
    """Tests of the `tensor_order_options` of a spatial dimension."""

    def _orders(self, tensor_order_options, explore_loop_orders=True):
        return list(
            spatial_loop_orders(
                {"m", "nI", "nA"},
                MATMUL_RELEVANCY,
                tensor_order_options,
                explore_loop_orders,
            )
        )

    def test_no_options_permits_every_tensor_order(self):
        self.assertCountEqual(
            permitted_tensor_orders(["A", "I", "WA"], []),
            [
                ("A", "I", "WA"),
                ("A", "WA", "I"),
                ("I", "A", "WA"),
                ("I", "WA", "A"),
                ("WA", "A", "I"),
                ("WA", "I", "A"),
            ],
        )

    def test_an_option_permits_the_orders_that_agree_with_it(self):
        self.assertCountEqual(
            permitted_tensor_orders(
                ["A", "I", "WA"], [[_tensors("I"), _tensors("WA", "A")]]
            ),
            [("I", "A", "WA"), ("I", "WA", "A")],
        )

    def test_the_orders_of_every_option_are_permitted(self):
        both = permitted_tensor_orders(
            ["A", "I", "WA"],
            [
                [_tensors("I"), _tensors("WA", "A")],
                [_tensors("A"), _tensors("WA", "I")],
            ],
        )
        self.assertCountEqual(
            both, [("I", "A", "WA"), ("I", "WA", "A"), ("A", "I", "WA"), ("A", "WA", "I")]
        )

    def test_tensors_an_option_does_not_name_are_unconstrained(self):
        # `A` is named by neither entry, so it may go anywhere; only `I` above `WA` is
        # required.
        self.assertCountEqual(
            permitted_tensor_orders(
                ["A", "I", "WA"], [[_tensors("I"), _tensors("WA")]]
            ),
            [("A", "I", "WA"), ("I", "A", "WA"), ("I", "WA", "A")],
        )

    def test_an_option_that_names_no_tensor_of_this_einsum_constrains_nothing(self):
        # A tensor of another Einsum does not evaluate against this Einsum's symbol
        # table, so it is left as a string. An option of nothing but those permits
        # every order, and so does the whole constraint.
        self.assertCountEqual(
            permitted_tensor_orders(
                ["A", "I", "WA"], [["SomeOtherEinsumsTensor", _tensors("I")]]
            ),
            permitted_tensor_orders(["A", "I", "WA"], []),
        )

    def test_options_narrow_the_loop_orders(self):
        # Of the three orders of an unconstrained matmul, `I` first gives the one whose
        # outermost loops tile `I`, and `A` first the one whose loops tile `A`.
        self.assertEqual(
            self._orders([[_tensors("I"), _tensors("WA", "A")]]), [("m", "nI", "nA")]
        )
        self.assertEqual(
            self._orders([[_tensors("A"), _tensors("WA", "I")]]), [("m", "nA", "nI")]
        )
        self.assertCountEqual(
            self._orders(
                [
                    [_tensors("I"), _tensors("WA", "A")],
                    [_tensors("A"), _tensors("WA", "I")],
                ]
            ),
            [("m", "nI", "nA"), ("m", "nA", "nI")],
        )

    def test_options_are_honored_even_if_not_exploring_loop_orders(self):
        # Not exploring takes the first order, which the constraint still chooses.
        self.assertEqual(
            self._orders(
                [[_tensors("A"), _tensors("WA", "I")]], explore_loop_orders=False
            ),
            [("m", "nA", "nI")],
        )


class TestTensorOrderOptionsInArch(unittest.TestCase):
    """Tests of `tensor_order_options` given in the architecture spec."""

    # The three orders of the batched matmul's spatial loops. `b` is relevant to every
    # tensor, so it is always highest.
    I_FIRST = ("b", "m", "nI", "nA")
    A_FIRST = ("b", "m", "nA", "nI")
    WA_FIRST = ("b", "nA", "nI", "m")

    def _orders_by_dimension(self, **jinja_parse_data):
        """The set of loop orders generated for each (component, spatial dimension)."""
        orders = {}
        for job in _jobs(
            BATCHED_MATMUL, arch=CONSTRAINABLE_ARCH, **jinja_parse_data
        ):
            for key, block in _blocks_by_dimension(job.mapping):
                orders.setdefault(key, set()).add(block)
        return orders

    def test_unconstrained_dimensions_explore_every_order(self):
        orders = self._orders_by_dimension()
        self.assertEqual(
            orders[("Array", "X")], {self.I_FIRST, self.A_FIRST, self.WA_FIRST}
        )
        self.assertEqual(orders[("Array", "X")], orders[("Array", "Y")])

    def test_an_option_narrows_the_dimension_it_is_given_for(self):
        orders = self._orders_by_dimension(X_TENSOR_ORDER_OPTIONS="[[I, WA | A]]")
        self.assertEqual(orders[("Array", "X")], {self.I_FIRST})
        # Y has no constraint of its own, so it still explores every order.
        self.assertEqual(
            orders[("Array", "Y")], {self.I_FIRST, self.A_FIRST, self.WA_FIRST}
        )

    def test_each_option_names_the_order_whose_loops_tile_it_first(self):
        self.assertEqual(
            self._orders_by_dimension(X_TENSOR_ORDER_OPTIONS="[[A, WA | I]]")[
                ("Array", "X")
            ],
            {self.A_FIRST},
        )
        self.assertEqual(
            self._orders_by_dimension(X_TENSOR_ORDER_OPTIONS="[[WA, I | A]]")[
                ("Array", "X")
            ],
            {self.WA_FIRST},
        )

    def test_several_options_are_all_explored(self):
        orders = self._orders_by_dimension(
            X_TENSOR_ORDER_OPTIONS="[[I, WA | A], [A, WA | I]]"
        )
        self.assertEqual(orders[("Array", "X")], {self.I_FIRST, self.A_FIRST})

    def test_intersecting_entries_in_one_option_are_rejected(self):
        # `A` is in both entries, so the option asks for the loops that tile it to be
        # in two places at once.
        with self.assertRaises(ValueError) as raised:
            _jobs(
                BATCHED_MATMUL,
                arch=CONSTRAINABLE_ARCH,
                X_TENSOR_ORDER_OPTIONS="[[I | A, A | WA]]",
            )
        self.assertIn("Intersecting entries", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
