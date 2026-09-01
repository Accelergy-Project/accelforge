"""
Tests for constraining which bindings of a physically-distributed memory the mapper
explores, so that the impact of a binding policy can be measured against another.
"""

import unittest
from pathlib import Path

import accelforge as af
from accelforge.frontend.arch import BindingConstraints
from accelforge.frontend.mapping import Binding, binding_summary
from accelforge.frontend.renames import RANK_DUPLICATE
from accelforge.frontend.spec import Spec
from accelforge.mapper import Metrics
from accelforge.mapper.FFM._make_pmappings.make_pmappings import get_jobs

ARCH = Path(__file__).parent / "input_files" / "distributed_binding.arch.yaml"

# The only physically-distributed memory of the fixture. Every tensor of the fixture has
# exactly two ranks, and the memory has two spatial dimensions (X and Y) with several
# spatial loops over each.
DISTRIBUTED_MEMORY = "SharedBuffer"


def _make_spec(binding=None, n_einsums: int = 2, **mapper_settings) -> Spec:
    """
    The fixture spec. `binding` is either one BindingConstraints, which goes on the
    fixture's only physically-distributed memory, or a dict of memory name to the
    constraints that memory carries. `n_einsums` is how long the chain of matmuls is,
    which is what decides how many tensors are shared between Einsums.
    """
    spec = Spec.from_yaml(
        ARCH,
        af.examples.workloads.basic.matmuls,
        jinja_parse_data={"N_EINSUMS": n_einsums, "M": 8, "KN": 8},
    )
    for k, v in mapper_settings.items():
        setattr(spec.mapper, k, v)
    if isinstance(binding, BindingConstraints):
        binding = {DISTRIBUTED_MEMORY: binding}
    for memory_name, constraints in (binding or {}).items():
        spec.arch[memory_name].binding = constraints
    return spec


def _get_jobs(n_einsums: int = 2, **mapper_settings) -> dict:
    spec = _make_spec(n_einsums=n_einsums, **mapper_settings)
    spec = spec._spec_eval_expressions(eval_arch=False, eval_non_arch=True)
    return get_jobs(
        spec,
        Metrics.ENERGY,
        spec.workload.einsum_names,
        fail_if_no_pmappings_for_einsum=True,
        print_progress=False,
    )


def _iter_jobs(jobs: dict):
    for compatibility2jobs in jobs.values():
        for job_list in compatibility2jobs.values():
            yield from job_list


def _explored_bindings(jobs: dict) -> set:
    """Every distinct binding of every template, as a hashable value."""
    return {
        tuple(
            (key, tuple(sorted(per_dimension.items())))
            for key, per_dimension in sorted(binding_summary(job.mapping.nodes).items())
        )
        for job in _iter_jobs(jobs)
    }


def _bound_ranks(jobs: dict, tensor: str | None = None) -> set:
    """Every rank named by any binding loop of any template, optionally of one tensor."""
    return {
        rank
        for job in _iter_jobs(jobs)
        for (_, bound_tensor), per_dimension in binding_summary(
            job.mapping.nodes
        ).items()
        if tensor is None or bound_tensor == tensor
        for rank in per_dimension.values()
    }


def _symbol_table(einsum_name: str = "MatmulA") -> dict:
    """The Einsum symbols that a tensor set expression is written against."""
    spec = _make_spec()
    spec = spec._spec_eval_expressions(eval_arch=False, eval_non_arch=True)
    einsum = spec.workload.einsums[einsum_name]
    return {r.name: r.source for r in einsum.renames}


def _evaluate(constraints: BindingConstraints) -> BindingConstraints:
    """One BindingConstraints evaluated against an Einsum, as the arch evaluates it."""
    return constraints._eval_expressions(_symbol_table())[0]


class TestBindingConstraintsModel(unittest.TestCase):
    """The constraint object on its own, without running the mapper."""

    def test_tensor_set_expressions(self):
        # "All" and "Nothing" resolve against the Einsum's tensors, so a constraint
        # written against them applies to whatever the workload happens to hold.
        spec = _make_spec()
        spec = spec._spec_eval_expressions(eval_arch=False, eval_non_arch=True)
        tensors = set(spec.workload.einsums["MatmulA"].tensor_names)

        self.assertEqual(
            set(_evaluate(BindingConstraints(may_duplicate="All")).may_duplicate),
            tensors,
        )
        self.assertEqual(
            set(_evaluate(BindingConstraints(may_duplicate="Nothing")).may_duplicate),
            set(),
        )

        # A union of tensor names, as in the rest of the constraints specification.
        two = sorted(tensors)[:2]
        evaluated = _evaluate(
            BindingConstraints(may_duplicate=" | ".join(two))
        )
        self.assertEqual(set(evaluated.may_duplicate), set(two))

        # Inversion and the named subsets work too.
        self.assertEqual(
            set(_evaluate(BindingConstraints(may_duplicate="~Inputs")).may_duplicate),
            tensors - set(spec.workload.einsums["MatmulA"].input_tensor_names),
        )

    def test_an_unevaluated_constraint_cannot_be_queried(self):
        # The arch holds the expression until an Einsum is in hand. Querying it before
        # then would test membership against the expression string itself, which would
        # silently answer for the wrong set of tensors.
        with self.assertRaises(ValueError):
            BindingConstraints().allowed_ranks("I", "X", ["M", "N"])

    def test_a_bad_expression_is_an_error(self):
        with self.assertRaises(Exception):
            _evaluate(BindingConstraints(may_duplicate="NotATensor"))

    def test_contradictory_duplication_is_rejected(self):
        # A tensor that must duplicate but may not would make every binding of it
        # invalid, dropping every template rather than reporting a bad configuration.
        with self.assertRaises(Exception):
            _evaluate(
                BindingConstraints(may_duplicate="Nothing", must_duplicate="All")
            )
        with self.assertRaises(Exception):
            _evaluate(
                BindingConstraints(may_duplicate="Inputs", must_duplicate="All")
            )
        # Consistent selections are fine.
        _evaluate(BindingConstraints(may_duplicate="All", must_duplicate="Inputs"))

    def test_allowed_ranks(self):
        ranks = ["M", "N"]

        unconstrained = _evaluate(BindingConstraints())
        self.assertEqual(
            unconstrained.allowed_ranks("I", "X", ranks), ["M", "N", RANK_DUPLICATE]
        )

        never = _evaluate(BindingConstraints(may_duplicate="Nothing"))
        self.assertEqual(never.allowed_ranks("I", "X", ranks), ["M", "N"])

        always = _evaluate(BindingConstraints(must_duplicate="All"))
        self.assertEqual(always.allowed_ranks("I", "X", ranks), [RANK_DUPLICATE])

        # A per-tensor selection applies only to the tensors it names.
        per_tensor = _evaluate(BindingConstraints(may_duplicate="I"))
        self.assertIn(RANK_DUPLICATE, per_tensor.allowed_ranks("I", "X", ranks))
        self.assertNotIn(RANK_DUPLICATE, per_tensor.allowed_ranks("WA", "X", ranks))

        # dimension_ranks constrains the dimensions it names and no others.
        per_dimension = _evaluate(BindingConstraints(dimension_ranks={"X": {"M"}}))
        self.assertEqual(per_dimension.allowed_ranks("I", "X", ranks), ["M"])
        self.assertEqual(
            per_dimension.allowed_ranks("I", "Y", ranks), ["M", "N", RANK_DUPLICATE]
        )

    def test_violations_describes_every_problem(self):
        constraints = _evaluate(BindingConstraints(may_duplicate="Nothing"))
        problems = constraints.violations(
            "I", ["X", "Y"], [RANK_DUPLICATE, RANK_DUPLICATE], ["M", "N"]
        )
        self.assertEqual(len(problems), 2)
        self.assertEqual(
            constraints.violations("I", ["X", "Y"], ["M", "N"], ["M", "N"]), []
        )


class TestBindingConstraints(unittest.TestCase):
    def test_never_duplicate(self):
        jobs = _get_jobs(binding=BindingConstraints(may_duplicate="Nothing"))
        self.assertTrue(list(_iter_jobs(jobs)))
        bound = _bound_ranks(jobs)
        self.assertTrue(bound, "The fixture has a distributed memory, so it has bindings.")
        self.assertNotIn(RANK_DUPLICATE, bound)

    def test_never_duplicate_removes_bindings_that_were_explored(self):
        # The constraint has to actually shrink the mapspace, not just be satisfiable.
        unconstrained = _explored_bindings(
            _get_jobs(binding=BindingConstraints(max_binding_rank_changes=1))
        )
        constrained_jobs = _get_jobs(
            binding=BindingConstraints(
                max_binding_rank_changes=1, may_duplicate="Nothing"
            )
        )
        self.assertTrue(unconstrained - _explored_bindings(constrained_jobs))
        self.assertNotIn(RANK_DUPLICATE, _bound_ranks(constrained_jobs))

    def test_must_duplicate(self):
        constraints = BindingConstraints(must_duplicate="All")
        jobs = _get_jobs(
            binding=BindingConstraints(must_duplicate="All", max_binding_rank_changes=1)
        )
        self.assertTrue(list(_iter_jobs(jobs)))
        self.assertEqual(_bound_ranks(jobs), {RANK_DUPLICATE})

        # Every binding loop is forced, so there is nothing left for
        # max_binding_rank_changes to explore and no extra templates to generate.
        without = _get_jobs(binding=constraints)
        self.assertEqual(len(list(_iter_jobs(jobs))), len(list(_iter_jobs(without))))

    def test_per_tensor_duplication(self):
        # Only the tensors named by the set expression may duplicate.
        jobs = _get_jobs(binding=BindingConstraints(may_duplicate="WA | WB"))
        self.assertNotIn(RANK_DUPLICATE, _bound_ranks(jobs, tensor="A"))
        self.assertIn(RANK_DUPLICATE, _bound_ranks(jobs, tensor="WA"))

    def test_dimension_ranks(self):
        # Y may only duplicate; X is left alone.
        jobs = _get_jobs(
            binding=BindingConstraints(dimension_ranks={"Y": {RANK_DUPLICATE}})
        )
        self.assertTrue(list(_iter_jobs(jobs)))
        for job in _iter_jobs(jobs):
            for per_dimension in binding_summary(job.mapping.nodes).values():
                for dimension, rank in per_dimension.items():
                    if dimension.startswith("Y"):
                        self.assertEqual(rank, RANK_DUPLICATE)

    def test_constraints_apply_only_to_the_memory_that_carries_them(self):
        # GlobalBuffer is not physically distributed, so it has no binding to constrain
        # and the same constraint there changes nothing.
        elsewhere = _get_jobs(
            binding={"GlobalBuffer": BindingConstraints(may_duplicate="Nothing")}
        )
        self.assertEqual(_explored_bindings(elsewhere), _explored_bindings(_get_jobs()))
        self.assertIn(RANK_DUPLICATE, _bound_ranks(elsewhere))

        # On the distributed memory, the same constraint takes effect.
        on_distributed = _get_jobs(
            binding={DISTRIBUTED_MEMORY: BindingConstraints(may_duplicate="Nothing")}
        )
        self.assertNotIn(RANK_DUPLICATE, _bound_ranks(on_distributed))

    def test_max_binding_rank_changes(self):
        one = _get_jobs(binding=BindingConstraints(max_binding_rank_changes=1))
        two = _get_jobs(binding=BindingConstraints(max_binding_rank_changes=2))
        self.assertGreater(
            len(list(_iter_jobs(two))), len(list(_iter_jobs(one)))
        )

    def test_max_binding_rank_changes_comes_from_the_bound_memory(self):
        # The budget of a memory that has no binding of its own buys nothing: only the
        # memories that are actually bound get to widen the exploration.
        pinned = _get_jobs(
            binding={"MainMemory": BindingConstraints(max_binding_rank_changes=1)}
        )
        unpinned = _get_jobs(binding=BindingConstraints(max_binding_rank_changes=1))
        self.assertEqual(
            _explored_bindings(pinned), _explored_bindings(_get_jobs())
        )
        self.assertLess(len(list(_iter_jobs(pinned))), len(list(_iter_jobs(unpinned))))

    def test_unknown_memory_is_an_error(self):
        # Constraints are addressed through the arch, so a name that matches no
        # component cannot quietly have no effect.
        with self.assertRaises(KeyError) as ctx:
            _make_spec(binding={"NotAMemory": BindingConstraints()})
        self.assertIn("NotAMemory", str(ctx.exception))

    def test_infeasible_constraints_drop_templates(self):
        # Nothing may duplicate and dimension X may bind nothing but DUPLICATE, so no
        # binding loop over X has any rank left to name.
        with self.assertRaises(Exception):
            _get_jobs(
                binding=BindingConstraints(
                    may_duplicate="Nothing",
                    dimension_ranks={"X": {RANK_DUPLICATE}},
                )
            )

    def test_constrained_default_is_used_without_exploration(self):
        # max_binding_rank_changes counts changes against the default binding. If the
        # default were still the unconstrained locally-optimal one, a constraint that
        # forbids it would leave nothing to explore at the default setting of 0.
        jobs = _get_jobs(binding=BindingConstraints(may_duplicate="Nothing"))
        self.assertTrue(list(_iter_jobs(jobs)))
        for job in _iter_jobs(jobs):
            for (_, tensor), per_dimension in binding_summary(job.mapping.nodes).items():
                allowed = set(job.einsum.tensor_accesses[tensor].ranks)
                self.assertTrue(set(per_dimension.values()) <= allowed)


class TestMayExploreBinding(unittest.TestCase):
    """
    Which tensors `max_binding_rank_changes` may be spent on. Everything else is held at
    the binding it would have had with no exploration at all, so one tensor's binding can
    be studied without the rest of the mapspace moving underneath it.
    """

    @staticmethod
    def _bindings_of_tensor(jobs: dict, tensor: str) -> set:
        """Every binding any template gives `tensor`, as (memory, dimension -> rank)."""
        return {
            (memory, tuple(sorted(per_dimension.items())))
            for job in _iter_jobs(jobs)
            for (memory, bound_tensor), per_dimension in binding_summary(
                job.mapping.nodes
            ).items()
            if bound_tensor == tensor
        }

    def test_exploring_nothing_is_the_same_as_having_no_budget(self):
        # The budget is there, but no tensor may spend it.
        held = _get_jobs(
            binding=BindingConstraints(
                max_binding_rank_changes=1, may_explore_binding="Nothing"
            )
        )
        self.assertEqual(_explored_bindings(held), _explored_bindings(_get_jobs()))

    def test_the_named_tensor_is_explored(self):
        # A and B are both shared in a three-Einsum chain, so naming one of them is a
        # real choice rather than the only option.
        named = _get_jobs(
            n_einsums=3,
            binding=BindingConstraints(
                max_binding_rank_changes=1, may_explore_binding="A"
            ),
        )
        unexplored = _get_jobs(n_einsums=3)
        self.assertTrue(
            self._bindings_of_tensor(named, "A")
            - self._bindings_of_tensor(unexplored, "A"),
            "The tensor named by may_explore_binding was not explored.",
        )

    def test_a_tensor_left_out_keeps_its_locally_optimal_binding(self):
        named = _get_jobs(
            n_einsums=3,
            binding=BindingConstraints(
                max_binding_rank_changes=1, may_explore_binding="A"
            ),
        )
        unexplored = _get_jobs(n_einsums=3)
        # B is shared, so it would have been explored had it been named.
        self.assertEqual(
            self._bindings_of_tensor(named, "B"),
            self._bindings_of_tensor(unexplored, "B"),
            "A tensor that may not be explored was explored anyway.",
        )

    def test_naming_a_tensor_does_not_explore_it_if_nobody_shares_it(self):
        # may_explore_binding bounds exploration; it does not widen it. WA is named but
        # no other Einsum touches it, so it stays at its locally-optimal binding.
        named = _get_jobs(
            binding=BindingConstraints(
                max_binding_rank_changes=1, may_explore_binding="WA"
            )
        )
        self.assertEqual(
            self._bindings_of_tensor(named, "WA"),
            self._bindings_of_tensor(_get_jobs(), "WA"),
        )

    def test_may_explore_reads_the_evaluated_set(self):
        constraints = _evaluate(BindingConstraints(may_explore_binding="A"))
        self.assertTrue(constraints.may_explore("A"))
        self.assertFalse(constraints.may_explore("I"))

    def test_may_explore_cannot_be_queried_before_evaluation(self):
        # As with the other tensor sets: the expression string is not a set of tensors,
        # and testing membership against it would quietly answer for the wrong ones.
        with self.assertRaises(ValueError) as ctx:
            BindingConstraints().may_explore("I")
        self.assertIn("may_explore_binding", str(ctx.exception))


class TestShardUnsharedTensorsModel(unittest.TestCase):
    """
    The two fields that say what a memory does with a tensor no other Einsum touches:
    replicate it along the loops that do not index it, or partition it along a rank.
    """

    def test_neither_is_set_by_default(self):
        constraints = _evaluate(BindingConstraints())
        self.assertEqual(set(constraints.must_shard_unshared_tensors), set())
        self.assertEqual(set(constraints.may_shard_unshared_tensors), set())

    def test_may_defaults_to_must(self):
        # A tensor named only in must_ is sharded outright rather than also measured
        # against replicating it, which is what naming it in may_ would ask for.
        constraints = _evaluate(
            BindingConstraints(must_shard_unshared_tensors="WA")
        )
        self.assertEqual(set(constraints.may_shard_unshared_tensors), {"WA"})

    def test_may_can_be_widened_past_must(self):
        constraints = _evaluate(
            BindingConstraints(may_shard_unshared_tensors="All")
        )
        self.assertEqual(set(constraints.must_shard_unshared_tensors), set())
        self.assertTrue(set(constraints.may_shard_unshared_tensors))

    def test_must_outside_may_is_a_contradiction(self):
        # As with must_duplicate: the mapper would otherwise read it as a tensor with no
        # valid binding at all rather than as the spec saying two opposite things.
        with self.assertRaises(ValueError) as ctx:
            _evaluate(
                BindingConstraints(
                    must_shard_unshared_tensors="All",
                    may_shard_unshared_tensors="Nothing",
                )
            )
        self.assertIn("must_shard_unshared_tensors", str(ctx.exception))

    def test_the_sets_read_as_evaluated(self):
        constraints = _evaluate(BindingConstraints(may_shard_unshared_tensors="WA"))
        self.assertTrue(constraints.may_shard_unshared("WA"))
        self.assertFalse(constraints.may_shard_unshared("I"))
        self.assertFalse(constraints.must_shard_unshared("WA"))

    def test_they_cannot_be_queried_before_evaluation(self):
        for query in ("may_shard_unshared", "must_shard_unshared"):
            with self.assertRaises(ValueError) as ctx:
                getattr(BindingConstraints(), query)("I")
            self.assertIn("shard_unshared_tensors", str(ctx.exception))


class TestBindingConstraintsEndToEnd(unittest.TestCase):
    """The mapper and the model, which is where an experiment actually reads results."""

    def test_mapper_returns_a_mapping_that_satisfies_the_constraints(self):
        spec = _make_spec(
            binding=BindingConstraints(
                may_duplicate="Nothing", max_binding_rank_changes=1
            )
        )
        result = spec.map_workload_to_arch(print_progress=False)
        self.assertTrue(len(result))
        mapping = result.mapping(0)
        summary = binding_summary(mapping.nodes)
        self.assertTrue(summary, "The mapper returned a mapping with no binding.")
        for per_dimension in summary.values():
            self.assertNotIn(RANK_DUPLICATE, per_dimension.values())

        # The mapping still models, with the binding the mapper chose for it.
        spec.mapping = mapping
        spec.evaluate_mapping()
        self.assertEqual(binding_summary(mapping.nodes), summary)

    def test_evaluating_a_violating_mapping_is_an_error(self):
        spec = _make_spec()
        result = spec.map_workload_to_arch(print_progress=False)
        mapping = result.mapping(0)
        bound = [
            node
            for node in mapping.nodes
            if isinstance(node, Binding) and node.bindings
        ]
        self.assertTrue(bound, "The mapper returned a mapping with no binding.")
        bound[0].entries()[0][2][0].rank_variable = RANK_DUPLICATE

        spec.arch[DISTRIBUTED_MEMORY].binding = BindingConstraints(
            may_duplicate="Nothing"
        )
        spec.mapping = mapping
        with self.assertRaises(ValueError) as ctx:
            spec.evaluate_mapping()
        self.assertIn("binding constraints", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
