"""
Tests for exploring the binding of a tensor to the physical instances of a
physically-distributed memory.
"""

import importlib
import unittest
from contextlib import contextmanager
from pathlib import Path

import accelforge as af
from accelforge.frontend.arch import BindingConstraints
from accelforge.frontend.mapping import (
    Binding,
    Loop,
    Spatial,
    Storage,
    TilePattern,
    iter_binding_blocks,
)
from accelforge.frontend.renames import RANK_DUPLICATE
from accelforge.frontend.spec import Spec
from accelforge.mapper import Metrics
from accelforge.mapper.FFM._make_pmappings.make_pmapping_templates.make_pmapping_templates import (
    _binding_rank_classes,
    default_binding_ranks,
    iter_binding_choices,
)
from accelforge.mapper.FFM._make_pmappings.make_pmappings import (
    get_jobs,
    make_pmappings,
)
from accelforge.mapper.FFM._make_pmappings.make_pmappings_from_templates.make_pmappings_from_templates import (
    _binding_expansions,
)
from accelforge.mapper.FFM._join_pmappings.compatibility import (
    Compatibility,
    Loop as CompatibilityLoop,
    TensorReservation,
)
from accelforge.frontend.renames import Rank
from accelforge.util._frozenset import fzs

ARCH = Path(__file__).parent / "input_files" / "distributed_binding.arch.yaml"

# The module, not the same-named function the package re-exports: the tests below
# swap out one of its functions to compare against the unmerged behavior.
templates_module = importlib.import_module(
    "accelforge.mapper.FFM._make_pmappings.make_pmapping_templates."
    "make_pmapping_templates"
)


# The only physically-distributed memory of the fixture, and so the only one whose
# binding constraints the mapper consults.
DISTRIBUTED_MEMORY = "SharedBuffer"


def _compatibility_with_binding(
    binding: list[tuple[str, str]], tensor: str = "T1", resource: str = "SharedBuffer"
) -> Compatibility:
    """A Compatibility whose only reservation binds `binding`, as (dimension, rank)."""
    return Compatibility(
        tensors=fzs(
            [
                TensorReservation(
                    loops=(),
                    name=tensor,
                    resource_name=resource,
                    physical_spatial_loops=tuple(
                        CompatibilityLoop(
                            rank_name=Rank(rank),
                            tile_pattern=TilePattern(calculated_n_iterations=2),
                            is_spatial=True,
                            spatial_dim=dimension,
                        )
                        for dimension, rank in binding
                    ),
                )
            ]
        ),
        # The reservation sits above every loop, so index zero is where it lives.
        reservation_indices=fzs([0]),
    )


def _binding_of(compatibility: Compatibility) -> list[tuple[str, str]]:
    """The (dimension, rank) of every binding loop of `compatibility`."""
    return [
        (loop.spatial_dim, loop.rank_name)
        for tensor in compatibility.tensors
        for loop in tensor.physical_spatial_loops
    ]


def _make_spec(binding: BindingConstraints | None = None, **mapper_settings) -> Spec:
    spec = Spec.from_yaml(
        ARCH,
        af.examples.workloads.basic.matmuls,
        jinja_parse_data={"N_EINSUMS": 2, "M": 8, "KN": 8},
    )
    for k, v in mapper_settings.items():
        setattr(spec.mapper, k, v)
    # Binding constraints live on the memory they constrain.
    if binding is not None:
        spec.arch[DISTRIBUTED_MEMORY].binding = binding
    return spec


def _get_jobs(**mapper_settings) -> dict:
    spec = _make_spec(**mapper_settings)
    spec = spec._spec_eval_expressions(eval_arch=False, eval_non_arch=True)
    return get_jobs(
        spec,
        Metrics.ENERGY,
        spec.workload.einsum_names,
        fail_if_no_pmappings_for_einsum=True,
        print_progress=False,
    )


def _first_job_with_binding():
    """The first job whose mapping binds a physically-distributed memory."""
    for compatibility2jobs in _get_jobs().values():
        for job_list in compatibility2jobs.values():
            for job in job_list:
                if any(
                    isinstance(node, Binding) and node.bindings
                    for node in job.mapping.nodes
                ):
                    return job
    raise AssertionError("No job of the fixture has a binding.")


def _bindings_of(job) -> tuple:
    """The (component, tensor, bound ranks) of every bound pair of a job."""
    return tuple(
        (component, tensor, tuple(l.rank_variable for l in loops))
        for node in job.mapping.nodes
        if isinstance(node, Binding)
        for component, tensor, loops in node.entries()
    )


class TestBindingExploration(unittest.TestCase):
    def test_default_binding_is_the_only_one_if_not_exploring(self):
        jobs = _get_jobs()
        for einsum_name, compatibility2jobs in jobs.items():
            for job_list in compatibility2jobs.values():
                for job in job_list:
                    self.assertTrue(
                        _bindings_of(job),
                        f"{einsum_name} has a physically-distributed memory, so its "
                        f"templates should have a binding.",
                    )
                    for block in iter_binding_blocks(job.mapping.nodes):
                        if not block.binding:
                            continue
                        irrelevant = job.einsum.tensor2irrelevant_rank_variables[
                            block.tensor
                        ]
                        rank_variable2ranks = job.einsum.tensor_accesses[
                            block.tensor
                        ].rank_variable2ranks
                        for spatial, bound in zip(block.spatials, block.binding):
                            expected = (
                                RANK_DUPLICATE
                                if spatial.rank_variable in irrelevant
                                else next(
                                    iter(rank_variable2ranks[spatial.rank_variable])
                                )
                            )
                            self.assertEqual(bound.rank_variable, expected)

    def test_exploring_bindings_adds_templates_and_compatibilities(self):
        without = _get_jobs()
        with_ = _get_jobs(binding=BindingConstraints(max_binding_rank_changes=1))

        for einsum_name in without:
            n_without = sum(len(j) for j in without[einsum_name].values())
            n_with = sum(len(j) for j in with_[einsum_name].values())
            self.assertGreater(
                n_with,
                n_without,
                f"Exploring bindings did not add templates for {einsum_name}.",
            )
            # The binding of a fused tensor's backing storage is part of the
            # compatibility, so exploring bindings gives more ways to fuse.
            self.assertGreater(
                len(with_[einsum_name]),
                len(without[einsum_name]),
                f"Exploring bindings did not add compatibilities for {einsum_name}.",
            )

            # Every binding explored without exploration is still explored with it.
            explored = {
                _bindings_of(j) for jl in with_[einsum_name].values() for j in jl
            }
            for job_list in without[einsum_name].values():
                for job in job_list:
                    self.assertIn(_bindings_of(job), explored)

    def test_explored_bindings_are_valid(self):
        for einsum_name, compatibility2jobs in _get_jobs(
            binding=BindingConstraints(max_binding_rank_changes=1)
        ).items():
            for job_list in compatibility2jobs.values():
                for job in job_list:
                    for block in iter_binding_blocks(job.mapping.nodes):
                        # One binding loop per spatial loop that binds the storage...
                        self.assertEqual(len(block.binding), len(block.spatials))
                        allowed = set(
                            job.einsum.tensor_accesses[block.tensor].ranks
                        ) | {RANK_DUPLICATE}
                        for spatial, bound in zip(block.spatials, block.binding):
                            # ...binding a rank of the tensor being bound...
                            self.assertIn(bound.rank_variable, allowed)
                            # ...and iterating as many times as its spatial loop...
                            self.assertEqual(
                                bound.calculated_n_iterations,
                                spatial.calculated_n_iterations,
                            )
                            # ...but with no tile shape of its own, since the bound rank
                            # may not have the shape of the rank variable that the
                            # spatial loop tiles.
                            self.assertIsNone(bound.tile_shape)
                            self.assertIsNone(bound.initial_tile_shape)

    def test_max_binding_rank_changes(self):
        one_change = _get_jobs(binding=BindingConstraints(max_binding_rank_changes=1))
        two_changes = _get_jobs(binding=BindingConstraints(max_binding_rank_changes=2))
        for einsum_name in one_change:
            self.assertGreater(
                sum(len(j) for j in two_changes[einsum_name].values()),
                sum(len(j) for j in one_change[einsum_name].values()),
            )

    def test_only_shared_tensors_are_explored(self):
        """
        A tensor that no other Einsum touches has no one to agree with, so exploring
        its binding only multiplies templates. Only the shared tensors are explored;
        every other tensor keeps its locally-optimal binding.
        """
        saw_unshared = False
        for compatibility2jobs in _get_jobs(
            # Two binding loops per storage in the fixture, so this is every binding.
            binding=BindingConstraints(max_binding_rank_changes=2)
        ).values():
            for job_list in compatibility2jobs.values():
                for job in job_list:
                    for block in iter_binding_blocks(job.mapping.nodes):
                        if block.tensor in job.fusable_tensors:
                            continue
                        saw_unshared = True
                        self.assertEqual(
                            [l.rank_variable for l in block.binding],
                            default_binding_ranks(block, job),
                        )
        self.assertTrue(saw_unshared, "The fixture binds no unshared tensor.")

    def test_beautify_loops_drops_the_binding_of_a_removed_spatial_loop(self):
        """
        Beautifying a mapping drops the loops that run a single iteration. A binding
        is one loop per spatial loop below its Storage, so a dropped spatial loop
        has to take its binding loop with it; otherwise nothing downstream can tell
        which binding loop belongs to which spatial loop.
        """
        job = _first_job_with_binding()
        mapping = job.mapping
        bounds = job.rank_variable_bounds

        # A loop is beautified away when it tiles its rank variable no finer than the
        # loop above it does, so giving every loop the full rank bound removes them
        # all. The two spatial loops of `m` are the exceptions: each tiles finer than
        # the loop above it, so both survive.
        m_spatials = [
            node
            for node in mapping.nodes
            if isinstance(node, Spatial) and node.rank_variable == "m"
        ]
        self.assertEqual(len(m_spatials), 2, "Fixture changed.")
        for node in mapping.nodes:
            if isinstance(node, Loop):
                node.tile_shape = bounds[node.rank_variable]
        m_spatials[0].tile_shape = bounds["m"] // 2
        m_spatials[1].tile_shape = bounds["m"] // 4

        kept = {id(node) for node in m_spatials}
        expected = {
            block.key: [
                bound
                for bound, spatial in zip(block.binding, block.spatials)
                if id(spatial) in kept
            ]
            for block in iter_binding_blocks(mapping.nodes)
        }
        self.assertTrue(any(expected.values()), "Fixture has no binding to prune.")

        mapping._beautify_loops(bounds)

        blocks = iter_binding_blocks(mapping.nodes)
        self.assertTrue(blocks, "Beautifying removed the bound storages themselves.")
        for block in blocks:
            # One binding loop per spatial loop, still...
            self.assertEqual(len(block.binding), len(block.spatials))
            self.assertEqual(
                [node.rank_variable for node in block.spatials], ["m", "m"]
            )
            # ...and the ones that were dropped are the ones whose spatial loop was.
            self.assertEqual(list(block.binding), expected[block.key])

    def test_mapper_evaluates_mappings_with_single_iteration_spatial_loops(self):
        """
        A mapping whose spatial loops are beautified away must still be modelable:
        `map_workload_to_arch` models every mapping it chooses in detail, and the
        model needs each binding to line up with the spatial loops that remain.
        """
        # One loop per spatial dimension makes single-iteration spatial loops the
        # norm, so the mappings the mapper returns have some to drop.
        spec = _make_spec(max_loops_per_spatial_dimension=1)
        result = spec.map_workload_to_arch(print_progress=False)
        self.assertTrue(len(result))
        self.assertTrue(
            any(
                isinstance(node, Binding) and node.bindings
                for node in result.mapping(0).nodes
            ),
            "The mapper returned a mapping with no binding.",
        )

    def test_mapper_keeps_the_binding_it_chose(self):
        """
        A mapping found by the mapper must be modeled with the binding the mapper chose
        for it, not with the locally-optimal binding.
        """
        spec = _make_spec(binding=BindingConstraints(max_binding_rank_changes=1))
        result = spec.map_workload_to_arch(print_progress=False)
        mapping = result.mapping(0)
        bindings = {
            (component, tensor): [l.rank_variable for l in loops]
            for node in mapping.nodes
            if isinstance(node, Binding)
            for component, tensor, loops in node.entries()
        }
        self.assertTrue(bindings, "The mapper returned a mapping with no binding.")

        spec.mapping = mapping
        spec.evaluate_mapping()
        for node in mapping.nodes:
            if not isinstance(node, Binding):
                continue
            for component, tensor, loops in node.entries():
                self.assertEqual(
                    [l.rank_variable for l in loops],
                    bindings[(component, tensor)],
                )


def _n_templates(jobs: dict) -> dict:
    """How many templates each Einsum has."""
    return {
        einsum_name: sum(len(job_list) for job_list in compatibility2jobs.values())
        for einsum_name, compatibility2jobs in jobs.items()
    }


def _iter_unshared_blocks(jobs: dict):
    """Every binding block of a tensor that no Einsum but its own touches."""
    for compatibility2jobs in jobs.values():
        for job_list in compatibility2jobs.values():
            for job in job_list:
                shared = job.fusable_tensors or set()
                for block in iter_binding_blocks(job.mapping.nodes):
                    if block.tensor not in shared and block.binding:
                        yield job, block


def _bindings_of_tensor(jobs: dict, tensor: str) -> set:
    """Every binding any template gives `tensor`, as a tuple of bound ranks."""
    return {
        tuple(loop.rank_variable for loop in block.binding)
        for compatibility2jobs in jobs.values()
        for job_list in compatibility2jobs.values()
        for job in job_list
        for block in iter_binding_blocks(job.mapping.nodes)
        if block.tensor == tensor and block.binding
    }


class TestShardUnsharedTensors(unittest.TestCase):
    """
    The one binding question an unshared tensor is asked. Which rank it partitions along
    is nobody else's business -- no other Einsum has to agree with it -- but whether it
    is partitioned at all is its memory's capacity problem, since replicating puts a
    whole copy of it in every physical instance.

    In the fixture, `A` is the only tensor two Einsums share; `I`, `WA`, `WB` and `B`
    are each touched by one.
    """

    UNSHARED = ("I", "WA", "WB", "B")

    def test_nothing_is_sharded_by_default(self):
        # Both fields default to Nothing, so an unshared tensor still replicates along
        # the loops that do not index it.
        self.assertTrue(
            any(
                RANK_DUPLICATE in {loop.rank_variable for loop in block.binding}
                for _, block in _iter_unshared_blocks(_get_jobs())
            ),
            "The fixture has no replicated binding loop left to shard.",
        )

    def test_must_shard_partitions_every_loop(self):
        jobs = _get_jobs(
            binding=BindingConstraints(must_shard_unshared_tensors="All")
        )
        blocks = list(_iter_unshared_blocks(jobs))
        self.assertTrue(blocks)
        for job, block in blocks:
            ranks = set(job.einsum.tensor_accesses[block.tensor].ranks)
            for loop in block.binding:
                self.assertIn(
                    loop.rank_variable,
                    ranks,
                    f"{block.tensor} still replicates along {loop.name}.",
                )

    def test_must_shard_leaves_shared_tensors_replicating(self):
        # `A` is shared, so this constraint has nothing to say about it.
        jobs = _get_jobs(
            binding=BindingConstraints(must_shard_unshared_tensors="All")
        )
        self.assertEqual(
            _bindings_of_tensor(jobs, "A"), _bindings_of_tensor(_get_jobs(), "A")
        )

    def test_must_shard_costs_no_templates(self):
        # It moves the binding every template starts from rather than adding templates.
        self.assertEqual(
            _n_templates(
                _get_jobs(binding=BindingConstraints(must_shard_unshared_tensors="All"))
            ),
            _n_templates(_get_jobs()),
        )

    def test_may_shard_explores_both_ways(self):
        without = _get_jobs()
        with_ = _get_jobs(
            binding=BindingConstraints(may_shard_unshared_tensors="All")
        )
        for einsum_name, n_with in _n_templates(with_).items():
            self.assertGreater(
                n_with,
                _n_templates(without)[einsum_name],
                f"Sharding unshared tensors added no templates for {einsum_name}.",
            )
        for tensor in self.UNSHARED:
            explored = _bindings_of_tensor(with_, tensor)
            # Everything the mapper explored without the field is still explored...
            self.assertTrue(_bindings_of_tensor(without, tensor) <= explored)
            # ...and a binding that partitions where the default replicated is too.
            self.assertTrue(
                any(RANK_DUPLICATE not in binding for binding in explored),
                f"{tensor} was never sharded.",
            )

    def test_may_shard_leaves_the_loops_that_index_the_tensor_alone(self):
        # Which rank such a loop partitions along is not a question an unshared tensor
        # gets asked, so it binds what it would have bound with no exploration at all.
        jobs = _get_jobs(
            binding=BindingConstraints(may_shard_unshared_tensors="All")
        )
        for job, block in _iter_unshared_blocks(jobs):
            irrelevant = job.einsum.tensor2irrelevant_rank_variables[block.tensor]
            rank_variable2ranks = job.einsum.tensor_accesses[
                block.tensor
            ].rank_variable2ranks
            for spatial, bound in zip(block.spatials, block.binding):
                if spatial.rank_variable in irrelevant:
                    continue
                self.assertEqual(
                    bound.rank_variable,
                    next(iter(rank_variable2ranks[spatial.rank_variable])),
                )

    def test_may_shard_leaves_shared_tensors_alone(self):
        jobs = _get_jobs(
            binding=BindingConstraints(may_shard_unshared_tensors="All")
        )
        self.assertEqual(
            _bindings_of_tensor(jobs, "A"), _bindings_of_tensor(_get_jobs(), "A")
        )

    def test_may_shard_is_not_budgeted(self):
        # max_binding_rank_changes buys the rank changes that let two Einsums disagree
        # about a shared tensor. An unshared tensor has nobody to disagree with, so its
        # one choice is not paid for out of that budget and is explored without one.
        jobs = _get_jobs(
            binding=BindingConstraints(
                may_shard_unshared_tensors="All", max_binding_rank_changes=0
            )
        )
        for einsum_name, n_with in _n_templates(jobs).items():
            self.assertGreater(n_with, _n_templates(_get_jobs())[einsum_name])

    def test_may_shard_defaults_to_must_shard(self):
        # Naming a tensor in must_ alone shards it rather than also measuring what
        # replicating it would have cost, so it adds no templates.
        self.assertEqual(
            _n_templates(
                _get_jobs(binding=BindingConstraints(must_shard_unshared_tensors="All"))
            ),
            _n_templates(_get_jobs()),
        )

    def test_a_tensor_that_must_duplicate_has_nothing_to_choose(self):
        # may_duplicate/must_duplicate settle the loop; there is no second way to bind
        # it left for this to explore.
        self.assertEqual(
            _n_templates(
                _get_jobs(
                    binding=BindingConstraints(
                        must_duplicate="All", may_shard_unshared_tensors="All"
                    )
                )
            ),
            _n_templates(_get_jobs(binding=BindingConstraints(must_duplicate="All"))),
        )


class TestBindingRankClasses(unittest.TestCase):
    """
    Which ranks a binding loop is modeled the same for. Partitioning along a rank other
    than the locally-optimal one costs the same whichever of those ranks it is, so they
    share a template; replicating does not, so it keeps its own.
    """

    def test_the_default_comes_first_and_alone(self):
        classes = _binding_rank_classes(["M", "N", RANK_DUPLICATE], "M")
        self.assertEqual(classes[0], ("M",))

    def test_other_partitioning_ranks_share_a_class(self):
        classes = _binding_rank_classes(["M", "N", "K", RANK_DUPLICATE], "M")
        self.assertIn(("N", "K"), classes)

    def test_duplicate_is_its_own_class(self):
        # Replicating puts a whole copy in every instance where partitioning gives each
        # a share, so it is not interchangeable with a partitioning rank.
        classes = _binding_rank_classes(["M", "N", "K", RANK_DUPLICATE], "M")
        self.assertIn((RANK_DUPLICATE,), classes)

    def test_a_duplicating_default_leaves_every_rank_in_one_class(self):
        classes = _binding_rank_classes(["M", "N", RANK_DUPLICATE], RANK_DUPLICATE)
        self.assertEqual(classes, [(RANK_DUPLICATE,), ("M", "N")])

    def test_every_allowed_rank_is_in_exactly_one_class(self):
        allowed = ["M", "N", "K", RANK_DUPLICATE]
        for default in allowed:
            classes = _binding_rank_classes(allowed, default)
            flat = [r for c in classes for r in c]
            self.assertEqual(sorted(flat), sorted(allowed), f"default={default}")

    def test_a_lone_rank_has_nothing_to_share_with(self):
        self.assertEqual(_binding_rank_classes(["M"], "M"), [("M",)])


class TestBindingExpansion(unittest.TestCase):
    """
    Collapsing the ranks that model alike into one template must not cost the mapper any
    binding: each merged rank has to come back as a compatibility of its own once the
    template has been modeled, because a rank is what a neighboring Einsum agrees with
    even when it is not what the numbers depend on.
    """

    EXPLORE = BindingConstraints(max_binding_rank_changes=1)

    @staticmethod
    def _one_class_per_rank(allowed, default):
        """How choices were enumerated before equivalent ranks were merged."""
        return [(default,)] + [(r,) for r in allowed if r != default]

    @contextmanager
    def _without_merging(self):
        """
        Run the body with every rank getting a choice to itself. Only usable on calls
        made in this process: template generation itself runs in worker processes, which
        would not see the swap.
        """
        original = templates_module._binding_rank_classes
        templates_module._binding_rank_classes = self._one_class_per_rank
        try:
            yield
        finally:
            templates_module._binding_rank_classes = original

    def _pmapping_groups_and_jobs(self, **mapper_settings):
        groups, _, jobs = make_pmappings(
            _make_spec(**mapper_settings),
            can_combine_multiple_runs=False,
            metrics=Metrics.ENERGY,
            fail_if_no_pmappings_for_einsum=True,
            print_progress=False,
        )
        return groups, jobs

    def test_merging_equivalent_ranks_offers_fewer_choices(self):
        """
        The point of the whole thing: a loop that could partition along any of several
        non-optimal ranks costs one choice between them, not one each.
        """
        compared = 0
        for compatibility2jobs in _get_jobs(binding=self.EXPLORE).values():
            for job_list in compatibility2jobs.values():
                for job in job_list:
                    merged = list(iter_binding_choices(job.mapping, job))
                    with self._without_merging():
                        unmerged = list(iter_binding_choices(job.mapping, job))
                    self.assertLessEqual(len(merged), len(unmerged))
                    if len(merged) < len(unmerged):
                        compared += 1
        self.assertTrue(
            compared,
            "No template of the fixture had two ranks to merge, so nothing here "
            "tested the merge.",
        )

    def test_expansions_are_every_combination_with_the_modeled_one_first(self):
        aliases = {("T1", "SharedBuffer", 0): ("M", "N1"),
                   ("T1", "SharedBuffer", 3): ("K", "L")}
        expansions = _binding_expansions(aliases)
        self.assertEqual(len(expansions), 4)
        # The first is what the template was actually modeled with, so the caller can
        # keep its group as it stands and expand only the rest.
        self.assertEqual(
            expansions[0],
            {("T1", "SharedBuffer", 0): "M", ("T1", "SharedBuffer", 3): "K"},
        )
        self.assertEqual(len(set(map(str, expansions))), 4, "combinations repeat")

    def test_nothing_to_expand_is_one_empty_expansion(self):
        self.assertEqual(_binding_expansions({}), [{}])

    def test_rebinding_a_compatibility_moves_only_the_loop_it_names(self):
        """
        The loop is named by position, not by dimension: a memory with several spatial
        loops over one dimension binds a rank for each of them, and naming them by
        dimension would rewrite them all together.
        """
        compatibility = _compatibility_with_binding(
            [("Y", "N1"), ("Y", RANK_DUPLICATE), ("X", "M")]
        )
        rebound = compatibility.rebind_physical_spatial_loops(
            {("T1", "SharedBuffer", 1): Rank("M")}
        )
        self.assertEqual(
            _binding_of(rebound), [("Y", "N1"), ("Y", "M"), ("X", "M")]
        )
        # ...and the one it was asked to rewrite really did change.
        self.assertNotEqual(_binding_of(rebound), _binding_of(compatibility))

    def test_rebinding_leaves_another_tensor_alone(self):
        compatibility = _compatibility_with_binding([("Y", "N1")], tensor="T2")
        self.assertEqual(
            _binding_of(
                compatibility.rebind_physical_spatial_loops(
                    {("T1", "SharedBuffer", 0): Rank("M")}
                )
            ),
            [("Y", "N1")],
        )

    def test_rebinding_leaves_the_same_tensor_in_another_memory_alone(self):
        compatibility = _compatibility_with_binding([("Y", "N1")], resource="OtherBuffer")
        self.assertEqual(
            _binding_of(
                compatibility.rebind_physical_spatial_loops(
                    {("T1", "SharedBuffer", 0): Rank("M")}
                )
            ),
            [("Y", "N1")],
        )

    def test_an_expanded_compatibility_matches_its_mapping(self):
        """
        A group's compatibility names the binding of the mapping its rows point at, with
        the degenerate loops dropped -- so its binding loops are a subsequence of the
        mapping's. An expansion that rewrote the wrong loop would break that, since
        several spatial loops of one memory can share a dimension.
        """
        groups, pmapping_objects, _ = make_pmappings(
            _make_spec(binding=self.EXPLORE),
            can_combine_multiple_runs=False,
            metrics=Metrics.ENERGY,
            fail_if_no_pmappings_for_einsum=True,
            print_progress=False,
        )

        def is_subsequence(short, long) -> bool:
            it = iter(long)
            return all(item in it for item in short)

        checked = 0
        for einsum_name, group_list in groups.items():
            for group in group_list:
                for tensor in group.compatibility.tensors:
                    if not tensor.physical_spatial_loops:
                        continue
                    wanted = [
                        (l.spatial_dim, l.rank_name)
                        for l in tensor.physical_spatial_loops
                    ]
                    column = f"{einsum_name}<SEP>mapping"
                    for job_id in group.mappings.data[column].unique():
                        mapping = pmapping_objects[einsum_name][job_id]
                        bound = [
                            (loop.name, loop.rank_variable)
                            for block in iter_binding_blocks(mapping.nodes)
                            if block.tensor == tensor.name
                            and block.component == tensor.resource_name
                            for loop in block.binding
                        ]
                        self.assertTrue(
                            is_subsequence(wanted, bound),
                            f"{einsum_name}: compatibility binds {wanted}, but the "
                            f"mapping it points at binds {bound}",
                        )
                        checked += 1
        self.assertTrue(checked, "No bound mapping was checked.")


if __name__ == "__main__":
    unittest.main()
