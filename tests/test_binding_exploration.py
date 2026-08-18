"""
Tests for exploring the binding of a tensor to the physical instances of a
physically-distributed memory.
"""

import unittest
from pathlib import Path

import accelforge as af
from accelforge.frontend.arch import BindingConstraints
from accelforge.frontend.mapping import Loop, Spatial, Storage, iter_binding_blocks
from accelforge.frontend.renames import RANK_DUPLICATE
from accelforge.frontend.spec import Spec
from accelforge.mapper import Metrics
from accelforge.mapper.FFM._make_pmappings.make_pmappings import get_jobs

ARCH = Path(__file__).parent / "input_files" / "distributed_binding.arch.yaml"


# The only physically-distributed memory of the fixture, and so the only one whose
# binding constraints the mapper consults.
DISTRIBUTED_MEMORY = "SharedBuffer"


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
    """The first job whose mapping has a bound, physically-distributed Storage."""
    for compatibility2jobs in _get_jobs().values():
        for job_list in compatibility2jobs.values():
            for job in job_list:
                if any(
                    isinstance(node, Storage) and node.binding
                    for node in job.mapping.nodes
                ):
                    return job
    raise AssertionError("No job of the fixture has a binding.")


def _bindings_of(job) -> tuple:
    """The (component, tensor, bound ranks) of every bound Storage node of a job."""
    return tuple(
        (node.component, node.tensor, tuple(l.rank_variable for l in node.binding))
        for node in job.mapping.nodes
        if isinstance(node, Storage) and node.binding
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
                    for node in job.mapping.nodes:
                        if not isinstance(node, Storage) or not node.binding:
                            continue
                        irrelevant = job.einsum.tensor2irrelevant_rank_variables[
                            node.tensor
                        ]
                        rank_variable2ranks = job.einsum.tensor_accesses[
                            node.tensor
                        ].rank_variable2ranks
                        _, spatials = next(
                            (s, l)
                            for s, l in iter_binding_blocks(job.mapping.nodes)
                            if s is node
                        )
                        for spatial, bound in zip(spatials, node.binding):
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
                    for storage, spatials in iter_binding_blocks(job.mapping.nodes):
                        # One binding loop per spatial loop that binds the storage...
                        self.assertEqual(len(storage.binding), len(spatials))
                        allowed = set(
                            job.einsum.tensor_accesses[storage.tensor].ranks
                        ) | {RANK_DUPLICATE}
                        for spatial, bound in zip(spatials, storage.binding):
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
            id(storage): [
                bound
                for bound, spatial in zip(storage.binding, spatials)
                if id(spatial) in kept
            ]
            for storage, spatials in iter_binding_blocks(mapping.nodes)
        }
        self.assertTrue(any(expected.values()), "Fixture has no binding to prune.")

        mapping._beautify_loops(bounds)

        blocks = iter_binding_blocks(mapping.nodes)
        self.assertTrue(blocks, "Beautifying removed the bound storages themselves.")
        for storage, spatials in blocks:
            # One binding loop per spatial loop, still...
            self.assertEqual(len(storage.binding), len(spatials))
            self.assertEqual([node.rank_variable for node in spatials], ["m", "m"])
            # ...and the ones that were dropped are the ones whose spatial loop was.
            self.assertEqual(list(storage.binding), expected[id(storage)])

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
                isinstance(node, Storage) and node.binding
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
            (node.component, node.tensor): [l.rank_variable for l in node.binding]
            for node in mapping.nodes
            if isinstance(node, Storage) and node.binding
        }
        self.assertTrue(bindings, "The mapper returned a mapping with no binding.")

        spec.mapping = mapping
        spec.evaluate_mapping()
        for node in mapping.nodes:
            if isinstance(node, Storage) and node.binding:
                self.assertEqual(
                    [l.rank_variable for l in node.binding],
                    bindings[(node.component, node.tensor)],
                )


if __name__ == "__main__":
    unittest.main()
