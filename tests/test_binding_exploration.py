"""
Tests for exploring the binding of a tensor to the physical instances of a
physically-distributed memory (see FFM.explore_bindings).
"""

import unittest
from pathlib import Path

import accelforge as af
from accelforge.frontend.mapping import Storage, iter_binding_blocks
from accelforge.frontend.renames import RANK_ANY
from accelforge.frontend.spec import Spec
from accelforge.mapper import Metrics
from accelforge.mapper.FFM._make_pmappings.make_pmappings import get_jobs

ARCH = Path(__file__).parent / "input_files" / "distributed_binding.arch.yaml"


def _make_spec(**mapper_settings) -> Spec:
    spec = Spec.from_yaml(
        ARCH,
        af.examples.workloads.basic.matmuls,
        jinja_parse_data={"N_EINSUMS": 2, "M": 8, "KN": 8},
    )
    for k, v in mapper_settings.items():
        setattr(spec.mapper, k, v)
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


def _bindings_of(job) -> tuple:
    """The (component, tensor, bound ranks) of every bound Storage node of a job."""
    return tuple(
        (node.component, node.tensor, tuple(l.rank_variable for l in node.binding))
        for node in job.mapping.nodes
        if isinstance(node, Storage) and node.binding
    )


class TestBindingExploration(unittest.TestCase):
    def test_default_binding_is_the_only_one_if_not_exploring(self):
        jobs = _get_jobs(explore_bindings=False)
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
                                RANK_ANY
                                if spatial.rank_variable in irrelevant
                                else next(
                                    iter(rank_variable2ranks[spatial.rank_variable])
                                )
                            )
                            self.assertEqual(bound.rank_variable, expected)

    def test_exploring_bindings_adds_templates_and_compatibilities(self):
        without = _get_jobs(explore_bindings=False)
        with_ = _get_jobs(explore_bindings=True)

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
            explore_bindings=True
        ).items():
            for job_list in compatibility2jobs.values():
                for job in job_list:
                    for storage, spatials in iter_binding_blocks(job.mapping.nodes):
                        # One binding loop per spatial loop that binds the storage...
                        self.assertEqual(len(storage.binding), len(spatials))
                        allowed = set(
                            job.einsum.tensor_accesses[storage.tensor].ranks
                        ) | {RANK_ANY}
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
        one_change = _get_jobs(explore_bindings=True, max_binding_rank_changes=1)
        two_changes = _get_jobs(explore_bindings=True, max_binding_rank_changes=2)
        for einsum_name in one_change:
            self.assertGreater(
                sum(len(j) for j in two_changes[einsum_name].values()),
                sum(len(j) for j in one_change[einsum_name].values()),
            )

    def test_max_binding_options_per_pmapping_template(self):
        capped = _get_jobs(
            explore_bindings=True, max_binding_options_per_pmapping_template=3
        )
        uncapped = _get_jobs(explore_bindings=False)
        for einsum_name in capped:
            n_capped = sum(len(j) for j in capped[einsum_name].values())
            n_templates = sum(len(j) for j in uncapped[einsum_name].values())
            self.assertEqual(n_capped, 3 * n_templates)

    def test_mapper_keeps_the_binding_it_chose(self):
        """
        A mapping found by the mapper must be modeled with the binding the mapper chose
        for it, not with the locally-optimal binding.
        """
        spec = _make_spec(
            explore_bindings=True, max_binding_options_per_pmapping_template=2
        )
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
