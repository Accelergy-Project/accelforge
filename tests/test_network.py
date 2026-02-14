from unittest import TestCase

import accelforge as af


class TestParsing(TestCase):
    def test_hierarchical_only(self):
        spec = af.Spec.from_yaml(
            af.examples.arches.networked.hierarchical_only,
        )
        self.assertIn("PeArray", spec.arch.nodes)
        self.assertEqual(spec.arch.nodes["PeArray"].get_fanout(), 4)
        self.assertIn("MacArray", spec.arch.nodes)
        self.assertEqual(spec.arch.nodes["MacArray"].get_fanout(), 4)

        try:
            spec = spec.calculate_component_area_energy_latency_leak()
        except af.EvaluationError as e:
            self.fail(e.message)

    def test_flat(self):
        spec = af.Spec.from_yaml(
            af.examples.arches.networked.flat,
        )
        self.assertIn("NoC", spec.arch.nodes)
        self.assertEqual(spec.arch.nodes["NoC"].get_fanout(), 16)
        self.assertEqual(
            {n.name for n in spec.arch.get_nodes_of_type(af.spec.Leaf)},
            {'MainMemory', 'NoC', 'GlobalBuffer', 'RowBuffer', 'ColumnBuffer', 'DistributedBuffer', 'Scratchpad', 'MAC'}
        )

        try:
            spec = spec.calculate_component_area_energy_latency_leak()
        except af.EvaluationError as e:
            self.fail(e.message)