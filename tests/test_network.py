from unittest import TestCase

import accelforge as af


class TestParsing(TestCase):
    def test_hierarchical(self):
        spec = af.Spec.from_yaml(
            af.examples.arches.networked.hierarchical,
        )
        self.assertIn("PeArray", spec.arch.nodes)
        self.assertEqual(spec.arch.nodes["PeArray"].get_fanout(), 1)
        self.assertIn("Scratchpad", spec.arch.nodes)
        self.assertEqual(spec.arch.nodes["Scratchpad"].get_fanout(), 4)
        self.assertIn("MacArray", spec.arch.nodes)
        self.assertEqual(spec.arch.nodes["MacArray"].get_fanout(), 1)

        try:
            spec = spec.calculate_component_area_energy_latency_leak()
        except af.EvaluationError as e:
            self.fail(e.message)

    def test_flat(self):
        spec = af.Spec.from_yaml(
            af.examples.arches.networked.flat,
        )
        self.assertIn("NoC", spec.arch.nodes)
        self.assertEqual(spec.arch.nodes["NoC"].get_fanout(), 1)
        self.assertEqual(
            {n.name for n in spec.arch.get_nodes_of_type(af.spec.Leaf)},
            {'MainMemory', 'GlobalBuffer', 'NoC', 'RowBuffer', 'ColumnBuffer', 'DistributedBuffer', 'Scratchpad', 'MAC'}
        )

        try:
            spec = spec.calculate_component_area_energy_latency_leak()
        except af.EvaluationError as e:
            self.fail(e.message)


class TestModel(TestCase):
    def test_hierarchical(self):
        M = 8
        KN = 8
        MAC_TILE = 2
        PE_TILE = KN//MAC_TILE
        M_TILE = 4
        BITS_PER_VALUE = 8

        spec = af.Spec.from_yaml(
            af.examples.workloads.matmuls,
            af.examples.arches.networked.hierarchical,
            af.examples.mappings.one_matmul_to_networked_hierarchical,
            jinja_parse_data={"N_EINSUMS": 1, "M": 8, "KN": 8, "MAC_TILE": MAC_TILE, "M_TILE": M_TILE}
        )
        result = spec.evaluate_mapping()
        self.assertEqual(
            result.data["Matmul0<SEP>action<SEP>MacArray<SEP>T0<SEP>hops"].iloc[0],
            (M/M_TILE)*(KN/MAC_TILE)**2 * M_TILE * (0.5*MAC_TILE*(MAC_TILE-1) + MAC_TILE*(MAC_TILE-1)) * BITS_PER_VALUE
        )
        # NOTE: assuming XY routing (as defined in mapping)
        self.assertEqual(
            result.data["Matmul0<SEP>action<SEP>MacArray<SEP>T1<SEP>hops"].iloc[0],
            (M/M_TILE)*(KN/MAC_TILE)**2 * M_TILE * (MAC_TILE*(MAC_TILE-1) + MAC_TILE*(MAC_TILE-1)) * BITS_PER_VALUE
        )
        self.assertEqual(
            result.data["Matmul0<SEP>action<SEP>MacArray<SEP>W0<SEP>hops"].iloc[0],
            (M/M_TILE)*(KN/MAC_TILE)**2 * M_TILE * (MAC_TILE*(MAC_TILE-1) + MAC_TILE*(MAC_TILE-1)) * BITS_PER_VALUE
        )

        self.assertEqual(
            result.data["Matmul0<SEP>action<SEP>PeArray<SEP>T0<SEP>hops"].iloc[0],
            (M/M_TILE) * (0.5*PE_TILE*(PE_TILE-1) + PE_TILE*(PE_TILE-1)) * M_TILE*MAC_TILE*BITS_PER_VALUE
        )
        # NOTE: assuming XY routing (as defined in mapping)
        self.assertEqual(
            result.data["Matmul0<SEP>action<SEP>PeArray<SEP>T1<SEP>hops"].iloc[0],
            (M/M_TILE) * (PE_TILE*(PE_TILE-1) + PE_TILE*0.5*PE_TILE*(PE_TILE-1)) * M_TILE*MAC_TILE*BITS_PER_VALUE
        )
        self.assertEqual(
            result.data["Matmul0<SEP>action<SEP>PeArray<SEP>W0<SEP>hops"].iloc[0],
            (M/M_TILE) * (PE_TILE*0.5*PE_TILE*(PE_TILE-1) + PE_TILE*0.5*PE_TILE*(PE_TILE-1)) * MAC_TILE**2*BITS_PER_VALUE
        )
