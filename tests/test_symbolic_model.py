import unittest
from pathlib import Path

from fastfusion.frontend import Specification
from fastfusion.frontend.mapping import Mapping
from fastfusion.frontend.workload import Workload

from fastfusion.model.looptree.accesses import buffer_accesses_from_buffet_actions, Accesses
from fastfusion.model.looptree.energy import gather_actions
from fastfusion.model.looptree.latency import get_latency
from fastfusion.model.looptree.reuse.summarized.symbolic_new import analyze_reuse, Compute


class TestSymbolicModel(unittest.TestCase):
    def test_q_mapping(self):
        mapping = Mapping.from_yaml(Path(__file__).parent / 'Q_mapping.yaml')
        workload = Workload.from_yaml(Path(__file__).parent / 'mha.yaml')

        result = analyze_reuse(mapping, workload)

        self.assertEqual(result.compute_stats[Compute('Q', 'MAC')].total_ops, 64)
        self.assertEqual(result.compute_stats[Compute('Q', 'MAC')].max_per_unit_ops, 16)

    def test_conv_mapping(self):
        mapping = Mapping.from_yaml(Path(__file__).parent / 'conv.mapping.yaml')
        workload = Workload.from_yaml(Path(__file__).parent / 'conv.workload.yaml')

        result = analyze_reuse(mapping, workload)

        self.assertEqual(result.compute_stats[Compute('conv', 'MAC')].total_ops, 120)
        self.assertEqual(result.compute_stats[Compute('conv', 'MAC')].max_per_unit_ops, 10)


class TestSymbolicAccesses(unittest.TestCase):
    def test_q_mapping(self):
        mapping = Mapping.from_yaml(Path(__file__).parent / 'Q_mapping.yaml')
        workload = Workload.from_yaml(Path(__file__).parent / 'mha.yaml')

        result = analyze_reuse(mapping, workload)

        accesses = buffer_accesses_from_buffet_actions(result, mapping, workload, is_path=True)

        main_memory_I_accesses = accesses.get_accesses('MainMemory', 'I', 'Q')
        self.assertEqual(main_memory_I_accesses,
                         Accesses(total_reads=64,
                                  total_writes=0,
                                  max_per_unit_reads=64,
                                  max_per_unit_writes=0))

        local_buffer_I_accesses = accesses.get_accesses('LocalBuffer', 'I', 'Q')
        self.assertEqual(local_buffer_I_accesses,
                         Accesses(total_reads=64,
                                  total_writes=64,
                                  max_per_unit_reads=16,
                                  max_per_unit_writes=16))

        local_buffer_Q_accesses = accesses.get_accesses('LocalBuffer', 'Q', 'Q')
        self.assertEqual(local_buffer_Q_accesses,
                         Accesses(total_reads=64,
                                  total_writes=128,
                                  max_per_unit_reads=16,
                                  max_per_unit_writes=32))


class TestSymbolicActions(unittest.TestCase):
    def test_q_mapping(self):
        mapping = Mapping.from_yaml(Path(__file__).parent / 'Q_mapping.yaml')
        workload = Workload.from_yaml(Path(__file__).parent / 'mha.yaml')

        result = analyze_reuse(mapping, workload)
        actions = gather_actions(result, mapping, workload, None, is_path=True, use_name=True)

        self.assertEqual(actions[('LocalBuffer', 'read')].total, 128)
        self.assertEqual(actions[('LocalBuffer', 'read')].max_per_unit, 32)
        self.assertEqual(actions[('LocalBuffer', 'write')].total, 192)
        self.assertEqual(actions[('LocalBuffer', 'write')].max_per_unit, 48)

        self.assertEqual(actions[('MAC', 'compute')].total, 64)
        self.assertEqual(actions[('MAC', 'compute')].max_per_unit, 16)


class TestSymbolicLatency(unittest.TestCase):
    def test_q_mapping(self):
        spec = Specification.from_yaml([
            # Path(__file__).parent / 'Q_mapping.yaml',
            Path(__file__).parent / 'mha.yaml',
            Path(__file__).parent / 'four_level.arch.yaml'
        ])
        workload = spec.workload
        architecture = spec.architecture
        mapping = Mapping.from_yaml(Path(__file__).parent / 'Q_mapping.yaml')

        result = analyze_reuse(mapping, workload)
        overall_latency, _, _ = get_latency(result, mapping, workload, architecture)

        self.assertEqual(overall_latency, 16)
