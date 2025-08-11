from math import isclose
import unittest
from pathlib import Path

from fastfusion.frontend import Specification
from fastfusion.frontend.mapping import Mapping
from fastfusion.frontend.workload import Workload

from fastfusion.mapper.FFM.exploration.mapper_one_einsum.mapper_job import Job
from fastfusion.model.looptree.accesses import isl_buffer_accesses_from_buffet_actions, Accesses
from fastfusion.model.looptree.energy import gather_actions
from fastfusion.model.looptree.latency import get_latency
from fastfusion.model.looptree.reuse.summarized.symbolic import BuffetStats, analyze_reuse_and_add_reservations_to_mapping, Compute, Buffet


PARENT_DIR = Path(__file__).parent

def make_job(mapping: Mapping, workload: Workload) -> Job:
    return Job(
        spec=None,
        tagger=None,
        mapping=mapping,
        workload=workload,
    )

class TestSymbolicModel(unittest.TestCase):
    def test_q_mapping(self):
        mapping = Mapping.from_yaml(Path(__file__).parent / 'Q_mapping.yaml')
        workload = Workload.from_yaml(Path(__file__).parent / 'mha.yaml')

        result = analyze_reuse_and_add_reservations_to_mapping(make_job(mapping, workload))

        self.assertAlmostEqual(result.compute_stats[Compute('Q', 'MAC')].total_ops, 64.0)
        self.assertAlmostEqual(result.compute_stats[Compute('Q', 'MAC')].max_per_unit_ops, 16.0)

    def test_conv_mapping(self):
        mapping = Mapping.from_yaml(Path(__file__).parent / 'conv.mapping.yaml')
        workload = Workload.from_yaml(Path(__file__).parent / 'conv.workload.yaml')

        result = analyze_reuse_and_add_reservations_to_mapping(make_job(mapping, workload))

        self.assertAlmostEqual(result.compute_stats[Compute('conv', 'MAC')].total_ops, 120.0)
        self.assertAlmostEqual(result.compute_stats[Compute('conv', 'MAC')].max_per_unit_ops, 10.0)

    def test_matmul_mapping(self):
        mapping = Mapping.from_yaml(Path(__file__).parent / 'matmul.mapping.yaml')
        workload = Workload.from_yaml(Path(__file__).parent / 'matmul.workload.yaml')

        result = analyze_reuse_and_add_reservations_to_mapping(make_job(mapping, workload))

        REF_OCCUPANCY = {
            'W0': 1,
            'T0': 128,
            'T1': 128*128
        }
        for tensor, ref_occupancy in REF_OCCUPANCY.items():
            self.assertAlmostEqual(
                result.buffet_stats[Buffet(tensor, 'Matmul1', 'LocalBuffer')].max_occupancy,
                ref_occupancy
            )

    def test_matmul_spatial(self):
        mapping = Mapping.from_yaml(PARENT_DIR / 'matmul_spatial.mapping.yaml')
        workload = Workload.from_yaml(PARENT_DIR / 'matmul.workload.yaml')

        result = analyze_reuse_and_add_reservations_to_mapping(make_job(mapping, workload))
        self.assertAlmostEqual(
            result.fanout,
            {('LocalBuffer', 'Matmul1'): {0: 128.0, 1: 4.0}, ('MainMemory', 'Matmul1'): {}}
        )
        
    def test_copy_mapping(self):
        mapping = Mapping.from_yaml(PARENT_DIR / 'copy.mapping.yaml')
        workload = Workload.from_yaml(PARENT_DIR / 'copy.workload.yaml')

        result = analyze_reuse_and_add_reservations_to_mapping(make_job(mapping, workload))

        self.assertAlmostEqual(result.compute_stats[Compute('copy', 'MAC')].total_ops, 0)
        self.assertAlmostEqual(result.compute_stats[Compute('copy', 'MAC')].max_per_unit_ops, 0)
        
        for tensor in ["O1", "O2", "O3", "O4"]:
            for memory in ["MainMemory", "GlobalBuffer", "LocalBuffer", "Register", "MAC"]:
                buffet = Buffet(level=memory, tensor=tensor, einsum='copy')
                if buffet not in result.buffet_stats:
                    continue
                stats = result.buffet_stats[buffet]
                self.assertAlmostEqual(stats.net_total_read_actions(), 0)
                self.assertAlmostEqual(stats.net_max_per_unit_read_actions(), 0)
                self.assertAlmostEqual(stats.net_total_write_actions(), 0)
                self.assertAlmostEqual(stats.net_max_per_unit_write_actions(), 0)
                self.assertAlmostEqual(stats.max_occupancy, 0)
                
        buffet = Buffet(level='GlobalBuffer', tensor='I', einsum='copy')
        stats = result.buffet_stats.get(buffet, BuffetStats())
        self.assertAlmostEqual(stats.net_total_read_actions(), 0)
        self.assertAlmostEqual(stats.net_max_per_unit_read_actions(), 0)
        self.assertAlmostEqual(stats.net_total_write_actions(), 0)
        self.assertAlmostEqual(stats.net_max_per_unit_write_actions(), 0)
        self.assertAlmostEqual(stats.max_occupancy, 0)

        stats = result.buffet_stats[Buffet(level='LocalBuffer', tensor='I', einsum='copy')]
        self.assertAlmostEqual(stats.net_total_read_actions(), 16)
        self.assertAlmostEqual(stats.net_max_per_unit_read_actions(), 16)
        self.assertAlmostEqual(stats.net_total_write_actions(), 0)
        self.assertAlmostEqual(stats.net_max_per_unit_write_actions(), 0)
        self.assertAlmostEqual(stats.max_occupancy, 2)

        stats = result.buffet_stats[Buffet(level='Register', tensor='I', einsum='copy')]
        self.assertAlmostEqual(stats.net_total_read_actions(), 0)
        self.assertAlmostEqual(stats.net_max_per_unit_read_actions(), 0)
        self.assertAlmostEqual(stats.net_total_write_actions(), 8)
        self.assertAlmostEqual(stats.net_max_per_unit_write_actions(), 8)
        self.assertAlmostEqual(stats.max_occupancy, 1)

        stats = result.buffet_stats[Buffet(level='MainMemory', tensor='I', einsum='copy')]
        self.assertAlmostEqual(stats.net_total_read_actions(), 8)
        self.assertAlmostEqual(stats.net_max_per_unit_read_actions(), 8)
        self.assertAlmostEqual(stats.net_total_write_actions(), 8)
        self.assertAlmostEqual(stats.net_max_per_unit_write_actions(), 8)
        self.assertAlmostEqual(stats.max_occupancy, 8)

        stats = result.buffet_stats[Buffet(level='Disk', tensor='I', einsum='copy')]
        self.assertAlmostEqual(stats.net_total_read_actions(), 0)
        self.assertAlmostEqual(stats.net_max_per_unit_read_actions(), 0)
        self.assertAlmostEqual(stats.net_total_write_actions(), 8)
        self.assertAlmostEqual(stats.net_max_per_unit_write_actions(), 8)
        self.assertAlmostEqual(stats.max_occupancy, 8)

class TestSymbolicAccesses(unittest.TestCase):
    def test_q_mapping(self):
        mapping = Mapping.from_yaml(Path(__file__).parent / 'Q_mapping.yaml')
        workload = Workload.from_yaml(Path(__file__).parent / 'mha.yaml')

        result = analyze_reuse_and_add_reservations_to_mapping(make_job(mapping, workload))

        # main_memory_I_accesses = accesses.get_accesses('MainMemory', 'I', 'Q')
        stats = result.buffet_stats[Buffet(level='MainMemory', tensor='I', einsum='Q')]
        self.assertAlmostEqual(stats.net_total_read_actions(), 64.0)
        self.assertAlmostEqual(stats.net_max_per_unit_read_actions(), 64.0)
        self.assertAlmostEqual(stats.net_total_write_actions(), 0.0)
        self.assertAlmostEqual(stats.net_max_per_unit_write_actions(), 0.0)

        stats = result.buffet_stats[Buffet(level='LocalBuffer', tensor='I', einsum='Q')]
        self.assertAlmostEqual(stats.net_total_read_actions(), 64.0)
        self.assertAlmostEqual(stats.net_max_per_unit_read_actions(), 16.0)
        self.assertAlmostEqual(stats.net_total_write_actions(), 64.0)
        self.assertAlmostEqual(stats.net_max_per_unit_write_actions(), 16.0)

        stats = result.buffet_stats[Buffet(level='MainMemory', tensor='Q', einsum='Q')]
        self.assertAlmostEqual(stats.net_total_read_actions(), 0)
        self.assertAlmostEqual(stats.net_max_per_unit_read_actions(), 0)
        self.assertAlmostEqual(stats.net_total_write_actions(), 16.0)
        self.assertAlmostEqual(stats.net_max_per_unit_write_actions(), 16.0)

        stats = result.buffet_stats[Buffet(level='LocalBuffer', tensor='Q', einsum='Q')]
        self.assertAlmostEqual(stats.net_total_read_actions(), 64.0)
        self.assertAlmostEqual(stats.net_max_per_unit_read_actions(), 16.0)
        self.assertAlmostEqual(stats.net_total_write_actions(), 64.0)
        self.assertAlmostEqual(stats.net_max_per_unit_write_actions(), 16.0)


class TestSymbolicActions(unittest.TestCase):
    def test_q_mapping(self):
        mapping = Mapping.from_yaml(Path(__file__).parent / 'Q_mapping.yaml')
        workload = Workload.from_yaml(Path(__file__).parent / 'mha.yaml')

        result = analyze_reuse_and_add_reservations_to_mapping(make_job(mapping, workload))
        actions = gather_actions(result, None, use_name=True)

        self.assertAlmostEqual(actions[('LocalBuffer', 'read')].total, 128.0)
        self.assertAlmostEqual(actions[('LocalBuffer', 'read')].max_per_unit, 32.0)
        self.assertAlmostEqual(actions[('LocalBuffer', 'write')].total, 128.0)
        self.assertAlmostEqual(actions[('LocalBuffer', 'write')].max_per_unit, 32.0)

        self.assertAlmostEqual(actions[('MAC', 'compute')].total, 64.0)
        self.assertAlmostEqual(actions[('MAC', 'compute')].max_per_unit, 16.0)


class TestSymbolicLatency(unittest.TestCase):
    def test_q_mapping(self):
        spec = Specification.from_yaml([
            # Path(__file__).parent / 'Q_mapping.yaml',
            Path(__file__).parent / 'mha.yaml',
            Path(__file__).parent / 'four_level.arch.yaml'
        ])
        workload = spec.workload
        architecture = spec.get_flattened_architecture()
        mapping = Mapping.from_yaml(Path(__file__).parent / 'Q_mapping.yaml')

        result = analyze_reuse_and_add_reservations_to_mapping(make_job(mapping, workload))
        overall_latency, _, _ = get_latency(result, mapping, workload, architecture)

        self.assertAlmostEqual(overall_latency, 16.0)

if __name__ == '__main__':
    unittest.main()