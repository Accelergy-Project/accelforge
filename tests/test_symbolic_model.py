import unittest
from pathlib import Path

from fastfusion.frontend.mapping import Mapping
from fastfusion.frontend.workload import Workload
from fastfusion.model.looptree.reuse.summarized.symbolic_new import analyze_reuse


class TestSymbolicModel(unittest.TestCase):
    def test_q_mapping(self):
        mapping = Mapping.from_yaml(Path(__file__).parent / 'Q_mapping.yaml')
        workload = Workload.from_yaml(Path(__file__).parent / 'mha.yaml')

        result = analyze_reuse(mapping, workload)

        self.assertEqual(result.per_einsum_ops['Q'], 64)

    def test_conv_mapping(self):
        mapping = Mapping.from_yaml(Path(__file__).parent / 'conv.mapping.yaml')
        workload = Workload.from_yaml(Path(__file__).parent / 'conv.workload.yaml')

        result = analyze_reuse(mapping, workload)

        self.assertEqual(result.per_einsum_ops['conv'], 120)