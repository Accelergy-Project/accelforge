from pathlib import Path
import unittest

from fastfusion.frontend import Workload


class TestWorkload(unittest.TestCase):
    def test_mha(self):
        workload = Workload.from_yaml(Path(__file__).parent / 'mha.yaml')

        TENSOR_NAMES_REF = {'I', 'WV', 'V', 'WK', 'K', 'WQ', 'Q', 'QK', 'AV',
                            'WZ', 'Z', 'WFFA', 'FFA', 'WFFB', 'FFB' }
        self.assertEqual(TENSOR_NAMES_REF, {t.name for t in workload.tensors})
