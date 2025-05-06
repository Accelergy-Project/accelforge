from pathlib import Path
import unittest

from fastfusion.frontend import Workload
from fastfusion.frontend.workload import get_rank_variable_bounds


class TestWorkload(unittest.TestCase):
    def test_mha(self):
        workload = Workload.from_yaml(Path(__file__).parent / 'mha.yaml')

        TENSOR_NAMES_REF = {'I', 'WV', 'V', 'WK', 'K', 'WQ', 'Q', 'QK', 'AV',
                            'WZ', 'Z', 'WFFA', 'FFA', 'WFFB', 'FFB' }
        self.assertEqual(TENSOR_NAMES_REF, {t.name for t in workload.tensors})

        rank_variable_bounds = get_rank_variable_bounds(workload, 'Q')
        REF_RANK_VARIABLE_BOUNDS = {
            'm': 1, 'b': 1, 'h': 1, 'e': 1, 'd': 1
        }
        self.assertEqual(REF_RANK_VARIABLE_BOUNDS, rank_variable_bounds)
