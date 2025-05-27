from pathlib import Path
import unittest

from fastfusion.frontend import Workload
from fastfusion.frontend.workload import *


class TestWorkload(unittest.TestCase):
    def test_mha(self):
        workload = Workload.from_yaml(Path(__file__).parent / 'mha.yaml')

        TENSOR_NAMES_REF = {'I', 'WV', 'V', 'WK', 'K', 'WQ', 'Q', 'QK', 'AV',
                            'WZ', 'Z', 'WFFA', 'FFA', 'WFFB', 'FFB' }
        self.assertEqual(TENSOR_NAMES_REF, {t for t in workload.tensors})

        rank_variable_bounds = get_rank_variable_bounds(workload, 'Q')
        REF_RANK_VARIABLE_BOUNDS = {
            'm': 2, 'b': 2, 'h': 2, 'e': 2, 'd': 4
        }
        self.assertEqual(REF_RANK_VARIABLE_BOUNDS, rank_variable_bounds)

        REF_TENSOR_Q_SIZE = 16  # B*H*M*E
        self.assertEqual(REF_TENSOR_Q_SIZE,
                         get_tensor_size(workload, TensorName('Q')))
