import unittest
from pathlib import Path

import accelforge as af

INPUT_FILES_DIR = Path(__file__).parent / "input_files" / "adapters"


class TestParsing(unittest.TestCase):
    def test_gpt3(self):
        spec = af.Spec.from_yaml(
            INPUT_FILES_DIR / "gpt3_6.7B.yaml"
        )
        self.assertEqual(spec.workload.einsum_names, ["V", "K", "Q", "QK_softmax", "Z", "FFA", "FFB"])

