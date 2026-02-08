import unittest

from accelforge.frontend.spec import Spec
from accelforge.model.main import evaluate_mapping
from accelforge.util.parallel import set_n_parallel_jobs

try:
    from .paths import EXAMPLES_DIR
except ImportError:
    from paths import EXAMPLES_DIR

set_n_parallel_jobs(1)


class TestModel(unittest.TestCase):
    def test_one_matmul(self):
        M = 64
        KN = 32
        BITS_PER_VALUE = 8
        spec = Spec.from_yaml(
            EXAMPLES_DIR / "arches" / "simple.yaml",
            EXAMPLES_DIR / "workloads" / "matmuls.yaml",
            EXAMPLES_DIR / "mappings" / "unfused_matmuls_to_simple.yaml",
            jinja_parse_data={"N_EINSUMS": 1, "M": M, "KN": KN},
        )

        result = evaluate_mapping(spec)
        energy_breakdown = result.energy(per_component=True, per_tensor=True)
        self.assertAlmostEqual(
            energy_breakdown[("MainMemory", "T0")], M * KN * BITS_PER_VALUE
        )
        self.assertAlmostEqual(
            energy_breakdown[("MainMemory", "T1")], M * KN * BITS_PER_VALUE
        )
        self.assertAlmostEqual(
            energy_breakdown[("MainMemory", "W0")], M * KN**2 * BITS_PER_VALUE
        )

    def test_two_matmuls(self):
        spec = Spec.from_yaml(
            EXAMPLES_DIR / "arches" / "simple.yaml",
            EXAMPLES_DIR / "workloads" / "matmuls.yaml",
            EXAMPLES_DIR / "mappings" / "unfused_matmuls_to_simple.yaml",
            jinja_parse_data={"N_EINSUMS": 2, "M": 64, "KN": 64},
        )

        result = evaluate_mapping(spec)


if __name__ == "__main__":
    unittest.main()
