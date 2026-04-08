import unittest

import accelforge as af
from accelforge.frontend.spec import Spec
from accelforge.model.main import evaluate_mapping, InvalidMappingError
from accelforge.util.parallel import set_n_parallel_jobs

try:
    from .paths import EXAMPLES_DIR
except ImportError:
    from paths import EXAMPLES_DIR

set_n_parallel_jobs(1)


class TestModelWithBinding(unittest.TestCase):
    def test_matmuls(self):
        spec = af.Spec.from_yaml(
            af.examples.arches.simple_heterogeneous,
            af.examples.workloads.matmuls,
            af.examples.mappings.with_binding.fused_matmuls_to_simple_heterogeneous,
            jinja_parse_data={"N_EINSUMS": 3},
        )
        spec.mapper.n_concurrent_threads = 2

        mappings = spec.evaluate_mapping()
        print(mappings["Total<SEP>latency"])


class TestInvalidMapping(unittest.TestCase):
    def test_matmul_to_simple(self):
        M = 64
        KN = 64
        spec = Spec.from_yaml(
            EXAMPLES_DIR / "arches" / "simple.yaml",
            EXAMPLES_DIR / "workloads" / "matmuls.yaml",
            "tests/input_files/mapping/invalid_matmul_to_simple.yaml",
            jinja_parse_data={"N_EINSUMS": 1, "M": M, "KN": KN},
        )
        self.assertRaises(InvalidMappingError, lambda: evaluate_mapping(spec))


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
            energy_breakdown[("MainMemory", "W0")], KN**2 * BITS_PER_VALUE
        )

    def test_skip_initial_output_write_false(self):
        M = 64
        KN = 32
        BITS_PER_VALUE = 8
        spec = Spec.from_yaml(
            EXAMPLES_DIR / "arches" / "simple.yaml",
            EXAMPLES_DIR / "workloads" / "matmuls.yaml",
            EXAMPLES_DIR / "mappings" / "unfused_matmuls_to_simple.yaml",
            jinja_parse_data={"N_EINSUMS": 1, "M": M, "KN": KN},
        )

        spec.arch["MainMemory"].skip_initial_output_write = False

        result = evaluate_mapping(spec)
        energy_breakdown = result.energy(per_component=True, per_tensor=True)
        # Input and weight energy unchanged
        self.assertAlmostEqual(
            energy_breakdown[("MainMemory", "T0")], M * KN * BITS_PER_VALUE
        )
        self.assertAlmostEqual(
            energy_breakdown[("MainMemory", "W0")], KN**2 * BITS_PER_VALUE
        )
        # Output energy doubles: the initial fill from MainMemory is no longer
        # skipped, so MainMemory sees both the fill read and the writeback write.
        self.assertAlmostEqual(
            energy_breakdown[("MainMemory", "T1")], 2 * M * KN * BITS_PER_VALUE
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
