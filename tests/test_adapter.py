import unittest
from pathlib import Path

import accelforge as af

INPUT_FILES_DIR = Path(__file__).parent / "input_files" / "adapters"


class TestParsing(unittest.TestCase):
    def test_gpt3(self):
        spec = af.Spec.from_yaml(INPUT_FILES_DIR / "gpt3_6.7B.yaml")
        self.assertEqual(
            spec.workload.einsum_names,
            ["V", "K", "Q", "QK", "QK_softmax", "AV", "Z", "FFA", "FFB"],
        )


class TestMangling(unittest.TestCase):
    def setUp(self):
        self.spec = af.Spec.from_yaml(INPUT_FILES_DIR / "gpt3_6.7B.yaml")
        self.workload = self.spec.workload
        self.adapted = self.workload.get_adapted_workload()

    def test_gpt3(self):
        self.assertIn("copy_I__I", self.adapted.einsums["Q"].input_tensor_names)

    def test_all_consumers_of_adapted_tensor_are_mangled(self):
        # Every Einsum that read the original input I should now read the mangled
        # name instead, and should no longer reference I directly.
        for name in ["V", "K", "Q"]:
            inputs = self.adapted.einsums[name].input_tensor_names
            self.assertIn("copy_I__I", inputs)
            self.assertNotIn("I", inputs)

    def test_copy_einsum_is_inserted(self):
        # The adapter is lowered into a copy Einsum named after the adapter.
        self.assertIn("copy_I", self.adapted.einsum_names)
        copy_einsum = self.adapted.einsums["copy_I"]
        self.assertTrue(copy_einsum.is_copy_operation)
        self.assertEqual(copy_einsum.input_tensor_names, {"I"})
        self.assertEqual(copy_einsum.output_tensor_names, {"copy_I__I"})

    def test_copy_einsum_mirrors_original_projection(self):
        # The copy reads/writes the same ranks the original tensor was accessed by.
        copy_einsum = self.adapted.einsums["copy_I"]
        src = next(t for t in copy_einsum.tensor_accesses if t.name == "I")
        dst = next(t for t in copy_einsum.tensor_accesses if t.name == "copy_I__I")
        self.assertEqual(set(src.ranks), {"B", "M", "D"})
        self.assertEqual(set(dst.ranks), {"B", "M", "D"})

    def test_original_tensor_only_remains_on_copy(self):
        # After adapting, the original I is produced/consumed only by the copy
        # Einsum; downstream Einsums use the mangled name.
        einsums_with_I = [e.name for e in self.adapted.einsums_with_tensor("I")]
        self.assertEqual(einsums_with_I, ["copy_I"])

    def test_downstream_einsums_unaffected(self):
        # Tensors unrelated to the adapter keep their names.
        qk_inputs = self.adapted.einsums["QK"].input_tensor_names
        self.assertEqual(qk_inputs, {"Q", "K"})

    def test_einsum_order_preserved(self):
        self.assertEqual(
            self.adapted.einsum_names,
            ["copy_I", "V", "K", "Q", "QK", "QK_softmax", "AV", "Z", "FFA", "FFB"],
        )

    def test_original_workload_unchanged(self):
        # get_adapted_workload returns a copy; the source workload is untouched.
        self.assertIn("I", self.workload.einsums["Q"].input_tensor_names)
        self.assertNotIn("copy_I__I", self.workload.einsums["Q"].input_tensor_names)
        self.assertNotIn("copy_I", self.workload.einsum_names)
