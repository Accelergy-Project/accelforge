import unittest
from pathlib import Path

from accelforge.frontend.workload import (
    Workload,
    _parse_einsum_string,
    _parse_einsum_entry,
)
from accelforge.frontend.spec import Spec

try:
    from ..paths import EXAMPLES_DIR
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from paths import EXAMPLES_DIR


class TestEinsumStringParser(unittest.TestCase):

    def test_simple_einsum_list_projection(self):
        einsum_str = "V[b, m, h, e] = I[b, m, d] * WV[h, e, d]"
        result = _parse_einsum_string(einsum_str)

        self.assertEqual(result["name"], "V")
        self.assertEqual(len(result["tensor_accesses"]), 3)

        input1 = result["tensor_accesses"][0]
        self.assertEqual(input1["name"], "I")
        self.assertEqual(input1["projection"], {"B": "b", "M": "m", "D": "d"})
        self.assertFalse(input1["output"])

        input2 = result["tensor_accesses"][1]
        self.assertEqual(input2["name"], "WV")
        self.assertEqual(input2["projection"], {"H": "h", "E": "e", "D": "d"})
        self.assertFalse(input2["output"])

        output = result["tensor_accesses"][2]
        self.assertEqual(output["name"], "V")
        self.assertEqual(output["projection"], {"B": "b", "M": "m", "H": "h", "E": "e"})
        self.assertTrue(output["output"])

    def test_einsum_with_dict_projection(self):
        einsum_str = "QK[b, m, p, h] = Q[b, m, h, e] * K[B:b, M:p, H:h, E:e]"
        result = _parse_einsum_string(einsum_str)

        self.assertEqual(result["name"], "QK")

        k_tensor = result["tensor_accesses"][1]
        self.assertEqual(k_tensor["name"], "K")
        self.assertEqual(
            k_tensor["projection"], {"B": "b", "M": "p", "H": "h", "E": "e"}
        )
        self.assertFalse(k_tensor["output"])

    def test_einsum_with_mixed_projection(self):
        """Test parsing einsum with mixed-style projection using equals."""
        einsum_str = "QK[b, m, p, h] = Q[b, m, h, e] * K[b, M:p, h, e]"
        result = _parse_einsum_string(einsum_str)

        # Check the K tensor with mixed projection
        k_tensor = result["tensor_accesses"][1]
        self.assertEqual(k_tensor["name"], "K")
        # Mixed style converts to dict with uppercase keys
        expected = {"B": "b", "M": "p", "H": "h", "E": "e"}
        self.assertEqual(k_tensor["projection"], expected)

    def test_einsum_single_input(self):
        """Test parsing einsum with a single input (like copy operation)."""
        einsum_str = "I[b, m, d] = I_in[b, m, d]"
        result = _parse_einsum_string(einsum_str)

        self.assertEqual(result["name"], "I")
        self.assertEqual(len(result["tensor_accesses"]), 2)

        input_tensor = result["tensor_accesses"][0]
        self.assertEqual(input_tensor["name"], "I_in")
        self.assertFalse(input_tensor["output"])

        output_tensor = result["tensor_accesses"][1]
        self.assertEqual(output_tensor["name"], "I")
        self.assertTrue(output_tensor["output"])

    def test_einsum_malformed_no_equals(self):
        """Test that parsing fails when there's no equals sign."""
        with self.assertRaises(ValueError) as ctx:
            _parse_einsum_string("V[b, m, h, e]")
        self.assertIn("Invalid einsum format", str(ctx.exception))

    def test_einsum_malformed_no_brackets(self):
        """Test that parsing fails when brackets are missing."""
        with self.assertRaises(ValueError) as ctx:
            _parse_einsum_string("V = I * WV")
        self.assertIn("Invalid einsum format", str(ctx.exception))

    def test_einsum_malformed_empty_projection(self):
        """Test that parsing fails with empty projection."""
        with self.assertRaises(ValueError) as ctx:
            _parse_einsum_string("V[] = I[b, m, d]")
        self.assertIn("cannot be empty", str(ctx.exception))

    def test_einsum_complex_expression(self):
        """Test parsing einsum with complex tensor names."""
        einsum_str = "QK_softmax[b, m, p, h] = QK[b, m, p, h]"
        result = _parse_einsum_string(einsum_str)

        self.assertEqual(result["name"], "QK_softmax")
        self.assertEqual(result["tensor_accesses"][0]["name"], "QK")


class TestInvalidInputs(unittest.TestCase):
    """Test that invalid inputs raise appropriate errors."""

    def test_einsum_no_output_tensor(self):
        """Test that missing output tensor raises an error."""
        with self.assertRaises(ValueError) as ctx:
            _parse_einsum_string("= I[b] * W[b]")
        self.assertIn("Invalid einsum format", str(ctx.exception))

    def test_einsum_no_input_tensors(self):
        """Test that missing input tensors raises an error."""
        with self.assertRaises(ValueError) as ctx:
            _parse_einsum_string("V[b] =")
        self.assertIn("Invalid einsum format", str(ctx.exception))

    def test_einsum_unmatched_brackets(self):
        """Test that unmatched brackets raise an error."""
        with self.assertRaises(ValueError) as ctx:
            _parse_einsum_string("V[b, m = I[b] * W[m]")
        self.assertIn("Invalid einsum format", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            _parse_einsum_string("V[b, m] = I[b * W[m]")
        self.assertIn("Invalid projection element", str(ctx.exception))

    def test_einsum_empty_tensor_name(self):
        """Test that empty tensor names raise an error."""
        with self.assertRaises(ValueError) as ctx:
            _parse_einsum_string("[b, m] = I[b] * W[m]")
        self.assertIn("Invalid einsum format", str(ctx.exception))

    def test_projection_invalid_identifier(self):
        with self.assertRaises(ValueError):
            _parse_einsum_string("V[b, 123, m] = I[b]")

        with self.assertRaises(ValueError):
            _parse_einsum_string("V[b, m-n] = I[b]")

    def test_projection_empty_key_or_value(self):
        with self.assertRaises(ValueError):
            _parse_einsum_string("V[b, m] = I[=b]")

        with self.assertRaises(ValueError):
            _parse_einsum_string("V[b, m] = I[B=]")

    def test_workload_duplicate_einsum_names(self):
        """Test that duplicate einsum names raise an error."""
        workload_data = {
            "rank_sizes": {"B": 4, "M": 8},
            "bits_per_value": {"All": 8},
            "einsums": [
                "V[b, m] = I[b]",
                "V[b, m] = X[b]",  # Duplicate name 'V'
            ],
        }

        with self.assertRaises(ValueError) as ctx:
            Workload(**workload_data)
        self.assertIn("not unique", str(ctx.exception))

    def test_workload_inconsistent_tensor_ranks(self):
        """Test that inconsistent ranks for same tensor raise an error."""
        workload_data = {
            "rank_sizes": {"B": 4, "M": 8, "D": 16},
            "bits_per_value": {"All": 8},
            "einsums": [
                "V[b, m] = I[b]",
                "K[b, m, d] = I[b, d]",  # I has different ranks (b vs b,d)
            ],
        }

        with self.assertRaises(ValueError) as ctx:
            Workload(**workload_data)
        self.assertIn("inconsistent ranks", str(ctx.exception))

    def test_einsum_entry_invalid_type(self):
        """Test that invalid types for einsum entries raise an error."""
        with self.assertRaises(ValueError) as ctx:
            _parse_einsum_entry(123)
        self.assertIn("must be dicts, strings, or Einsum objects.", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            _parse_einsum_entry([1, 2, 3])
        self.assertIn("must be dicts, strings, or Einsum objects", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            _parse_einsum_entry(None)
        self.assertIn("must be dicts, strings, or Einsum objects.", str(ctx.exception))

    def test_projection_with_special_characters(self):
        """Test that special characters in identifiers raise errors."""
        invalid_chars = ["@", "#", "$", "%", "!", "&", "*", "(", ")", "+"]

        for char in invalid_chars:
            with self.assertRaises(ValueError) as ctx:
                _parse_einsum_string(f"V[b{char}m] = I[b]")
            self.assertIn("Invalid projection element:", str(ctx.exception))

    def test_einsum_whitespace_only_tensor_name(self):
        """Test that whitespace-only tensor names raise an error."""
        with self.assertRaises(ValueError) as ctx:
            _parse_einsum_string("   [b, m] = I[b]")
        self.assertIn("Invalid einsum format", str(ctx.exception))

    def test_projection_trailing_comma(self):
        """Test that trailing commas in projection raise an error."""
        # Trailing commas create empty elements which should fail
        with self.assertRaises(ValueError) as ctx:
            _parse_einsum_string("V[b, m, ] = I[b]")
        self.assertIn("Invalid projection element", str(ctx.exception))

    def test_projection_double_comma(self):
        """Test that double commas (empty elements) raise an error."""
        with self.assertRaises(ValueError) as ctx:
            _parse_einsum_string("V[b,, m] = I[b]")
        self.assertIn("Invalid projection element", str(ctx.exception))

    def test_workload_reserved_einsum_name(self):
        """Test that reserved einsum name 'Total' raises an error."""
        workload_data = {
            "rank_sizes": {"B": 4},
            "bits_per_value": {"All": 8},
            "einsums": [
                "Total[b] = I[b]",  # 'Total' is reserved
            ],
        }

        with self.assertRaises(ValueError) as ctx:
            Workload(**workload_data)
        self.assertIn("reserved", str(ctx.exception).lower())

    def test_malformed_yaml_structure(self):
        """Test that malformed workload structures are caught."""
        from pydantic import ValidationError

        # Workload with invalid rank_sizes type should fail validation
        with self.assertRaises(ValidationError):
            Workload(rank_sizes="not a dict", bits_per_value={"All": 8}, einsums=[])

    def test_einsum_with_curly_braces(self):
        """Test that curly braces instead of square brackets raise an error."""
        with self.assertRaises(ValueError) as ctx:
            _parse_einsum_string("V{b, m} = I{b}")
        self.assertIn("Invalid einsum format", str(ctx.exception))

    def test_projection_key_not_identifier(self):
        with self.assertRaises(ValueError):
            _parse_einsum_string("V[b] = I[1B=b]")

        with self.assertRaises(ValueError):
            _parse_einsum_string("V[b] = I[B-2=b]")

    def test_einsum_only_whitespace(self):
        """Test that whitespace-only einsum string raises an error."""
        with self.assertRaises(ValueError) as ctx:
            _parse_einsum_string("   ")
        self.assertIn("Einsum string cannot be empty", str(ctx.exception))

    def test_einsum_empty_string(self):
        """Test that empty einsum string raises an error."""
        with self.assertRaises(ValueError) as ctx:
            _parse_einsum_string("")
        self.assertIn("Einsum string cannot be empty", str(ctx.exception))

    def test_projection_only_spaces(self):
        """Test projection with only spaces between commas."""
        with self.assertRaises(ValueError) as ctx:
            _parse_einsum_string("V[b,  , m] = I[b]")
        self.assertIn("Invalid projection element:", str(ctx.exception))

    def test_tensor_name_with_numbers_ok(self):
        """Test that tensor names with numbers are valid."""
        # This should work - numbers are OK in identifiers (just not at start)
        result = _parse_einsum_string("V2[b, m] = I1[b] * W2[m]")
        self.assertEqual(result["name"], "V2")
        input_names = [t["name"] for t in result["tensor_accesses"] if not t["output"]]
        self.assertIn("I1", input_names)
        self.assertIn("W2", input_names)

    def test_tensor_name_starting_with_number_invalid(self):
        """Test that tensor names starting with numbers are invalid."""
        with self.assertRaises(ValueError):
            _parse_einsum_string("2V[b] = I[b]")

    def test_nested_brackets(self):
        """Test that nested brackets raise an error."""
        with self.assertRaises(ValueError):
            _parse_einsum_string("V[[b]] = I[b]")
        # This will fail because the projection parsing expects valid content

    def test_unicode_characters_in_names(self):
        """Test that unicode characters in tensor names work (if valid identifiers)."""
        # Python allows unicode in identifiers
        result = _parse_einsum_string("Vélocity[b] = Iñput[b]")
        self.assertEqual(result["name"], "Vélocity")

    def test_very_long_tensor_name(self):
        """Test that very long tensor names are handled."""
        long_name = "A" * 1000
        result = _parse_einsum_string(f"{long_name}[b] = I[b]")
        self.assertEqual(result["name"], long_name)

    def test_projection_equals_without_space(self):
        """Test mixed projection with no spaces around equals."""
        result = _parse_einsum_string("V[b] = I[B:b]")
        self.assertEqual(result["tensor_accesses"][0]["projection"], {"B": "b"})

    def test_multiple_asterisks(self):
        """Test that multiple input tensors separated by * work."""
        result = _parse_einsum_string("Y[b] = A[b] * B[b] * C[b] * D[b]")
        self.assertEqual(len(result["tensor_accesses"]), 5)  # 4 inputs + 1 output
        input_names = [t["name"] for t in result["tensor_accesses"] if not t["output"]]
        self.assertEqual(input_names, ["A", "B", "C", "D"])

    def test_asterisk_with_extra_spaces(self):
        """Test that extra spaces around * are handled."""
        result = _parse_einsum_string("V[b] = I[b]    *    W[b]")
        self.assertEqual(len(result["tensor_accesses"]), 3)


class TestEinsumEntryParser(unittest.TestCase):
    """Test parsing of einsum entries (dict or string)."""

    def test_parse_string_entry(self):
        entry = {"einsum": "V[b, m, h, e] = I[b, m, d] * WV[h, e, d]"}
        result = _parse_einsum_entry(entry)

        self.assertEqual(result["name"], "V")
        self.assertEqual(len(result["tensor_accesses"]), 3)

    def test_parse_dict_entry_with_einsum_key(self):
        entry = {
            "einsum": "QK[b, m, p, h] = Q[b, m, h, e] * K[b, M:p, h, e]",
        }
        result = _parse_einsum_entry(entry)

        self.assertEqual(result["name"], "QK")

    def test_parse_dict_entry_full_format(self):
        entry = {
            "name": "V",
            "tensor_accesses": [
                {"name": "I", "projection": ["b", "m", "d"]},
                {"name": "WV", "projection": ["h", "e", "d"]},
                {"name": "V", "projection": ["b", "m", "h", "e"], "output": True},
            ],
        }
        result = _parse_einsum_entry(entry)

        self.assertEqual(result["name"], "V")
        self.assertEqual(len(result["tensor_accesses"]), 3)

    def test_invalid_entry_type(self):
        with self.assertRaises(ValueError):
            _parse_einsum_entry(123)


class TestWorkloadYAMLParsing(unittest.TestCase):
    """Test parsing of complete workload YAML files."""

    def test_parse_concise_yaml(self):
        """Test parsing the concise GPT-3 6.7B workload YAML."""
        yaml_path = EXAMPLES_DIR / "workloads" / "gpt3_6.7B.yaml"

        # Load and parse the spec
        spec = Spec.from_yaml(yaml_path)
        workload = spec.workload

        # Check that workload was parsed correctly
        self.assertIsNotNone(workload)
        self.assertGreater(len(workload.einsums), 0)

        # Check rank sizes
        self.assertIn("H", workload.rank_sizes)
        self.assertEqual(workload.rank_sizes["H"], 32)

        # Check that einsums were created
        einsum_names = [e.name for e in workload.einsums]
        self.assertIn("I", einsum_names)
        self.assertIn("V", einsum_names)
        self.assertIn("K", einsum_names)
        self.assertIn("Q", einsum_names)
        self.assertIn("QK", einsum_names)

        # Check a specific einsum
        v_einsum = [e for e in workload.einsums if e.name == "V"][0]
        self.assertEqual(len(v_einsum.tensor_accesses), 3)

        # Check input tensors
        input_names = [t.name for t in v_einsum.tensor_accesses if not t.output]
        self.assertIn("I", input_names)
        self.assertIn("WV", input_names)

        # Check output tensor
        output_names = [t.name for t in v_einsum.tensor_accesses if t.output]
        self.assertIn("V", output_names)

    def test_persistent_tensors_in_concise_yaml(self):
        """Test that persistent_tensors field marks weight tensors as persistent."""
        yaml_path = EXAMPLES_DIR / "workloads" / "gpt3_6.7B.yaml"

        # Load and parse the spec
        spec = Spec.from_yaml(yaml_path)
        spec = spec._spec_eval_expressions()
        workload = spec.workload

        # Check that weight tensors are marked as persistent
        weight_tensors = ["WV", "WK", "WQ", "WZ", "WFFA", "WFFB"]

        for einsum in workload.einsums:
            for tensor_access in einsum.tensor_accesses:
                if tensor_access.name in weight_tensors:
                    self.assertTrue(
                        tensor_access.persistent,
                        f"Tensor {tensor_access.name} in einsum {einsum.name} should be persistent",
                    )

        # Check that non-weight tensors are not marked as persistent (unless explicitly set)
        v_einsum = [e for e in workload.einsums if e.name == "V"][0]
        i_tensor = [t for t in v_einsum.tensor_accesses if t.name == "I"][0]
        # I tensor should not be persistent (it's an intermediate)
        # Note: this might be False or not set depending on the YAML
        self.assertFalse(i_tensor.persistent)

    def test_parse_regular_yaml(self):
        """Test parsing the regular GPT-3 6.7B workload YAML."""
        yaml_path = EXAMPLES_DIR / "workloads" / "gpt3_6.7B.yaml"

        # Load and parse the spec
        spec = Spec.from_yaml(yaml_path)
        workload = spec.workload

        # Check that workload was parsed correctly
        self.assertIsNotNone(workload)
        self.assertGreater(len(workload.einsums), 0)

        # Check rank sizes
        self.assertIn("H", workload.rank_sizes)
        self.assertEqual(workload.rank_sizes["H"], 32)

        # Check that einsums were created
        einsum_names = [e.name for e in workload.einsums]
        self.assertIn("I", einsum_names)
        self.assertIn("V", einsum_names)
        self.assertIn("K", einsum_names)
        self.assertIn("Q", einsum_names)
        self.assertIn("QK", einsum_names)

    def test_persistent_tensors_in_regular_yaml(self):
        """Test that persistent flags in regular YAML are preserved."""
        yaml_path = EXAMPLES_DIR / "workloads" / "gpt3_6.7B.yaml"

        # Load and parse the spec
        spec = Spec.from_yaml(yaml_path)
        spec = spec._spec_eval_expressions()
        workload = spec.workload

        # Check that weight tensors have persistent=True explicitly set
        weight_tensors = ["WV", "WK", "WQ", "WZ", "WFFA", "WFFB"]

        for einsum in workload.einsums:
            for tensor_access in einsum.tensor_accesses:
                if tensor_access.name in weight_tensors:
                    self.assertTrue(
                        tensor_access.persistent,
                        f"Tensor {tensor_access.name} in einsum {einsum.name} should be persistent",
                    )

    def test_concise_and_regular_same_structure(self):
        concise_path = EXAMPLES_DIR / "workloads" / "gpt3_6.7B.yaml"
        regular_path = EXAMPLES_DIR / "misc" / "gpt3_6.7B_verbose_annotated.yaml"

        concise_spec = Spec.from_yaml(concise_path)
        concise_spec = concise_spec._spec_eval_expressions()
        regular_spec = Spec.from_yaml(regular_path)
        regular_spec = regular_spec._spec_eval_expressions()

        concise_workload = concise_spec.workload
        regular_workload = regular_spec.workload

        concise_names = set(e.name for e in concise_workload.einsums)
        regular_names = set(e.name for e in regular_workload.einsums)
        self.assertEqual(concise_names, regular_names)

        self.assertEqual(
            set(concise_workload.rank_sizes.keys()),
            set(regular_workload.rank_sizes.keys()),
        )


if __name__ == "__main__":
    unittest.main()
