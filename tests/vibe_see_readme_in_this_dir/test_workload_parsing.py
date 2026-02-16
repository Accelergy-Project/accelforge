"""
Tests for workload parsing functions:
  - _parse_einsum_string
  - _parse_einsum_entry
  - _parse_projection
  - _projection_factory
  - isl_expression_has_variable
"""

import unittest

from accelforge.frontend.workload import (
    _parse_einsum_string,
    _parse_einsum_entry,
    _parse_projection,
    _projection_factory,
    isl_expression_has_variable,
    ImpliedProjection,
    TensorAccess,
)


# ============================================================================
# _parse_projection
# ============================================================================


class TestParseProjection(unittest.TestCase):
    """Tests for _parse_projection which converts a projection string to dict/list."""

    def test_simple_identifiers_become_dict(self):
        result = _parse_projection("a, b, c")
        self.assertEqual(result, {"A": "a", "B": "b", "C": "c"})

    def test_explicit_equals_projection(self):
        result = _parse_projection("B:b, M:p, H:h, E:e")
        self.assertEqual(result, {"B": "b", "M": "p", "H": "h", "E": "e"})

    def test_mixed_implicit_and_explicit(self):
        result = _parse_projection("b, M:p, h, e")
        self.assertEqual(result, {"B": "b", "M": "p", "H": "h", "E": "e"})

    def test_single_element(self):
        result = _parse_projection("x")
        self.assertEqual(result, {"X": "x"})

    def test_single_explicit(self):
        result = _parse_projection("X:y")
        self.assertEqual(result, {"X": "y"})

    def test_empty_raises_value_error(self):
        with self.assertRaises(ValueError):
            _parse_projection("")

    def test_invalid_element_raises(self):
        with self.assertRaises(ValueError):
            _parse_projection("123")

    def test_element_with_operator_raises(self):
        with self.assertRaises(ValueError):
            _parse_projection("a+b")

    def test_trailing_comma_raises(self):
        with self.assertRaises(ValueError):
            _parse_projection("a, b, ")

    def test_double_comma_raises(self):
        with self.assertRaises(ValueError):
            _parse_projection("a,, b")

    def test_spaces_stripped(self):
        result = _parse_projection("  a ,  b ,  c  ")
        self.assertEqual(result, {"A": "a", "B": "b", "C": "c"})


# ============================================================================
# _projection_factory
# ============================================================================


class TestProjectionFactory(unittest.TestCase):
    """Tests for _projection_factory which handles list/dict projection input."""

    def test_list_projection_creates_implied(self):
        result = _projection_factory(["a", "b", "c"])
        self.assertIsInstance(result, ImpliedProjection)
        self.assertEqual(result, {"A": "a", "B": "b", "C": "c"})

    def test_dict_projection_returned_as_is(self):
        d = {"M": "m", "N": "n"}
        result = _projection_factory(d)
        self.assertEqual(result, {"M": "m", "N": "n"})
        self.assertNotIsInstance(result, ImpliedProjection)

    def test_invalid_type_raises(self):
        with self.assertRaises(TypeError):
            _projection_factory(123)

    def test_invalid_type_string_raises(self):
        with self.assertRaises(TypeError):
            _projection_factory("abc")

    def test_list_with_non_string_raises(self):
        with self.assertRaises(TypeError):
            _projection_factory([1, 2, 3])

    def test_list_with_expression_raises(self):
        with self.assertRaises(ValueError):
            _projection_factory(["a+b"])

    def test_dict_with_non_string_key_raises(self):
        with self.assertRaises(TypeError):
            _projection_factory({1: "a"})

    def test_dict_with_non_identifier_key_raises(self):
        with self.assertRaises(ValueError):
            _projection_factory({"1invalid": "a"})

    def test_empty_list(self):
        result = _projection_factory([])
        self.assertIsInstance(result, ImpliedProjection)
        self.assertEqual(len(result), 0)

    def test_empty_dict(self):
        result = _projection_factory({})
        self.assertEqual(len(result), 0)


# ============================================================================
# isl_expression_has_variable
# ============================================================================


class TestIslExpressionHasVariable(unittest.TestCase):
    """Tests for isl_expression_has_variable."""

    def test_simple_variable_present(self):
        self.assertTrue(isl_expression_has_variable("0 <= m < 10", "m"))

    def test_simple_variable_absent(self):
        self.assertFalse(isl_expression_has_variable("0 <= m < 10", "n"))

    def test_variable_in_expression(self):
        self.assertTrue(isl_expression_has_variable("a + b < 10", "a"))
        self.assertTrue(isl_expression_has_variable("a + b < 10", "b"))

    def test_operator_not_matched(self):
        """ISL operators like EQ, NE, etc. should not be matched."""
        self.assertFalse(isl_expression_has_variable("EQ", "EQ"))
        self.assertFalse(isl_expression_has_variable("NE", "NE"))
        self.assertFalse(isl_expression_has_variable("AND", "AND"))

    def test_partial_match_not_confused(self):
        """Variable 'm' should not match 'mm' or 'my_var'."""
        self.assertFalse(isl_expression_has_variable("mm + 5", "m"))

    def test_underscore_variable(self):
        self.assertTrue(isl_expression_has_variable("my_var < 5", "my_var"))

    def test_multiple_occurrences(self):
        self.assertTrue(isl_expression_has_variable("m + m < 10", "m"))


# ============================================================================
# _parse_einsum_string
# ============================================================================


class TestParseEinsumString(unittest.TestCase):
    """Tests for _parse_einsum_string."""

    def test_basic_matmul(self):
        result = _parse_einsum_string("C[m, n] = A[m, k] * B[k, n]")
        self.assertEqual(result["name"], "C")
        self.assertEqual(len(result["tensor_accesses"]), 3)

        inputs = [t for t in result["tensor_accesses"] if not t["output"]]
        outputs = [t for t in result["tensor_accesses"] if t["output"]]
        self.assertEqual(len(inputs), 2)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]["name"], "C")

    def test_three_input_tensors(self):
        result = _parse_einsum_string("Y[b] = A[b] * B[b] * C[b]")
        inputs = [t for t in result["tensor_accesses"] if not t["output"]]
        self.assertEqual(len(inputs), 3)
        self.assertEqual([t["name"] for t in inputs], ["A", "B", "C"])

    def test_single_input_tensor(self):
        result = _parse_einsum_string("O[b, m] = I[b, m]")
        self.assertEqual(result["name"], "O")
        self.assertEqual(len(result["tensor_accesses"]), 2)

    def test_dict_projection(self):
        result = _parse_einsum_string(
            "QK[b, m, p, h] = Q[b, m, h, e] * K[B:b, M:p, H:h, E:e]"
        )
        k_tensor = result["tensor_accesses"][1]
        self.assertEqual(k_tensor["name"], "K")
        self.assertEqual(
            k_tensor["projection"], {"B": "b", "M": "p", "H": "h", "E": "e"}
        )

    def test_mixed_projection(self):
        result = _parse_einsum_string(
            "QK[b, m, p, h] = Q[b, m, h, e] * K[b, M:p, h, e]"
        )
        k_tensor = result["tensor_accesses"][1]
        expected = {"B": "b", "M": "p", "H": "h", "E": "e"}
        self.assertEqual(k_tensor["projection"], expected)


    def test_copy_operation(self):
        result = _parse_einsum_string("I[b, m, d] = I_in[b, m, d]")
        self.assertEqual(result["name"], "I")
        self.assertEqual(result["tensor_accesses"][0]["name"], "I_in")

    def test_underscore_in_name(self):
        result = _parse_einsum_string("QK_softmax[b] = QK[b]")
        self.assertEqual(result["name"], "QK_softmax")

    def test_whitespace_handling(self):
        result = _parse_einsum_string("  C[m, n]  =  A[m, k]  *  B[k, n]  ")
        self.assertEqual(result["name"], "C")
        self.assertEqual(len(result["tensor_accesses"]), 3)

    def test_empty_string_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parse_einsum_string("")
        self.assertIn("cannot be empty", str(ctx.exception))

    def test_whitespace_only_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parse_einsum_string("   ")
        self.assertIn("cannot be empty", str(ctx.exception))

    def test_no_equals_raises(self):
        with self.assertRaises(ValueError):
            _parse_einsum_string("V[b, m, h, e]")

    def test_no_brackets_raises(self):
        with self.assertRaises(ValueError):
            _parse_einsum_string("V = I * WV")

    def test_no_input_tensors_raises(self):
        with self.assertRaises(ValueError):
            _parse_einsum_string("V[b] =")

    def test_empty_projection_raises(self):
        with self.assertRaises(ValueError):
            _parse_einsum_string("V[] = I[b]")

    def test_tensor_name_starting_with_number_raises(self):
        with self.assertRaises(ValueError):
            _parse_einsum_string("2V[b] = I[b]")

    def test_curly_braces_raise(self):
        with self.assertRaises(ValueError):
            _parse_einsum_string("V{b, m} = I{b}")

    def test_unmatched_brackets_raise(self):
        with self.assertRaises(ValueError):
            _parse_einsum_string("V[b, m = I[b] * W[m]")

    def test_nested_brackets_raise(self):
        with self.assertRaises(ValueError):
            _parse_einsum_string("V[[b]] = I[b]")

    def test_numbers_in_tensor_name_ok(self):
        result = _parse_einsum_string("T2[b] = T1[b] * W1[b]")
        self.assertEqual(result["name"], "T2")

    def test_very_long_tensor_name(self):
        long = "A" * 100
        result = _parse_einsum_string(f"{long}[b] = I[b]")
        self.assertEqual(result["name"], long)


# ============================================================================
# _parse_einsum_entry
# ============================================================================


class TestParseEinsumEntry(unittest.TestCase):
    """Tests for _parse_einsum_entry."""

    def test_entry_with_einsum_key(self):
        entry = {"einsum": "V[b, m, h, e] = I[b, m, d] * WV[h, e, d]"}
        result = _parse_einsum_entry(entry)
        self.assertEqual(result["name"], "V")
        self.assertEqual(len(result["tensor_accesses"]), 3)

    def test_entry_with_einsum_and_renames(self):
        entry = {
            "einsum": "QK[b, m, p, h] = Q[b, m, h, e] * K[b, M:p, h, e]",
            "renames": {"weight": "K", "input": "Q"},
        }
        result = _parse_einsum_entry(entry)
        self.assertEqual(result["name"], "QK")
        self.assertIn("weight", result["renames"])
        self.assertIn("input", result["renames"])

    def test_entry_with_einsum_and_tensor_accesses_extras(self):
        entry = {
            "einsum": "I[b, m, d] = I_in[b, m, d]",
            "tensor_accesses": [{"name": "I_in", "bits_per_value": 16}],
        }
        result = _parse_einsum_entry(entry)
        self.assertEqual(result["name"], "I")
        i_in = [t for t in result["tensor_accesses"] if t["name"] == "I_in"][0]
        self.assertEqual(i_in["bits_per_value"], 16)

    def test_entry_without_einsum_key_passes_through(self):
        entry = {
            "name": "V",
            "tensor_accesses": [
                {"name": "I", "projection": ["b", "m", "d"]},
                {"name": "V", "projection": ["b", "m"], "output": True},
            ],
        }
        result = _parse_einsum_entry(entry)
        self.assertEqual(result["name"], "V")

    def test_invalid_type_raises(self):
        with self.assertRaises(ValueError):
            _parse_einsum_entry(123)

        with self.assertRaises(ValueError):
            _parse_einsum_entry([1, 2, 3])

        with self.assertRaises(ValueError):
            _parse_einsum_entry(None)

    def test_conflicting_tensor_access_attribute_raises(self):
        """If the einsum string sets a field and the extra tensor_accesses also set it, error."""
        entry = {
            "einsum": "V[b, m] = I[b, m]",
            "tensor_accesses": [{"name": "I", "projection": {"B": "b", "M": "m"}}],
        }
        with self.assertRaises(ValueError) as ctx:
            _parse_einsum_entry(entry)
        self.assertIn("already set", str(ctx.exception))

    def test_tensor_access_not_in_einsum_raises(self):
        entry = {
            "einsum": "V[b] = I[b]",
            "tensor_accesses": [{"name": "NotHere", "bits_per_value": 8}],
        }
        with self.assertRaises(ValueError) as ctx:
            _parse_einsum_entry(entry)
        self.assertIn("not found", str(ctx.exception))

    def test_tensor_access_missing_name_raises(self):
        entry = {
            "einsum": "V[b] = I[b]",
            "tensor_accesses": [{"bits_per_value": 8}],
        }
        with self.assertRaises(ValueError) as ctx:
            _parse_einsum_entry(entry)
        self.assertIn("missing a name", str(ctx.exception))

    def test_invalid_tensor_accesses_type_raises(self):
        entry = {
            "einsum": "V[b] = I[b]",
            "tensor_accesses": "not_a_list",
        }
        with self.assertRaises(ValueError):
            _parse_einsum_entry(entry)


if __name__ == "__main__":
    unittest.main()
