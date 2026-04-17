import unittest

from accelforge.frontend.workload import _parse_einsum_string


class TestEinsumParsingValid(unittest.TestCase):
    # --- VALID CASES ---
    def test_binary(self):
        r = _parse_einsum_string("X[a] = Y[b] + Z[a,c]")
        self.assertEqual(r["map"], "+")
        self.assertEqual(r["name"], "X")

    def test_multi_char_binary_op(self):
        r = _parse_einsum_string("X[a] = Y[b] ** Z[C:a+b]")
        self.assertEqual(r["map"], "**")
        self.assertEqual(r["name"], "X")

    def test_no_spaces_binary(self):
        r = _parse_einsum_string("AX[a]=Y[b]+Z[c]")
        self.assertEqual(r["map"], "+")
        self.assertEqual(r["name"], "AX")

    def test_unary(self):
        r = _parse_einsum_string("X[a] = exp(Y[a])")
        self.assertEqual(r["map"], "exp")
        self.assertEqual(r["name"], "X")

    def test_unary_multiple_rank_var(self):
        r = _parse_einsum_string("I[b, m, d] = copy(I_in[b, m, d])")
        self.assertEqual(r["map"], "copy")
        self.assertEqual(r["name"], "I")

    def test_unary_multiple_rank_var(self):
        r = _parse_einsum_string("I[b, m, d] = copy(I_in[b, m, d])")
        self.assertEqual(r["map"], "copy")
        self.assertEqual(r["name"], "I")

    def test_unary_affine(self):
        r = _parse_einsum_string("_Einsum[a] = exp(Y[B:2*a+3])")
        self.assertEqual(r["map"], "exp")
        self.assertEqual(r["name"], "_Einsum")

    def test_whitespace_variations(self):
        r = _parse_einsum_string("  X[a]   =   Y[b]   +   Z[c]  ")
        self.assertEqual(r["map"], "+")
        self.assertEqual(r["name"], "X")


class TestEinsumParsingInvalid(unittest.TestCase):
    def test_missing_equals(self):
        with self.assertRaises(ValueError):
            _parse_einsum_string("X[a] Y[b]")

    def test_missing_brackets(self):
        with self.assertRaises(ValueError):
            _parse_einsum_string("X a = Y[b]")

    def test_bad_index_list(self):
        with self.assertRaises(ValueError):
            _parse_einsum_string("X[a,,b] = Y[c]")

    def test_missing_rhs(self):
        with self.assertRaises(ValueError):
            _parse_einsum_string("X[a] = ")

    def test_invalid_tensor_name(self):
        with self.assertRaises(ValueError):
            _parse_einsum_string("1X[a] = Y[b]")

    def test_missing_unary_op(self):
        with self.assertRaises(ValueError):
            _parse_einsum_string("X[a] = Y[b]")

    def test_empty_index(self):
        with self.assertRaises(ValueError):
            _parse_einsum_string("X[] = Y[a]")


if __name__ == "__main__":
    unittest.main()