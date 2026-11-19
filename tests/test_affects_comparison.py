"""Tests for the witness search in affects_comparison.

affects_comparison first looks for a witness that s affects the comparison
(_has_witness_that_s_affects_comparison), and only expands symbolically when it
finds none. These tests check that the answer is the same as the symbolic check's
alone, and that the witness search never claims a symbol that cancels out.
"""

import unittest
from unittest import mock

import sympy

from accelforge.mapper.FFM._make_pmappings.make_pmappings_from_templates import (
    make_tile_shapes as mts,
)

# Positive like the mapper's symbols, with no integer=True. Ctrl-F SYMPY_BUG.
x, y, z, n = sympy.symbols("x y z n", positive=True)


def symbolic_check_only(f, s, symbols_enumerated):
    """Runs affects_comparison with the witness search turned off."""
    with mock.patch.object(
        mts, "_has_witness_that_s_affects_comparison", return_value=False
    ):
        return mts.affects_comparison.__wrapped__(f, s, symbols_enumerated)


class TestAffectsComparisonWitness(unittest.TestCase):
    formulas = [
        # A plain polynomial.
        x * y + 3 * z,
        # Max, and a Float coefficient.
        sympy.Max(x * n, y) + sympy.Float(0.37) * z * x,
        # ceiling and Min, and a very small Float coefficient.
        sympy.ceiling(n / x) * y + sympy.Min(x, z) * sympy.Float(1.5e-12),
        # A quotient.
        (x + y) ** 2 / z + n,
        # With x enumerated, y cancels out of f(E_2) - f(E).
        x + y,
        # With x enumerated, y cancels out of f(E_2) / f(E).
        x * y,
        # A Max with a constant.
        sympy.Max(x, 4) * n + y,
    ]

    def test_same_answer_as_the_symbolic_check(self):
        for enumerated in (frozenset(), frozenset({x, z}), frozenset({x, y, z, n})):
            for f in self.formulas:
                # n is not in every formula, which covers an s that f doesn't use.
                for s in sorted(f.free_symbols | {n}, key=str):
                    with self.subTest(f=f, s=s, enumerated=enumerated):
                        self.assertEqual(
                            mts.affects_comparison.__wrapped__(f, s, enumerated),
                            symbolic_check_only(f, s, enumerated),
                        )

    def test_no_witness_for_a_symbol_that_cancels(self):
        witness = mts._has_witness_that_s_affects_comparison
        self.assertFalse(witness(x + y, y, frozenset({x})))
        self.assertFalse(witness(x * y, y, frozenset({x})))

    def test_witness_for_a_shared_symbol(self):
        f = sympy.Max(x, y) * y + x
        self.assertTrue(
            mts._has_witness_that_s_affects_comparison(f, y, frozenset({x}))
        )

    def test_witness_for_an_enumerated_symbol(self):
        # E_2 has its own copy of x, so only f(E) changes with x.
        f = x * y + y
        self.assertTrue(
            mts._has_witness_that_s_affects_comparison(f, x, frozenset({x}))
        )

    def test_exact_value_rejects_undefined_and_irrational_values(self):
        three, two = sympy.Integer(3), sympy.Integer(2)
        # y equal to x puts a pole inside the Max, and sympy raises.
        self.assertIsNone(
            mts._exact_value(sympy.Max(2, 1 / (y - x)), {x: three, y: three})
        )
        self.assertIsNone(mts._exact_value(sympy.sqrt(x), {x: two}))
        self.assertEqual(mts._exact_value(x / 3, {x: two}), sympy.Rational(2, 3))

    def test_undefined_values_fall_back_to_the_symbolic_check(self):
        f = sympy.Max(2, 1 / (y - x))
        self.assertTrue(mts.affects_comparison.__wrapped__(f, y, frozenset({x})))

    def test_no_search_without_positive_assumptions(self):
        w = sympy.Symbol("w")
        self.assertFalse(
            mts._has_witness_that_s_affects_comparison(x * w + w**2, w, frozenset({x}))
        )


if __name__ == "__main__":
    unittest.main()
