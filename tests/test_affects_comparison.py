"""
Checks the fast path in affects_comparison against the full symbolic check.
"""

import unittest
from unittest import mock

import sympy

from accelforge.mapper.FFM._make_pmappings.make_pmappings_from_templates import (
    make_tile_shapes as mts,
)

x, y, z, n = sympy.symbols("x y z n", integer=True, positive=True)


def symbolic_only(f, s, symbols_enumerated):
    # affects_comparison with the fast path turned off
    with mock.patch.object(mts, "_provably_changes_with", return_value=False):
        return mts.affects_comparison.__wrapped__(f, s, symbols_enumerated)


class TestAffectsComparisonFastPath(unittest.TestCase):
    formulas = [
        x * y + 3 * z,
        sympy.Max(x * n, y) + sympy.Float(0.37) * z * x,
        sympy.ceiling(n / x) * y + sympy.Min(x, z) * sympy.Float(1.5e-12),
        (x + y) ** 2 / z + n,
        x + y,  # y cancels in f(E_2) - f(E)
        x * y,  # y cancels in f(E_2) / f(E)
        sympy.Max(x, 4) * n + y,
    ]

    def test_matches_symbolic_check(self):
        for enumerated in (frozenset(), frozenset({x, z}), frozenset({x, y, z, n})):
            for f in self.formulas:
                # n is missing from some formulas, which covers s not in f
                for s in sorted(f.free_symbols | {n}, key=str):
                    with self.subTest(f=f, s=s, enumerated=enumerated):
                        self.assertEqual(
                            mts.affects_comparison.__wrapped__(f, s, enumerated),
                            symbolic_only(f, s, enumerated),
                        )

    def test_never_claims_a_cancelled_symbol(self):
        self.assertFalse(mts._provably_changes_with(x + y, y, frozenset({x})))
        self.assertFalse(mts._provably_changes_with(x * y, y, frozenset({x})))

    def test_proves_a_symbol_that_matters(self):
        f = sympy.Max(x, y) * y + x
        self.assertTrue(mts._provably_changes_with(f, y, frozenset({x})))

    def test_skips_points_where_f_is_undefined(self):
        # y == x puts a pole inside Max, which sympy refuses to compare
        f = sympy.Max(2, 1 / (y - x))
        self.assertTrue(mts.affects_comparison.__wrapped__(f, y, frozenset({x})))

    def test_declines_without_positive_assumptions(self):
        w = sympy.Symbol("w")
        self.assertFalse(mts._provably_changes_with(x * w + w**2, w, frozenset({x})))


if __name__ == "__main__":
    unittest.main()
