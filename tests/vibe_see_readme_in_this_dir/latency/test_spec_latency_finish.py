"""
The final free and fence (finish() in pmapping_dataframe.py): after them no per-level
total latency, per-component latency or wind-up column remains, total latency is
Total<SEP>latency, and each resource keeps one reservation at above_loop_index -1 on
the left. Reservations are fractions of the component's size.

GlobalBuffer bits with M = KN = 4: a W tensor 128, a one-row T tile 32.
"""

import unittest

from tests.vibe_see_readme_in_this_dir.latency.test_spec_latency_helpers import (
    DEEP,
    DEEP_THEN_SHALLOW,
    FUSED,
    GB,
    LATENCY,
    MAC_X_PARALLEL,
    evaluate,
)

PER_LEVEL_PREFIXES = (
    "total_latency<SEP>",
    "level_latency<SEP>",
    "descent_to_above_loop_index",
    "ascent_from_above_loop_index",
)
MAC_X = "reservation<SEP>MAC X<SEP>-1<SEP>left"
MM = "reservation<SEP>MainMemory<SEP>-1<SEP>left"
BUSY_WITH_WINDUP = {
    "MM_READ_THROUGHPUT": 1,
    "COMPUTE_THROUGHPUT": 0.5,
    "MM_READ_LATENCY": 100,
    "COMPUTE_LATENCY": 10,
}


class TestFinish(unittest.TestCase):
    def test_output_columns(self):
        """Only finished columns remain, for the sequential and the parallel arm."""
        for parallel in (False, True):
            with self.subTest(parallel=parallel):
                data = evaluate(
                    FUSED, 2, MAC_FANOUT=4, MAC_PARALLEL=parallel, **BUSY_WITH_WINDUP
                )
                self.assertIn(LATENCY, data.columns)
                self.assertNotIn("Total<SEP>leak_power", data.columns)
                for c in data.columns:
                    self.assertFalse(c.startswith(PER_LEVEL_PREFIXES), c)
                    if c.startswith("reservation<SEP>"):
                        self.assertTrue(c.endswith("<SEP>-1<SEP>left"), c)

    def test_reservations_fused_pair(self):
        """
        GlobalBuffer holds W0 + W1 = 256 above the shared loop plus the larger
        sequential branch, T0 + T1 tile = 64 (same for T1 + T2): 320 of 1000. MAC X
        uses 1 of fanout 1. MainMemory is infinite: 0.
        """
        data = evaluate(FUSED, 2, GB_SIZE=1000)
        self.assertAlmostEqual(data[GB].iloc[0], 0.32, places=4)
        self.assertEqual(data[MAC_X].iloc[0], 1)
        self.assertEqual(data[MM].iloc[0], 0)

    def test_reservations_three_einsums(self):
        """W0 + W1 + W2 = 384 plus one branch's 64: 448 of 1000."""
        data = evaluate(FUSED, 3, GB_SIZE=1000)
        self.assertAlmostEqual(data[GB].iloc[0], 0.448, places=4)

    def test_reservations_deep_then_shallow(self):
        """
        T2's 4-row tile 128 at level 1, plus the larger branch: Matmul0 Matmul1 hold
        W0 + W1 256 and T0 + T1 one-row tiles 64 (320) against Matmul2's W2 128 + T3
        4-row tile 128 (256). 128 + 320 = 448 of 1000.
        """
        data = evaluate(DEEP_THEN_SHALLOW, 3, GB_SIZE=1000, **DEEP)
        self.assertAlmostEqual(data[GB].iloc[0], 0.448, places=4)

    def test_reservations_parallel_arm(self):
        """
        RULING: at a parallel join the GlobalBuffer, above the multi-Einsum fanout, is
        alive in both branches at once, so the two branches' tiles sum (Ctrl-F for
        ALWAYS_PARALLEL_BELOW_MULTI_EINSUM_PARALLEL_SPATIAL): 320 of 1000 on the
        sequential arm (the weights 256 plus one branch's tiles 64) and 384 on the
        parallel arm (both branches' tiles). MAC X usage sums to 2 of 4.
        """
        data = evaluate(
            FUSED,
            2,
            GB_SIZE=1000,
            MAC_FANOUT=4,
            MAC_PARALLEL=True,
            COMPUTE_THROUGHPUT=0.5,
        )
        rows = {round(u * 4): g for u, g in zip(data[MAC_X_PARALLEL], data[GB])}
        self.assertEqual(sorted(rows), [1, 2])
        self.assertAlmostEqual(rows[1], 0.32, places=4)
        self.assertAlmostEqual(rows[2], 0.384, places=4)


if __name__ == "__main__":
    unittest.main()
