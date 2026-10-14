"""
What the final free and fence leave in the evaluated mappings: the total latency, the
reservations at the top level (-1, left), and per-Einsum columns. Every per-level
latency, descent/ascent, right total latency and leak power column is consumed. Also
the latency timeline, which lays the Einsums out one after another.

All tests use the matmul-chain workload on the latency test arch with M = KN = 4.
"""

import unittest

from tests.vibe_see_readme_in_this_dir.latency.test_latency import (
    LATENCY_ARCH,
    make_spec,
)
from tests.vibe_see_readme_in_this_dir.latency.test_latency_windup_private import (
    SIMPLE_MAPPING,
)
from tests.vibe_see_readme_in_this_dir.latency.test_parallel_mappings import (
    GB,
    MAC_X,
    SLOW_MM_THROUGHPUTS,
    WEIGHTS_INSIDE_MAPPING,
    evaluate,
)

# Per-level columns that the final free and fence consume.
PER_LEVEL_PREFIXES = (
    "total_latency<SEP>",
    "level_latency<SEP>",
    "descent_to_above_loop_index<SEP>",
    "ascent_from_above_loop_index<SEP>",
)


class TestFinish(unittest.TestCase):
    def test_only_top_level_columns_survive(self):
        """The final free takes every level: per-level latencies are maxed into the
        total latency and descent/ascent latencies added to it. The final fence adds
        the right total latency into the left one, which becomes Total<SEP>latency,
        and leak power is paid into energy over it. Every reservation is read at the
        top level, -1 left. This holds for one Einsum, which never joins, and for
        every sequential and parallel join of two and three."""
        for n_einsums in [1, 2, 3]:
            for explore_parallel in [False, True]:
                data = evaluate(
                    SIMPLE_MAPPING,
                    explore_parallel,
                    N_EINSUMS=n_einsums,
                    **SLOW_MM_THROUGHPUTS,
                )
                self.assertIn("Total<SEP>latency", data.columns)
                self.assertNotIn("Total<SEP>leak_power", data.columns)
                for c in data.columns:
                    self.assertFalse(c.startswith(PER_LEVEL_PREFIXES), c)
                    if c.startswith("reservation<SEP>"):
                        self.assertTrue(c.endswith("<SEP>-1<SEP>left"), c)

    def test_top_level_reservations(self):
        """Both weights' GlobalBuffer nodes are above the fused m loop, so the two
        Einsums' 128-bit weights are summed (256). The T tiles are below the split:
        sequentially only one branch's count is held (32 + 32 = 64), 320 bits; in
        parallel both branches' tiles are alive at once, 384 bits (Ctrl-F for
        ALWAYS_PARALLEL_BELOW_MULTI_EINSUM_PARALLEL_SPATIAL). The MAC X dimension holds
        one instance sequentially and two in parallel. With these throughputs the
        parallel result is faster (1376 against 1600,
        test_above_split_latency_uses_branch_slack), so both stay on the front of a
        384-bit GlobalBuffer; a 320-bit one keeps only the sequential arm."""
        data = evaluate(SIMPLE_MAPPING, True, GB_SIZE=384, **SLOW_MM_THROUGHPUTS)
        rows = sorted(zip(data["Total<SEP>latency"], data[MAC_X] * 4, data[GB] * 384))
        self.assertEqual(rows, [(1376, 2, 384), (1600, 1, 320)])
        data = evaluate(SIMPLE_MAPPING, True, GB_SIZE=320, **SLOW_MM_THROUGHPUTS)
        self.assertEqual(list(data[MAC_X] * 4), [1])

    def test_timeline_shows_the_sequential_result(self):
        """The latency timeline lays the Einsums out one after another, so even with
        the MAC X dimension allowing parallelism it shows the sequential result:
        Matmul0's branch is compute-bound (320) and Matmul1's is bound by its
        MainMemory traffic (1408), 320 + 1408 = 1728, not the parallel result's 1568
        on the front (test_parallel_overlaps_busy_time)."""
        from accelforge.plotting.latency import _latency_timeline

        spec = make_spec(
            LATENCY_ARCH,
            WEIGHTS_INSIDE_MAPPING,
            2,
            MAC_FANOUT=4,
            MAC_PARALLEL=True,
            **SLOW_MM_THROUGHPUTS,
        )
        blocks, _, total = _latency_timeline(spec)
        self.assertEqual(total, 1728)
        self.assertEqual(len(blocks), 2)
