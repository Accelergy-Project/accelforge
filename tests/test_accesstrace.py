import unittest

import matplotlib

matplotlib.use("Agg")

import numpy as np

from accelforge.frontend.spec import Spec
from accelforge.model.main import evaluate_mapping
from accelforge.plotting.accesstrace import plot_access_trace
from accelforge.tracegen import AccessTrace, trace_accesses
from accelforge.util.parallel import set_n_parallel_jobs

set_n_parallel_jobs(1)

try:
    from .paths import CURRENT_DIR, EXAMPLES_DIR
except ImportError:
    from paths import CURRENT_DIR, EXAMPLES_DIR

INPUT_FILES = CURRENT_DIR / "input_files"


def _matmul_spec(mapping: str, **jinja):
    return Spec.from_yaml(
        EXAMPLES_DIR / "arches" / "simple.yaml",
        EXAMPLES_DIR / "workloads" / "basic" / "matmuls.yaml",
        EXAMPLES_DIR / "mappings" / f"{mapping}.yaml",
        jinja_parse_data={"N_EINSUMS": 2, "M": 8, "KN": 4, **jinja},
    )


def _blocks(axis):
    """The single PatchCollection of memory-level tile blocks drawn on ``axis``."""
    from matplotlib.collections import PatchCollection

    (blocks,) = [c for c in axis.collections if isinstance(c, PatchCollection)]
    return blocks


def _conv_spec():
    return Spec.from_yaml(
        EXAMPLES_DIR / "arches" / "simple.yaml",
        INPUT_FILES / "conv1d.workload.yaml",
        INPUT_FILES / "conv1d.mapping.yaml",
    )


class TestTraceAccesses(unittest.TestCase):
    def test_one_timestep_per_compute(self):
        """An untiled, unparallelized mapping runs exactly one compute per timestep."""
        spec = _matmul_spec("unfused_matmuls_to_simple")
        trace = trace_accesses(evaluate_mapping(spec).mapping(), workload=spec.workload)
        self.assertEqual(trace.n_timesteps, spec.workload.n_computes())

    def test_covers_every_tensor_element(self):
        """Every element of every tensor is touched at some point."""
        spec = _matmul_spec("unfused_matmuls_to_simple")
        trace = trace_accesses(spec)
        for tensor in trace.tensors:
            touched = np.unique(
                np.concatenate([t.element for t in trace.for_tensor(tensor)])
            )
            size = int(np.prod(trace.tensor_shapes[tensor]))
            self.assertEqual(touched.tolist(), list(range(size)), tensor)

    def test_unfused_einsums_do_not_overlap(self):
        spec = _matmul_spec("unfused_matmuls_to_simple")
        trace = trace_accesses(spec)
        (_, end0), (start1, _) = trace.einsum_timespans.values()
        self.assertLessEqual(end0, start1)

    def test_fused_einsums_interleave(self):
        """Fusing shares outer loops, so the two Einsums' timespans overlap."""
        spec = _matmul_spec("fused_matmuls_to_simple")
        trace = trace_accesses(spec)
        (start0, end0), (start1, end1) = trace.einsum_timespans.values()
        self.assertLess(start1, end0)
        self.assertLess(start0, end1)
        # Fusion changes the schedule, not the amount of work.
        self.assertEqual(trace.n_timesteps, spec.workload.n_computes())

    def test_halo_projection(self):
        """I[p + r] slides a 3-wide window across a 10-element tensor."""
        spec = _conv_spec()
        trace = trace_accesses(spec)
        self.assertEqual(trace.tensor_shapes["I"], (10,))

        (inputs,) = trace.for_tensor("I")
        self.assertEqual(inputs.timestep.tolist(), list(range(24)))
        self.assertEqual(
            inputs.element.tolist(),
            [p + r for p in range(8) for r in range(3)],
        )

        (outputs,) = trace.for_tensor("O")
        self.assertTrue(outputs.is_output)
        self.assertEqual(outputs.element.tolist(), [p for p in range(8) for _ in range(3)])

    def test_spatial_loops_share_a_timestep(self):
        """Spatial iterations run concurrently, so they do not advance the clock."""
        spec = Spec.from_yaml(
            EXAMPLES_DIR / "arches" / "fanout_variations" / "at_glb.yaml",
            EXAMPLES_DIR / "workloads" / "basic" / "matmuls.yaml",
            jinja_parse_data={"N_EINSUMS": 1, "M": 8, "KN": 4},
        )
        mapping = spec.map_workload_to_arch().mapping()
        trace = trace_accesses(mapping, workload=spec.workload)
        self.assertLess(trace.n_timesteps, spec.workload.n_computes())

    def test_max_timesteps_gives_a_prefix(self):
        spec = _conv_spec()
        full = trace_accesses(spec)
        prefix = trace_accesses(spec, max_timesteps=9)

        self.assertTrue(prefix.truncated)
        self.assertFalse(full.truncated)
        self.assertLessEqual(prefix.n_timesteps, 9)

        (full_i,) = full.for_tensor("I")
        (prefix_i,) = prefix.for_tensor("I")
        keep = full_i.timestep < prefix.n_timesteps
        self.assertEqual(prefix_i.timestep.tolist(), full_i.timestep[keep].tolist())
        self.assertEqual(prefix_i.element.tolist(), full_i.element[keep].tolist())

    def test_max_points_guard(self):
        spec = _conv_spec()
        with self.assertRaises(ValueError):
            trace_accesses(spec, max_points=4)

    def test_filters(self):
        spec = _matmul_spec("unfused_matmuls_to_simple")
        self.assertEqual(trace_accesses(spec, tensors=["T1"]).tensors, ["T1"])
        self.assertEqual(
            trace_accesses(spec, einsums=["Matmul0"]).einsums, ["Matmul0"]
        )

    def test_dataframe(self):
        spec = _conv_spec()
        trace = trace_accesses(spec)
        df = trace.to_dataframe()
        self.assertEqual(
            sorted(df.columns),
            ["einsum", "element", "is_output", "tensor", "timestep"],
        )
        self.assertEqual(len(df), sum(t.n_accesses for t in trace.traces))

    def test_requires_a_workload(self):
        spec = _conv_spec()
        with self.assertRaises(ValueError):
            trace_accesses(spec.mapping)

    def test_memory_levels_are_outermost_first(self):
        trace = trace_accesses(_conv_spec())
        self.assertEqual(trace.memory_levels, ["MainMemory", "GlobalBuffer"])

    def test_backing_storage_holds_one_tile_for_the_whole_run(self):
        """No loop is above MainMemory here, so its tile never changes."""
        trace = trace_accesses(_conv_spec())
        for tensor in trace.tensors:
            self.assertEqual(
                trace.tile_lifetime("MainMemory", tensor), [(0, trace.n_timesteps)], tensor
            )

    def test_a_loop_above_a_storage_splits_its_tile(self):
        """The `p` loop is above GlobalBuffer, so it gets one tile per iteration."""
        trace = trace_accesses(_conv_spec())
        windows = trace.tile_lifetime("GlobalBuffer", "I")
        self.assertEqual(len(windows), trace.tensor_shapes["O"][0])
        self.assertEqual(
            windows, [(t, t + 3) for t in range(0, trace.n_timesteps, 3)]
        )

    def test_windows_tile_the_timeline_without_gaps(self):
        trace = trace_accesses(_matmul_spec("fused_matmuls_to_simple"))
        for level in trace.memory_levels:
            for tensor, windows in trace.tile_windows[level].items():
                ends = [0] + [end for _, end in windows[:-1]]
                self.assertEqual([s for s, _ in windows], ends, (level, tensor))
                self.assertEqual(windows[-1][1], trace.n_timesteps, (level, tensor))

    def test_fusion_shows_up_as_tiled_intermediates(self):
        """
        Fusing puts the `m` loop above the GlobalBuffer holding T1, so T1 is resident one
        tile at a time. Unfused, the whole of T1 is live for a whole Einsum.
        """
        fused = trace_accesses(_matmul_spec("fused_matmuls_to_simple"))
        unfused = trace_accesses(_matmul_spec("unfused_matmuls_to_simple"))
        self.assertGreater(
            len(fused.tile_lifetime("GlobalBuffer", "T1")),
            len(unfused.tile_lifetime("GlobalBuffer", "T1")),
        )
        # Fusion keeps intermediates out of backing storage entirely.
        self.assertEqual(fused.tile_lifetime("MainMemory", "T1"), [])
        self.assertNotEqual(unfused.tile_lifetime("MainMemory", "T1"), [])

    def test_windows_respect_max_timesteps(self):
        trace = trace_accesses(_conv_spec(), max_timesteps=7)
        for windows in trace.tile_windows["GlobalBuffer"].values():
            self.assertTrue(all(end <= trace.n_timesteps for _, end in windows))
            self.assertEqual(windows[-1][1], trace.n_timesteps)

    def test_windows_respect_the_tensor_filter(self):
        trace = trace_accesses(_conv_spec(), tensors=["I"])
        self.assertEqual(list(trace.tile_windows["GlobalBuffer"]), ["I"])


class TestPlotAccessTrace(unittest.TestCase):
    def test_plots_one_axes_per_tensor(self):
        spec = _matmul_spec("fused_matmuls_to_simple")
        trace = trace_accesses(spec)
        fig, axes = plot_access_trace(trace)
        self.assertEqual(len(axes), len(trace.tensors))
        self.assertEqual(axes[-1].get_xlabel(), "Timestep")

    def test_accepts_a_spec_directly(self):
        fig, axes = plot_access_trace(_conv_spec())
        self.assertEqual(len(axes), 3)

    def test_color_by_access_and_rank(self):
        spec = _matmul_spec("fused_matmuls_to_simple")
        fig, axes = plot_access_trace(
            spec, tensors=["T1"], color_by="access", rank="M"
        )
        self.assertEqual(axes[0].get_ylabel(), "M")

    def test_rejects_bad_color_by(self):
        with self.assertRaises(ValueError):
            plot_access_trace(_conv_spec(), color_by="tensor")

    def test_memory_level_shades_one_block_per_resident_tile(self):
        """Every tile of this mapping is contiguous, so it is one block."""
        spec = _matmul_spec("fused_matmuls_to_simple")
        trace = trace_accesses(spec)
        _, axes = plot_access_trace(trace, memory_level="GlobalBuffer")

        for axis, tensor in zip(axes, trace.tensors):
            self.assertEqual(
                len(_blocks(axis).get_paths()),
                len(trace.tile_lifetime("GlobalBuffer", tensor)),
                tensor,
            )

    def test_memory_level_blocks_cover_the_resident_elements(self):
        spec = _matmul_spec("fused_matmuls_to_simple")
        trace = trace_accesses(spec)
        _, axes = plot_access_trace(trace, tensors=["T1"], memory_level="GlobalBuffer")

        (written,) = [t for t in trace.for_tensor("T1") if t.is_output]
        windows = trace.tile_lifetime("GlobalBuffer", "T1")
        for (start, end), path in zip(windows, _blocks(axes[0]).get_paths()):
            resident = written.element[
                (written.timestep >= start) & (written.timestep < end)
            ]
            extents = path.get_extents()
            self.assertEqual(tuple(extents.min), (start - 0.5, resident.min() - 0.5))
            self.assertEqual(tuple(extents.max), (end - 0.5, resident.max() + 0.5))

    def test_memory_level_blocks_stop_at_the_last_live_use(self):
        """
        The GlobalBuffer reservation for T0 spans a whole `m` iteration, but T0 is only
        touched by Matmul0, the first branch of the split, so its block covers only that
        branch. T1 crosses the branches, so its block covers the whole reservation.
        """
        spec = _matmul_spec("fused_matmuls_to_simple")
        trace = trace_accesses(spec)
        _, axes = plot_access_trace(
            trace, tensors=["T0", "T1"], memory_level="GlobalBuffer"
        )

        (read,) = trace.for_tensor("T0")
        windows = trace.tile_lifetime("GlobalBuffer", "T0")
        self.assertGreater(len(windows), 1)
        for (start, end), path in zip(windows, _blocks(axes[0]).get_paths()):
            live = read.timestep[(read.timestep >= start) & (read.timestep < end)]
            interval = path.get_extents().intervalx
            self.assertEqual(tuple(interval), (live.min() - 0.5, live.max() + 0.5))
            self.assertLess(interval[1], end - 0.5)  # strictly inside the reservation

        for (start, end), path in zip(
            trace.tile_lifetime("GlobalBuffer", "T1"), _blocks(axes[1]).get_paths()
        ):
            self.assertEqual(
                tuple(path.get_extents().intervalx), (start - 0.5, end - 0.5)
            )

    def test_memory_level_adds_a_legend_entry(self):
        spec = _matmul_spec("fused_matmuls_to_simple")
        fig, _ = plot_access_trace(spec, memory_level="GlobalBuffer")
        labels = [t.get_text() for t in fig.legends[0].get_texts()]
        self.assertIn("tile in GlobalBuffer", labels)

    def test_rejects_a_memory_level_that_holds_nothing_plotted(self):
        spec = _matmul_spec("fused_matmuls_to_simple")
        with self.assertRaises(ValueError):
            plot_access_trace(spec, memory_level="Nowhere")
        # T1 is fused, so it never reaches MainMemory.
        with self.assertRaises(ValueError):
            plot_access_trace(spec, tensors=["T1"], memory_level="MainMemory")

    def test_existing_axes(self):
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        fig, axes = plot_access_trace(_conv_spec(), tensors=["W"], ax=ax)
        self.assertIs(axes[0], ax)

        with self.assertRaises(ValueError):
            plot_access_trace(_conv_spec(), ax=ax)


if __name__ == "__main__":
    unittest.main()
