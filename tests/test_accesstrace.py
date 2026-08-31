import unittest

import matplotlib

matplotlib.use("Agg")

import numpy as np

import accelforge as af
from accelforge.frontend.spec import Spec
from accelforge.model.main import evaluate_mapping
from accelforge.plotting.accesstrace import plot_access_trace
from accelforge.tracegen import trace_accesses

try:
    from .paths import CURRENT_DIR, EXAMPLES_DIR
except ImportError:
    from paths import CURRENT_DIR, EXAMPLES_DIR

INPUT_FILES = CURRENT_DIR / "input_files"


def _matmul_spec(mapping: str, **jinja):
    return Spec.from_yaml(
        af.examples.arches.simple,
        af.examples.workloads.basic.matmuls,
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
        af.examples.arches.simple,
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
        # Fusion changes the ordering but not the amount of work.
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

    def test_a_loop_above_a_storage_creates_tiles(self):
        """The `p` loop is above GlobalBuffer, so it gets one tile per iteration."""
        trace = trace_accesses(_conv_spec())
        windows = trace.tile_lifetime("GlobalBuffer", "I")
        self.assertEqual(len(windows), trace.tensor_shapes["O"][0])
        self.assertEqual(
            windows, [(t, t + 3) for t in range(0, trace.n_timesteps, 3)]
        )


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
            spec, tensors=["A"], color_by="access", rank="M"
        )
        self.assertEqual(axes[0].get_ylabel(), "M")

    def test_memory_level_shades_one_block_per_tile(self):
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

    def test_memory_level_blocks_cover_the_tile(self):
        spec = _matmul_spec("fused_matmuls_to_simple")
        trace = trace_accesses(spec)
        _, axes = plot_access_trace(trace, tensors=["A"], memory_level="GlobalBuffer")

        (written,) = [t for t in trace.for_tensor("A") if t.is_output]
        windows = trace.tile_lifetime("GlobalBuffer", "A")
        for (start, end), path in zip(windows, _blocks(axes[0]).get_paths()):
            resident = written.element[
                (written.timestep >= start) & (written.timestep < end)
            ]
            extents = path.get_extents()
            self.assertEqual(tuple(extents.min), (start - 0.5, resident.min() - 0.5))
            self.assertEqual(tuple(extents.max), (end - 0.5, resident.max() + 0.5))


if __name__ == "__main__":
    unittest.main()
