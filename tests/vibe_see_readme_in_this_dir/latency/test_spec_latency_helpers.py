"""
Helpers for the spec-derived latency tests (test_spec_latency_*.py): spec construction
from tests/input_files/latency.arch.yaml, the two model flags, and reading the arms of a
parallel exploration. Every expected number in those files is derived by hand from the
Latency Model docstring in accelforge/frontend/model.py and the column-family docstrings
in accelforge/mapper/FFM/_join_pmappings/pmapping_dataframe.py.
"""

from pathlib import Path

import accelforge as af
from accelforge.frontend.spec import Spec
from accelforge.model.main import evaluate_mapping

INPUT_FILES_DIR = Path(__file__).resolve().parent.parent.parent / "input_files"
LATENCY_ARCH = INPUT_FILES_DIR / "latency.arch.yaml"

FUSED = af.examples.mappings.fused_matmuls_to_simple
UNFUSED = af.examples.mappings.unfused_matmuls_to_simple
WEIGHTS_INSIDE = INPUT_FILES_DIR / "fused_matmuls_weights_inside.mapping.yaml"
TWO_FUSED_LOOPS = INPUT_FILES_DIR / "fused_matmuls_two_fused_loops.mapping.yaml"
DEEP_THEN_SHALLOW = INPUT_FILES_DIR / "fused_matmuls_deep_then_shallow.mapping.yaml"
SHALLOW_THEN_DEEP = INPUT_FILES_DIR / "fused_matmuls_shallow_then_deep.mapping.yaml"
FOUR_MATMULS = INPUT_FILES_DIR / "four_matmuls_three_loops.mapping.yaml"
PAIR_THEN_DEEP = INPUT_FILES_DIR / "fused_matmuls_pair_then_deep.mapping.yaml"
PAIR_THEN_SHALLOW = INPUT_FILES_DIR / "spec_pair_then_shallow.mapping.yaml"
GB_ONLY_FIRST = INPUT_FILES_DIR / "spec_gb_only_first.mapping.yaml"
UNFUSED_THEN_FUSED = INPUT_FILES_DIR / "spec_unfused_then_fused.mapping.yaml"

INF = float("inf")
# 64 MACs per Einsum, 128 bits per tensor.
SMALL = {"M": 4, "KN": 4}
# 256 MACs per Einsum, 512 bits per tensor, outer m tile of 4 rows.
DEEP = {"M": 16, "KN": 4, "M_TILE": 4}
# 32 MACs per Einsum; T tensors 128 bits, W tensors 32 bits.
FOUR = {"M": 8, "KN": 2}

LATENCY = "Total<SEP>latency"
ENERGY = "Total<SEP>energy"
MAC_X_PARALLEL = "reservation<SEP>MAC MULTI EINSUM PARALLEL X<SEP>-1<SEP>left"
GB = "reservation<SEP>GlobalBuffer<SEP>-1<SEP>left"


def evaluate(mapping, n_einsums=2, share=True, windup=True, metrics=None, **jinja):
    """All throughputs infinite and all latencies zero unless given in jinja."""
    spec = Spec.from_yaml(
        af.examples.workloads.basic.matmuls,
        LATENCY_ARCH,
        mapping,
        jinja_parse_data={
            "N_EINSUMS": n_einsums,
            **SMALL,
            "COMPUTE_THROUGHPUT": INF,
            **jinja,
        },
    )
    if metrics is not None:
        spec.model.metrics = metrics
    spec.model._share_resource_latency_above_splits = share
    spec.model._include_wind_up_latency = windup
    return evaluate_mapping(spec).data


def latency(mapping, n_einsums=2, **kwargs):
    data = evaluate(mapping, n_einsums, **kwargs)
    assert len(data) == 1, f"expected one row, got {len(data)}"
    return data[LATENCY].iloc[0]


def arms(mapping, n_einsums=2, fanout=4, **kwargs):
    """{PEs used: total latency} for each arm the join kept, MAC_PARALLEL on."""
    data = evaluate(mapping, n_einsums, MAC_FANOUT=fanout, MAC_PARALLEL=True, **kwargs)
    return {
        round(usage * fanout): total
        for usage, total in zip(data[MAC_X_PARALLEL], data[LATENCY])
    }
