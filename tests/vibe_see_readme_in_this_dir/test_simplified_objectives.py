import numpy as np
import pandas as pd
import pytest

from accelforge.frontend.mapper.ffm import FFM
from accelforge.frontend.mapper.metrics import Metrics
from accelforge.mapper.FFM._join_pmappings.compatibility import Compatibility
from accelforge.mapper.FFM._join_pmappings.join_pmappings import prune_with_tolerance
from accelforge.mapper.FFM._join_pmappings.pmapping_dataframe import PmappingDataframe
from accelforge.mapper.FFM._join_pmappings.pmapping_group import PmappingGroup
from accelforge.mapper.FFM._pareto_df.df_convention import (
    commlatency2col,
    complatency2col,
    make_fused_loop_col,
    reservation2col,
)
from accelforge.util import fzs


ENERGY = "Total<SEP>energy"
LATENCY = "Total<SEP>latency"
EDP = "Total<SEP>energy_delay_product"


def pmapping(columns, enabled=True, metrics=Metrics.ENERGY_DELAY_PRODUCT):
    data = pd.DataFrame(columns)
    return PmappingDataframe(
        data,
        len(data),
        len(data),
        fzs(),
        False,
        skip_pareto=True,
        use_simplified_objectives=metrics if enabled else None,
    )


def test_greedy_product_optimization_levels():
    mapper = FFM()
    assert not mapper.greedy_prune_simplified_objectives
    for level in [0, 1, 2, 3, 4, 2, 3]:
        mapper.set_optimization_level(level)
        assert mapper.greedy_prune_simplified_objectives == (level <= 2)


@pytest.mark.parametrize("enabled", [False, True])
def test_greedy_product_tradeoff_and_initial_leaves(enabled):
    mapping = pmapping({ENERGY: [2.0, 1.0], LATENCY: [1.0, 3.0]}, enabled)
    original = mapping.data.copy()
    compatibility = Compatibility(tensors=fzs())
    leaves = {"E": [PmappingGroup(compatibility, mapping)]}
    pruned = prune_with_tolerance(leaves, 0, 0, print_progress=False, is_last=True)
    expected = original.iloc[[0]] if enabled else original
    pd.testing.assert_frame_equal(pruned["E"][0].mappings.data, expected)
    direct = mapping.copy().make_pareto()
    pd.testing.assert_frame_equal(direct.data, expected)
    pd.testing.assert_frame_equal(mapping.data, original)


def test_greedy_product_keeps_resources_and_fused_shapes_with_duplicate_indices():
    mapping = pmapping(
        {
            ENERGY: [2.0, 1.0, 4.0, 10.0],
            LATENCY: [1.0, 3.0, 1.0, 1.0],
            reservation2col("GB", 0): [0.5, 0.5, 0.25, 0.5],
            make_fused_loop_col("tile_shape"): [16, 16, 16, 32],
            make_fused_loop_col("n_iterations"): [4, 4, 4, 2],
        }
    )
    mapping.data.index = [7, 7, 40, 7]
    original = mapping.data.copy()
    mapping.make_pareto()
    pd.testing.assert_frame_equal(mapping.data, original.iloc[[0, 2, 3]])


def test_greedy_product_folds_shared_busy_time_and_both_windups():
    mapping = pmapping(
        {
            ENERGY: [1.0, 2.0],
            LATENCY: [1e-6, 4e-6],
            complatency2col("Compute", 0): [10e-6, 4e-6],
            commlatency2col("descent", 0): [3e-6, 1e-6],
            commlatency2col("ascent", 0): [2e-6, 1e-6],
        }
    )
    original = mapping.data.copy()
    np.testing.assert_allclose(
        mapping._simplify_objectives_for_prune(Metrics.ENERGY_DELAY_PRODUCT)[EDP],
        [15e-6, 12e-6],
    )
    pd.testing.assert_frame_equal(mapping.data, original)
    mapping.make_pareto()
    pd.testing.assert_frame_equal(mapping.data, original.iloc[[1]])


@pytest.mark.parametrize(
    "metrics, positions",
    [
        (Metrics.ENERGY, [0]),
        (Metrics.LATENCY, [1]),
        (Metrics.ENERGY | Metrics.LATENCY, [0, 1]),
        (Metrics.ENERGY_DELAY_PRODUCT | Metrics.ENERGY | Metrics.LATENCY, [0, 1]),
    ],
)
def test_simplified_objectives_respect_requested_metrics(metrics, positions):
    columns = {}
    if metrics.includes_latency():
        columns = {
            LATENCY: [1e-6, 4e-6],
            complatency2col("Compute", 0): [10e-6, 4e-6],
            commlatency2col("descent", 0): [3e-6, 1e-6],
            commlatency2col("ascent", 0): [2e-6, 1e-6],
        }
    if metrics.includes_energy():
        columns[ENERGY] = [1.0, 2.0]
    mapping = pmapping(columns, metrics=metrics)
    original = mapping.data.copy()
    simplified = mapping._simplify_objectives_for_prune(metrics)
    expected = set()
    if metrics & Metrics.LATENCY:
        expected.add(LATENCY)
        np.testing.assert_allclose(simplified[LATENCY], [15e-6, 6e-6])
    if metrics & Metrics.ENERGY:
        expected.add(ENERGY)
    if metrics & Metrics.ENERGY_DELAY_PRODUCT:
        expected.add(EDP)
        np.testing.assert_allclose(simplified[EDP], [15e-6, 12e-6])
    assert set(simplified) == expected
    pd.testing.assert_frame_equal(mapping.data, original)
    mapping.make_pareto()
    pd.testing.assert_frame_equal(mapping.data, original.iloc[positions])


def test_greedy_product_energy_already_includes_leakage():
    mapping = pmapping(
        {
            ENERGY: [5.0, 3.0],
            LATENCY: [1.0, 2.0],
            "Total<SEP>leak_energy": [4.0, 0.2],
        }
    )
    original = mapping.data.copy()
    np.testing.assert_array_equal(
        mapping._simplify_objectives_for_prune(Metrics.ENERGY_DELAY_PRODUCT)[EDP],
        [5.0, 6.0],
    )
    mapping.make_pareto()
    pd.testing.assert_frame_equal(mapping.data, original.iloc[[0]])


def test_prune_projection_preserves_unfinished_reservations_and_rows():
    data = pd.DataFrame(
        {
            ENERGY: [1.0, 2.0],
            LATENCY: [1e-6, 1e-6],
            reservation2col("GB", 0): [0.5, 1.25],
            reservation2col("GB", 1): [0.25, 0.5],
            make_fused_loop_col("tile_shape"): [16, 16],
            make_fused_loop_col("n_iterations"): [4, 4],
        },
        index=[7, 7],
    )
    mapping = PmappingDataframe(data, 2, 2, fzs(), True, skip_pareto=True)
    original = mapping.data.copy()
    simplified = mapping._simplify_objectives_for_prune(
        Metrics.ENERGY_DELAY_PRODUCT | Metrics.RESOURCE_USAGE
    )
    expected = original.drop(columns=[ENERGY, LATENCY]).reset_index(drop=True)
    expected[EDP] = (original[ENERGY].astype(float) * original[LATENCY]).to_numpy()
    pd.testing.assert_frame_equal(simplified, expected)
    pd.testing.assert_frame_equal(mapping.data, original)


def test_greedy_product_survives_dataframe_operations_and_merge():
    mapping = pmapping({ENERGY: [2.0, 1.0], LATENCY: [1.0, 3.0]})
    original = mapping.data.copy()
    variants = [
        mapping.copy(),
        mapping.update(skip_pareto=True),
        PmappingDataframe.concat([mapping.copy(), mapping.copy()]),
    ]
    for variant in variants:
        assert variant.use_simplified_objectives == Metrics.ENERGY_DELAY_PRODUCT
        variant.make_pareto()
        pd.testing.assert_frame_equal(
            variant.data.reset_index(drop=True), original.iloc[[0]]
        )
    compatibility = Compatibility(tensors=fzs())
    merged = mapping.merge_next(
        pmapping({ENERGY: [1.0], LATENCY: [1.0]}),
        fzs(),
        compatibility,
        compatibility,
        compatibility,
        fzs(),
    )
    assert merged.use_simplified_objectives == Metrics.ENERGY_DELAY_PRODUCT
    assert merged.data[ENERGY].tolist() == [3.0]
    assert merged.data[LATENCY].tolist() == [2.0]
    pd.testing.assert_frame_equal(mapping.data, original)
