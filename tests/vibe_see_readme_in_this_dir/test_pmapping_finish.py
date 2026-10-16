import numpy as np
import pandas as pd
import pytest

from accelforge.frontend.mapper.metrics import Metrics
from accelforge.mapper.FFM._join_pmappings.pmapping_dataframe import PmappingDataframe
from accelforge.mapper.FFM._pareto_df.df_convention import (
    make_fused_loop_col,
    reservation2col,
)


ENERGY = "Total<SEP>energy"
LATENCY = "Total<SEP>latency"
EDP = "Total<SEP>energy_delay_product"


def test_finish_prunes_across_fused_tile_shapes():
    mapping = PmappingDataframe(
        pd.DataFrame(
            {
                "Total<SEP>energy": [2.0, 1.0],
                "Total<SEP>latency": [3e-6, 2e-6],
                make_fused_loop_col("tile_shape"): [16, 32],
                make_fused_loop_col("n_iterations"): [4, 2],
            }
        ),
        n_total_pmappings=2,
        n_valid_pmappings=2,
        ignored_resources=set(),
        drop_valid_reservations=True,
        skip_pareto=True,
    )
    mapping.finish(Metrics.ENERGY | Metrics.LATENCY)
    assert set(mapping.data) == {"Total<SEP>energy", "Total<SEP>latency"}
    assert mapping.data["Total<SEP>energy"].tolist() == [1.0]
    assert mapping.data["Total<SEP>latency"].tolist() == [2e-6]


@pytest.mark.parametrize("greedy", [False, True])
@pytest.mark.parametrize(
    "metrics, positions, objectives",
    [
        (Metrics.ENERGY, [1, 2], {ENERGY}),
        (Metrics.LATENCY, [0, 2], {LATENCY}),
        (Metrics.ENERGY_DELAY_PRODUCT, [0, 2], {EDP}),
        (Metrics.ENERGY | Metrics.LATENCY, [0, 1, 2], {ENERGY, LATENCY}),
        (
            Metrics.ENERGY_DELAY_PRODUCT | Metrics.ENERGY | Metrics.LATENCY,
            [0, 1, 2],
            {ENERGY, LATENCY, EDP},
        ),
    ],
)
def test_finish_uses_requested_objectives_and_resources(
    greedy, metrics, positions, objectives
):
    resource = reservation2col("GB", -1, left=True)
    columns = {
        ENERGY: [2.0, 1.0, 4.0],
        LATENCY: [1e-6, 3e-6, 2e-6],
        "Total<SEP>dynamic_energy": [2.0, 1.0, 4.0],
        resource: [0.5, 0.5, 0.25],
        make_fused_loop_col("tile_shape"): [16, 32, 64],
    }
    mapping = PmappingDataframe(
        pd.DataFrame(columns),
        n_total_pmappings=3,
        n_valid_pmappings=3,
        ignored_resources=set(),
        drop_valid_reservations=False,
        skip_pareto=True,
        use_simplified_objectives=metrics if greedy else None,
    )
    mapping.finish(metrics | Metrics.RESOURCE_USAGE)
    assert mapping.use_simplified_objectives is None
    assert {c for c in mapping.data if c.startswith("Total<SEP>")} == objectives
    assert not any(c.startswith("fused_loop<SEP>") for c in mapping.data)
    assert mapping.data[reservation2col("GB", -1)].tolist() == [
        columns[resource][p] for p in positions
    ]
    for column in objectives:
        expected = (
            np.array(columns[ENERGY]) * np.array(columns[LATENCY])
            if column == EDP
            else np.array(columns[column])
        )
        np.testing.assert_allclose(mapping.data[column], expected[positions])


def test_finish_preserves_selected_mapping_metadata():
    mapping = PmappingDataframe(
        pd.DataFrame(
            {
                ENERGY: [2.0, 1.0],
                LATENCY: [1e-6, 3e-6],
                "A<SEP>mapping": ["first", "second"],
                "compressed_index<SEP>A": [7, 9],
                make_fused_loop_col("tile_shape"): [16, 32],
            }
        ),
        n_total_pmappings=2,
        n_valid_pmappings=2,
        ignored_resources=set(),
        drop_valid_reservations=True,
        skip_pareto=True,
    )
    mapping.finish(Metrics.ENERGY_DELAY_PRODUCT)
    assert mapping.data.drop(columns=EDP).to_dict(orient="list") == {
        "A<SEP>mapping": ["first"],
        "compressed_index<SEP>A": [7],
    }
    np.testing.assert_allclose(mapping.data[EDP], [2e-6])
