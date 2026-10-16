import pandas as pd

from accelforge.mapper.FFM._join_pmappings.pmapping_dataframe import PmappingDataframe
from accelforge.mapper.FFM._pareto_df.df_convention import make_fused_loop_col


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
    mapping.finish()
    assert set(mapping.data) == {"Total<SEP>energy", "Total<SEP>latency"}
    assert mapping.data["Total<SEP>energy"].tolist() == [1.0]
    assert mapping.data["Total<SEP>latency"].tolist() == [2e-6]
