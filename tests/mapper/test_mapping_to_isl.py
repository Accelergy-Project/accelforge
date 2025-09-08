from pathlib import Path
import unittest

import islpy as isl

from fastfusion.frontend.workload import Workload
from fastfusion.frontend.workload._isl import get_rank_variable_bounds
from fastfusion.frontend.workload.workload import RankVariableName

from fastfusion.frontend.constraints import LoopOrder
from fastfusion.frontend.mapping import Mapping

from fastfusion.model.looptree.reuse.isl.mapping_to_isl import analyze_mapping
from fastfusion.model.looptree.reuse.isl.mapping_to_isl.types import MappingAnalysisResult

TEST_CONFIG_PATH: Path = Path(__file__).parent / "configs"

class TestMappingToIsl(unittest.TestCase):

    def test_conv1d(self):
        print("Here")
        # Loads in the CONV1D Config
        CONV1D_CONFIG_PATH: Path = TEST_CONFIG_PATH
        workload: Workload = Workload.from_yaml(CONV1D_CONFIG_PATH / "conv1d.workload.yaml")

        rank_var_bounds: dict = get_rank_variable_bounds(workload, "conv")        
        # rank_P = rank_var_bounds["p"]
        # rank_R = rank_var_bounds["r"]

        # print(rank_P)
        # print(rank_R)

        # # loop_nest: LooptreeWorkload = LooptreeWorkload()
        # loop_order: LoopOrder = LoopOrder(
        #     [RankVariableName('p'), RankVariableName('r')]
        # )
        mapping: Mapping = Mapping.from_yaml(CONV1D_CONFIG_PATH / "conv1d.mapping.yaml")
        print("Here")
        occupancies: MappingAnalysisResult = analyze_mapping.occupancies_from_mapping(
            mapping, workload
        )
        print("Here")

        for buffer, occupancy in occupancies.buffet_to_occupancy.items():
            if buffer == list(occupancies.buffet_to_occupancy.keys())[-1]:
                assert occupancy.map_ == isl.Map.read_from_str(
                    isl.DEFAULT_CONTEXT,
                    "{ [0, 0, P1, 0, 0, P0, R, 0] -> [10*P1 + P0] : "
                    "0 <= R < 3 and "
                    "((P1 = 0 and 0 <= P0 < 10) or (P1 = 1 and 0 <= P0 < 6)) }"
                )

        









                

        