from pathlib import Path
from pprint import pformat
import unittest

import islpy as isl

from fastfusion.frontend.workload import Workload
from fastfusion.frontend.workload._isl import get_rank_variable_bounds

from fastfusion.frontend.mapping import Mapping

from fastfusion.model.looptree.reuse.isl.mapping_to_isl import analyze_mapping
from fastfusion.model.looptree.reuse.isl.mapping_to_isl.types import MappingAnalysisResult

TEST_CONFIG_PATH: Path = Path(__file__).parent / "configs"

class TestMappingToIsl(unittest.TestCase):

    def test_conv1d(self):
        # Loads in the CONV1D Config
        CONV1D_CONFIG_PATH: Path = TEST_CONFIG_PATH
        workload: Workload = Workload.from_yaml(CONV1D_CONFIG_PATH / "conv1d.workload.yaml")

        mapping: Mapping = Mapping.from_yaml(CONV1D_CONFIG_PATH / "conv1d.mapping.yaml")
        occupancies: MappingAnalysisResult = analyze_mapping.occupancies_from_mapping(
            mapping, workload
        )

        for buffer, occupancy in occupancies.buffet_to_occupancy.items():
            if buffer == list(occupancies.buffet_to_occupancy.keys())[-1]:
                soln: isl.Map = isl.Map.read_from_str(
                    isl.DEFAULT_CONTEXT,
                    "{ [P1, P0, R] -> [P=8*P1 + P0] : "
                    "0 <= R < 3 and 0 <= P1 < 2 and 0 <= P0 < 8}"
                )
                print(occupancy.map_)
                assert occupancy.map_ == soln

    def test_two_conv1d(self):
        # Loads in the CONV1D Config
        CONV1D_CONFIG_PATH: Path = TEST_CONFIG_PATH
        workload: Workload = Workload.from_yaml(CONV1D_CONFIG_PATH / "two_conv1d.workload.yaml")

        mapping: Mapping = Mapping.from_yaml(CONV1D_CONFIG_PATH / "two_conv1d.mapping.yaml")
        occupancies: MappingAnalysisResult = analyze_mapping.occupancies_from_mapping(
            mapping, workload
        )

        print(pformat(occupancies))
        for buffer, occupancy in occupancies.buffet_to_occupancy.items():
            if buffer == list(occupancies.buffet_to_occupancy.keys())[-1]:
                print(occupancy.tags)
                soln: isl.Map = isl.Map.read_from_str(
                    isl.DEFAULT_CONTEXT,
                    "{ [P1, P0, R] -> [P=8*P1 + P0] : "
                    "0 <= R < 3 and 0 <= P1 < 2 and 0 <= P0 < 8}"
                )
                # assert occupancy.map_ == soln       
                print(occupancy.map_) 









                

        