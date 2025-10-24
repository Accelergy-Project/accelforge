import os
from pathlib import Path
from pprint import pformat
import unittest

from ruamel.yaml import YAML
import islpy as isl

from fastfusion.frontend.workload import Workload
from fastfusion.frontend.workload._isl import get_rank_variable_bounds

from fastfusion.frontend.mapping import Mapping

from fastfusion.model.looptree.reuse.isl.mapping_to_isl import analyze_mapping
from fastfusion.model.looptree.reuse.isl.mapping_to_isl.types import (
    MappingAnalysisResult,
)

TEST_CONFIG_PATH: Path = Path(__file__).parent / "configs"


def to_isl_maps(obj) -> dict:
    def _to_isl_maps(obj):
        """Recursively convert string ISL maps to isl.Map; leave others alone."""
        if isinstance(obj, str):
            return isl.Map.read_from_str(isl.DEFAULT_CONTEXT, obj)
        if isinstance(obj, dict):
            return {k: _to_isl_maps(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_isl_maps(v) for v in obj]
        return obj
    return _to_isl_maps(obj) # type: ignore

class TestMappingToIsl(unittest.TestCase):

    def test_conv1d(self):
        # Loads in the CONV1D Config
        CONV1D_CONFIG_PATH: Path = TEST_CONFIG_PATH
        workload: Workload = Workload.from_yaml(
            CONV1D_CONFIG_PATH / "conv1d.workload.yaml"
        )

        mapping: Mapping = Mapping.from_yaml(CONV1D_CONFIG_PATH / "conv1d.mapping.yaml")
        occupancies: MappingAnalysisResult = analyze_mapping.occupancies_from_mapping(
            mapping, workload
        )

        for buffer, occupancy in occupancies.buffet_to_occupancy.items():
            if buffer == list(occupancies.buffet_to_occupancy.keys())[-1]:
                soln: isl.Map = isl.Map.read_from_str(
                    isl.DEFAULT_CONTEXT,
                    "{ [P1, P0, R] -> O[P=8*P1 + P0] : "
                    "0 <= R < 3 and 0 <= P1 < 2 and 0 <= P0 < 8}",
                )
                assert occupancy.map_ == soln

    def test_two_conv1d(self):
        # Loads in the CONV1D Config
        CONV1D_CONFIG_PATH: Path = TEST_CONFIG_PATH
        workload: Workload = Workload.from_yaml(
            CONV1D_CONFIG_PATH / "two_conv1d.workload.yaml"
        )

        mapping: Mapping = Mapping.from_yaml(
            CONV1D_CONFIG_PATH / "two_conv1d.mapping.yaml"
    )
        occupancies: MappingAnalysisResult = analyze_mapping.occupancies_from_mapping(
            mapping, workload
        )
        # Load expected solutions (YAML file with string ISL maps)
        expected_path: Path = CONV1D_CONFIG_PATH / "two_conv1d.expected.yaml"
        yaml: YAML = YAML(typ='safe')

        with open(expected_path, 'r') as f:
            solns: dict = to_isl_maps(yaml.load(f))['two_conv1d.mapping.yaml']
        # pprint(solns)

        errors: list = []
        try:
            for buffer, occupancy in occupancies.buffet_to_occupancy.items():
                soln = solns[repr(buffer)]
                assert occupancy.map_ == soln, (
                    f"{buffer} should hold:\n{soln}\ninstead holds:\n{occupancy.map_}"
                )
        except AssertionError as e:
            errors.append(e)
        
        assert len(errors) == 0, pformat(errors)
