"""
Some non-exhaustive tests to make sure core functionality of the non-fusing and
fusing portions of ISL tiling is not violated under changes.
"""

from pathlib import Path
from pprint import pformat
import unittest

import islpy as isl

from fastfusion.frontend.workload import Workload
from fastfusion.frontend.mapping import Mapping

from fastfusion.model._looptree.reuse.isl.mapping_to_isl import analyze_mapping
from fastfusion.model._looptree.reuse.isl.mapping_to_isl.types import (
    MappingAnalysisResult,
)

from .util import TEST_CONFIG_PATH, load_solutions


class TestMappingToIsl(unittest.TestCase):
    """
    Tests taking a `Mapping` and `Workload` and converting it into relevant isl
    objects.
    """

    def test_conv1d(self):
        """
        Non-fusing one 1-dimensional convolution test.
        """
        # Loads in the CONV1D Config
        config_path: Path = TEST_CONFIG_PATH / "conv1d"
        workload: Workload = Workload.from_yaml(config_path / "conv1d.workload.yaml")

        mapping: Mapping = Mapping.from_yaml(config_path / "conv1d.mapping.yaml")
        occupancies: MappingAnalysisResult = analyze_mapping.occupancies_from_mapping(
            mapping, workload
        )

        for buffer, occupancy in occupancies.buffet_to_occupancy.items():
            if buffer == list(occupancies.buffet_to_occupancy.keys())[-1]:
                soln: isl.Map = isl.Map.read_from_str(
                    isl.DEFAULT_CONTEXT,
                    "{ MAC_spacetime[P1, P0, R] -> O[P=8*P1 + P0] : "
                    "0 <= R < 3 and 0 <= P1 < 2 and 0 <= P0 < 8}",
                )
                assert occupancy.map_ == soln

    def test_two_conv1d(self):
        """
        Fusing two 1-dimensional convolutions test.
        """
        # Loads in the CONV1D Config
        config_path: Path = TEST_CONFIG_PATH / "two_conv1d"
        workload: Workload = Workload.from_yaml(
            config_path / "two_conv1d.workload.yaml"
        )

        mapping: Mapping = Mapping.from_yaml(config_path / "two_conv1d.mapping.yaml")
        occupancies: MappingAnalysisResult = analyze_mapping.occupancies_from_mapping(
            mapping, workload
        )
        solns: dict = load_solutions(config_path / "two_conv1d.expected.yaml")[
            "mapping_to_isl"
        ]

        errors: list = []
        buffers_seen: set = set()
        for buffer, occupancy in occupancies.buffet_to_occupancy.items():
            try:
                soln = solns[repr(buffer)]
                assert (
                    occupancy.map_ == soln
                ), f"{buffer} should hold:\n{soln}\ninstead holds:\n{occupancy.map_}"
            except (AssertionError, KeyError) as e:
                errors.append(e)
            buffers_seen.add(repr(buffer))

        buffers_known: set = set(solns.keys())
        try:
            assert buffers_seen == buffers_known, (
                f"Buffers Missing: {pformat(buffers_seen - buffers_known)}\n"
                f"Buffers Extra: {pformat(buffers_known - buffers_seen)}"
            )
        except AssertionError as e:
            errors.append(e)

        assert len(errors) == 0, pformat(errors)
