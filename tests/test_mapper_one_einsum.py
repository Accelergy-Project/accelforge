from pathlib import Path
import unittest

from fastfusion.frontend import Specification

from fastfusion.mapper.FFM.exploration.mapper_one_einsum import get_single_einsum_sims
from fastfusion.mapper.FFM.exploration.mapping_filter_tags import get_one_split_tag
from fastfusion.mapper.FFM.exploration.mapping_filter_tags.onesplit import ONE_SPLIT
from fastfusion.mapper.FFM.tags import Tags, TagMatch


PARENT_DIR = Path(__file__).parent


class TestExploration(unittest.TestCase):
    def test_mha(self):
        spec = Specification.from_yaml(
            PARENT_DIR / "four_level.arch.yaml",
            PARENT_DIR / "mha.workload.yaml",
            PARENT_DIR / "mha.renames.yaml",
        )
        spec.estimate_energy_area()

        workload = spec.workload

        einsum_name = "K"
        einsum = workload.einsums[einsum_name]
        rank_variables = einsum.rank_variables
        rank_variable_to_size = {r: 16 for r in rank_variables}

        sims, decompress_data = get_single_einsum_sims(spec, "Q", rank_variable_to_size)

    def test_mha_with_tags(self):
        spec = Specification.from_yaml(
            PARENT_DIR / "four_level.arch.yaml",
            PARENT_DIR / "mha.workload.yaml",
            PARENT_DIR / "mha.renames.yaml",
        )
        spec.estimate_energy_area()

        workload = spec.workload

        einsum_name = "K"
        einsum = workload.einsums[einsum_name]
        rank_variables = einsum.rank_variables
        rank_variable_to_size = {r: 16 for r in rank_variables}

        def tagger(pmapping):
            return get_one_split_tag(pmapping,
                                     workload.intermediate_tensors(),
                                     "MainMemory")

        sims, decompress_data = get_single_einsum_sims(spec, "Q", rank_variable_to_size, tagger=tagger)
        for sim in sims['Q']:
            self.assertEqual(
                TagMatch(sim.compatibility.tags),
                TagMatch(Tags((ONE_SPLIT,)))
            )