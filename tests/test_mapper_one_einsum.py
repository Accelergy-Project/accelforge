from pathlib import Path
import unittest

from fastfusion.frontend import Specification

from fastfusion.mapper.FFM.exploration.mapper_multi_einsum import get_sims
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

        sims, decompress_data = get_sims(spec, einsum_names=[einsum_name])

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

        def tagger(pmapping):
            return get_one_split_tag(pmapping,
                                     workload.intermediate_tensors(),
                                     "MainMemory")

        sims, decompress_data = get_sims(spec, einsum_names=["Q"], tagger=tagger)
        for sim in sims['Q']:
            self.assertEqual(
                TagMatch(sim.compatibility.tags),
                TagMatch(Tags((ONE_SPLIT,)))
            )
