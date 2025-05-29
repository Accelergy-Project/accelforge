from pathlib import Path
import unittest

from fastfusion.frontend import Specification

from fastfusion.mapper.FFM.exploration.mapper_one_einsum import get_single_einsum_sims


class TestExploration(unittest.TestCase):
    def test_mha(self):
        PARENT_DIR = Path(__file__).parent
        spec = Specification.from_yaml(
            PARENT_DIR / "four_level.arch.yaml",
            PARENT_DIR / "mha.workload.yaml",
            PARENT_DIR / "mha.renames.yaml",
        )

        workload = spec.workload

        einsum_name = "K"
        einsum = workload.einsums[einsum_name]
        rank_variables = einsum.rank_variables
        rank_variable_to_size = {r: 16 for r in rank_variables}

        # If there are two back-to-back storages for the same tensor & the outer is
        # optional, then it is invalid.
        import time

        t0 = time.time()
        mappings_count = 0
        n_mappings = 0

        # pr = cProfile.Profile()
        # pr.enable()

        sims = get_single_einsum_sims(spec, "Q", rank_variable_to_size)
        print(sims)