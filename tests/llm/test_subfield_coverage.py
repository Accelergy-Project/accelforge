import unittest
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parents[1]
LLM_DIR = Path(__file__).resolve().parent


def _extract_subfields() -> set[str]:
    subfields: set[str] = set()

    for test_file in (TESTS_DIR).glob("test_*.py"):
        subfields.add(test_file.stem.replace("test_", "", 1))

    for test_file in (TESTS_DIR / "isl" / "mapper").glob("test_*.py"):
        stem = test_file.stem.replace("test_", "", 1)
        subfields.add(f"isl/mapper/{stem}")

    for test_file in (TESTS_DIR / "isl" / "distributed").glob("test_*.py"):
        stem = test_file.stem.replace("test_", "", 1)
        subfields.add(f"isl/distributed/{stem}")

    for test_file in (TESTS_DIR / "vibe_see_readme_in_this_dir").glob("test_*.py"):
        stem = test_file.stem.replace("test_", "", 1)
        subfields.add(f"vibe_see_readme_in_this_dir/{stem}")

    return subfields


class TestLLMSubfieldCoverage(unittest.TestCase):
    def test_every_existing_subfield_has_llm_cases_manifest(self):
        missing: list[str] = []
        for subfield in sorted(_extract_subfields()):
            manifest = LLM_DIR / subfield / "cases.yaml"
            if not manifest.is_file():
                missing.append(subfield)

        self.assertEqual(
            missing,
            [],
            f"Missing llm cases.yaml for subfields: {missing}",
        )


if __name__ == "__main__":
    unittest.main()
