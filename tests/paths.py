import os
from pathlib import Path

# Use __file__ to get absolute paths relative to this file's location
_TESTS_DIR = Path(__file__).parent
_REPO_ROOT = _TESTS_DIR.parent

EXAMPLES_DIR = _REPO_ROOT / "examples"
CURRENT_DIR = _TESTS_DIR
NOTEBOOKS_DIR = _REPO_ROOT / "notebooks"
