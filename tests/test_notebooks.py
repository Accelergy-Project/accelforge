import unittest
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError


try:
    from .paths import NOTEBOOKS_DIR
except ImportError:
    from paths import NOTEBOOKS_DIR

TUTORIALS_DIR = NOTEBOOKS_DIR / "tutorials"


class TestNotebooks(unittest.TestCase):
    """Test that all tutorial notebooks execute without errors."""

    @classmethod
    def setUpClass(cls):
        """Discover all notebooks in the tutorials directory."""
        all_notebooks = TUTORIALS_DIR.rglob("*.ipynb")
        # Filter out .ipynb_checkpoints directories
        cls.notebooks = sorted(
            [nb for nb in all_notebooks if ".ipynb_checkpoints" not in nb.parts]
        )
        if not cls.notebooks:
            raise ValueError(f"No notebooks found in {NOTEBOOKS_DIR}")

    def _execute_notebook(self, notebook_path: Path):
        """Execute a single notebook and return any errors."""
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        # Configure the executor
        ep = ExecutePreprocessor(
            timeout=600,  # 10 minutes per cell
            kernel_name="python3",
            allow_errors=False,  # Fail on first error
        )

        # Execute the notebook
        try:
            ep.preprocess(nb, {"metadata": {"path": str(notebook_path.parent)}})
            return None  # Success
        except CellExecutionError as e:
            return str(e)
        except Exception as e:
            return f"Unexpected error: {type(e).__name__}: {str(e)}"

    def test_all_notebooks(self):
        """Test that all notebooks execute successfully."""
        errors = {}

        for notebook_path in self.notebooks:
            with self.subTest(notebook=notebook_path.name):
                error = self._execute_notebook(notebook_path)
                if error:
                    errors[notebook_path.name] = error
                    self.fail(f"Notebook {notebook_path.name} failed:\n{error}")

        # If we get here, all notebooks passed
        self.assertEqual(
            len(errors), 0, f"Some notebooks failed: {list(errors.keys())}"
        )


if __name__ == "__main__":
    unittest.main()
