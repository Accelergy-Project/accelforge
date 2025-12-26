from docutils import nodes
from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList
import json
import os
import sys
from paths import PROJECT_ROOT

class IncludeNotebook(Directive):
    """
    Directive to include marked sections from Jupyter notebooks.

    Usage:
        .. include-notebook:: path/to/notebook.ipynb
           :name: example_mac
           :language: python
    """

    required_arguments = 1  # path to notebook
    option_spec = {
        "name": directives.unchanged_required,
        "language": directives.unchanged,
    }

    def run(self):
        env = self.state.document.settings.env
        notebook_path = self.arguments[0]
        marker_name = self.options.get("name")
        language = self.options.get("language", "python")

        notebook_path = os.path.normpath(
            os.path.join(PROJECT_ROOT, notebook_path)
        )

        # Resolve relative paths
        if not os.path.isabs(notebook_path):
            source_dir = os.path.dirname(self.state.document.current_source)
            notebook_path = os.path.normpath(os.path.join(source_dir, notebook_path))

        if not os.path.exists(notebook_path):
            error = self.state_machine.reporter.error(
                f"Notebook file not found: {notebook_path}",
                nodes.literal_block("", ""),
                line=self.lineno,
            )
            return [error]

        # Read and parse the notebook
        try:
            with open(notebook_path, "r", encoding="utf-8") as f:
                notebook = json.load(f)
        except Exception as e:
            error = self.state_machine.reporter.error(
                f"Error reading notebook: {e}",
                nodes.literal_block("", ""),
                line=self.lineno,
            )
            return [error]

        # Extract all marked sections
        content_sections = self._extract_marked_sections(notebook, marker_name)

        if not content_sections:
            warning = self.state_machine.reporter.warning(
                f'Marker "# < DOC_INCLUDE_MARKER > {marker_name}" not found in {notebook_path}',
                nodes.literal_block("", ""),
                line=self.lineno,
            )
            return [warning]

        # Concatenate all sections with double newline separator
        content = "\n\n".join(content_sections)

        # Create a code block node
        code_block = nodes.literal_block(content, content)
        code_block["language"] = language
        return [code_block]

    def _extract_marked_sections(self, notebook, marker_name):
        """
        Extract content between DOC_INCLUDE_MARKER and DOC_INCLUDE_END (or end of cell).
        Returns a list of all matching sections.
        """
        start_marker = f"# < DOC_INCLUDE_MARKER > {marker_name}"
        end_marker = "# < DOC_INCLUDE_END >"
        sections = []

        for cell_idx, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue

            source = cell.get("source", [])

            # Handle both string and list formats
            if isinstance(source, str):
                lines = source.split("\n")
            else:
                lines = source

            # Look for all start markers in this cell
            for i, line in enumerate(lines):
                # Remove trailing newlines for comparison
                line_stripped = line.rstrip("\n")

                if start_marker in line_stripped:
                    # Found the start marker, collect lines until end marker or end of cell
                    content_lines = []
                    for j in range(i + 1, len(lines)):
                        if end_marker in lines[j]:
                            break
                        content_lines.append(lines[j].rstrip("\n"))

                    # Join lines and strip trailing whitespace
                    content = "\n".join(content_lines).rstrip()
                    if content:  # Only add non-empty sections
                        sections.append(content)

        return sections


def setup(app):
    app.add_directive("include-notebook", IncludeNotebook)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }