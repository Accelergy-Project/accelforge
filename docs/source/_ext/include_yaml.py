from docutils import nodes
from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList
import re

from paths import PROJECT_ROOT
import os


def grab_from_yaml_file(
    yaml_file: str,
    startfrom: str | None = None,
    same_indent: bool = True,
    include_lines_before: int = 0,
) -> str:
    """
    Grab a section from a YAML file.

    Args:
        yaml_file:
            The path to the YAML file.
        startfrom:
            The string to start from. If None, return the entire file.
        same_indent:
            Whether to include lines with the same indentation as the startfrom line, or
            only lines with >= indentation.
        include_lines_before:
            The number of lines to include before the startfrom line.

    Returns:
        The section of the YAML file as a string.
    """
    with open(yaml_file, "r") as f:
        contents = f.readlines()
    start, end = 0, len(contents)
    n_whitespace = 0

    if startfrom is not None:
        for i, line in enumerate(contents):
            if re.findall(r"\b\s*" + startfrom + r"\b", line):
                start = i
                n_whitespace = len(re.findall(r"^\s*", line)[0])
                break
        else:
            raise ValueError(f"{startfrom} not found in {yaml_file}")
        for i, line in enumerate(contents[start + 1 :]):
            if not line.strip():
                continue
            ws = len(re.findall(r"^\s*", line)[0])
            if ws < n_whitespace or (not same_indent and ws == n_whitespace):
                end = start + i + 1
                break

    contents = [c[n_whitespace:] for c in contents[start - include_lines_before : end]]
    return "".join(contents)


class IncludeYaml(Directive):
    """
    Directive to include content from a YAML file with optional filtering.

    Usage:
        .. include-yaml:: path/to/file.yaml
           :startfrom: section_name
           :same-indent:
           :include-lines-before: 2
    """

    required_arguments = 1  # The YAML file path
    optional_arguments = 0
    option_spec = {
        'startfrom': directives.unchanged,
        'same-indent': directives.flag,
        'include-lines-before': directives.nonnegative_int,
    }
    has_content = False

    def run(self):
        yaml_file = os.path.join(PROJECT_ROOT, self.arguments[0])
        startfrom = self.options.get('startfrom', None)
        same_indent = 'same-indent' in self.options
        include_lines_before = self.options.get('include-lines-before', 0)

        try:
            content = grab_from_yaml_file(
                yaml_file=yaml_file,
                startfrom=startfrom,
                same_indent=same_indent,
                include_lines_before=include_lines_before,
            )
        except Exception as e:
            error = self.state_machine.reporter.error(
                f'Error reading YAML file "{yaml_file}": {str(e)}',
                nodes.literal_block('', ''),
                line=self.lineno
            )
            return [error]

        # Create a literal block with YAML syntax highlighting
        literal = nodes.literal_block(content, content)
        literal['language'] = 'yaml'

        return [literal]


def setup(app):
    """
    Setup function for the Sphinx extension.

    Add this to your conf.py:
        extensions = ['path.to.this.module']
    """
    app.add_directive('include-yaml', IncludeYaml)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }