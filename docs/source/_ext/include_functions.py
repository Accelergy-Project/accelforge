from docutils import nodes
from docutils.parsers.rst import Directive
import importlib
import inspect


class IncludeFunctions(Directive):
    required_arguments = 1  # fully-qualified name

    def run(self):
        fqname = self.arguments[0]
        parts = fqname.split(".")

        # --- progressively import the longest valid module ---
        module = None
        for i in range(len(parts), 0, -1):
            try:
                module = importlib.import_module(".".join(parts[:i]))
                rest = parts[i:]
                break
            except ImportError:
                continue

        if module is None:
            return []

        obj = module
        for part in rest:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return []

        # --- Collect all functions ---
        functions = {}  # {func_name: {'signature': ..., 'doc': ...}}

        # Get all members
        for name, member in inspect.getmembers(obj):
            # Skip underscore-prefixed and special functions
            if self._should_skip_function(name):
                continue

            # Check if it's a function or method
            if inspect.isfunction(member) or inspect.ismethod(member):
                doc = self._extract_summary_doc(member)
                functions[name] = {
                    'signature': self._get_signature(member),
                    'doc': doc
                }

        # --- Build bullet list ---
        if not functions:
            return []

        bullet_list = nodes.bullet_list()
        for func_name in sorted(functions.keys()):
            func_info = functions[func_name]
            list_item = nodes.list_item()
            para = nodes.paragraph()

            # Function name as :py:func: role for clickable links
            from sphinx.addnodes import pending_xref
            refnode = pending_xref(
                '',
                refdomain='py',
                reftype='func',
                reftarget=fqname + '.' + func_name,
                refwarn=True
            )
            # refnode += nodes.literal('', f"{func_name}{func_info['signature']}", classes=['xref', 'py', 'py-func'])
            refnode += nodes.literal('', f"{func_name}()", classes=['xref', 'py', 'py-func'])
            para += refnode

            # Docstring
            if func_info['doc']:
                para += nodes.Text(f": {func_info['doc']}")

            list_item += para
            bullet_list += list_item

        return [bullet_list]

    def _should_skip_function(self, func_name):
        """Check if a function should be skipped."""
        return (
            func_name.startswith('_') or
            func_name.startswith('__')
        )

    def _get_signature(self, func):
        """Get the function signature as a string."""
        try:
            sig = inspect.signature(func)
            return str(sig)
        except (ValueError, TypeError):
            return "()"

    def _extract_summary_doc(self, func):
        """Extract the summary part of the docstring, stopping at section headers."""
        # Get raw docstring
        doc = getattr(func, "__doc__", None)
        if not doc:
            return None

        # Clean and process the docstring
        doc = inspect.cleandoc(doc)

        section_headers = [
            "Parameters", "Returns", "Postcondition", "Raises",
            "Yields", "Examples", "Notes", "See Also", "Attributes",
            "Methods", "References", "Warnings", "Args", "Return",
            "Keyword Arguments", "Other Parameters"
        ]

        lines = doc.split('\n')
        summary_lines = []

        for line in lines:
            stripped = line.strip()

            # Stop at empty line followed by section header pattern
            if not stripped:
                # Check if we've already collected some content
                if summary_lines:
                    # Peek ahead to see if next non-empty line is a section
                    continue

            # Check if this line is a section header
            if stripped in section_headers:
                break

            # Check if this line ends with a colon and might be a section header
            if stripped.endswith(':'):
                header_candidate = stripped[:-1].strip()
                if header_candidate in section_headers:
                    break

            # Check for common docstring section patterns (underlined headers)
            if stripped and all(c in '-=~' for c in stripped):
                break

            summary_lines.append(line)

        # Join lines and clean up
        summary = '\n'.join(summary_lines).strip()

        # Replace multiple newlines/spaces with single space for better formatting
        summary = ' '.join(summary.split())

        return summary if summary else None


def setup(app):
    app.add_directive("include-functions", IncludeFunctions)
