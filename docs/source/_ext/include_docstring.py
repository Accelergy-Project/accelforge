from docutils import nodes
from docutils.parsers.rst import Directive, roles
from docutils.statemachine import ViewList
from sphinx.util.nodes import nested_parse_with_titles
import importlib
import inspect
import ast
import re


class IncludeDocstring(Directive):
    required_arguments = 1  # fully-qualified name
    option_spec = {
        'decapitalize': lambda x: True,  # Flag option, presence means True
        'inline': lambda x: True  # Flag option for inline rendering
    }

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
        for idx, part in enumerate(rest):
            # Check if we're at the last part and current obj might have the part as a field
            is_last = (idx == len(rest) - 1)

            # Try to get docstring from Pydantic/annotated field BEFORE moving to next attribute
            docstring = self._try_get_field_docstring(obj, part)
            if docstring:
                if is_last:
                    return self._parse_docstring(docstring)
                # If not last, we found the field but need to continue traversing
                # Get the field's type and continue with that
                field_type = self._get_field_type(obj, part)
                if field_type:
                    obj = field_type
                    continue

            # Normal attribute access - move to next level
            if hasattr(obj, part):
                next_obj = getattr(obj, part)
                # If we got an instance, use its class for field lookups
                if not inspect.isclass(next_obj) and not inspect.ismodule(next_obj):
                    obj = type(next_obj)
                else:
                    obj = next_obj
                continue

            # If we couldn't find it as an attribute and already checked fields, give up
            return []

        # Fallback: normal __doc__ for the final object
        doc = getattr(obj, "__doc__", None)
        return self._parse_docstring(doc) if doc else []

    def _try_get_field_docstring(self, obj, field_name):
        """Try to extract docstring from a field in obj."""
        # --- Check if obj is a class with annotations ---
        if inspect.isclass(obj) and hasattr(obj, "__annotations__") and field_name in obj.__annotations__:
            # Try to extract inline docstring using AST
            try:
                source = inspect.getsource(obj)
                tree = ast.parse(source)

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        for i, item in enumerate(node.body):
                            # Look for annotated assignment (attribute with type hint)
                            if isinstance(item, ast.AnnAssign):
                                if isinstance(item.target, ast.Name) and item.target.id == field_name:
                                    # Check if next item is a string (docstring)
                                    if i + 1 < len(node.body):
                                        next_item = node.body[i + 1]
                                        if isinstance(next_item, ast.Expr) and isinstance(next_item.value, ast.Constant):
                                            if isinstance(next_item.value.value, str):
                                                return next_item.value.value
            except (OSError, TypeError, SyntaxError):
                pass

        # --- Pydantic v2 field ---
        if hasattr(obj, "model_fields") and field_name in obj.model_fields:
            field = obj.model_fields[field_name]
            return (
                field.description
                or (field.json_schema_extra or {}).get("description")
            )

        # --- Pydantic v1 field ---
        if hasattr(obj, "__fields__") and field_name in obj.__fields__:
            field = obj.__fields__[field_name]
            if hasattr(field, "description"):
                return field.description

        return None

    def _get_field_type(self, obj, field_name):
        """Get the type/class of a field so we can continue traversing."""
        # Pydantic v2
        if hasattr(obj, "model_fields") and field_name in obj.model_fields:
            field = obj.model_fields[field_name]
            if hasattr(field, "annotation"):
                return field.annotation

        # Pydantic v1
        if hasattr(obj, "__fields__") and field_name in obj.__fields__:
            field = obj.__fields__[field_name]
            if hasattr(field, "outer_type_"):
                return field.outer_type_
            elif hasattr(field, "type_"):
                return field.type_

        # Regular annotations
        if inspect.isclass(obj) and hasattr(obj, "__annotations__") and field_name in obj.__annotations__:
            return obj.__annotations__[field_name]

        return None

    def _parse_docstring(self, docstring):
        """Parse a docstring as reStructuredText."""
        if not docstring:
            return []

        # Decapitalize first letter if option is set
        if 'decapitalize' in self.options:
            docstring = self._decapitalize_first_letter(docstring)

        # If inline mode, return just the text without block parsing
        if 'inline' in self.options:
            # Strip whitespace and collapse to single line
            text = ' '.join(docstring.split())
            # Parse as inline RST to handle inline markup like ``code``
            result = ViewList()
            result.append(text, '<include-docstring>', 0)

            # Use a paragraph node for inline parsing
            para = nodes.paragraph()
            para.document = self.state.document
            self.state.nested_parse(result, 0, para)

            # Return the inline contents of the paragraph
            return para.children

        # Dedent the docstring while preserving blank lines and relative indentation
        lines = docstring.splitlines()

        # Find the minimum indentation (excluding blank lines)
        min_indent = float('inf')
        for line in lines:
            stripped = line.lstrip()
            if stripped:  # Only consider non-blank lines
                indent = len(line) - len(stripped)
                min_indent = min(min_indent, indent)

        # Remove the common indentation
        if min_indent < float('inf'):
            dedented_lines = []
            for line in lines:
                if line.strip():  # Non-blank line
                    dedented_lines.append(line[min_indent:])
                else:  # Blank line
                    dedented_lines.append('')
        else:
            dedented_lines = lines

        # Parse the docstring as reStructuredText
        result = ViewList()

        for i, line in enumerate(dedented_lines):
            result.append(line, '<include-docstring>', i)

        node = nodes.section()
        node.document = self.state.document
        nested_parse_with_titles(self.state, result, node)

        return node.children

    def _decapitalize_first_letter(self, text):
        """Decapitalize the first letter of the text."""
        if not text:
            return text

        # Find the first letter (skip whitespace)
        for i, char in enumerate(text):
            if char.isalpha():
                return text[:i] + char.lower() + text[i+1:]

        return text


def docstring_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """
    Inline role to include docstrings.
    Usage: :docstring:`module.Class.attribute` or :docstring-lower:`module.Class.attribute`
    """
    fqname = text.strip()
    decapitalize = name == 'docstring-lower'

    # Get the docstring
    docstring = _get_docstring(fqname)

    if not docstring:
        msg = inliner.reporter.warning(
            f'Could not find docstring for {fqname}',
            line=lineno)
        prb = inliner.problematic(rawtext, rawtext, msg)
        return [prb], [msg]

    # Decapitalize if requested
    if decapitalize:
        docstring = _decapitalize_first_letter(docstring)

    # Collapse to single line
    processed_text = ' '.join(docstring.split())

    # Parse the text as inline RST
    # The inliner.parse() method handles inline markup
    nodes_list, messages = inliner.parse(processed_text, lineno, inliner, inliner.parent)

    return nodes_list, messages


def _get_docstring(fqname):
    """Get docstring from a fully qualified name."""
    parts = fqname.split(".")

    # Import the module
    module = None
    for i in range(len(parts), 0, -1):
        try:
            module = importlib.import_module(".".join(parts[:i]))
            rest = parts[i:]
            break
        except ImportError:
            continue

    if module is None:
        return None

    obj = module
    for idx, part in enumerate(rest):
        # Check if we're at the last part
        is_last = (idx == len(rest) - 1)

        # Try to get docstring from Pydantic/annotated field BEFORE moving to next attribute
        docstring = _try_get_field_docstring(obj, part)
        if docstring:
            if is_last:
                return docstring
            # If not last, we found the field but need to continue traversing
            # Don't return yet, but also don't try normal attribute access for this field
            # Instead, get the field's type and continue with that
            field_type = _get_field_type(obj, part)
            if field_type:
                obj = field_type
                continue

        # Normal attribute access - move to next level
        if hasattr(obj, part):
            next_obj = getattr(obj, part)
            # If we got an instance, use its class for field lookups
            if not inspect.isclass(next_obj) and not inspect.ismodule(next_obj):
                obj = type(next_obj)
            else:
                obj = next_obj
            continue

        # If we couldn't find it as an attribute and already checked fields, give up
        return None

    # Fallback: normal __doc__ for the final object
    return getattr(obj, "__doc__", None)


def _try_get_field_docstring(obj, field_name):
    """Try to extract docstring from a field in obj."""
    # --- Check if obj is a class with annotations ---
    if inspect.isclass(obj) and hasattr(obj, "__annotations__") and field_name in obj.__annotations__:
        # Try to extract inline docstring using AST
        try:
            source = inspect.getsource(obj)
            tree = ast.parse(source)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for i, item in enumerate(node.body):
                        if isinstance(item, ast.AnnAssign):
                            if isinstance(item.target, ast.Name) and item.target.id == field_name:
                                if i + 1 < len(node.body):
                                    next_item = node.body[i + 1]
                                    if isinstance(next_item, ast.Expr) and isinstance(next_item.value, ast.Constant):
                                        if isinstance(next_item.value.value, str):
                                            return next_item.value.value
        except (OSError, TypeError, SyntaxError):
            pass

    # --- Pydantic v2 field ---
    if hasattr(obj, "model_fields") and field_name in obj.model_fields:
        field = obj.model_fields[field_name]
        return (
            field.description
            or (field.json_schema_extra or {}).get("description")
        )

    # --- Pydantic v1 field ---
    if hasattr(obj, "__fields__") and field_name in obj.__fields__:
        field = obj.__fields__[field_name]
        if hasattr(field, "description"):
            return field.description

    return None


def _get_field_type(obj, field_name):
    """Get the type/class of a field so we can continue traversing."""
    # Pydantic v2
    if hasattr(obj, "model_fields") and field_name in obj.model_fields:
        field = obj.model_fields[field_name]
        if hasattr(field, "annotation"):
            return field.annotation

    # Pydantic v1
    if hasattr(obj, "__fields__") and field_name in obj.__fields__:
        field = obj.__fields__[field_name]
        if hasattr(field, "outer_type_"):
            return field.outer_type_
        elif hasattr(field, "type_"):
            return field.type_

    # Regular annotations
    if inspect.isclass(obj) and hasattr(obj, "__annotations__") and field_name in obj.__annotations__:
        return obj.__annotations__[field_name]

    return None


def _decapitalize_first_letter(text):
    """Decapitalize the first letter of the text."""
    if not text:
        return text

    for i, char in enumerate(text):
        if char.isalpha():
            return text[:i] + char.lower() + text[i+1:]

    return text


def setup(app):
    app.add_directive("include-docstring", IncludeDocstring)
    app.add_role("docstring", docstring_role)
    app.add_role("docstring-lower", docstring_role)