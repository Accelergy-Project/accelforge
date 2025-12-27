from docutils import nodes
from docutils.parsers.rst import Directive
import importlib
import inspect
import ast
import typing


class IncludeAttrs(Directive):
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

        # --- Collect all attributes with their metadata ---
        attrs = {}  # {attr_name: {'type': ..., 'default': ..., 'doc': ...}}

        # Check if obj is a class
        if not inspect.isclass(obj):
            return []

        # --- Get type annotations ---
        annotations = getattr(obj, "__annotations__", {})

        # --- Extract inline docstrings and defaults using AST ---
        try:
            source = inspect.getsource(obj)
            tree = ast.parse(source)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for i, item in enumerate(node.body):
                        # Look for annotated assignment (attribute with type hint)
                        if isinstance(item, ast.AnnAssign):
                            if isinstance(item.target, ast.Name):
                                attr_name = item.target.id

                                # Skip underscore-prefixed and excluded attributes
                                if self._should_skip_attr(attr_name):
                                    continue

                                if attr_name not in attrs:
                                    attrs[attr_name] = {'type': None, 'default': None, 'doc': None}

                                # Get type annotation
                                if attr_name in annotations:
                                    attrs[attr_name]['type'] = annotations[attr_name]

                                # Get default value
                                if item.value is not None:
                                    try:
                                        attrs[attr_name]['default'] = ast.unparse(item.value)
                                    except:
                                        attrs[attr_name]['default'] = repr(item.value)

                                # Check if next item is a string (docstring)
                                if i + 1 < len(node.body):
                                    next_item = node.body[i + 1]
                                    if isinstance(next_item, ast.Expr) and isinstance(next_item.value, ast.Constant):
                                        if isinstance(next_item.value.value, str):
                                            attrs[attr_name]['doc'] = next_item.value.value.strip()
        except (OSError, TypeError, SyntaxError):
            pass

        # --- Pydantic v2 fields ---
        if hasattr(obj, "model_fields"):
            for field_name, field in obj.model_fields.items():
                # Skip underscore-prefixed and excluded attributes
                if self._should_skip_attr(field_name):
                    continue

                if field_name not in attrs:
                    attrs[field_name] = {'type': None, 'default': None, 'doc': None}

                # Get type
                if hasattr(field, "annotation"):
                    attrs[field_name]['type'] = field.annotation

                # Get default
                if hasattr(field, "default") and field.default is not None:
                    attrs[field_name]['default'] = repr(field.default)
                elif hasattr(field, "default_factory") and field.default_factory is not None:
                    attrs[field_name]['default'] = f"{field.default_factory.__name__}()"

                # Get docstring
                doc = field.description or (field.json_schema_extra or {}).get("description")
                if doc:
                    attrs[field_name]['doc'] = doc

        # --- Pydantic v1 fields ---
        if hasattr(obj, "__fields__"):
            for field_name, field in obj.__fields__.items():
                # Skip underscore-prefixed and excluded attributes
                if self._should_skip_attr(field_name):
                    continue

                if field_name not in attrs:
                    attrs[field_name] = {'type': None, 'default': None, 'doc': None}

                # Get type
                if hasattr(field, "outer_type_"):
                    attrs[field_name]['type'] = field.outer_type_

                # Get default
                if hasattr(field, "default") and field.default is not None:
                    attrs[field_name]['default'] = repr(field.default)
                elif hasattr(field, "default_factory") and field.default_factory is not None:
                    attrs[field_name]['default'] = f"{field.default_factory.__name__}()"

                # Get docstring - field is already a FieldInfo object
                if hasattr(field, "description"):
                    doc = field.description
                    if doc:
                        attrs[field_name]['doc'] = doc

        # --- Build bullet list ---
        if not attrs:
            return []

        bullet_list = nodes.bullet_list()
        for attr_name in sorted(attrs.keys()):
            attr_info = attrs[attr_name]
            list_item = nodes.list_item()
            para = nodes.paragraph()

            # Attribute name as :py:attr: role for clickable links
            from sphinx.addnodes import pending_xref
            refnode = pending_xref(
                '',
                refdomain='py',
                reftype='attr',
                reftarget=fqname + '.' + attr_name,
                refwarn=True
            )
            refnode += nodes.literal('', attr_name, classes=['xref', 'py', 'py-attr'])
            para += refnode

            # # Type
            # if attr_info['type'] is not None:
            #     type_str = self._format_type(attr_info['type'])
            #     para += nodes.Text(f" ({type_str})")

            # # Default
            # if attr_info['default'] is not None:
            #     para += nodes.Text(f", default: {attr_info['default']}")

            # Docstring
            if attr_info['doc']:
                para += nodes.Text(f": {attr_info['doc']}")

            list_item += para
            bullet_list += list_item

        return [bullet_list]

    def _should_skip_attr(self, attr_name):
        """Check if an attribute should be skipped."""
        return (
            attr_name.startswith('_') or
            attr_name in ('type', 'version')
        )

    def _format_type(self, type_hint):
        """Format a type hint into a readable string."""
        if type_hint is None:
            return "Any"

        # Handle string annotations
        if isinstance(type_hint, str):
            return type_hint

        # Get the type name
        if hasattr(type_hint, "__name__"):
            return type_hint.__name__

        # Handle typing module types
        if hasattr(type_hint, "__origin__"):
            origin = type_hint.__origin__
            args = getattr(type_hint, "__args__", ())

            origin_name = getattr(origin, "__name__", str(origin))

            if args:
                args_str = ", ".join(self._format_type(arg) for arg in args)
                return f"{origin_name}[{args_str}]"
            return origin_name

        return str(type_hint)


def setup(app):
    app.add_directive("include-attrs", IncludeAttrs)
