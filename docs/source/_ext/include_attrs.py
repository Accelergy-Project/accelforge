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

        # --- Look up inheritance chain for missing docstrings ---
        for attr_name in list(attrs.keys()):
            if attrs[attr_name]['doc'] is None:
                doc = self._find_docstring_in_mro(obj, attr_name)
                if doc:
                    attrs[attr_name]['doc'] = doc

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

        # --- Look up inheritance chain for missing docstrings and track defining class ---
        attr_defining_class = {}  # Maps attr_name to the class that defines it
        for attr_name in list(attrs.keys()):
            # Find which class in the MRO actually defines this attribute
            defining_class = self._find_defining_class(obj, attr_name)
            attr_defining_class[attr_name] = defining_class

            if attrs[attr_name]['doc'] is None:
                doc = self._find_docstring_in_mro(obj, attr_name)
                if doc:
                    attrs[attr_name]['doc'] = doc

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

            # Use the defining class for the link target
            defining_class = attr_defining_class.get(attr_name)
            if defining_class:
                defining_class_name = f"{defining_class.__module__}.{defining_class.__qualname__}"
                link_target = f"{defining_class_name}.{attr_name}"
            else:
                link_target = f"{fqname}.{attr_name}"

            refnode = pending_xref(
                '',
                refdomain='py',
                reftype='obj',
                reftarget=link_target,
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

    def _find_defining_class(self, obj, attr_name):
        """Find where attribute is first defined by checking __annotations__ in __dict__."""
        mro_list = list(inspect.getmro(obj))

        # Walk MRO backwards to find first class that defines this in its own __annotations__
        for i in range(len(mro_list) - 1, -1, -1):
            base_class = mro_list[i]
            if '__annotations__' in base_class.__dict__:
                if attr_name in base_class.__dict__['__annotations__']:
                    return base_class

        return None

    def _find_docstring_in_mro(self, obj, attr_name):
        """Find docstring for an attribute by walking the MRO."""
        for base_class in inspect.getmro(obj):
            # Try Pydantic v2 first
            if hasattr(base_class, "model_fields"):
                if attr_name in base_class.model_fields:
                    field = base_class.model_fields[attr_name]
                    doc = field.description or (field.json_schema_extra or {}).get("description")
                    if doc:
                        return doc

            # Try Pydantic v1
            if hasattr(base_class, "__fields__"):
                if attr_name in base_class.__fields__:
                    field = base_class.__fields__[attr_name]
                    if hasattr(field, "description") and field.description:
                        return field.description

            # Try AST parsing for inline docstrings
            try:
                source = inspect.getsource(base_class)
                tree = ast.parse(source)

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        for i, item in enumerate(node.body):
                            if isinstance(item, ast.AnnAssign):
                                if isinstance(item.target, ast.Name) and item.target.id == attr_name:
                                    # Check if next item is a docstring
                                    if i + 1 < len(node.body):
                                        next_item = node.body[i + 1]
                                        if isinstance(next_item, ast.Expr) and isinstance(next_item.value, ast.Constant):
                                            if isinstance(next_item.value.value, str):
                                                return next_item.value.value.strip()
            except (OSError, TypeError, SyntaxError):
                pass

        return None

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


class IncludeAttrsExcept(Directive):
    """Include attributes for all fields except those from specified base classes."""
    required_arguments = 2  # main class and classes to exclude

    def run(self):
        main_class_name = self.arguments[0]
        exclude_class_names = self.arguments[1:]

        # Get the main class
        main_class = self._get_class(main_class_name)
        if main_class is None:
            return []

        # Get the exclude classes
        exclude_classes = []
        for name in exclude_class_names:
            cls = self._get_class(name)
            if cls:
                exclude_classes.append(cls)

        # Get all fields from the main class
        main_attrs = self._get_class_attrs(main_class)

        # Get all fields from exclude classes
        exclude_fields = set()
        for exclude_class in exclude_classes:
            exclude_attrs = self._get_class_attrs(exclude_class)
            exclude_fields.update(exclude_attrs.keys())

        # Filter to only fields unique to main class
        unique_fields = {k: v for k, v in main_attrs.items() if k not in exclude_fields}

        if not unique_fields:
            return []

        # Build the output using same format as IncludeAttrs
        bullet_list = nodes.bullet_list()
        for attr_name in sorted(unique_fields.keys()):
            attr_info = unique_fields[attr_name]
            list_item = nodes.list_item()
            para = nodes.paragraph()

            # Attribute name as :py:attr: role for clickable links
            from sphinx.addnodes import pending_xref

            # Find which class actually defines this attribute
            defining_class = self._find_defining_class(main_class, attr_name)
            if defining_class:
                defining_class_name = f"{defining_class.__module__}.{defining_class.__qualname__}"
                link_target = f"{defining_class_name}.{attr_name}"
            else:
                link_target = f"{main_class_name}.{attr_name}"

            refnode = pending_xref(
                '',
                refdomain='py',
                reftype='attr',
                reftarget=link_target,
                refwarn=True
            )
            refnode += nodes.literal('', attr_name, classes=['xref', 'py', 'py-attr'])
            para += refnode

            # Docstring
            if attr_info['doc']:
                para += nodes.Text(f": {attr_info['doc']}")

            list_item += para
            bullet_list += list_item

        return [bullet_list]

    def _get_class(self, fqname):
        """Get a class from a fully qualified name."""
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
        for part in rest:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return None

        return obj if inspect.isclass(obj) else None

    def _get_class_attrs(self, obj):
        """Get all attributes from a class with their metadata."""
        attrs = {}

        if not inspect.isclass(obj):
            return attrs

        # Get annotations
        annotations = getattr(obj, "__annotations__", {})

        # Extract inline docstrings using AST
        try:
            source = inspect.getsource(obj)
            tree = ast.parse(source)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for i, item in enumerate(node.body):
                        if isinstance(item, ast.AnnAssign):
                            if isinstance(item.target, ast.Name):
                                attr_name = item.target.id

                                if self._should_skip_attr(attr_name):
                                    continue

                                if attr_name not in attrs:
                                    attrs[attr_name] = {'type': None, 'default': None, 'doc': None}

                                if attr_name in annotations:
                                    attrs[attr_name]['type'] = annotations[attr_name]

                                if item.value is not None:
                                    try:
                                        attrs[attr_name]['default'] = ast.unparse(item.value)
                                    except:
                                        attrs[attr_name]['default'] = repr(item.value)

                                # Check for inline docstring
                                if i + 1 < len(node.body):
                                    next_item = node.body[i + 1]
                                    if isinstance(next_item, ast.Expr) and isinstance(next_item.value, ast.Constant):
                                        if isinstance(next_item.value.value, str):
                                            attrs[attr_name]['doc'] = next_item.value.value.strip()
        except (OSError, TypeError, SyntaxError):
            pass

        # Pydantic v2 fields
        if hasattr(obj, "model_fields"):
            for field_name, field in obj.model_fields.items():
                if self._should_skip_attr(field_name):
                    continue

                if field_name not in attrs:
                    attrs[field_name] = {'type': None, 'default': None, 'doc': None}

                if hasattr(field, "annotation"):
                    attrs[field_name]['type'] = field.annotation

                if hasattr(field, "default") and field.default is not None:
                    attrs[field_name]['default'] = repr(field.default)
                elif hasattr(field, "default_factory") and field.default_factory is not None:
                    attrs[field_name]['default'] = f"{field.default_factory.__name__}()"

                doc = field.description or (field.json_schema_extra or {}).get("description")
                if doc:
                    attrs[field_name]['doc'] = doc

        # Pydantic v1 fields
        if hasattr(obj, "__fields__"):
            for field_name, field in obj.__fields__.items():
                if self._should_skip_attr(field_name):
                    continue

                if field_name not in attrs:
                    attrs[field_name] = {'type': None, 'default': None, 'doc': None}

                if hasattr(field, "outer_type_"):
                    attrs[field_name]['type'] = field.outer_type_

                if hasattr(field, "default") and field.default is not None:
                    attrs[field_name]['default'] = repr(field.default)
                elif hasattr(field, "default_factory") and field.default_factory is not None:
                    attrs[field_name]['default'] = f"{field.default_factory.__name__}()"

                if hasattr(field, "description"):
                    doc = field.description
                    if doc:
                        attrs[field_name]['doc'] = doc

        # Look up inheritance chain for missing docstrings
        for attr_name in list(attrs.keys()):
            if attrs[attr_name]['doc'] is None:
                doc = self._find_docstring_in_mro(obj, attr_name)
                if doc:
                    attrs[attr_name]['doc'] = doc

        return attrs

    def _find_defining_class(self, obj, attr_name):
        """Find where attribute is first defined by checking __annotations__ in __dict__."""
        mro_list = list(inspect.getmro(obj))

        # Walk MRO backwards to find first class that defines this in its own __annotations__
        for i in range(len(mro_list) - 1, -1, -1):
            base_class = mro_list[i]
            if '__annotations__' in base_class.__dict__:
                if attr_name in base_class.__dict__['__annotations__']:
                    return base_class

        return None

    def _find_docstring_in_mro(self, obj, attr_name):
        """Find docstring for an attribute by walking the MRO."""
        for base_class in inspect.getmro(obj):
            # Try Pydantic v2 first
            if hasattr(base_class, "model_fields"):
                if attr_name in base_class.model_fields:
                    field = base_class.model_fields[attr_name]
                    doc = field.description or (field.json_schema_extra or {}).get("description")
                    if doc:
                        return doc

            # Try Pydantic v1
            if hasattr(base_class, "__fields__"):
                if attr_name in base_class.__fields__:
                    field = base_class.__fields__[attr_name]
                    if hasattr(field, "description") and field.description:
                        return field.description

            # Try AST parsing for inline docstrings
            try:
                source = inspect.getsource(base_class)
                tree = ast.parse(source)

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        for i, item in enumerate(node.body):
                            if isinstance(item, ast.AnnAssign):
                                if isinstance(item.target, ast.Name) and item.target.id == attr_name:
                                    # Check if next item is a docstring
                                    if i + 1 < len(node.body):
                                        next_item = node.body[i + 1]
                                        if isinstance(next_item, ast.Expr) and isinstance(next_item.value, ast.Constant):
                                            if isinstance(next_item.value.value, str):
                                                return next_item.value.value.strip()
            except (OSError, TypeError, SyntaxError):
                pass

        return None

    def _should_skip_attr(self, attr_name):
        """Check if an attribute should be skipped."""
        return (
            attr_name.startswith('_') or
            attr_name in ('type', 'version')
        )


def setup(app):
    app.add_directive("include-attrs", IncludeAttrs)
    app.add_directive("include-attrs-except", IncludeAttrsExcept)
