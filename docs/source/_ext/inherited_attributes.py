from sphinx.ext.autodoc import ClassDocumenter, INSTANCEATTR, ObjectMember
from sphinx.util.inspect import safe_getattr
import inspect
from pydantic import BaseModel


# Default ignore lists
DEFAULT_IGNORE = [
    'object',
    'ABC',
    'BaseModel',
    'Enum',
    'Flag',
    'IntEnum',
    'IntFlag',
    'Generic',
    'object',
    'ABC',
    'BaseModel',
    'ParsableModel',
    'ParsableList',
    'ParsableDict',
    'ParseExtras',
    'NonParsableModel',
    'FromYAMLAble',
    'ParsesTo',
    'Enum',
    'Flag',
    'IntEnum',
    'IntFlag',
    'Generic',
    'model_validate',
    'model_validate_json',
    'model_validate_strings',
    'parse_file',
    'parse_obj',
    'parse_raw',
    'schema',
    'schema_json',
    'model_dump',
    'model_dump_json',
    'model_copy',
    'construct',
    'copy',
    'dict',
    'json',
    'update_forward_refs',
    'model_post_init',
    'model_config',
    'model_fields',
    'model_fields_set',
    'model_extra',
    'model_private',
    'model_rebuild',
    'get_fields',
    'get_validator',
    'all_fields_default',
    'model_dump_non_none',
] + list(BaseModel.__dict__.keys())

class InheritedAttributesClassDocumenter(ClassDocumenter):
    """Enhanced ClassDocumenter that includes inherited attributes."""

    priority = ClassDocumenter.priority + 1

    def get_object_members(self, want_all):
        """Override to include inherited members."""
        # Get the normal members first
        members_check_module, members = super().get_object_members(want_all)

        # Get configuration
        config = self.env.config
        ignore = getattr(config, 'inherited_attributes_ignore', []) + DEFAULT_IGNORE

        # Collect ALL attributes defined directly on this class (not inherited)
        own_members = set()
        if hasattr(self.object, '__dict__'):
            own_members.update(self.object.__dict__.keys())
        if hasattr(self.object, 'model_fields'):
            own_members.update(self.object.model_fields.keys())
        if hasattr(self.object, '__fields__'):
            own_members.update(self.object.__fields__.keys())

        # Filter out members from ignored classes that were already included
        # BUT keep them if they're overridden in the current class
        filtered_members = []
        for member in members:
            if hasattr(member, '__getitem__') and len(member) >= 4:
                name = member[0]
                member_class = member[3] if len(member) > 3 else None

                # Keep if it's defined directly on this class
                if name in own_members:
                    filtered_members.append(member)
                    continue

                # Skip if from an ignored class (and not overridden)
                if member_class and self._should_ignore_class(member_class, ignore):
                    continue

                # Skip ignored functions/attributes (only if not overridden)
                if name in ignore:
                    continue

                filtered_members.append(member)
            else:
                filtered_members.append(member)

        members = filtered_members

        # Get MRO
        try:
            mro = inspect.getmro(self.object)
        except (AttributeError, TypeError):
            return members_check_module, members

        # Track seen names - includes both members from super() and own_members
        # ObjectMember is a namedtuple, access by index
        seen_names = set()
        for member in members:
            if hasattr(member, '__getitem__'):
                seen_names.add(member[0])
        # Add all own members to seen_names to prevent inheritance
        seen_names.update(own_members)

        # Collect inherited members
        inherited = []

        for parent_class in mro[1:]:  # Skip self
            # Check if we should stop FIRST, before processing any members
            if self._should_ignore_class(parent_class, ignore):
                break

            # Get members from parent
            for name in dir(parent_class):
                # Skip underscore-prefixed
                if name.startswith('_'):
                    continue

                # Skip if already seen (including own_members)
                if name in seen_names:
                    continue

                # Skip ignored
                if name in ignore:
                    continue

                try:
                    obj = safe_getattr(parent_class, name)

                    # Skip callables in ignore list
                    if callable(obj) and name in ignore:
                        continue

                    # Create ObjectMember
                    inherited.append(ObjectMember(name, obj))
                    seen_names.add(name)

                except (AttributeError, TypeError):
                    continue

            # Check Pydantic fields
            if hasattr(parent_class, 'model_fields'):
                for field_name in parent_class.model_fields:
                    if field_name.startswith('_'):
                        continue
                    # Skip if already seen (including own_members)
                    if field_name in seen_names:
                        continue
                    if field_name in ignore:
                        continue

                    inherited.append(ObjectMember(field_name, INSTANCEATTR))
                    seen_names.add(field_name)

            elif hasattr(parent_class, '__fields__'):
                for field_name in parent_class.__fields__:
                    if field_name.startswith('_'):
                        continue
                    # Skip if already seen (including own_members)
                    if field_name in seen_names:
                        continue
                    if field_name in ignore:
                        continue

                    inherited.append(ObjectMember(field_name, INSTANCEATTR))
                    seen_names.add(field_name)

        # Combine
        all_members = list(members) + inherited

        return members_check_module, all_members

    def _should_ignore_class(self, cls, ignore_classes):
        """Check if a class should be ignored."""
        name = getattr(cls, '__name__', '')
        module = getattr(cls, '__module__', '')

        # Ignore if not part of fastfusion package
        if module and not module.startswith('fastfusion'):
            return True

        # Check against ignore list using just the class name
        for ignore_pattern in ignore_classes:
            if name == ignore_pattern:
                return True

        return False

def setup(app):
    """Setup the extension."""
    # Only add config value if it doesn't exist
    if not hasattr(app.config, 'inherited_attributes_ignore'):
        app.add_config_value('inherited_attributes_ignore', [], 'env')

    app.add_autodocumenter(InheritedAttributesClassDocumenter, override=True)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
