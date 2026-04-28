"""
Render Pydantic v2 field default values in autodoc output.

Pydantic v2 stores field defaults in `Model.model_fields[name].default`, not at
the class attribute level, so Sphinx's stock AttributeDocumenter can't see them
and emits just the type annotation. This extension subclasses AttributeDocumenter,
looks up the default in `model_fields`, and adds a `:value:` line when the repr
is short enough to be worth showing inline.
"""

from sphinx.ext.autodoc import AttributeDocumenter

# Defaults with reprs longer than this are skipped. A user reading the docs
# wants a quick "what's the default if I leave it alone" answer; huge nested
# reprs clutter the page and are better left to the source.
MAX_DEFAULT_LENGTH = 80


class PydanticDefaultAttributeDocumenter(AttributeDocumenter):
    """AttributeDocumenter that adds `:value:` for Pydantic v2 fields."""

    priority = AttributeDocumenter.priority + 1

    def should_suppress_value_header(self) -> bool:
        # If we have a Pydantic default to show, suppress the stock :value:
        # line so we can emit our own. Otherwise, defer to the parent.
        if self._pydantic_default_repr() is not None:
            return True
        return super().should_suppress_value_header()

    def add_directive_header(self, sig: str) -> None:
        super().add_directive_header(sig)
        text = self._pydantic_default_repr()
        if text is not None:
            self.add_line(f"   :value: {text}", self.get_sourcename())

    def _pydantic_default_repr(self) -> str | None:
        from pydantic_core import PydanticUndefined

        parent = self.parent
        if not hasattr(parent, "model_fields"):
            return None
        name = self.objpath[-1] if self.objpath else None
        field = parent.model_fields.get(name)
        if field is None:
            return None

        default = field.default
        if default is PydanticUndefined:
            return None

        try:
            text = repr(default)
        except Exception:
            return None
        if len(text) > MAX_DEFAULT_LENGTH:
            return None
        return text


def setup(app):
    # Sphinx 9's `_register_directives` re-registers AttributeDocumenter at
    # config-inited time, which would clobber anything we register in setup().
    # Hook the same event with a later priority so ours lands last.
    def _install(app, config):
        app.add_autodocumenter(PydanticDefaultAttributeDocumenter, override=True)

    app.connect("config-inited", _install, priority=1000)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
