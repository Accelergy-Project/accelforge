from fastfusion.yamlparse.nodes import DictNode
from .version import assert_version


class Variables(DictNode):
    """
    A top-level class for variables. These will be available to parsing
    at all other points in the specification.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        cls.recognize_all()
        super().add_attr("version", default="0.5", callfunc=assert_version)
        super().add_attr("", part_name_match=True, no_change_key=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._default_parse = True
        self.version: str = self["version"]
