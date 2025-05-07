from fastfusion.yamlparse.nodes import DictNode, ListNode
from .version import assert_version
from typing import Union


class Mapping(DictNode):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("version", default="0.5", callfunc=assert_version)
        super().add_attr("nodes", MappingNodeList)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.version: str = self["version"]
        self.nodes: MappingNodeList = self["nodes"]


class MappingNodeList(ListNode):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", MappingNode)
        
    def __getitem__(self, key: Union[str, int]) -> "MappingNode":
        return super().__getitem__(key)


class MappingNode(DictNode):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("type")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type: str = self["type"]
