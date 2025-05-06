from fastfusion.yamlparse.nodes import DictNode, ListNode
from fastfusion.frontend.version import assert_version

class Renames(DictNode):
    """
    A factory for creating confusion matrices.
    """
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("version", default="0.5", callfunc=assert_version)
        super().add_attr("einsums", EinsumRenameList, [])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.einsums: EinsumRenameList = self["einsums"]
        
class EinsumRenameList(ListNode):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", EinsumRename)
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class EinsumRename(DictNode):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("name", str)
        super().add_attr("tensor_accesses", TensorAccessRenameList, [])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name: str = self["name"]
        self.tensor_accesses: TensorAccessRenameList = self["tensor_accesses"]
        
class TensorAccessRenameList(ListNode):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", TensorRename)
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
class TensorRename(DictNode):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("name", str)
        super().add_attr("source", str)
        super().add_attr("injective", bool, False)
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name: str = self["name"]
        self.source: str = self["source"]
        self.injective: bool = self["injective"]
