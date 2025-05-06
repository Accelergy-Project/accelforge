import copy
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
        
    def get_renames_for_einsum(self, einsum_name: str) -> "EinsumRename":
        if einsum_name not in self.einsums:
            rename = EinsumRename(name=einsum_name)
        else:
            rename = copy.deepcopy(self.einsums[einsum_name])
        if "default" in self.einsums:
            default = self.einsums["default"]
            rename.tensor_accesses.extend_no_name_repeat(default.tensor_accesses)
        return rename
        
class EinsumRenameList(ListNode):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", EinsumRename)
        
    def __getitem__(self, key: str) -> "EinsumRename":
        return super().__getitem__(key)

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
        
    def __getitem__(self, key: str) -> "TensorRename":
        return super().__getitem__(key)
    
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