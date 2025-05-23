import copy
from typing import Annotated
from fastfusion.util.basetypes import ParsableList, ParsableModel
from fastfusion.version import assert_version, __version__

class TensorRename(ParsableModel):
    """
    Represents a tensor rename operation.
    
    Attributes:
        name (str): The new name for the tensor
        source (str): The original name of the tensor
        injective (bool): Whether the rename is injective
    """
    name: str
    source: str
    injective: bool = False

class EinsumRename(ParsableModel):
    name: str
    tensor_accesses: ParsableList[TensorRename] = ParsableList()

class Renames(ParsableModel):
    version:  Annotated[str, assert_version] = __version__
    einsums: ParsableList[EinsumRename] = ParsableList()

    def __init__(self, **data):
        super().__init__(**data)
        assert_version(self.version)

    def get_renames_for_einsum(self, einsum_name: str) -> EinsumRename:
        if einsum_name not in self.einsums:
            rename = EinsumRename(name=einsum_name)
        else:
            rename = copy.deepcopy(self.einsums[einsum_name])
        if "default" in self.einsums:
            default = self.einsums["default"]
            for tensor_rename in default.tensor_accesses:
                if tensor_rename.name not in rename.tensor_accesses:
                    rename.tensor_accesses.append(tensor_rename)
        return rename