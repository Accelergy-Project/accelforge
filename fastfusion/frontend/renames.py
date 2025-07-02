import copy
from typing import Annotated
from fastfusion.frontend.workload.workload import EinsumName, RankVariableName, TensorName
from fastfusion.util.basetypes import ParsableList, ParsableModel
from fastfusion.version import assert_version, __version__

class TensorRename(ParsableModel):
    name: TensorName
    source: str
    expected_count: int | None = None
    
class RankVariableRename(ParsableModel):
    name: RankVariableName
    source: str
    expected_count: int | None = None

class EinsumRename(ParsableModel):
    name: EinsumName
    tensor_accesses: ParsableList[TensorRename] = ParsableList()
    rank_variables: ParsableList[RankVariableRename] = ParsableList()

class Renames(ParsableModel):
    version:  Annotated[str, assert_version] = __version__
    einsums: ParsableList[EinsumRename] = ParsableList()

    def __init__(self, **data):
        super().__init__(**data)
        assert_version(self.version)

    def get_renames_for_einsum(self, einsum_name: EinsumName) -> EinsumRename:
        if einsum_name not in self.einsums:
            rename = EinsumRename(name=einsum_name)
        else:
            rename = copy.deepcopy(self.einsums[einsum_name])
        if "default" in self.einsums:
            default = self.einsums["default"]
            for tensor_rename in default.tensor_accesses:
                if tensor_rename.name not in rename.tensor_accesses:
                    rename.tensor_accesses.append(tensor_rename)
            for rank_variable_rename in default.rank_variables:
                if rank_variable_rename.name not in rename.rank_variables:
                    rename.rank_variables.append(rank_variable_rename)
        return rename