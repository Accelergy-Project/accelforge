import copy
from typing import Annotated, TypeAlias
from fastfusion.util.basetypes import ParsableList, ParsableModel
from fastfusion.version import assert_version, __version__

TensorName: TypeAlias = str
RankVariableName: TypeAlias = str
RankName: TypeAlias = str
EinsumName: TypeAlias = str


class Rename(ParsableModel):
    name: str
    source: str
    expected_count: int | None = None


def rename_list_factory(rename_list: list | dict) -> "RenameList":
    if isinstance(rename_list, list):
        return RenameList(rename_list)

    if not isinstance(rename_list, dict):
        raise TypeError(
            f"Expected a list or dict, got {type(rename_list)}: {rename_list}"
        )

    return RenameList(
        Rename(name=k, source=v, expected_count=None) for k, v in rename_list.items()
    )


class RenameList(ParsableList[Rename]):
    pass


class EinsumRename(ParsableModel):
    name: EinsumName
    tensor_accesses: ParsableList[Rename] = ParsableList()
    rank_variables: ParsableList[Rename] = ParsableList()

    def __init__(self, *args, **kwargs) -> None:
        if "tensor_accesses" in kwargs:
            kwargs["tensor_accesses"] = rename_list_factory(kwargs["tensor_accesses"])
        if "rank_variables" in kwargs:
            kwargs["rank_variables"] = rename_list_factory(kwargs["rank_variables"])
        super().__init__(*args, **kwargs)


class Renames(ParsableModel):
    version: Annotated[str, assert_version] = __version__
    einsums: ParsableList[EinsumRename] = ParsableList()

    def model_post_init(self, __context__=None) -> None:
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
