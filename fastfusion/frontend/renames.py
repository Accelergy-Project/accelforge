import copy
from typing import Annotated, Any, TypeAlias
from fastfusion.util._basetypes import ParsableList, ParsableModel, ParsesTo, TryParseTo, _PostCall
from fastfusion._version import assert_version, __version__
from fastfusion.util._parse_expressions import ParseError
from fastfusion.util._setexpressions import InvertibleSet

TensorName: TypeAlias = str
RankVariable: TypeAlias = str
Rank: TypeAlias = str
EinsumName: TypeAlias = str


class Rename(ParsableModel):
    """
    A rename of something into something else.
    """

    name: str
    """ The name of the thing to be renamed. This is a string representing the new name."""

    source: TryParseTo[InvertibleSet[TensorName | RankVariable]]
    """ The source of the rename. This is a set expression that can be parsed, yielding
    a set that can be referenced by the new name. """

    expected_count: ParsesTo[int] | None = None
    """
    The expected count of the source set expression. If this is set, then the source
    expression must resolve to the expected count or an error will be raised. Otherwise,
    any count (including zero for an empty set) is allowed.
    """

    def _parse_expressions(self, symbol_table: dict[str, Any], *args, **kwargs):
        parsed, symbol_table = super()._parse_expressions(symbol_table, *args, **kwargs)
        expected_count = parsed.expected_count
        if expected_count is not None and isinstance(parsed.source, InvertibleSet) and len(parsed.source) != expected_count:
            parsed, symbol_table = super()._parse_expressions(symbol_table, *args, **kwargs)
            raise ParseError(
                f"Expected count is {parsed.expected_count}, but got "
                f"{len(parsed.source)}: {parsed.source}",
                source_field="source",
            )
        return parsed, symbol_table


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
    """A list of renames."""

    def __dict__(self) -> dict[str, Any]:
        return {r.name: r.source for r in self}

    def _parse_expressions(self, symbol_table: dict[str, Any], *args, **kwargs):

        cur_symbol_table = symbol_table.copy()
        class PostCallRenameList(_PostCall[Rename]):
            def __call__(self, field, value, parsed, symbol_table):
                symbol_table[parsed.name] = parsed.source
                return parsed

        new, _ = super()._parse_expressions(cur_symbol_table, *args, **kwargs, post_calls=(PostCallRenameList(),))
        return new, symbol_table


class EinsumRename(ParsableModel):
    """
    Renames for a single Einsum.
    """

    name: EinsumName
    """ The name of the Einsum. Set this to "default" to apply the renames to all
    Einsums, unless overridden. Overriding is specific to a single name, so every rename
    in the default must be overridden independently. """

    tensor_accesses: ParsableList[Rename] = ParsableList()
    """ Renames for the tensor accesses of this Einsum. This may be given either as a
    dictionary ``{new_name: source_set_expression}`` expressions, or as a list of
    dictionaries, each one having the structure ``{name: new_name, source:
    source_set_expression, expected_count: 1}``, where expected count is optional for
    each and may be set to any integer. """

    rank_variables: ParsableList[Rename] = ParsableList()
    """ Renames for the rank variables of this Einsum. This may be given either as a
    dictionary ``{new_name: source_set_expression}`` expressions, or as a list of
    dictionaries, each one having the structure ``{name: new_name, source:
    source_set_expression, expected_count: 1}``, where expected count is optional for
    each and may be set to any integer. """

    def __init__(self, *args, **kwargs) -> None:
        if "tensor_accesses" in kwargs:
            kwargs["tensor_accesses"] = rename_list_factory(kwargs["tensor_accesses"])
        if "rank_variables" in kwargs:
            kwargs["rank_variables"] = rename_list_factory(kwargs["rank_variables"])
        super().__init__(*args, **kwargs)


class Renames(ParsableModel):
    # version: Annotated[str, assert_version] = __version__
    einsums: list[EinsumRename] = list()
    """
    Renames for a workload. The Einsum list is a list of EinsumRename objects, and
    renames will be applied to Einsums whose names match the EinsumRename.name. If an
    EinsumRename is named "default", then its renames are applied to every Einsum unless
    overridden. Overriding is specific to a single name, so every rename in the default
    must be overridden independently.
    """

    def get_renames_for_einsum(self, einsum_name: EinsumName) -> EinsumRename:
        if einsum_name not in self.einsums:
            rename = EinsumRename(name=einsum_name)
        else:
            rename = copy.deepcopy(self.einsums[einsum_name])
        for einsum in self.einsums:
            if einsum.name != "default":
                continue
            for tensor_rename in einsum.tensor_accesses:
                if tensor_rename.name not in rename.tensor_accesses:
                    rename.tensor_accesses.append(tensor_rename)
            for rank_variable_rename in einsum.rank_variables:
                if rank_variable_rename.name not in rename.rank_variables:
                    rename.rank_variables.append(rank_variable_rename)
        return rename
