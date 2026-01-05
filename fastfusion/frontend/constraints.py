import copy
import logging
import re
from abc import ABC
from typing import Annotated, Any, Callable, List, Optional

from fastfusion._accelerated_imports import np
from fastfusion.util._basetypes import ParsableList, ParsableModel, ParsesTo
from fastfusion.util._parse_expressions import parse_expression
from fastfusion.util._setexpressions import InvertibleSet, eval_set_expression
from fastfusion.frontend.workload.workload import RankVariable, TensorName
from fastfusion._version import assert_version, __version__


# class LoopOrder(ParsableList[RankVariable]):
#     """
#     A loop_order of ranks.
#     """

#     def _parse(self, symbol_table: dict[str, Any], location: str):
#         # return [x._parse(symbol_table) for x in self]
#         return type(self)(
#             [
#                 eval_set_expression(x, symbol_table, "rank_variables", location)
#                 for x in self
#             ],
#         )






# class Spatial(ParsableModel):
#     """
#     A :class:`~.Loop` constraints that apply to spatial loops.
#     """

#     name: str
#     """ The dimension name across which different spatial iterations occur. """



# class Temporal(Loop):
#     """
#     :class:`~.Loop` constraints that apply to temporal loops.

#     CURRENTLY NOT USED. NO TEMPORAL CONSTRAINTS ARE SUPPORTED.

#     """

#     # rmw_first_update: str | InvertibleSet[TensorName] | set[TensorName] = "Nothing"
#     # """ A set of tensors or a set expression representing tensors that incur a
#     # read-modify-write the first time they are updated in a memory. For tensors outputted
#     # by an Einsum, the first update of a value only incurs a read, because the previous
#     # value is null. If a tensor is given here, then the first update of that tensor will
#     # incur a read and write.
#     # """

#     def _parse(self, symbol_table: dict[str, Any], location: str):
#         new_temporal = super()._parse(symbol_table, location)
#         # new_temporal.rmw_first_update = eval_set_expression(
#         #     self.rmw_first_update, symbol_table, "tensors", location
#         # )
#         return new_temporal


# class Misc(ParsableModel):
#     """
#     Miscellaneous constraints that do not fit into the other categories.
#     """


# class MiscOnlyConstraints(ParsableModel):
#     """
#     Miscellaneous constraints that do not fit into the other categories.
#     """

#     misc: Misc = Misc()
#     """ Miscellaneous constraints that do not fit into the other categories. """


# class ConstraintGroup(MiscOnlyConstraints):
#     """A group of constraints that apply to a component."""

#     spatial: ParsableList[Spatial] = ParsableList()
#     """ Constraints that apply to spatial loops across spatial instances of this
#     component. """

#     # temporal: Temporal = Temporal()

#     tensors: Tensors = Tensors()
#     """ Constraints that apply to tensors stored in this component. """



# class Constraints(ParsableModel):
#     # version: Annotated[str, assert_version] = __version__
#     constraints: ParsableList[ConstraintGroup] = ParsableList()
