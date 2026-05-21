import copy
import math

from accelforge.util._basetypes import (
    EvalableModel,
    EvalableList,
    EvalsTo,
    TryEvalTo,
    NoParse,
)

from accelforge.util._setexpressions import InvertibleSet
from accelforge.frontend.renames import TensorName
from accelforge.frontend.arch.constraints import Comparison


class Spatial(EvalableModel):
    """A one-dimensional spatial fanout in the architecture."""

    name: str
    """
    The name of the dimension over which this spatial fanout is occurring (e.g., X or Y).
    """

    fanout: EvalsTo[int]
    """ The size of this fanout. """

    may_reuse: TryEvalTo[InvertibleSet[TensorName]] = (
        "<Same as reuse if reuse is defined, else All>"
    )
    """ 
    A set of a tensors or set expression representing tensors that the hardware may
    reuse across spatial iterations. If a tensor is not in this set, then hardware must
    unicast its values to all spatial instances that use it. If this is not defined and
    ``reuse`` is also not defined, all tensors may be reused. If ``reuse`` is defined,
    then ``may_reuse`` defaults to the same value as ``reuse``.
    
    Note that this behaves differently from ``reuse`` in that ``may_reuse`` has no
    effect on whether a particular spatial loop is valid, only on how data is moved
    to/from spatial instances. On the other hand, ``reuse`` constrains which loops are
    valid.
    """

    loop_bounds: EvalableList[Comparison] = EvalableList()
    """ Bounds for loops over this dimension. This is a list of :class:`~.Comparison`
    objects, all of which must be satisfied by the loops to which this constraint
    applies.

    Note: Loops may be removed if they are constrained to only one iteration.
    """

    min_usage: EvalsTo[int | float] = 0.0
    """ The minimum usage of spatial instances, as a value from 0 to 1. A mapping
    is invalid if less than this porportion of this dimension's fanout is utilized.
    Mappers that support it (e.g., FFM) may, if no mappings satisfy this constraint,
    return the highest-usage mappings. These constraints are disabled for copy Einsums.
    """

    reuse: TryEvalTo[InvertibleSet[TensorName]] = "<Defaults to Nothing>"
    """
    A set of tensors or a set expression representing tensors that must be reused across
    spatial iterations. Spatial loops may only be placed that reuse ALL tensors given
    here.

    Note that this behaves differently from ``may_reuse`` in that ``may_reuse`` has no
    effect on whether a particular spatial loop is valid, only on how data is moved
    to/from spatial instances. On the other hand, ``reuse`` constrains which loops are
    valid.

    Note that loops may be removed from the mapping if they do not reuse a tensor given
    here and they do not appear in another loop bound constraint.
    """

    usage_scale: EvalsTo[int | float | str] = 1
    """
    This factor scales the usage in this dimension. For example, if usage_scale is 2 and
    10/20 spatial instances are used, then the usage will be scaled to 20/20.
    """

    power_gateable: EvalsTo[bool] = False
    """
    Whether this spatial fanout has power gating. If True, then unused spatial instances
    will be power gated if not used by a particular Einsum.
    """

    def _eval_expressions(self, *args, **kwargs):
        self = copy.copy(self)
        reuse, may_reuse = self.reuse, self.may_reuse
        if may_reuse == "<Same as reuse if reuse is defined, else All>":
            self.may_reuse = "All" if reuse == "<Defaults to Nothing>" else reuse
        if reuse == "<Defaults to Nothing>":
            self.reuse = "Nothing"
        return super(self.__class__, self)._eval_expressions(*args, **kwargs)


class Spatialable(EvalableModel):
    """Something that can be duplicated to create an array of."""

    spatial: EvalableList[Spatial] = EvalableList()
    """
    The spatial fanouts of this `Leaf`.

    Spatial fanouts describe the spatial organization of components in the architecture.
    A spatial fanout of size N for this node means that there are N instances of this
    node. Multiple spatial fanouts lead to a multi-dimensional fanout. Spatial
    constraints apply to the data exchange across these instances. Spatial fanouts
    specified at this level also apply to lower-level `Leaf` nodes in the architecture.
    """

    _physical_spatial: NoParse[Spatial] = EvalableList()
    """
    The physical spatial fanout of this node. Should only have a value for a
    flattened arch. Otherwise, the `spatial` attribute is authoritative.
    """

    def get_fanout(self) -> int:
        """The spatial fanout of this node."""
        return int(math.prod(x.fanout for x in self.spatial))

    def get_fanout_along(self, dim_name: str, default: int = 1) -> int:
        for s in self.spatial:
            if s.name == dim_name:
                return s.fanout
        return default

    def _get_physical_fanout_along(self, dim_name: str, default: int = 1) -> int:
        for s in self._physical_spatial:
            if s.name == dim_name:
                return s.fanout
        return default

    def _spatial_str(self, include_newline=True) -> str:
        if not self.spatial:
            return ""
        result = ", ".join(f"{s.fanout}× {s.name}" for s in self.spatial)
        return f"\n[{result}]" if include_newline else result
