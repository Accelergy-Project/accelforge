import math

from accelforge.util._basetypes import (
    EvalableModel,
    EvalableList,
    EvalsTo,
    TryEvalTo,
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

    may_reuse: TryEvalTo[InvertibleSet[TensorName]] = "All"
    """ The tensors that can be reused spatially across instances of this fanout. This
    expression will be evaluated for each mapping template. """

    loop_bounds: EvalableList[Comparison] = EvalableList()
    """ Bounds for loops over this dimension. This is a list of :class:`~.Comparison`
    objects, all of which must be satisfied by the loops to which this constraint
    applies.

    Note: Loops may be removed if they are constrained to only one iteration.
    """

    min_usage: int | float | str = 0.0
    """ The minimum usage of spatial instances, as a value from 0 to 1. A mapping
    is invalid if less than this porportion of this dimension's fanout is utilized.
    Mappers that support it (e.g., FFM) may, if no mappings satisfy this constraint,
    return the highest-usage mappings.
    """

    reuse: TryEvalTo[InvertibleSet[TensorName]] = "Nothing"
    """ A set of tensors or a set expression representing tensors that must be reused
    across spatial iterations. Spatial loops may only be placed that reuse ALL tensors
    given here.

    Note: Loops may be removed if they do not reuse a tensor given here and they do not
    appear in another loop bound constraint.
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

    def get_fanout(self) -> int:
        """The spatial fanout of this node."""
        return int(math.prod(x.fanout for x in self.spatial))

    def _spatial_str(self, include_newline=True) -> str:
        if not self.spatial:
            return ""
        result = ", ".join(f"{s.fanout}× {s.name}" for s in self.spatial)
        return f"\n[{result}]" if include_newline else result
