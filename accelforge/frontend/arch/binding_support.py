from accelforge.util._basetypes import (
    EvalableModel,
    EvalsTo,
    _uninstantiable,
)

@_uninstantiable
class ConcurrentlyBoundable(EvalableModel):
    """
    An architecture node that supports binding multiple Einsums
    concurrently. A concurrently-boundable node within a hierarchy
    implies concurrently-boundable parents.
    """

    support_concurrent_binding: EvalsTo[bool] = False
    """
    Whether different Einsums can be concurrently bound to use this unit.
    """
