"""
Constraints on how a tensor may be bound to the physical instances of a
physically-distributed memory.

A *binding* pairs one loop with each spatial loop below a physically-distributed
:class:`~accelforge.frontend.mapping.Storage`. Each of those loops names either a rank
of the tensor -- the memory instances along that spatial dimension each hold a slice of
the tensor along that rank -- or ``RANK_DUPLICATE``, meaning every instance along that
dimension holds a whole copy.

:class:`BindingConstraints` restricts which of those choices the mapper may explore, so
that the impact of a binding policy (e.g. never duplicating) can be measured against
another. Each memory carries its own constraints in
:py:attr:`~accelforge.frontend.arch.Memory.binding`, and, like the rest of the
architecture, they are evaluated per-Einsum.
"""

from typing import Iterable

from accelforge.frontend.renames import Rank, RANK_DUPLICATE, TensorName
from accelforge.util._basetypes import EvalableModel, EvalsTo
from accelforge.util._setexpressions import InvertibleSet


class BindingConstraints(EvalableModel):
    """
    Which bindings of a physically-distributed memory the mapper may explore.

    The defaults place no constraint on the binding, which reproduces the behavior of a
    mapper that has never heard of these constraints.
    """

    max_binding_rank_changes: float | int = 0
    """
    By default, a binding that results in minimum memory usage and movement over
    network (in that order) per Einsum is used, referred to as "locally-optimal."

    If non-zero, `max_binding_rank_changes` sets the maximum number of binding
    loops of a pmapping template that may bind a rank other than the one the
    locally-optimal binding gives them. Set to float("inf") to explore all bindings.

    Changes are counted against the locally-optimal binding *that satisfies* the rest of
    these constraints, which may not be the unconstrained locally-optimal one.

    When different memories are given different values, a pmapping template may change
    at most the largest of them in total, and at most each memory's own value in that
    memory's binding.
    """

    may_duplicate: EvalsTo[InvertibleSet[TensorName]] = "All"
    """
    The tensors that may be replicated across a spatial dimension rather than
    partitioned along one of their ranks. ``"Nothing"`` forbids duplication entirely:
    every binding loop must partition its tensor, even when the spatial loop it pairs
    with iterates a rank variable that does not index that tensor.

    Note that forbidding duplication can make a pmapping template unmappable, if the
    memory has a spatial dimension along which the tensor cannot be partitioned at all.
    Such templates are dropped, and the number dropped is logged.
    """

    must_duplicate: EvalsTo[InvertibleSet[TensorName]] = "Nothing"
    """
    The tensors that must be replicated across every spatial dimension that binds them.
    This is the opposite extreme of ``may_duplicate="Nothing"``, and is useful as an
    experimental baseline: no tensor is ever partitioned.

    Every tensor named here must also be named by `may_duplicate`.
    """

    dimension_ranks: dict[str, set[Rank]] = {}
    """
    Per spatial dimension name (e.g. ``X``), the only ranks that a binding loop over that
    dimension may name. Dimensions absent from this dict are unconstrained.

    ``RANK_DUPLICATE`` is treated like any other entry: a dimension listed here may
    duplicate only if the literal ``"DUPLICATE"`` is among its ranks. This is why the
    dict is empty rather than exhaustive by default.
    """

    def _eval_expressions(self, *args, **kwargs):
        """
        The constraints with their tensor set expressions evaluated, checked for a
        contradiction between `may_duplicate` and `must_duplicate` while the two are in
        hand. The sets are only comparable once evaluated, and the mapper would otherwise
        read the contradiction as a memory whose tensor has no valid binding at all.
        """
        evaluated, symbol_table = super()._eval_expressions(*args, **kwargs)

        if isinstance(evaluated.must_duplicate, InvertibleSet) and isinstance(
            evaluated.may_duplicate, InvertibleSet
        ):
            contradictory = evaluated.must_duplicate - evaluated.may_duplicate
            if contradictory:
                raise ValueError(
                    f"In the binding constraints of a memory, tensors "
                    f"{sorted(contradictory)} must duplicate but are not among the "
                    f"tensors that may duplicate ({sorted(evaluated.may_duplicate)}). "
                    f"may_duplicate must include every tensor of must_duplicate."
                )
        return evaluated, symbol_table

    # =============================================================================
    # Queries. These need expressions to have been evaluated.
    # =============================================================================
    def _assert_evaluated(self) -> None:
        if not isinstance(self.may_duplicate, InvertibleSet) or not isinstance(
            self.must_duplicate, InvertibleSet
        ):
            raise ValueError(
                f"The tensor set expressions of these BindingConstraints have not been "
                f"evaluated (may_duplicate={self.may_duplicate!r}, "
                f"must_duplicate={self.must_duplicate!r}). The architecture must be "
                f"evaluated against an Einsum before its bindings can be queried."
            )

    def allowed_ranks(
        self, tensor: TensorName, dimension: str, tensor_ranks: Iterable[Rank]
    ) -> list[Rank]:
        """
        The ranks that one binding loop of `tensor` over spatial dimension `dimension`
        may name, in a deterministic order. An empty result means that no binding of
        `tensor` over `dimension` satisfies these constraints.
        """
        self._assert_evaluated()
        if tensor in self.must_duplicate:
            allowed = [RANK_DUPLICATE]
        else:
            allowed = list(tensor_ranks)
            if tensor in self.may_duplicate:
                allowed.append(RANK_DUPLICATE)

        permitted = self.dimension_ranks.get(dimension)
        if permitted is not None:
            allowed = [r for r in allowed if r in permitted]
        return allowed

    def violations(
        self,
        tensor: TensorName,
        dimensions: Iterable[str],
        ranks: Iterable[Rank],
        tensor_ranks: Iterable[Rank],
    ) -> list[str]:
        """
        Human-readable descriptions of the ways in which binding `tensor` so that its
        loop over each of `dimensions` names the corresponding rank of `ranks` violates
        these constraints. Empty if the binding is allowed.
        """
        problems = []
        for dimension, rank in zip(dimensions, ranks):
            allowed = self.allowed_ranks(tensor, dimension, tensor_ranks)
            if rank not in allowed:
                problems.append(
                    f"its loop over dimension {dimension} binds {rank}, but only "
                    f"{sorted(allowed) if allowed else 'no rank'} may be bound there"
                )
        return problems
