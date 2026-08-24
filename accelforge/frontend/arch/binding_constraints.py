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

    Only the bindings of shared tensors -- those passed between Einsums -- are explored,
    and only those of the tensors named by `may_explore_binding`. A tensor that only one
    Einsum touches has no other Einsum to agree with, so its binding is always the
    locally-optimal one no matter what this is set to.

    Changes are counted against the locally-optimal binding *that satisfies* the rest of
    these constraints, which may not be the unconstrained locally-optimal one.

    When different memories are given different values, a pmapping template may change
    at most the largest of them in total, and at most each memory's own value in that
    memory's binding.
    """

    may_explore_binding: EvalsTo[InvertibleSet[TensorName]] = "All"
    """
    The tensors whose binding `max_binding_rank_changes` may spend its budget on. A
    tensor left out of this set always takes its locally-optimal binding, however large
    the budget is, and costs no templates.

    Narrowing this is how one tensor's binding is studied on its own: give the memory a
    budget and name only that tensor here, and every other tensor is held at the binding
    it would have had with no exploration at all, so what the mapper reports differs only
    in the one binding under study.

    This bounds exploration; it does not widen it. A tensor named here is still only
    explored if it is shared with another Einsum, since a tensor nobody else touches has
    no other Einsum to agree with.
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

    may_adapt: EvalsTo[InvertibleSet[TensorName]] = "Nothing"
    """
    The tensors whose binding in this memory two neighboring Einsums may *disagree*
    about. A tensor named here can be left by one Einsum bound one way and picked up by
    the next bound another; the data is shuffled among the memory's physical instances in
    between, and the mapper is charged the latency and energy of that shuffle over the
    network below this memory (see
    :mod:`accelforge.mapper.FFM._join_pmappings.binding_adapter`).

    A tensor left out of this set has to be bound the same way by every Einsum that
    shares it, which is what happens with no adaptation at all: the two Einsums simply
    do not fuse unless their bindings already agree.

    Adapting is only ever *needed* when `max_binding_rank_changes` lets the two sides
    reach different bindings in the first place, and it is only possible where there is a
    network below this memory to charge and this memory's size is finite, since the size
    is what turns the mapper's fractional occupancies back into bits.
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
        unevaluated = {
            name: value
            for name, value in (
                ("may_duplicate", self.may_duplicate),
                ("must_duplicate", self.must_duplicate),
                ("may_explore_binding", self.may_explore_binding),
                ("may_adapt", self.may_adapt),
            )
            if not isinstance(value, InvertibleSet)
        }
        if unevaluated:
            described = ", ".join(f"{n}={v!r}" for n, v in unevaluated.items())
            raise ValueError(
                f"The tensor set expressions of these BindingConstraints have not been "
                f"evaluated ({described}). The architecture must be evaluated against "
                f"an Einsum before its bindings can be queried."
            )

    def may_explore(self, tensor: TensorName) -> bool:
        """
        Whether the mapper may spend `max_binding_rank_changes` on `tensor`'s binding in
        this memory. False holds it at its locally-optimal binding; it says nothing about
        which ranks that binding may name, which is `allowed_ranks`.
        """
        self._assert_evaluated()
        return tensor in self.may_explore_binding

    def may_adapt_binding(self, tensor: TensorName) -> bool:
        """
        Whether two Einsums sharing `tensor` in this memory may disagree on its binding,
        paying for the data to be moved from one binding to the other rather than
        failing to fuse.
        """
        self._assert_evaluated()
        return tensor in self.may_adapt

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
