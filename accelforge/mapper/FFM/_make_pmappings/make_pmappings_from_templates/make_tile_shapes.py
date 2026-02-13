from enum import Enum
from functools import lru_cache
from math import ceil, log2, prod
import copy
import re
import resource
import time
from typing import Callable, Iterator, Optional
from sympy import Expr, Symbol, factorint, lambdify
from accelforge import util
from accelforge._accelerated_imports import np
from accelforge._accelerated_imports import pd
import accelforge.frontend.arch as arch
from accelforge.frontend._workload_isl._isl import get_rank_variable_bounds
from accelforge.frontend._workload_isl._symbolic import get_projection_expr
from accelforge.frontend.mapping.mapping import MappingNode
from accelforge.frontend.workload import Einsum
from accelforge.frontend.mapping import (
    Loop,
    Mapping,
    Temporal,
    Spatial,
    TensorHolder,
)
from accelforge.mapper.FFM._make_pmappings.pmapper_job import Job
from accelforge.mapper.FFM._pareto_df.df_convention import (
    stride2col,
    initial2col,
    iterations2col,
)
from accelforge.mapper.FFM._pareto_df.pareto import makepareto_numpy
from accelforge.model._looptree.reuse.symbolic import IMPERFECT, PRINT_FORMULAS
from accelforge.mapper.FFM._join_pmappings.pmapping_dataframe import (
    nameloop2col,
    tensor2col,
    firstlatency2col,
)
from accelforge.frontend.mapper.metrics import Metrics
from accelforge.util._frozenset import fzs
import math
import sympy
import numpy as np
from numbers import Number

from accelforge.mapper.FFM._make_pmappings.make_pmappings_from_templates.symbol_relations import (
    SymbolRelations,
)
from accelforge.util._sympy.broadcast_max import Max
from accelforge.mapper.FFM._make_pmappings.make_pmappings_from_templates.run_model import (
    run_model,
)

import getpass

DEBUG = getpass.getuser() == "tanner"


class ComparisonResult(Enum):
    ALWAYS_GEQ_THAN_ZERO = "ALWAYS_GEQ_THAN_ZERO"
    ALWAYS_LEQ_THAN_ZERO = "ALWAYS_LEQ_THAN_ZERO"
    ALWAYS_EQUAL_TO_ZERO = "ALWAYS_EQUAL_TO_ZERO"
    UNKNOWN = "unknown"

    def __or__(self, other: "ComparisonResult"):
        if self == other:
            return self
        if self == ComparisonResult.ALWAYS_EQUAL_TO_ZERO:
            return other
        if other == ComparisonResult.ALWAYS_EQUAL_TO_ZERO:
            return self
        return ComparisonResult.UNKNOWN


@lru_cache(maxsize=10000)
def diff(f: Expr, s: Symbol):
    return sympy.diff(f, s)


@lru_cache(maxsize=10000)
def diff_geq_leq_zero(f: Expr, s: Symbol, bounds: tuple[tuple[Symbol, int, int], ...]):
    # Assume ceiling won't affect the sign of the derivative. Changing from positive to
    # zero or negative to zero is OK and does not count as changing the sign.
    if isinstance(f, sympy.Expr):
        f = f.replace(
            lambda expr: expr.is_Function and expr.func == sympy.ceiling,
            lambda expr: expr.args[0],
        )
    return geq_leq_zero(diff(f, s), bounds)


@lru_cache(maxsize=10000)
def function_range(f: Expr, s: Symbol, lo: int, hi: int):
    return sympy.calculus.util.function_range(f, s, domain=sympy.Interval(lo, hi))


def expr_replace(f: Expr, old: sympy.Function, new: Expr) -> Expr:
    return f.replace(
        lambda expr: expr.is_Function and expr.func == old,
        lambda expr: new,
    )


def partition_heaviside(f: Expr) -> tuple[Expr, ...]:
    if f.has(sympy.Heaviside):
        return expr_replace(f, sympy.Heaviside, 1), expr_replace(f, sympy.Heaviside, 0)
    return (f,)


# @lru_cache(maxsize=10000)
# def _get_function_range(
#     f: Expr,
#     check_symbols: tuple[Symbol, ...],
#     bounds: tuple[tuple[Symbol, int, int], ...],
#     return_min: bool,
# ) -> list:
#     if isinstance(f, sympy.Expr):
#         f = f.replace(
#             lambda expr: expr.is_Function and expr.func == sympy.ceiling,
#             lambda expr: expr.args[0],
#         )
#         fs = list(partition_heaviside(f))
#     else:
#         fs = [f]

#     if len(fs) > 1:
#         return [f3 for f2 in fs for f3 in _get_function_range(f2, check_symbols, bounds, return_min)]

#     f = fs[0]
#     check_symbol = check_symbols[0]
#     check_symbols = check_symbols[1:]
#     bounds = None
#     for s, lo, hi in bounds:
#         if s == check_symbol:
#             bounds = (s, lo, hi)
#             break
#     else:
#         raise ValueError(f"Symbol {check_symbol} not found in bounds")

#     f_range = sympy.calculus.util.function_range(f, check_symbol, domain=sympy.Interval(lo, hi))

#     if isinstance(f_range, sympy.FiniteSet):
#         return [f3 for f2 in f_range for f3 in _get_function_range(f2, check_symbols, bounds, return_min)]

#     target = f_range.left if return_min else f_range.right
#     return _get_function_range(target, check_symbols, bounds, return_min)


@lru_cache(maxsize=10000)
def _compare_to_zero(
    f: Expr, bounds: tuple[tuple[Symbol, int, int], ...], check_lt_zero: bool
) -> bool:
    """
    Returns True if the function may possibly be less than zero or greater than zero.

    If check_lt_zero is True, then we're checking if the function may possibly be less
    than zero. Otherwise, we're checking if the function may possibly be greater than
    zero.

    If we can't tell, then conservatively return True.
    """
    if isinstance(f, sympy.Expr):
        f = f.replace(
            lambda expr: expr.is_Function and expr.func == sympy.ceiling,
            lambda expr: expr.args[0],
        )
        fs = list(partition_heaviside(f))
    else:
        fs = [f]

    if len(fs) > 1:
        return any(_compare_to_zero(f2, bounds, check_lt_zero) for f2 in fs)

    f = fs[0]
    try:
        if check_lt_zero:
            # Less than zero anywhere == NOT geq zero everywhere
            return not f >= 0
        else:
            # Greater than zero anywhere == NOT leq zero everywhere
            return not f <= 0
    except TypeError:
        pass

    min_check, max_check = (any, all) if check_lt_zero else (all, any)
    if isinstance(f, sympy.Min):
        return min_check(_compare_to_zero(g, bounds, check_lt_zero) for g in f.args)
    if isinstance(f, sympy.Max):
        return max_check(_compare_to_zero(g, bounds, check_lt_zero) for g in f.args)

    # Tried this on one workload and had marginally faster speeds with choosing the
    # symbol that appears the least times. Also tried the symbol that appears the most
    # times and the symbol that appears first in the bounds list. They had equivalent
    # speeds, approx. 3% slower overall tile shape exploration than min.
    chosen_s = min(f.free_symbols, key=lambda s: f.count(s))
    for s, lo, hi in bounds:
        if s == chosen_s:
            break
    else:
        raise ValueError(f"Symbol {chosen_s} not found in bounds")

    try:
        f_range = function_range(f, s, lo, hi)
    except (NotImplementedError, TypeError):
        return True

    if isinstance(f_range, sympy.FiniteSet):
        return any(_compare_to_zero(f2, bounds, check_lt_zero) for f2 in f_range)
    else:
        return _compare_to_zero(
            f_range.left if check_lt_zero else f_range.right,
            bounds,
            check_lt_zero,
        )


@lru_cache(maxsize=10000)
def geq_leq_zero(
    f: Expr,
    bounds: tuple[tuple[Symbol, int, int], ...],
):
    # return geq_leq_than_zero(f, bounds)
    lt_zero = _compare_to_zero(f, bounds, check_lt_zero=True)
    gt_zero = _compare_to_zero(f, bounds, check_lt_zero=False)

    if lt_zero and gt_zero:
        return ComparisonResult.UNKNOWN
    if lt_zero and not gt_zero:
        return ComparisonResult.ALWAYS_LEQ_THAN_ZERO
    if gt_zero and not lt_zero:
        return ComparisonResult.ALWAYS_GEQ_THAN_ZERO
    return ComparisonResult.ALWAYS_EQUAL_TO_ZERO


def compile_dict(symbols, dictionary):
    def lambdify(key, value):
        x = util._lambdify_type_check(symbols, value)
        return x

    return {k: lambdify(symbols, v) for k, v in dictionary.items()}


class Goal:
    """
    X subset Y means that Y will block pruning for all cases that X will block pruning.

    - min is a subset of min_per_prime_factor is a subset of diff
    - max is a subset of max_per_prime_factor is a subset of diff

    If we're combining goals and they disagree, use the larger space.
    """

    def __init__(
        self,
        goal: str = None,
        max_value: Optional[float] = None,
        only_care_if_valid: bool = False,
    ):
        self.goal = goal
        self.max_value = max_value
        self.only_care_if_valid = only_care_if_valid

    def __or__(self, other: "Goal"):
        if self.goal is None:
            return copy.copy(other)
        if other.goal is None:
            return copy.copy(self)
        assert self.max_value == other.max_value
        assert self.only_care_if_valid == other.only_care_if_valid
        mv = self.max_value
        care = self.only_care_if_valid or other.only_care_if_valid

        # If the goals are the same, space doesn't change
        if self.goal == other.goal:
            return Goal(self.goal, max_value=mv, only_care_if_valid=care)

        # min_per_prime_factor is a superset of min, so we can just keep the min_per_prime_factor goal
        if {self.goal, other.goal} == {"min", "min_per_prime_factor"}:
            return Goal("min_per_prime_factor", max_value=mv, only_care_if_valid=care)

        # max_per_prime_factor is a superset of max, so we can just keep the max_per_prime_factor goal
        if {self.goal, other.goal} == {"max", "max_per_prime_factor"}:
            return Goal("max_per_prime_factor", max_value=mv, only_care_if_valid=care)

        # Otherwise, there's a disagreement and the only space we're both in can be diff
        return Goal("diff", max_value=mv, only_care_if_valid=care)

    def __str__(self):
        return f"{self.goal} {self.max_value} {self.only_care_if_valid}"

    def __repr__(self):
        return f"Goal({self.goal}, {self.max_value}, {self.only_care_if_valid})"

    def __invert__(self):
        if self.goal == "min":
            return Goal("max", self.max_value, self.only_care_if_valid)
        elif self.goal == "max":
            return Goal("min", self.max_value, self.only_care_if_valid)
        elif self.goal == "min_per_prime_factor":
            raise ValueError("Can't invert min_per_prime_factor")
        elif self.goal == "max_per_prime_factor":
            raise ValueError("Can't invert max_per_prime_factor")
        else:
            return copy.copy(self)

    def __eq__(self, other: "Goal"):
        return (
            isinstance(other, Goal)
            and self.goal == other.goal
            and self.max_value == other.max_value
            and self.only_care_if_valid == other.only_care_if_valid
        )


class Objective:
    def __init__(
        self,
        name: str,
        formula: Expr | Number,
        max_value: float = None,
        symbols: list[str] = None,
        only_care_if_valid: bool = False,
        min_value: float = None,
        inclusive: bool = True,
        try_best_if_none_reaches_min: bool = False,
    ):
        if isinstance(formula, Number):
            formula = sympy.Number(formula)
        self.name: str = name
        self.formula: Expr = simplify(formula)
        self._symbols: list[str] = symbols
        self.max_value: float = max_value
        self.min_value: float = min_value
        self.only_care_if_valid: bool = only_care_if_valid
        if only_care_if_valid:
            assert max_value is not None or min_value is not None
        self.inclusive: bool = inclusive
        self.try_best_if_none_reaches_min: bool = try_best_if_none_reaches_min


def is_constant(f: Expr) -> bool:
    try:
        return f.is_constant()
    except ValueError:
        return all(is_constant(arg) for arg in f.args)


@lru_cache(maxsize=10000)
def _try_replace_single_term(
    t: Expr,
    symbols_enumerated: fzs[Symbol],
    bounds: tuple[tuple[Symbol, int, int], ...],
):
    goal = None
    if len(t.free_symbols & symbols_enumerated) == 1:
        s = next(iter(t.free_symbols & symbols_enumerated))
        try:
            diff_result = diff_geq_leq_zero(t, s, bounds)
            if diff_result == ComparisonResult.ALWAYS_GEQ_THAN_ZERO:
                goal = Goal("min")
            elif diff_result == ComparisonResult.ALWAYS_LEQ_THAN_ZERO:
                goal = Goal("max")
            elif diff_result == ComparisonResult.UNKNOWN:
                goal = Goal("diff")
            elif diff_result == ComparisonResult.ALWAYS_EQUAL_TO_ZERO:
                pass
            else:
                raise ValueError(
                    f"Comparison result {diff_result} is not a valid comparison result"
                )
            return s, goal
        except (TypeError, ValueError):
            pass
    return t, None


def try_replace_single_term(
    t: Expr,
    symbols_enumerated: fzs[Symbol],
    bounds: tuple[tuple[Symbol, int, int], ...],
):
    return _try_replace_single_term(t, symbols_enumerated & t.free_symbols, bounds)


@lru_cache(maxsize=10000)
def _partition_formula(
    f: Expr,
    symbols_enumerated: set[Symbol],
    bounds: tuple[tuple[Symbol, int, int], ...],
) -> dict[Symbol, Goal]:
    goals: dict[Symbol, Goal] = {}

    def update_goal(symbol: Symbol, goal: str, **kwargs):
        goals[symbol] = Goal(goal) | goals.get(symbol, Goal())

    negate = False

    if not f.free_symbols & symbols_enumerated:
        return goals

    if f.free_symbols.issubset(symbols_enumerated):
        return {f: Goal("min")}

    def _try_replace_unknowns(t: Expr):
        for s in t.free_symbols - symbols_enumerated:
            if not affects_comparison(t, s, symbols_enumerated):
                t = t.subs(s, 1)
        return t

    def _recombine_terms(terms: list[Expr], transform_terms: bool = True):
        can_evaluate = []
        no_relation = []
        others = {}
        for t in terms:
            if transform_terms:
                t = _try_replace_unknowns(t)
            try:
                if not t.free_symbols & symbols_enumerated:
                    continue
            except (TypeError, ValueError):
                pass
            if t.free_symbols.issubset(symbols_enumerated):
                can_evaluate.append(t)
            elif t.free_symbols.isdisjoint(symbols_enumerated):
                no_relation.append(t)
            else:
                others.setdefault(fzs(t.free_symbols - symbols_enumerated), []).append(
                    t
                )

        # Grab the terms that we can evaluate directly first
        chosen = []
        if can_evaluate:
            chosen.append(type(f)(*can_evaluate))

        # Ignore no relation
        chosen.extend([x for v in others.values() for x in v])

        return chosen

    # NOTE: Recombine terms will partition the formula into terms. It also drops unknown
    # terms from each of the given terms. We'll partition f(a,b,c,d) into a, b, c, d,
    # then we'll transform them into a1, b1, c1, d1, then we'll recombine them into
    # f(a1, b1, c1, d1). We must ensure that minimizing f(a1, b1, c1, d1) is the same as
    # minimizing f(a,b,c,d). Min and max: can't transform terms, because if a1 < b1 but
    # a > b, then it will mask which is biggest. Ceiling is OK because it's one term.
    # Add and mul are OK to transform if we're sure that minimizing a1 also minimizes a,
    # but for mul, we need to check the sign of terms, and if any signs are unknown, we
    # don't know if we're minimizing or maximizing.

    if isinstance(f, (sympy.Max, sympy.Min)):
        # We can't just break this up, simplfy, and put back together because we have no
        # guarantee, if we replace variables in the terms with constants, that the
        # result will be the same, because it may change which term is the long pole.
        terms = _recombine_terms(f.args, transform_terms=False)
    elif isinstance(f, (sympy.Add, sympy.ceiling)):
        # CAN break this up because if we can guarantee that for any settings of all the
        # unknown terms, then if one known term is lower, the sum is lower.
        terms = _recombine_terms(f.args)
    elif isinstance(f, sympy.Mul):
        # CAN break this up because if we can guarantee that for any settings of all the
        # unknown terms, then if one known term is lower, the sum is lower.
        terms = _recombine_terms(f.args)
        # If the formula is a product:
        # - Divide the max value by the constant factors
        # - For non-constant factors, if they're >1 then we can keep the max.
        #   Otherwise we have to drop it.
        for t in f.args:
            geq_result = geq_leq_zero(t, bounds)
            if geq_result == ComparisonResult.ALWAYS_LEQ_THAN_ZERO:
                negate = not negate
            elif geq_result == ComparisonResult.UNKNOWN:
                negate = None
                break
            elif geq_result == ComparisonResult.ALWAYS_GEQ_THAN_ZERO:
                pass
            elif geq_result == ComparisonResult.ALWAYS_EQUAL_TO_ZERO:
                pass
            else:
                raise ValueError(
                    f"Comparison result {geq_result} is not a valid comparison result"
                )
    else:
        terms = [_try_replace_unknowns(f)]

    for term in terms:
        term, goal = try_replace_single_term(term, fzs(symbols_enumerated), bounds)
        if goal is not None:
            update_goal(term, goal.goal)
            continue

        # Constant! Don't care
        if len(term.free_symbols & symbols_enumerated) == 0:
            continue

        if term.free_symbols.issubset(symbols_enumerated):
            update_goal(term, "min")
            continue

        # Don't recurse with the same formula. If we got here without simplifying it,
        # give up and mark everything "diff".
        if term == f:
            for symbol in term.free_symbols:
                update_goal(symbol, "diff")
        else:
            for subterm, subgoal in partition_formula(
                term, symbols_enumerated, bounds
            ).items():
                goals[subterm] = subgoal | goals.get(subterm, Goal())

    for k, v in goals.items():
        if negate:
            goals[k] = ~v
        if negate is None:
            v.goal = "diff"

    return goals


@lru_cache(maxsize=10000)
def _get_n_prime_factors(n: int) -> int:
    return len(factorint(n))


def partition_formula(
    f: Expr,
    symbols_enumerated: set[Symbol],
    bounds: tuple[tuple[Symbol, int, int], ...],
) -> dict[Symbol, Goal]:
    return _partition_formula(f, fzs(symbols_enumerated & f.free_symbols), bounds)


def get_possible_factor_sizes(n: int, imperfect: bool = False) -> list[int]:
    factors = []
    for i in range(1, math.ceil(n**0.5) + 1):
        if not imperfect and n % i != 0:
            continue
        factors.append(i)
        factors.append(math.ceil(n / i))
    return sorted(set(factors))


def append_vector(matrix: np.ndarray, vector: np.ndarray):
    if matrix is None:
        return vector.reshape(-1, 1)
    a = np.repeat(matrix, vector.shape[0], axis=0)
    b = np.tile(vector.reshape(-1, 1), (matrix.shape[0], 1))
    max_val = max(np.max(a), np.max(b))
    min_val = min(np.min(a), np.min(b))
    assert min_val >= 0, f"min_val is {min_val}"
    assert max_val <= 2**64, f"max_val is {max_val}"
    if max_val >= 2**32:
        dtype = np.uint64
    elif max_val >= 2**16:
        dtype = np.uint32
    elif max_val >= 2**8:
        dtype = np.uint16
    else:
        dtype = np.uint8

    return np.concatenate((a.astype(dtype), b.astype(dtype)), axis=1)


@lru_cache(maxsize=10000)
def simplify(f: Expr):
    return f.simplify()


def symbol2int(symbol: Symbol):
    return int(re.findall(r"(\d+)", symbol.name)[0])


@lru_cache(maxsize=10000)
def f_minus_other_f(f: Expr, symbols_enumerated: set[Symbol]):
    f2 = f
    for s in f.free_symbols & symbols_enumerated:
        f2 = f2.subs(s, sympy.Symbol(f"{s}_2", integer=True, positive=True))
    return f2 - f > 0


@lru_cache(maxsize=10000)
def affects_comparison(f: Expr, s: Symbol, symbols_enumerated: set[Symbol]):
    if not isinstance(f, sympy.Expr):
        return False
    delta = f_minus_other_f(f, symbols_enumerated)
    if not isinstance(delta, sympy.Expr) or s not in delta.free_symbols:
        return False

    delta = simplify(delta)
    if s not in delta.free_symbols:
        return False

    return True


def get_padded_choices(
    symbols_enumerated: list[Symbol],
    symbols_non_enumerated_set: set[Symbol],
    choices_enumerated: np.ndarray,
    what_tiles_symbol: SymbolRelations,
    minimize_formula: Expr = None,
    maximize_formula: Expr = None,
):
    choices_padded = {}
    ones = np.ones(choices_enumerated.shape[0], choices_enumerated.dtype)
    for symbol in symbols_enumerated:
        choices_padded[symbol] = choices_enumerated[:, symbols_enumerated.index(symbol)]
    for symbol in symbols_non_enumerated_set:
        choices_padded[symbol] = ones
        if minimize_formula is not None or maximize_formula is not None:
            if minimize_formula is None:
                formula = maximize_formula
                sign = -1
            elif maximize_formula is None:
                formula = minimize_formula
                sign = 1
            else:
                raise ValueError(
                    "Both minimize_formula and maximize_formula are not None"
                )
            diff_result = diff_geq_leq_zero(
                sign * formula, symbol, what_tiles_symbol.bounds
            )
            if diff_result == ComparisonResult.ALWAYS_LEQ_THAN_ZERO:
                choices_padded[symbol] = ones * what_tiles_symbol.get_max_size(symbol)
            elif diff_result == ComparisonResult.ALWAYS_GEQ_THAN_ZERO:
                pass
            elif diff_result == ComparisonResult.ALWAYS_EQUAL_TO_ZERO:
                pass
            elif diff_result == ComparisonResult.UNKNOWN:
                raise ValueError(f"Can't tell if {symbol} is increasing or decreasing")
            else:
                raise ValueError(
                    f"Comparison result {diff_result} is not a valid comparison result"
                )

    return choices_padded


def check_loops(
    symbols_enumerated: list[Symbol],
    choices_enumerated: np.ndarray,
    max_loop_check_groups: list[tuple[Number, list[Symbol]]],
    what_tiles_symbol: SymbolRelations,
):
    def get_size(x: Symbol | int):
        if isinstance(x, Symbol) and x in symbols_enumerated:
            return choices_enumerated[:, symbols_enumerated.index(x)]
        elif isinstance(x, Symbol):
            return what_tiles_symbol.get_max_size(x)
        else:
            return x

    def has_fanout(x: Symbol | int):
        try:
            outer = get_size(what_tiles_symbol.get_outer_tiles(x))
        except ValueError:
            return False
        inner = get_size(x)
        return outer != inner

    for limit, group in max_loop_check_groups:
        if len(group) <= limit:
            continue

        n = 0
        for g in group:
            n += has_fanout(g)

        if isinstance(n, np.ndarray):
            choices_enumerated = choices_enumerated[n <= limit]
        elif n > limit:
            choices_enumerated = choices_enumerated[0:0, :]

    return choices_enumerated


def coalesce_symbols(
    update_symbol2goal: Callable,
    symbols_enumerated: list[Symbol],
    symbol2goal: dict[Symbol, Goal],
    log_message: Callable,
    bounds: tuple[tuple[Symbol, int, int], ...],
):
    sym_enumerated_set = fzs(symbols_enumerated)
    new_symbol2goal = {}

    log_message("coalesce symbols", f"initial")
    for s, g in symbol2goal.items():
        log_message(f"\t{g.goal}: {s}")

    changed = True
    while changed:
        new_symbol2goal = {}

        def latest(s=None):
            if s is None:
                x = dict(symbol2goal)
                x.update(new_symbol2goal)
                return x
            return new_symbol2goal[s] if s in new_symbol2goal else symbol2goal[s]

        for formula, goal in list(symbol2goal.items()):
            # Not dependent on any enumerated symbols, so drop it
            if not formula.free_symbols & sym_enumerated_set:
                log_message("coalesce symbols", f"dropping constant: {formula}")
                continue

            # It is an enumerated symbol, so just keep it
            if formula in symbols_enumerated:
                update_symbol2goal(formula, goal, new_symbol2goal)
                continue

            # If it's a sum, remove any terms that are constant
            if isinstance(formula, sympy.Add):
                for term in formula.args:
                    if len(term.free_symbols) == 0:
                        formula = formula.subs(term, 0)
                        log_message("coalesce symbols", f"dropping constant: {term}")
                        continue
                if len(formula.args) == 1:
                    formula = formula.args[0]

            # If it's a product, remove any terms that are constant
            if isinstance(formula, sympy.Mul):
                for term in formula.args:
                    if len(term.free_symbols) == 0:
                        formula = formula.subs(term, 1)
                        if term < 0:
                            goal = ~goal
                        log_message("coalesce symbols", f"dropping constant: {term}")
                        continue
                if len(formula.args) == 1:
                    formula = formula.args[0]

            # If it's a function of a non-enumerated symbol or a symbol that we can't
            # compare and it won't affect comparisons, then we can drop it.

            # If it's a function of a non-enumerated symbol &
            for s in formula.free_symbols:
                if s in symbols_enumerated and latest().get(s, Goal()).goal != "diff":
                    continue

                if not affects_comparison(formula, s, sym_enumerated_set):
                    formula = formula.subs(s, 1)
                    log_message(
                        "coalesce symbols",
                        f"dropping non-comparable symbol that does not affect comparison {s}: {formula}",
                    )
                    continue
                else:
                    log_message(
                        "coalesce symbols",
                        f"keeping dropping symbol that affects comparison {s}: {formula}",
                    )

            # If there's only one symbol in the formula, we can try to replace it with
            # just the symbol.
            if len(formula.free_symbols & sym_enumerated_set) == 1:
                formula, new_goal = try_replace_single_term(
                    formula, sym_enumerated_set, bounds
                )
                if new_goal is not None:
                    log_message("coalesce symbols", f"replacing single term: {formula}")
                    update_symbol2goal(formula, new_goal, new_symbol2goal)

            # If we're a fraction and all of our symbols are in the denominator, replace
            # it with the reciprocal and change the goal
            if isinstance(formula, sympy.Mul):
                for term in formula.args:
                    if len(term.free_symbols) == 0:
                        continue
                    if isinstance(term, sympy.Pow) and term.args[1] == -1:
                        continue
                    break
                else:
                    log_message("coalesce symbols", f"replacing reciprocal: {formula}")
                    formula = 1 / formula
                    goal = ~goal

            # If a formula agrees entirely with other goals, then we can remove it
            disagrees = []
            for s in formula.free_symbols:
                g = latest(s).goal if s in latest() else None
                if g in ["min", "max"]:
                    log_message(
                        "coalesce symbols", f"checking agreement with {s} and goal {g}"
                    )
                    diff_result = diff_geq_leq_zero(formula, s, bounds)
                    if diff_result == ComparisonResult.ALWAYS_LEQ_THAN_ZERO:
                        this_goal = (~goal).goal
                        log_message("coalesce symbols", f"diff result for {s} <= 0")
                    elif diff_result == ComparisonResult.ALWAYS_GEQ_THAN_ZERO:
                        this_goal = (goal).goal
                        log_message("coalesce symbols", f"diff result for {s} >= 0")
                    elif diff_result == ComparisonResult.UNKNOWN:
                        log_message(
                            "coalesce symbols", f"diff result for {s} is unknown"
                        )
                        break
                    elif diff_result == ComparisonResult.ALWAYS_EQUAL_TO_ZERO:
                        log_message("coalesce symbols", f"diff result for {s} = 0")
                        this_goal = g  # Make it agree
                    else:
                        diff_geq_leq_zero(formula, s, bounds)
                        raise ValueError(
                            f"Comparison result {diff_result} is not a valid comparison result"
                        )
                    if g != this_goal:
                        disagrees.append(s)
                    continue
                break
            else:
                # We didn't break! This formula agrees with all other goals, so we can
                # remove it.
                log_message(
                    "coalesce symbols",
                    f"removing formula that agrees with all other goals: {formula}",
                )
                for s in disagrees:
                    log_message(
                        "coalesce symbols",
                        f"previous formula disagreed with {s}. Changing goal to diff",
                    )
                    update_symbol2goal(s, Goal("diff"), new_symbol2goal)
                continue
            log_message(
                "coalesce symbols",
                f"can't change formula: {formula}",
            )
            update_symbol2goal(formula, goal, new_symbol2goal)

        changed = symbol2goal != new_symbol2goal
        symbol2goal = new_symbol2goal

    log_message("coalesce symbols", f"final")
    for s, g in symbol2goal.items():
        log_message(f"\t{g.goal}: {s}")

    return symbol2goal


def get_tile_shape_choices(
    objectives: list[Objective],
    symbols: list[Symbol],
    what_tiles_symbol: SymbolRelations,
    job: "Job",
    keep_symbols: list[Symbol] = (),
    max_loop_check_groups: list[tuple[Number, list[Symbol]]] = (),
):
    objectives = [copy.deepcopy(o) for o in objectives]

    import time

    objectives = objectives.copy()

    symbols_enumerated: list[Symbol] = []
    choices_enumerated: np.ndarray = None

    symbols_remaining = list(symbols)

    imperfect = IMPERFECT

    # Inner to outer faster if there's symbols to keep because those symbols end up in
    # the outer loops, so it does those symbols (which end up multiplying our choices)
    # last. Outer to inner is faster if there's no symbols to keep because that's what
    # happened on exactly one workload that Tanner tested.
    TILE_SHAPE_ORDER = "inner_to_outer_prod_others"

    # For imperfect, we make inner tile shapes, then create outer tile shapes that are
    # multiples of the non-residual part of the inner tile shape. This way, the very
    # last iteration of the outer tile shape fully contains the reisudal part of the
    # inner tile shape, and we don't have any cases where there are residuals stacking
    # across multiple loop levels.
    if IMPERFECT:
        assert "inner_to_outer" in TILE_SHAPE_ORDER

    paretoed_by = []

    prev_time, start_time = time.time(), time.time()
    times = {}

    def time_end(s):
        nonlocal prev_time
        cur_time = time.time()
        times.setdefault(s, 0)
        times[s] += cur_time - prev_time
        prev_time = cur_time

    def log_message(message: str, *args: str):
        t = time.time() - prev_time
        s = "**" if t > 1 else ""
        job.log_message(f"{s}{t:.2f}s: {message} {' '.join(args)}")
        if PRINT_FORMULAS:
            print(f"{time.time() - prev_time:.2f}s: {message} {' '.join(args)}")
        time_end(message)

    log_message("init")

    def eval_objective(
        formula: Expr | Objective,
        choices: np.ndarray,
        minimize_formula: Expr = None,
        maximize_formula: Expr = None,
    ):
        if isinstance(formula, Objective):
            formula = formula.formula
        if formula in symbols_enumerated:
            return choices[:, symbols_enumerated.index(formula)]

        padded_choices = get_padded_choices(
            symbols_enumerated=symbols_enumerated,
            symbols_non_enumerated_set=symbols_non_enumerated_set,
            choices_enumerated=choices,
            what_tiles_symbol=what_tiles_symbol,
            minimize_formula=minimize_formula,
            maximize_formula=maximize_formula,
        )
        return util._lambdify_type_check(symbols, formula)(
            **{str(k): v for k, v in padded_choices.items()},
        )

    def grab_symbol(prev_symbol: Symbol = None):
        strides = [s for s in symbols_remaining if what_tiles_symbol.is_stride(s)]
        if not strides:
            return symbols_remaining.pop(0)

        tile_shape_order = TILE_SHAPE_ORDER

        if "inner_to_outer" in tile_shape_order:
            strides = strides[::-1]
            prev_tile = what_tiles_symbol.get_inner_tiles
            next_tile = what_tiles_symbol.get_outer_tiles
        elif "outer_to_inner" in tile_shape_order:
            prev_tile = what_tiles_symbol.get_outer_tiles
            next_tile = what_tiles_symbol.get_inner_tiles
        else:
            raise RuntimeError(f"BUG: invalid TILE_SHAPE_ORDER: {TILE_SHAPE_ORDER}")

        if "one_rv_at_a_time" in tile_shape_order:
            n = next_tile(prev_symbol, none_if_fail=True)
            if n is not None and n in symbols_remaining:
                symbols_remaining.remove(n)
                return n
            tile_shape_order = tile_shape_order.replace(
                "one_rv_at_a_time", "smallest_first"
            )

        def max_formulas(s):
            prev = prev_tile(s, none_if_fail=True)
            constrains_prev = isinstance(prev, Symbol) and prev not in keep_symbols
            return (
                # Sum number of times it appears in all objectives
                sum(o.formula.count(s) for o in objectives) + 4**constrains_prev,
                # Number of objectives that have this symbol
                len([o for o in objectives if s in o.formula.free_symbols])
                + 4**constrains_prev,
                # Minimize size
                min_size(s),
            )

        def _porp_of_formula(s):
            prev = prev_tile(s, none_if_fail=True)
            n = 0
            for o in objectives:
                f = o.formula
                if s in f.free_symbols:
                    # n += f.count(s) / sum(1 for _ in f.atoms(Symbol))
                    n += 1 / len(f.free_symbols)
            constrains_prev = isinstance(prev, Symbol) and prev not in keep_symbols
            if constrains_prev:
                n += 1
            return n

        def porp_of_formulas(s):
            n = 0
            scale = 1
            cur = s
            while cur is not None and isinstance(cur, Symbol):
                n += _porp_of_formula(cur)
                break
                cur = next_tile(cur, none_if_fail=True)
                scale /= 2

            return n / what_tiles_symbol.get_max_size(s)  # (
            #     n,
            #     *max_formulas(s)
            # )

        def prod_others(s):
            return prod(max_formulas(s))

        def min_size(s):
            return 1 / what_tiles_symbol.get_max_size(s)

        if "smallest_first" in tile_shape_order:
            objective = min_size
        elif "most_formulas_first" in tile_shape_order:
            objective = max_formulas
        elif "prod_others" in tile_shape_order:
            objective = prod_others
        elif tile_shape_order in ["inner_to_outer", "outer_to_inner"]:
            objective = lambda s: 1
        elif "porp_of_formulas" in tile_shape_order:
            objective = porp_of_formulas
        else:
            raise RuntimeError(f"BUG: invalid tile_shape_order: {tile_shape_order}")

        choice, best = 0, objective(strides[0])
        for i, s in enumerate(strides):
            prev = prev_tile(s, none_if_fail=True)
            if prev is not None and prev in symbols_remaining:
                # Haven't made the prerequisite tile shape yet, so skip this symbol
                continue
            value = objective(s)
            if value > best:
                best = value
                choice = i
        choice = symbols_remaining.index(strides[choice])
        log_message("Selected symbol", f"{symbols_remaining[choice]} with value {best}")
        return symbols_remaining.pop(choice)

        raise RuntimeError(f"BUG: invalid TILE_SHAPE_ORDER: {TILE_SHAPE_ORDER}")

    last_stride_symbol = None  # track the last stride symbol to select next symbol
    symbol = None
    while symbols_remaining:
        # ==============================================================================
        # Enumerate choices for a new symbol
        # ==============================================================================
        symbol = grab_symbol(last_stride_symbol)

        choices = []
        if what_tiles_symbol.is_stride(symbol):
            last_stride_symbol = symbol
            inner_tiles = what_tiles_symbol.get_inner_tiles(symbol, none_if_fail=True)
            outer_tiles = what_tiles_symbol.get_outer_tiles(symbol, none_if_fail=True)

            # Figure out inner size and outer size
            if inner_tiles in symbols_enumerated:
                inner_tiles_type = "enumerated"
                inner_size = None
            elif isinstance(inner_tiles, int):
                inner_tiles_type = "set"
                inner_size = inner_tiles
            else:
                inner_tiles_type = "unknown"
                inner_size = 1

            if outer_tiles in symbols_enumerated:
                outer_tiles_type = "enumerated"
                outer_size = None
            elif isinstance(outer_tiles, int):
                outer_tiles_type = "set"
                outer_size = outer_tiles
            else:
                outer_tiles_type = "unknown"
                outer_size = what_tiles_symbol.get_max_size(outer_tiles)

            if inner_tiles_type == "enumerated" and outer_tiles_type == "enumerated":
                raise RuntimeError(
                    f"BUG: both inner, {inner_tiles}, and outer, {outer_tiles},"
                    f"tiles of {symbol} are enumerated (thus far: {symbols_enumerated})"
                )
            if inner_tiles_type == "unknown" and outer_tiles_type == "unknown":
                raise RuntimeError("BUG: both inner and outer tiles are unknown")

            # Use inner size and outer size to generate choices
            if inner_tiles_type in {"set", "unknown"} and outer_tiles_type in {
                "set",
                "unknown",
            }:
                factorize = math.ceil(outer_size / inner_size)
                factors = list(get_possible_factor_sizes(factorize, imperfect))
                scaled = np.array(factors) * inner_size
                choices.append(append_vector(choices_enumerated, scaled))
            elif inner_tiles_type == "enumerated":
                assert isinstance(outer_size, int)
                i = symbols_enumerated.index(inner_tiles)
                for inner_choice in np.unique(choices_enumerated[:, i]):
                    partition = choices_enumerated[
                        np.where(choices_enumerated[:, i] == inner_choice)
                    ]
                    factorize = math.ceil(outer_size / inner_choice)
                    factors = list(get_possible_factor_sizes(factorize, imperfect))
                    scaled = np.array(factors) * inner_choice
                    choices.append(append_vector(partition, scaled))
            else:
                assert outer_tiles_type == "enumerated"
                assert isinstance(inner_size, int)
                i = symbols_enumerated.index(outer_tiles)
                for outer_choice in np.unique(choices_enumerated[:, i]):
                    partition = choices_enumerated[
                        np.where(choices_enumerated[:, i] == outer_choice)
                    ]
                    factorize = math.ceil(outer_choice / inner_size)
                    factors = list(get_possible_factor_sizes(factorize, imperfect))
                    scaled = np.array(factors) * inner_size
                    choices.append(append_vector(partition, scaled))
        elif what_tiles_symbol.is_initial_tile_shape(symbol):
            stride = what_tiles_symbol.get_stride(symbol)
            delta_choices = np.array(list(what_tiles_symbol.get_delta_choices(symbol)))

            outer_stride = what_tiles_symbol.get_outer_tiles(stride, none_if_fail=True)
            assert outer_stride is None or isinstance(
                outer_stride, int
            ), f"outer stride is symbol {outer_stride}"
            if outer_stride is None:
                outer_size = what_tiles_symbol.get_max_size(stride)
            else:
                outer_size = outer_stride

            if not stride in symbols_enumerated and not isinstance(stride, int):
                raise RuntimeError(
                    f"BUG: stride {stride} of initial tile shape "
                    f"{symbol} is neither enumerated nor a specified value"
                )

            if isinstance(stride, int):
                initial_choices = delta_choices + stride
                initial_choices = initial_choices[initial_choices <= outer_size]
                choices.append(append_vector(choices_enumerated, initial_choices))
            else:
                i = symbols_enumerated.index(stride)
                for stride_choice in np.unique(choices_enumerated[:, i]):
                    partition = choices_enumerated[
                        np.where(choices_enumerated[:, i] == stride_choice)
                    ]
                    initial_choices = delta_choices + stride_choice
                    initial_choices = initial_choices[initial_choices <= outer_size]
                    choices.append(append_vector(partition, initial_choices))
        else:
            raise RuntimeError(
                f"BUG: symbol {symbol} is neither stride nor initial tile shape"
            )

        prev_size = choices_enumerated.shape[0] if choices_enumerated is not None else 1
        choices_enumerated = np.concatenate(choices, axis=0)
        choices = None  # Let it be freed by the garbage collector
        job.n_total_pmappings *= choices_enumerated.shape[0] / max(1, prev_size)
        symbols_enumerated.append(symbol)
        log_message("enumerate", f"{symbol}", f"size={choices_enumerated.shape[0]}")

        # ==============================================================================
        # Max fused loops per rank check
        # ==============================================================================

        prev_size = choices_enumerated.shape[0]
        choices_enumerated = check_loops(
            symbols_enumerated,
            choices_enumerated,
            max_loop_check_groups,
            what_tiles_symbol,
        )
        job.log_porp_pmappings_kept(
            f"max_fused_loops_or_max_per_rank_variable",
            choices_enumerated.shape[0] / max(1, prev_size),
        )
        log_message(
            "max_fused_loops_or_max_per_rank_variable",
            f"size {prev_size} -> {choices_enumerated.shape[0]}",
        )

        # ==============================================================================
        # Create initial Pareto-finding goals
        # ==============================================================================
        symbol2goal = {}

        def update_symbol2goal(
            symbol: Symbol, goal: Goal, s2g: dict[Symbol, Goal] = None
        ):
            if s2g is None:
                s2g = symbol2goal
            s2g[symbol] = goal | s2g.get(symbol, Goal())

        # If we're a symbol and a non-enumerated outer loop depends on us, then we need
        # to track this loop. Minimize it if we're imperfect (giving the outer the most
        # choices possible), or diff if we're perfect (since perfect constrains choices
        # so we can't just min).
        for s in symbols_enumerated:
            per_prime_factor = not (
                IMPERFECT
                or _get_n_prime_factors(what_tiles_symbol.get_max_size(s)) == 1
            )
            tiles = what_tiles_symbol.get_outer_tiles(s, none_if_fail=True)
            if isinstance(tiles, Symbol) and tiles not in symbols_enumerated:
                update_symbol2goal(
                    s, Goal("min_per_prime_factor" if per_prime_factor else "min")
                )

            # Same for inner loops depending on us, but maximize if we're imperfect
            tiled_by = what_tiles_symbol.get_inner_tiles(s, none_if_fail=True)
            if isinstance(tiled_by, Symbol) and tiled_by not in symbols_enumerated:
                update_symbol2goal(
                    s, Goal("max_per_prime_factor" if per_prime_factor else "max")
                )

        # If we need to keep this symbol, must preserve all choices for it
        for s in set(symbols_enumerated) & set(keep_symbols):
            update_symbol2goal(s, Goal("diff"))

        symbols_non_enumerated_set = set(symbols) - set(symbols_enumerated)
        sym_enumerated_set = set(symbols_enumerated)

        if job.spec.mapper._count_option_for_mapsapce_size_evaluation != ():
            choices_enumerated = choices_enumerated[:1, :]
            continue

        choices_enumerated_float = choices_enumerated.astype(util.NUMPY_FLOAT_TYPE)

        # ==============================================================================
        # Create functions to Pareto using objectives
        # ==============================================================================
        for objective in list(objectives):
            # ==========================================================================
            # If there's a max value, then check for validity
            # ==========================================================================
            complete = objective.formula.free_symbols.issubset(sym_enumerated_set)
            if objective.max_value is not None:
                try:
                    # minimize_for_objective may raise a TypeError if there's unknown
                    # symbols
                    result = eval_objective(
                        objective.formula,
                        choices_enumerated_float,
                        minimize_formula=objective.formula,
                    )
                    if objective.inclusive:
                        valid = result <= objective.max_value
                    else:
                        valid = result < objective.max_value
                    if not isinstance(valid, np.ndarray):
                        valid = (
                            np.zeros(choices_enumerated.shape[0], dtype=bool) + valid
                        )
                    choices_enumerated = choices_enumerated[valid]
                    choices_enumerated_float = choices_enumerated_float[valid]
                except (TypeError, ValueError):
                    pass

                porp = sum(valid) / max(1, choices_enumerated.shape[0])
                job.log_porp_pmappings_kept(
                    f"{objective.name}",
                    sum(valid) / max(1, prev_size),
                )
                log_message(f"Valid check", f"{objective.name}", f"porp={porp:.2%}")
                if complete:
                    objective.max_value = None  # We don't care anymore
                    objective.min_value = None
                    if objective.only_care_if_valid:
                        objectives.remove(objective)
                        log_message(
                            f"Removed {objective.name} because it is always valid"
                        )

            if objective.min_value is not None:
                try:
                    # minimize_for_objective may raise a TypeError if there's unknown
                    # symbols
                    result = eval_objective(
                        objective.formula,
                        choices_enumerated_float,
                        maximize_formula=objective.formula,
                    )
                    if objective.inclusive:
                        valid = result >= objective.min_value
                    else:
                        valid = result > objective.min_value
                    if not isinstance(valid, np.ndarray):
                        valid = (
                            np.zeros(choices_enumerated.shape[0], dtype=bool) + valid
                        )

                    if not objective.try_best_if_none_reaches_min:
                        choices_enumerated = choices_enumerated[valid]
                        choices_enumerated_float = choices_enumerated_float[valid]
                    else:
                        if valid.any():
                            choices_enumerated = choices_enumerated[valid]
                            choices_enumerated_float = choices_enumerated_float[valid]
                        elif complete:
                            valid |= result == (
                                result.min()
                                if isinstance(result, np.ndarray)
                                else result
                            )
                            choices_enumerated = choices_enumerated[valid]
                            choices_enumerated_float = choices_enumerated_float[valid]
                except (TypeError, ValueError):
                    pass

                porp = sum(valid) / max(1, choices_enumerated.shape[0])
                job.log_porp_pmappings_kept(
                    f"{objective.name}",
                    sum(valid) / max(1, prev_size),
                )
                log_message(f"Valid check", f"{objective.name}", f"porp={porp:.2%}")
                if complete:
                    objective.max_value = None  # We don't care anymore
                    objective.min_value = None
                    if objective.only_care_if_valid:
                        objectives.remove(objective)
                        log_message(
                            f"Removed {objective.name} because it is always valid"
                        )

        if not choices_enumerated.shape[0]:
            return np.array([]).reshape(-1, len(symbols))

        if choices_enumerated.shape[0] < 1000:
            continue

        for objective in objectives:
            # ==========================================================================
            # If there's a max value, then check for validity
            # ==========================================================================
            complete = objective.formula.free_symbols.issubset(sym_enumerated_set)
            prev_size = choices_enumerated.shape[0]

            log_message(
                "Partitioning formula", f"{objective.name}: {objective.formula}"
            )

            goals = partition_formula(
                objective.formula, sym_enumerated_set, what_tiles_symbol.bounds
            )

            log_message(f"formula", f"{objective.formula}")
            for k, v in goals.items():
                log_message("formula", f"\t -> {k}: {v}")

            for symbol, goal in goals.items():
                update_symbol2goal(symbol, goal)

        job.n_evaluated_pmappings += choices_enumerated.shape[0]
        if not choices_enumerated.shape[0]:
            return np.array([]).reshape(-1, len(symbols))

        # ==============================================================================
        # Coalesce symbols. This simplifies our tracked goals. It also breaks down
        # partially-unknown goals into fully-known and/or fully-unknown goals.
        # ==============================================================================
        symbol2goal = coalesce_symbols(
            symbols_enumerated=symbols_enumerated,
            symbol2goal=symbol2goal,
            update_symbol2goal=update_symbol2goal,
            log_message=log_message,
            bounds=what_tiles_symbol.bounds,
        )

        log_message("coalesce symbols", f"{symbol2goal}")

        paretoed_by_key = fzs((f, g.goal) for f, g in symbol2goal.items())
        if any(p.issubset(paretoed_by_key) for p in paretoed_by):
            job.log_message(
                "Skipping Pareto because we've already found a Pareto with these objectives."
            )
            continue
        paretoed_by.append(paretoed_by_key)

        objective_values = {}
        for formula, goal in list(symbol2goal.items()):
            objective_values[formula] = eval_objective(
                formula, choices_enumerated_float
            )
            symbol2goal[formula] = goal
            log_message("eval", f"{goal.goal}", f"{formula}")

        if not objective_values:
            # Objective values don't depend on tile shapes
            choices_enumerated = choices_enumerated[:1, :]
            choices_enumerated_float = choices_enumerated_float[:1, :]

        elif not all(
            symbol2goal.get(s, None) == Goal("diff") for s in symbols_enumerated
        ):
            to_pareto = np.concatenate(
                [v.reshape(-1, 1) for v in objective_values.values()], axis=1
            )
            log_message("Pareto", f"size {to_pareto.shape[0]}", "with objectives:")
            for obj in objectives:
                log_message(f"\t{obj.name}: {obj.formula}")
            log_message("Formulas:")
            for formula, goal in symbol2goal.items():
                log_message(f"\t{goal.goal}: {formula}")

            drop_cols = []
            pareto_goals = []
            for i, (formula, goal) in enumerate(objective_values.items()):
                goal = symbol2goal[formula]
                if i not in drop_cols:
                    pareto_goals.append(goal.goal)
            to_pareto = to_pareto[
                :, [i for i in range(to_pareto.shape[1]) if i not in drop_cols]
            ]
            keep = makepareto_numpy(to_pareto, pareto_goals, dirty=True)
            prev_size = choices_enumerated.shape[0]
            choices_enumerated = choices_enumerated[keep]
            job.log_porp_pmappings_kept(
                f"Pareto", sum(keep) / choices_enumerated.shape[0]
            )
            log_message("pareto", f"size {prev_size} -> {choices_enumerated.shape[0]}")

        # expected_shapes = {
        #     'stride0': 64,
        #     'stride1': 128,
        #     'stride2': 24,
        #     'stride3': 24,
        #     'stride4': 3,
        #     'stride5': 3,
        #     'stride6': 2,
        #     'stride7': 24,
        #     'stride8': 24,
        #     'stride9': 3,
        #     'stride12': 3,
        #     'stride13': 128,
        #     'stride14': 32
        # }
        # matched = np.ones(len(choices_enumerated), dtype=bool)
        # for i, s in enumerate(symbols_enumerated):
        #     s = str(s)
        #     if s in expected_shapes:
        #         cur_symbol = choices_enumerated[:, i]
        #         matched &= cur_symbol == expected_shapes[s]
        # assert np.sum(matched) > 0

    # ==================================================================================
    # Return the choices
    # ==================================================================================
    t = time.time() - start_time
    if t > 60 and DEBUG:
        a = [
            f"Total time: {t:.2f}s",
            f"Pmapping: {job.mapping.compact_str()}",
        ]
        print("\n\t" + f"\n\t".join(a + job.messages))

    # Rearrange in tile shape order
    if choices_enumerated is None:
        return np.array([])
    choices_enumerated = choices_enumerated.astype(np.int64)
    return choices_enumerated[:, [symbols_enumerated.index(s) for s in symbols]]


def makesymbol(name: str):
    # TODO: Do the solve() calls work with integer=True?
    return Symbol(name, positive=True, integer=True)


def make_keep_symbols(pmapping: Mapping) -> set[Symbol]:
    keep_symbols = set()
    for node in pmapping.nodes:
        if isinstance(node, Loop) and node._fused:
            if isinstance(node.initial_tile_shape, Symbol):
                keep_symbols.add(node.initial_tile_shape)
            if isinstance(node.tile_shape, Symbol):
                keep_symbols.add(node.tile_shape)
    return keep_symbols


def get_rank_var_to_fused_loops(
    pmapping: Mapping, shape: dict[str, int]
) -> dict[str, list[Symbol]]:
    rank_var_to_fused_loops: dict[str, list[Symbol]] = {}
    for node in [n for n in pmapping.nodes if isinstance(n, Loop) and n._fused]:
        rank_var_to_fused_loops.setdefault(node.rank_variable, []).append(
            node.tile_shape
        )
    return rank_var_to_fused_loops


def set_last_tile_shape_to_one(pmapping):
    pmapping = pmapping.nodes

    rank_var_to_last_node = {}
    for node in pmapping:
        if isinstance(node, Temporal) or isinstance(node, Spatial):
            rank_var_to_last_node[node.rank_variable] = node

    for last_node in rank_var_to_last_node.values():
        last_node.initial_tile_shape = None
        last_node.tile_shape = 1


# This was made only so we could do some counting of the time.
def call_compiled_objective(f, *args):
    return f(*args)


def _calculate_iterations_and_rank_columns(
    pmapping: list[MappingNode], job: "Job", df: pd.DataFrame, shape: dict[str, int]
):
    loops = [n for n in pmapping if isinstance(n, Loop)]
    # Some initial tile shapes are invalid
    for nloops, n in enumerate(loops):
        if not n._fused:
            break
        stride = n.tile_pattern.tile_shape
        initial = (
            n.tile_pattern.initial_tile_shape
            if n.tile_pattern.initial_tile_shape is not None
            else stride
        )
        outer_stride = job.rank_variable_bounds[n.rank_variable]
        outer_initial = job.rank_variable_bounds[n.rank_variable]
        for l in loops[:nloops]:
            if l.rank_variable == n.rank_variable:
                outer_stride = l.tile_shape
                outer_initial = l.initial_tile_shape
                if outer_initial is None:
                    outer_initial = outer_stride

        outer_initial = (
            df[outer_initial.name]
            if isinstance(outer_initial, Symbol)
            else outer_stride
        )

        rank_var_stride = df[stride.name] if isinstance(stride, Symbol) else stride
        rank_var_initial = df[initial.name] if isinstance(initial, Symbol) else initial

        # NOTE: The concept of having one "n_iterations" is precarious when imperfect
        # factorization in involved
        df[iterations2col(nloops)] = np.ceil(
            (outer_initial - rank_var_initial) / rank_var_stride + 1
        )
        df[f"lower_iterations<SEP>{nloops}"] = outer_stride - rank_var_initial

        # Generate rank columns
        einsum: Einsum = job.spec.workload.einsums[job.einsum_name]
        for tensor_access in einsum.tensor_accesses:
            tensor = tensor_access.name
            projections = get_projection_expr(einsum, tensor)
            for rank, expr in projections.items():
                free_symbols = tuple(expr.free_symbols)
                free_symbols_str = tuple(symbol.name for symbol in free_symbols)
                if n.rank_variable not in free_symbols_str:
                    continue

                rank_stride = expr.coeff(n.rank_variable) * rank_var_stride

                args = []
                for free_rank_var in free_symbols:
                    if free_rank_var.name == n.rank_variable:
                        args.append(rank_var_initial)
                    else:
                        args.append(shape[free_rank_var.name])
                rank_initial = lambdify(free_symbols, expr)(*args)

                df[stride2col(rank, nloops)] = rank_stride
                df[initial2col(rank, nloops)] = rank_initial


def _make_tile_shapes(job: "Job"):
    # We're going to convert the job into a list of symbols and objectives
    pmapping = job.mapping
    constraints = job.constraints
    constraints.set_loop_indices(pmapping.nodes)
    set_last_tile_shape_to_one(pmapping)
    t0 = time.time()
    (
        symbols,
        symbolic_df,
        per_memory_usage_df,
        usage_df,
        tensor2mapping,
    ) = run_model(job)

    model_time = time.time() - t0
    shape = job.rank_variable_bounds
    what_tiles_symbol = SymbolRelations.from_pmapping_and_shape(
        pmapping, shape, job.spec.workload
    )
    keep_symbols = make_keep_symbols(pmapping)
    rank_var_to_fused_loops = get_rank_var_to_fused_loops(pmapping, shape)
    all_fused_loops = set(sum(rank_var_to_fused_loops.values(), []))

    objectives = []

    # ==================================================================================
    # Loop bounds constraints. Put these before the other objectives so that hopefully
    # if 100% of the pmappings are pruned, then we're given the actual architecture
    # component that caused it and not the loop bound constraint.
    # ==================================================================================
    loops = [n for n in pmapping.nodes if isinstance(n, Loop)]
    for c in constraints.loop_bounds_constraints:
        min_value, max_value, inclusive = None, None, True
        is_product = "product" in c.constraint.operator
        operator = c.constraint.operator.replace("product", "")
        if operator in ["==", "<=", "<"]:
            max_value = c.constraint.value
        if operator in [">=", ">", "=="]:
            min_value = c.constraint.value
        if operator in ["<", ">"]:
            inclusive = False

        targets = []
        for i in c._target_loop_indices:
            n = loops[i]
            size = job.rank_variable_bounds[n.rank_variable]
            for l in loops[:i]:
                if l.rank_variable == n.rank_variable:
                    size = l.tile_shape
            targets.append(size / n.tile_shape)

        # targets = [loops[i]._calculated_n_iterations for i in c._target_loop_indices]
        if not targets:
            continue

        if is_product:
            targets = [sympy.Mul(*targets)]

        if max_value is None and min_value is not None:
            max_value = -min_value
            targets = [-target for target in targets]
            min_value = None

        for target in targets:
            objectives.append(
                Objective(
                    name=f"loop_bounds_{c.constraint}",
                    formula=target,
                    symbols=symbols,
                    only_care_if_valid=True,
                    max_value=max_value,
                    min_value=min_value,
                    inclusive=inclusive,
                )
            )

    # ==================================================================================
    # Memory usage and usage constraints.
    # ==================================================================================
    for k, v in {**per_memory_usage_df, **usage_df}.items():
        # If we only track for pmappings, we only care if it's valid. If we track for
        # all, we care about the value too.

        split = k.split("<SEP>")
        assert split[0] == "usage", f"invalid {split}"
        if split[1] == "spatial":
            assert len(split) == 4
        elif split[1] == "memory":
            assert len(split) == 3
        else:
            assert False, f"invalid {split}"

        only_care_if_valid = False
        if split[2] in job.memories_track_pmappings_only:
            only_care_if_valid = True

        # TODO: Update check to see if we may be sharing usage with other
        # pmappings in parallel/pipeline.
        if k in usage_df:
            only_care_if_valid = True

        objectives.append(
            Objective(
                name=k,
                formula=v,
                symbols=symbols,
                only_care_if_valid=only_care_if_valid,
                max_value=1,
            )
        )

    # ==================================================================================
    # Min usage constraints. Put this last because it has some try best if none reach
    # min logic.
    # ==================================================================================
    for (
        component_name,
        name,
    ), constraint in job.constraints.min_usage_constraints.items():
        objectives.append(
            Objective(
                name=f"min_usage_{component_name}_{name}",
                formula=v,
                symbols=symbols,
                only_care_if_valid=True,
                min_value=constraint.min_usage,
                try_best_if_none_reaches_min=True,
            )
        )

    for k, v in symbolic_df.items():
        if "Total" not in k:
            continue

        objectives.append(
            Objective(
                name=k,
                formula=v,
                symbols=symbols,
            )
        )

    rank2symbols = {}
    for node in pmapping.nodes:
        if isinstance(node, (Temporal, Spatial)):
            if node.tile_shape in symbols:
                rank2symbols.setdefault(node.rank_variable, []).append(node.tile_shape)

    max_loop_check_groups = [
        (job.spec.mapper.max_fused_loops, all_fused_loops),
        *[
            (job.spec.mapper.max_fused_loops_per_rank_variable, x)
            for x in rank_var_to_fused_loops.values()
        ],
    ]

    max_loop_check_groups = [g for g in max_loop_check_groups if g[1]]

    choices_enumerated = get_tile_shape_choices(
        objectives=objectives,
        symbols=symbols,
        what_tiles_symbol=what_tiles_symbol,
        job=job,
        keep_symbols=keep_symbols,
        max_loop_check_groups=max_loop_check_groups,
    )

    try:
        compiled_df = compile_dict(symbols, symbolic_df)
        compiled_per_memory_usage_df = compile_dict(symbols, per_memory_usage_df)
        compiled_usage_df = compile_dict(symbols, usage_df)
    except Exception as e:
        print("Compilation failed for this mapping:")
        for node in pmapping.nodes:
            if hasattr(node, "compact_str"):
                print(node.compact_str())
        print(symbolic_df)
        e.add_note("Compilation failed")
        raise

    choices_float = choices_enumerated.astype(util.NUMPY_FLOAT_TYPE)
    # choices_float = np.tile(choices_float, (1000000, 1))
    # choices_enumerated = np.tile(choices_enumerated, (1000000, 1))

    df = {}
    for i, symbol in enumerate(symbols):
        df[symbol.name] = choices_enumerated[:, i]

    t0 = time.time()
    for key in compiled_df:
        df[key] = call_compiled_objective(compiled_df[key], *choices_float.T)
        if "latency" in key and "first_latency" not in key:
            val = [df[key]] if isinstance(df[key], Number) else df[key]
            if any(l < 0 for l in val):
                raise ValueError(f"Negative latency for {key}: {val}")
        if "energy" in key:
            val = [df[key]] if isinstance(df[key], Number) else df[key]
            if any(l < 0 for l in val):
                raise ValueError(f"Negative energy for {key}: {val}")

    _calculate_iterations_and_rank_columns(pmapping.nodes, job, df, shape)

    try:
        df = pd.DataFrame(df, columns=df.keys())
    except ValueError as e:
        df = pd.DataFrame(df, columns=df.keys(), index=[0])
    assert not df.isna().any().any()

    energy_cols = [c for c in df.columns if "Total<SEP>energy" in c]
    if (df[energy_cols] < 0).any(axis=None):
        mapping_with_negative_energy = df[(df[energy_cols] < 0).any(axis=1)]
        print(df.columns)
        msg = ""
        for _, row in mapping_with_negative_energy.iterrows():
            for k, v in row.items():
                msg += f"{k}: {v}\n"
            msg += "\n"
        raise RuntimeError(f"negative energy:\n{msg}")

    job.n_valid_pmappings = job.n_total_pmappings * prod(
        job.pmapping_keep_rates.values()
    )
    return df, tensor2mapping


def make_tile_shapes(job: "Job"):
    memory_limit = job.memory_limit // 8  # Bytes -> bits
    if job.memory_limit != float("inf"):
        try:
            resource.setrlimit(resource.RLIMIT_AS, (job.memory_limit, job.memory_limit))
        except (ValueError, OSError):
            # Ignore permission errors when trying to set memory limits
            pass

    if job.time_limit != float("inf"):
        try:
            resource.setrlimit(
                resource.RLIMIT_CPU, (ceil(job.time_limit), ceil(job.time_limit))
            )
        except (ValueError, OSError):
            # Ignore permission errors when trying to set CPU limits
            pass

    def format_memory_limit() -> str:
        if memory_limit == float("inf"):
            return "infinite"
        if memory_limit > 1024 * 1024 * 1024:
            return f"{memory_limit / (1024 * 1024 * 1024):.2f} GB"
        elif memory_limit > 1024 * 1024:
            return f"{memory_limit / (1024 * 1024):.2f} MB"
        elif memory_limit > 1024:
            return f"{memory_limit / 1024:.2f} KB"
        else:
            return f"{memory_limit:.2f} B"

    try:
        return _make_tile_shapes(job)
    except MemoryError as e:
        s = f"Job ran out of memory with memory limit {format_memory_limit()}"
        job.log_message(f"Tile shape exploration failed: {s}")
        raise RuntimeError(job.pretty_str()) from e
    except TimeoutError as e:
        s = f"Job timed out with time limit {job.time_limit:.2f} seconds"
        job.log_message(f"Tile shape exploration failed: {s}")
        raise RuntimeError(job.pretty_str()) from e

    finally:
        try:
            resource.setrlimit(
                resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
            )
        except (ValueError, OSError):
            # Ignore permission errors when trying to reset memory limits
            pass
        try:
            resource.setrlimit(
                resource.RLIMIT_CPU, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
            )
        except (ValueError, OSError):
            # Ignore permission errors when trying to reset CPU limits
            pass
