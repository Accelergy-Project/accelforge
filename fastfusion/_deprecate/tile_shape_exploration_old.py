from collections import defaultdict
from functools import lru_cache
import itertools
from math import ceil, prod
import copy
import random
import re
import resource
from typing import Callable, Optional

# from combinatorics.integer import integer_factorizations_to_n_parts
from dataclasses import dataclass, field

from fastfusion import util
from fastfusion._accelerated_imports import np

import sympy

from fastfusion._accelerated_imports import pd

from fastfusion.frontend import mapping
import fastfusion.frontend.arch as arch
from fastfusion.frontend.arch import Memory
from fastfusion.frontend.spec import Spec
from fastfusion.frontend.workload import Workload
from fastfusion.frontend.workload._isl import get_rank_variable_bounds
from fastfusion.frontend.workload._symbolic import get_stride_and_halo
from fastfusion.frontend.mapping import (
    Loop,
    Mapping,
    MappingNode,
    Temporal,
    Spatial,
    TensorHolder,
    TilePattern,
)

from fastfusion.mapper.FFM._make_pmappings.contraints.constraints import (
    MappingConstraints,
)
from fastfusion.mapper.FFM._make_pmappings.mapper_one_einsum.mapper_job import Job
from fastfusion.model._looptree.reuse.summarized.symbolic import (
    analyze_reuse_and_add_reservations_to_mapping,
)
from fastfusion.model._looptree.energy import (
    compute_energy_from_actions,
    gather_actions,
)
from fastfusion.model._looptree.latency import get_latency

from fastfusion.mapper.FFM._pmapping_group import (
    nameloop2col,
    tensor2col,
    firstlatency2col,
)
from fastfusion.frontend.mapper.metrics import Metrics
from fastfusion.util._frozenset import fzs


def run_model(job: Job):
    pmapping = job.mapping
    spec = job.spec
    metrics = job.metrics
    is_copy_op = job.is_copy_operation
    workload = spec.workload
    ert = spec.component_energy

    component_to_max_fanout = {}
    memory_to_size = {}
    for node in job.flattened_arch:
        if isinstance(node, arch.TensorHolder):
            if isinstance(node, arch.Memory):
                memory_to_size[node.name] = node.attributes.size
        component_to_max_fanout[node.name] = {s.name: s.fanout for s in node.spatial}

    df = {}

    reuse = analyze_reuse_and_add_reservations_to_mapping(job)
    overall_latency, comp_latency, mem_latency = get_latency(
        reuse, pmapping, workload, job.flattened_arch
    )
    actions = gather_actions(reuse, None, use_name=True)
    energy = compute_energy_from_actions(actions, ert, overall_latency)

    intermediate_tensors = workload.intermediate_tensor_names
    tensor_to_backing = {}
    for node in pmapping.nodes:
        if isinstance(node, TensorHolder):
            for tensor in node.tensors:
                if tensor not in tensor_to_backing and tensor in intermediate_tensors:
                    tensor_to_backing[tensor] = node.component

    total_occupancy = {}
    compute_unit = pmapping.nodes[-1].compute
    max_n_loops = 0

    for buffet, stats in reuse.buffet_stats.items():
        if buffet.level == compute_unit:
            continue

        occupancy = stats.max_occupancy

        if occupancy == 0:
            continue

        for tensor, backing in tensor_to_backing.items():
            if (is_copy_op or buffet.tensor == tensor) and buffet.level == backing:
                df[tensor2col(tensor)] = occupancy

        if buffet.level not in total_occupancy:
            total_occupancy[buffet.level] = {stats.n_loops_above: occupancy}
        else:
            occupancy_per_level = total_occupancy[buffet.level]
            if stats.n_loops_above not in occupancy_per_level:
                occupancy_per_level[stats.n_loops_above] = occupancy
            else:
                occupancy_per_level[stats.n_loops_above] += occupancy

        max_n_loops = max(max_n_loops, stats.n_loops_above + 1)

    for memory, occupancies in total_occupancy.items():
        running_total = 0
        for n_loop in range(max_n_loops):
            if n_loop in occupancies:
                running_total += occupancies[n_loop]
                if True or memory in job.memories_track_all:
                    df[nameloop2col(memory, n_loop)] = running_total

    if metrics & Metrics.LATENCY:
        df[f"Total<SEP>latency"] = overall_latency
        df[f"latency<SEP>compute"] = comp_latency
        # For first latency, we'll follow the convention of treating compute
        # as a component, similarly to memory (see below).
        for compute_level, stats in reuse.compute_stats.items():  # FIRST LATENCY
            for idx, max_first_latency in stats.max_first_latency.items():
                df[firstlatency2col(compute_level, idx)] = max_first_latency
        for component, latency in mem_latency.items():
            df[f"latency<SEP>{component}"] = latency

    if metrics & Metrics.ENERGY:
        df[f"Total<SEP>energy"] = sum(energy.values())
        for (component, action), energy in energy.items():
            df[f"energy<SEP>{component}<SEP>{action}"] = energy

    if metrics & Metrics.RESERVATIONS:
        for memory, occupancies in total_occupancy.items():
            df[f"reservations<SEP>{memory}"] = sum(occupancies.values())

    per_memory_usage_df = {}
    for memory, occupancies in total_occupancy.items():
        per_memory_usage_df[memory] = sum(occupancies.values()) / memory_to_size[memory]

    utilization_df = {}
    for (component, einsum), per_dim_fanout in reuse.fanout.items():
        for dim, fanout in per_dim_fanout.items():
            utilization_df[f"utilization<SEP>{component}<SEP>{dim}"] = (
                fanout / component_to_max_fanout[component][dim]
            )

    return reuse.symbols, df, per_memory_usage_df, utilization_df


def compile_dict(symbols, dictionary):
    def lambdify(key, value):
        x = util._lambdify_type_check(symbols, value)
        # x._loops = [int(loop) for loop in re.findall(r'loop(\d+)_', key)]
        return x

    return {k: lambdify(symbols, v) for k, v in dictionary.items()}


def get_initial_delta_choices(einsum_name: str, workload: Workload):
    stride_and_halo = get_stride_and_halo(workload)
    einsum = workload.einsums[einsum_name]

    choices = defaultdict(lambda: set([0]))
    consumer_chains = []
    stack = [[(None, einsum)]]
    while stack:
        cur_chain = stack.pop()
        last_tensor, last_einsum = cur_chain[-1]
        for tensor in last_einsum.output_tensor_names:
            einsums_with_tensor_as_input = workload.einsums_with_tensor_as_input(tensor)

            if len(einsums_with_tensor_as_input) == 0:
                consumer_chains.append(cur_chain)

            for next_einsum in einsums_with_tensor_as_input:
                stack.append(cur_chain + [(tensor, next_einsum)])

    for chain in consumer_chains:
        for (_, producer), (tensor, consumer) in zip(
            list(reversed(chain))[1:], reversed(chain)
        ):
            rank_stride_and_halo = stride_and_halo[(consumer.name, tensor)]
            if tensor is None:
                break  # done

            for cons_rank_var in consumer.rank_variables:
                for prod_rank_var in producer.rank_variables:
                    for cons_choice in choices[cons_rank_var]:
                        if (prod_rank_var, cons_rank_var) not in rank_stride_and_halo:
                            continue
                        stride, halo = rank_stride_and_halo[
                            (prod_rank_var, cons_rank_var)
                        ]
                        choices[prod_rank_var].add(cons_choice * stride + halo)

    return choices


"""

# FFM-Style Pmapper

This doc is an attempt to formalize using FFM-style pruning for pmappings that only have
some loops' tile shapes chosen.

Suppose that we're selecting tile shapes $t_0...t_n$ to populate the loops in our
pmapping template. We have a set of objectives $f_i(t_0,...t_n)$ to minimize. Assume
that we've enumerated all Pareto-optimal pmapping choices for $t_0..t_i$, and we'd like
to construct critera by which we can compare two chosen sets of tile shapes.

We can construct criteria for a pmapping using the following:

- For each $f_i$:

  - Calculate $f_i(t_0...t_n)-f_i(u_0...u_n)>0$. If the function does not depend on
    $t_{i+1}...t_n$, then we can compare $f_i$ values directly by plugging in dummy
    numbers for $t_{i+1}...t_n$. This case may occur if we're minimizing one term of a
    sum-of-products, such as when summing energy for each component in the architecture.
  - Otherwise, calculate $df_i/dt_j$ for each $t_0..t_i$. If the derivative $=0$, we no
    longer need to consider $t_j$ for this $f_i$.

    - If the derivative is $>0$, mark $t_j$ as "maximize".
    - If the derivative is $<0$, mark $t_j$ as "minimize".
    - If the derivative depends on $t_{k\neq j}$, mark $t_j$ as "can't compare".

- Mark $t_i$ as "can't compare" if any of the following are true:

  - $t_i$ is the outermost enumerated loop for a given rank, since it determine the
    options for future tile shapes. This check can be skipped if the next-outermost loop
    is not symbolic.
  - $t_i$ is a fused loop
  - $t_i$ is marked with conflicting "minimize" and "maximize".

To extend to constraints, make each constraint an objective with a max value. Once a
constraint can be fully evaluated, remove all pmappings that violate the constraint and
remove the constraint from the list of objectives.

"""

import math
import sympy
from sympy import Symbol
import numpy as np
import paretoset
from enum import Enum
from numbers import Number


class EvaluationState(Enum):
    EVALUATABLE = "evaluatable"
    COMPARABLE = "comparable"
    INCOMPARABLE = "incomparable"
    IRRELEVANT = "irrelevant"


class Goal:
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
        if self.goal == other.goal:
            return Goal(self.goal, max_value=mv, only_care_if_valid=care)
        return Goal("diff", only_care_if_valid=care)

    def __str__(self):
        return f"{self.goal} {self.max_value} {self.only_care_if_valid}"

    def __repr__(self):
        return f"Goal({self.goal}, {self.max_value}, {self.only_care_if_valid})"

    def __invert__(self):
        if self.goal == "min":
            return Goal("max", self.max_value, self.only_care_if_valid)
        elif self.goal == "max":
            return Goal("min", self.max_value, self.only_care_if_valid)
        else:
            return copy.copy(self)

    def __eq__(self, other: "Goal"):
        return (
            isinstance(other, Goal)
            and self.goal == other.goal
            and self.max_value == other.max_value
            and self.only_care_if_valid == other.only_care_if_valid
        )


def get_denoms(f: sympy.Expr) -> list[sympy.Expr]:
    denoms = []
    for term in f.args:
        cur_denoms = []
        if isinstance(term, sympy.Mul):
            for factor in term.args:
                if isinstance(factor, sympy.Pow) and factor.args[1] == -1:
                    cur_denoms.append(factor.args[0])
        for d in denoms:
            for c in cur_denoms:
                if d == c:
                    cur_denoms.remove(c)
                    break
        denoms.extend(cur_denoms)
    return denoms


@lru_cache(maxsize=1000)
def simplify_denoms(f: sympy.Expr) -> sympy.Expr:
    return f * sympy.denom(f)


def is_constant(f: sympy.Expr) -> bool:
    try:
        return f.is_constant()
    except ValueError:
        return all(is_constant(arg) for arg in f.args)


@lru_cache(maxsize=10000)
def _try_replace_single_term(t: sympy.Expr, symbols_enumerated: fzs[Symbol]):
    goal = None
    if len(t.free_symbols & symbols_enumerated) == 1:
        s = next(iter(t.free_symbols & symbols_enumerated))
        try:
            diffed = sympy.diff(t, s)
            if diffed > 0:
                goal = Goal("min")
            elif diffed < 0:
                goal = Goal("max")
            else:
                goal = Goal("diff")
            t = t.subs(s, 1)
            return s, goal
        except (TypeError, ValueError):
            pass
    return t, None


@lru_cache(maxsize=10000)
def _partition_formula(
    f: sympy.Expr,
    symbols_enumerated: set[Symbol],
) -> dict[Symbol, Goal]:
    goals: dict[Symbol, Goal] = {}

    def update_goal(symbol: Symbol, goal: str, **kwargs):
        goals[symbol] = Goal(goal) | goals.get(symbol, Goal())

    negate = False

    if not f.free_symbols & symbols_enumerated:
        return goals

    def _try_replace_unknowns(t: sympy.Expr):
        for s in t.free_symbols:
            if s in symbols_enumerated:
                continue
            try:
                diffed = sympy.diff(t, s)
                if diffed > 0 or diffed < 0 or diffed == 0:
                    t = t.subs(s, 1)
            except (TypeError, ValueError):
                pass
        return t

    def _recombine_terms(terms: list[sympy.Expr]):
        symbols2terms = {}
        for t in terms:
            t = _try_replace_unknowns(t)
            try:
                if not t.free_symbols & symbols_enumerated:
                    continue
            except (TypeError, ValueError):
                pass
            symbols2terms.setdefault(
                fzs(t.free_symbols - symbols_enumerated), []
            ).append(t)
        return [type(f)(*terms) for terms in symbols2terms.values()]

    if isinstance(f, sympy.Max):
        terms = _recombine_terms(f.args)
    elif isinstance(f, sympy.Add):
        terms = _recombine_terms(f.args)
    elif isinstance(f, sympy.Mul):
        terms = _recombine_terms(f.args)
        # If the formula is a product:
        # - Divide the max value by the constant factors
        # - For non-constant factors, if they're >1 then we can keep the max.
        #   Otherwise we have to drop it.
        for t in f.args:
            try:
                if negate is not None and t < 0:
                    negate = not negate
            except TypeError:
                negate = None
    else:
        terms = [_try_replace_unknowns(f)]

    for term in terms:
        term, goal = _try_replace_single_term(term, fzs(symbols_enumerated))
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
            for subterm, subgoal in partition_formula(term, symbols_enumerated).items():
                goals[subterm] = subgoal | goals.get(subterm, Goal())

    for k, v in goals.items():
        if negate:
            goals[k] = ~v
        if negate is None:
            v.goal = "diff"

    return goals


def partition_formula(
    f: sympy.Expr,
    symbols_enumerated: set[Symbol],
) -> dict[Symbol, Goal]:
    return _partition_formula(f, fzs(symbols_enumerated & f.free_symbols))


class Objective:
    def __init__(
        self,
        name: str,
        formula: sympy.Expr,
        max_value: float = None,
        symbols: list[str] = None,
        only_care_if_valid: bool = False,
        min_value: float = None,
    ):
        self.name: str = name
        self.formula: sympy.Expr = formula.simplify()
        self._formula_compiled: Callable = None
        self._symbols: list[str] = symbols
        self.max_value: float = max_value
        self.only_care_if_valid: bool = only_care_if_valid
        if only_care_if_valid:
            assert self.max_value is not None
        self.min_value: float = min_value

    def is_formula(self):
        return not isinstance(self.formula, (float, int))

    def differentiate(self, symbol: Symbol):
        return sympy.diff(self.formula, symbol)

    @property
    def formula_compiled(self):
        if self._formula_compiled is None:
            self._formula_compiled = util._lambdify_type_check(
                self._symbols, self.formula
            )
        return self._formula_compiled

    def __call__(self, values: dict[Symbol, np.array]):
        return self._formula_compiled(**{k.name: v for k, v in values.items()})


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
    return np.concatenate(
        (
            np.repeat(matrix, vector.shape[0], axis=0),
            np.tile(vector.reshape(-1, 1), (matrix.shape[0], 1)),
        ),
        axis=1,
    )


def symbol2int(symbol: Symbol):
    return int(re.findall(r"(\d+)", symbol.name)[0])


def get_padded_choices(
    symbols_enumerated: list[Symbol],
    symbols_non_enumerated_set: set[Symbol],
    choices_enumerated: np.ndarray,
    what_tiles_symbol: list[tuple[Symbol | int, Symbol | int]],
    minimize_formula: sympy.Expr = None,
):
    choices_padded = {}
    ones = np.ones(choices_enumerated.shape[0])
    for symbol in symbols_enumerated:
        choices_padded[symbol] = choices_enumerated[:, symbols_enumerated.index(symbol)]
    for symbol in symbols_non_enumerated_set:
        choices_padded[symbol] = ones
        if minimize_formula is not None:
            if sympy.diff(minimize_formula, symbol) < 0:
                choices_padded[symbol] = ones * get_max_size(symbol, what_tiles_symbol)
            elif sympy.diff(minimize_formula, symbol) > 0:
                pass
            elif sympy.diff(minimize_formula, symbol) == 0:
                pass
            else:
                raise ValueError(f"Can't tell if {symbol} is increasing or decreasing")

    return choices_padded


def get_tiled_by(
    symbol: Symbol,
    what_tiles_symbol: list[tuple[Symbol | int, Symbol | int]],
):
    for tiled_by, what_tiles in what_tiles_symbol:
        if tiled_by == symbol:
            return what_tiles
    raise ValueError(f"Symbol {symbol} not found in what_tiles_symbol")


def get_tiles(
    symbol: Symbol,
    what_tiles_symbol: list[tuple[Symbol | int, Symbol | int]],
):
    for tiled_by, what_tiles in what_tiles_symbol:
        if what_tiles == symbol:
            return tiled_by
    raise ValueError(f"Symbol {symbol} not found in what_tiles_symbol")


def get_max_size(
    symbol: Symbol,
    what_tiles_symbol: list[tuple[Symbol | int, Symbol | int]],
):
    while not isinstance(symbol, Number):
        symbol = get_tiles(symbol, what_tiles_symbol)
    return symbol


def check_max_fused_loops_per_rank(
    symbols: list[Symbol],
    symbols_enumerated: list[Symbol],
    choices_enumerated: np.ndarray,
    max_fused_loops_per_rank_check_groups: list[list[Symbol]],
    max_fused_loops_per_rank_variable: Number,
):
    if max_fused_loops_per_rank_variable >= len(symbols_enumerated):
        return choices_enumerated

    def get_size(x: Symbol | int):
        if isinstance(x, Symbol):
            x = choices_enumerated[:, symbols_enumerated.index(x)]
        return x

    def can_check(x: Symbol | int):
        return not isinstance(x, Symbol) or x in symbols_enumerated

    for group in max_fused_loops_per_rank_check_groups:
        if len(group) <= max_fused_loops_per_rank_variable + 1:
            continue

        n = 0
        for i, a in enumerate(group[:-1]):
            b = group[i + 1]
            if not (can_check(a) and can_check(b)):
                continue
            n += get_size(a) != get_size(b)
        if isinstance(n, np.ndarray):
            choices_enumerated = choices_enumerated[
                n <= max_fused_loops_per_rank_variable
            ]
        elif n > max_fused_loops_per_rank_variable:
            choices_enumerated = choices_enumerated[0:0, :]

    return choices_enumerated


def coalesce_symbols(
    symbols: list[Symbol],
    update_symbol2goal: Callable,
    symbols_enumerated: list[Symbol],
    symbol2goal: dict[Symbol, Goal],
    log_message: Callable,
):
    sym_enumerated_set = set(symbols_enumerated)
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

            # If it's a function of a non-enumerated symbol or a symbol that we can't
            # compare and the derivative of the formula WRT the symbol does not change
            # sign (i.e., the formula is monotonic WRT the symbol), then we can drop it
            # because it won't affect comparisons.
            for s in formula.free_symbols:
                goal_str = latest(s).goal if s in latest() else None
                # If we don't know the symbol yet, we can't compare it
                if s not in symbols_enumerated:
                    goal_str = "diff"

                try:
                    diffed = sympy.diff(formula, s)
                    # it anyway or we can't compare this symbol, we don't need to
                    # include it here.
                    if (diffed < 0 or diffed > 0 or diffed == 0) and goal_str == "diff":
                        log_message(
                            "coalesce symbols", f"dropping symbol {s}: {formula}"
                        )
                        formula = formula.subs(s, 1)
                except TypeError:
                    pass

            # If there's only one symbol in the formula, we can try to replace it with
            # just the symbol.
            if len(formula.free_symbols & sym_enumerated_set) == 1:
                prev = formula
                formula, goal = _try_replace_single_term(
                    formula, fzs(symbols_enumerated)
                )
                if goal is not None:
                    log_message("coalesce symbols", f"replacing single term: {formula}")
                    update_symbol2goal(formula, goal, new_symbol2goal)
                    continue

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
                    try:
                        diffed = sympy.diff(formula, s)
                        if diffed > 0:
                            if g == goal.goal:
                                disagrees.append(s)
                            continue
                        if diffed < 0:
                            if g != goal.goal:
                                disagrees.append(s)
                            continue
                    except TypeError:
                        diffed = None
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
    what_tiles_symbol: list[tuple[Symbol, Symbol]],
    job: "Job",
    keep_symbols: list[Symbol] = (),
    max_fused_loops_per_rank_check_groups: list[list[Symbol]] = (),
    max_fused_loops_per_rank_variable: Number = float("inf"),
):
    objectives = [copy.deepcopy(o) for o in objectives]

    import time

    objectives = objectives.copy()

    symbols_enumerated: list[Symbol] = []
    choices_enumerated: np.ndarray = None

    symbols_remaining = list(symbols)

    objectives_finished_valid_check = []

    imperfect = False

    paretoed_by = []

    prev_time, start_time = time.time(), time.time()
    times = {}

    def time_end(s):
        nonlocal prev_time
        cur_time = time.time()
        times.setdefault(s, 0)
        times[s] += cur_time - prev_time
        prev_time = cur_time

    log = []

    def log_message(message: str, *args: str):
        log.append(f"{time.time() - prev_time:.2f}s: {message} {' '.join(args)}")
        # print(f"{time.time() - prev_time:.2f}s: {message} {' '.join(args)}")
        time_end(message)

    log_message("init")

    def eval_objective(
        formula: sympy.Expr | Objective,
        choices: np.ndarray,
        minimize_formula: sympy.Expr = None,
    ):
        if isinstance(formula, Objective):
            formula = formula.formula
        if formula in symbols_enumerated:
            return choices[:, symbols_enumerated.index(formula)]

        padded_choices = get_padded_choices(
            symbols_enumerated=symbols_enumerated,
            symbols_non_enumerated_set=symbols_non_enumerated_set,
            choices_enumerated=choices_enumerated,
            what_tiles_symbol=what_tiles_symbol,
            minimize_formula=minimize_formula,
        )
        return util._lambdify_type_check(symbols, formula)(
            **{str(k): v for k, v in padded_choices.items()},
        )

    while symbols_remaining:
        # ==============================================================================
        # Enumerate choices for a new symbol
        # ==============================================================================
        symbol = symbols_remaining.pop()

        choices = []
        tiled_by = get_tiled_by(symbol, what_tiles_symbol)
        if isinstance(tiled_by, Symbol):
            assert tiled_by in symbols_enumerated
            i = symbols_enumerated.index(tiled_by)
            partitions = {
                v: choices_enumerated[np.where(choices_enumerated[:, i] == v)]
                for v in np.unique(choices_enumerated[:, i])
            }
        else:
            partitions = {tiled_by: choices_enumerated}

        for v, partition in partitions.items():
            remaining_factor = math.ceil(get_max_size(symbol, what_tiles_symbol) / v)
            shapes = (
                np.array(list(get_possible_factor_sizes(remaining_factor, imperfect)))
                * v
            )
            # print(f'\tFor {s} with value {v}, the possible shapes are {shapes}')
            choices.append(append_vector(partition, shapes))

        if not partitions:
            return np.array([]).reshape(-1, len(symbols))

        prev_size = choices_enumerated.shape[0] if choices_enumerated is not None else 1
        choices_enumerated = np.concatenate(choices, axis=0)
        job.n_total_pmappings *= choices_enumerated.shape[0] / prev_size
        symbols_enumerated.append(symbol)
        log_message("enumerate", f"{symbol}", f"size={choices_enumerated.shape[0]}")

        # ==============================================================================
        # Max fused loops per rank check
        # ==============================================================================

        prev_size = choices_enumerated.shape[0]
        choices_enumerated = check_max_fused_loops_per_rank(
            symbols,
            symbols_enumerated,
            choices_enumerated,
            max_fused_loops_per_rank_check_groups,
            max_fused_loops_per_rank_variable,
        )
        job.log_porp_pmappings_kept(
            f"max_fused_loops_per_rank_variable",
            choices_enumerated.shape[0] / prev_size,
        )
        log_message(
            "max_fused_loops_per_rank_variable", f"size={choices_enumerated.shape[0]}"
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

        # If we're a symbol and an outer loop depends on us, then we need to track this
        # loop. Minimize it if we're imperfect (giving the outer the most choices
        # possible), or diff if we're perfect (since perfect constrains choices so we
        # can't just min).
        for s in symbols_enumerated:
            tiles = get_tiles(s, what_tiles_symbol)
            if isinstance(tiles, Symbol) and tiles not in symbols_enumerated:
                update_symbol2goal(s, Goal("min" if imperfect else "diff"))

        # If we need to keep this symbol, must preserve all choices for it
        for s in set(symbols_enumerated) & set(keep_symbols):
            update_symbol2goal(s, Goal("diff"))

        # For each objective, check for validity, and use it to mark whether we should
        # minimize or maximize each symbol.
        symbols_non_enumerated_set = set(symbols) - set(symbols_enumerated)
        sym_enumerated_set = set(symbols_enumerated)

        # ==============================================================================
        # Create functions to Pareto using objectives
        # ==============================================================================
        for objective in list(objectives):
            goals = partition_formula(objective.formula, sym_enumerated_set)

            # ==========================================================================
            # If there's a max value, then check for validity
            # ==========================================================================
            if objective.max_value is not None:
                try:
                    # minimize_for_objective may raise a TypeError if there's unknown symbols
                    result = eval_objective(
                        objective.formula,
                        choices_enumerated,
                        minimize_formula=objective.formula,
                    )
                    valid = result <= objective.max_value
                    if not isinstance(valid, np.ndarray):
                        valid = (
                            np.zeros(choices_enumerated.shape[0], dtype=bool) + valid
                        )
                    if objective.min_value is not None and valid.sum():
                        past_min = valid & (result >= objective.min_value)
                        if past_min.sum() == 0:
                            best = (result[valid]).max()
                            past_min = valid & (result == best)
                        valid = valid & past_min

                    porp = sum(valid) / max(1, choices_enumerated.shape[0])
                    job.log_porp_pmappings_kept(
                        f"{objective.name}",
                        sum(valid) / max(1, choices_enumerated.shape[0]),
                    )
                    choices_enumerated = choices_enumerated[valid]
                    log_message(f"Valid check", f"{objective.name}", f"porp={porp:.2%}")
                    if objective.formula.free_symbols.issubset(sym_enumerated_set):
                        objective.max_value = None  # We don't care anymore
                        if objective.only_care_if_valid:
                            objectives.remove(objective)
                            log_message(
                                f"Removed {objective.name} because it is always valid"
                            )
                            goals.clear()
                except TypeError:
                    pass

            log_message(f"formula", f"{objective.formula}", f"{goals}")

            for symbol, goal in goals.items():
                update_symbol2goal(symbol, goal)

        job.n_evaluated_pmappings += choices_enumerated.shape[0]
        if not choices_enumerated.shape[0]:
            return np.array([]).reshape(-1, len(symbols))

        symbol2goal_before = dict(symbol2goal)

        symbol2goal = coalesce_symbols(
            symbols=symbols,
            symbols_enumerated=symbols_enumerated,
            symbol2goal=symbol2goal,
            update_symbol2goal=update_symbol2goal,
            log_message=log_message,
        )

        log_message("coalesce symbols", f"{symbol2goal}")

        paretoed_by_key = fzs((f, g.goal) for f, g in symbol2goal.items())
        if any(p.issubset(paretoed_by_key) for p in paretoed_by):
            continue

        objective_values = {}
        for formula, goal in list(symbol2goal.items()):
            objective_values[formula] = eval_objective(formula, choices_enumerated)
            symbol2goal[formula] = goal
            log_message("eval", f"{goal.goal}", f"{formula}")

        if not objective_values:
            # Objective values don't depend on tile shapes
            choices_enumerated = choices_enumerated[:1, :]
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
            keep = paretoset.paretoset(to_pareto, pareto_goals)
            prev_size = choices_enumerated.shape[0]
            choices_enumerated = choices_enumerated[keep]
            paretoed_by.append(paretoed_by_key)
            job.log_porp_pmappings_kept(
                f"Pareto", sum(keep) / choices_enumerated.shape[0]
            )
            log_message("pareto", f"size={choices_enumerated.shape[0]}")

    total_time = sum(times.values())

    t = time.time() - start_time
    if t > 60:
        a = [
            f"Total time: {t:.2f}s",
            f"Pmapping: {job.mapping.compact_str()}",
        ]
        print("\n\t" + f"\n\t".join(a + log))

    # Rearrange in tile shape order
    return choices_enumerated[:, [symbols_enumerated.index(s) for s in symbols]]


def makesymbol(name: str):
    # TODO: Do the solve() calls work with integer=True?
    return Symbol(name, positive=True, integer=True)


def make_what_tiles_symbol(
    pmapping: Mapping, shape: dict[str, int]
) -> list[tuple[Symbol | int, Symbol | int]]:
    what_tiles_symbol: list[tuple[Symbol | int, Symbol | int]] = []
    last_seen_loop_per_rank_var: dict[str, Symbol | int] = dict(shape)
    for node in pmapping.nodes:
        if not isinstance(node, Loop):
            continue
        prev = last_seen_loop_per_rank_var.get(node.rank_variable, None)
        # If we're a symbol and we've seen an outer loop with the same rank variable,
        # then we tile that one.
        if prev is not None:
            what_tiles_symbol.append((prev, node.stride))
        last_seen_loop_per_rank_var[node.rank_variable] = node.stride

    for r, s in last_seen_loop_per_rank_var.items():
        if isinstance(s, Symbol):
            what_tiles_symbol.append((s, 1))
    return what_tiles_symbol


def make_keep_symbols(pmapping: Mapping) -> set[Symbol]:
    keep_symbols = set()
    for node in pmapping.nodes:
        if isinstance(node, Loop) and node._fused:
            if isinstance(node.initial_tile_shape, Symbol):
                keep_symbols.add(node.initial_tile_shape)
            if isinstance(node.stride, Symbol):
                keep_symbols.add(node.stride)
    return keep_symbols


def get_rank_var_to_fused_loops(
    pmapping: Mapping, shape: dict[str, int]
) -> dict[str, list[Symbol]]:
    rank_var_to_fused_loops: dict[str, list[Symbol]] = {}
    for node in [n for n in pmapping.nodes if isinstance(n, Loop) and n._fused]:
        rank_var_to_fused_loops.setdefault(node.rank_variable, []).append(node.stride)

    # Max fused loops per rank
    for k, v in rank_var_to_fused_loops.items():
        v.insert(0, shape[k])

    return rank_var_to_fused_loops


def _explore_tile_shapes_new(job: "Job"):
    # We're going to convert the job into a list of symbols and objectives
    pmapping = job.mapping
    constraints = job.constraints
    constraints.set_loop_indices(pmapping.nodes)
    set_last_tile_shape_to_one(pmapping)
    symbols, symbolic_df, per_memory_usage_df, utilization_df = run_model(job)
    shape = get_rank_variable_bounds(job.spec.workload, pmapping.nodes[-1].einsum)
    what_tiles_symbol = make_what_tiles_symbol(pmapping, shape)
    keep_symbols = make_keep_symbols(pmapping)
    rank_var_to_fused_loops = get_rank_var_to_fused_loops(pmapping, shape)

    objectives = []
    for k, v in {**per_memory_usage_df, **utilization_df}.items():
        # If we only track for pmappings, we only care if it's valid. If we track for
        # all, we care about the value too.

        only_care_if_valid = False
        if k in job.memories_track_pmappings_only:
            only_care_if_valid = True

        # TODO: Update check to see if we may be sharing utilization with other
        # pmappings in parallel/pipeline.
        if k in utilization_df:
            only_care_if_valid = True

        min_value = None
        if k in utilization_df:
            component_name, name = k.split("<SEP>")[1:]
            if (component_name, name) in job.constraints.min_utilization_constraints:
                min_value = job.constraints.min_utilization_constraints[
                    (component_name, name)
                ].min_utilization

        objectives.append(
            Objective(
                name=k,
                formula=v,
                symbols=symbols,
                only_care_if_valid=only_care_if_valid,
                max_value=1,
                min_value=min_value,
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
            if node.stride in symbols:
                rank2symbols.setdefault(node.rank_variable, []).append(node.stride)

    choices_enumerated = get_tile_shape_choices(
        objectives=objectives,
        symbols=symbols,
        what_tiles_symbol=what_tiles_symbol,
        job=job,
        keep_symbols=keep_symbols,
        max_fused_loops_per_rank_check_groups=list(rank_var_to_fused_loops.values()),
        max_fused_loops_per_rank_variable=job.spec.mapper.ffm.max_fused_loops_per_rank_variable,
    )

    try:
        compiled_df = compile_dict(symbols, symbolic_df)
        compiled_per_memory_usage_df = compile_dict(symbols, per_memory_usage_df)
        compiled_utilization_df = compile_dict(symbols, utilization_df)
    except Exception as e:
        print("Compilation failed for this mapping:")
        for node in pmapping.nodes:
            if hasattr(node, "compact_str"):
                print(node.compact_str())
        print(symbolic_df)
        raise RuntimeError("Compilation failed") from e

    df = {}
    for i, symbol in enumerate(symbols):
        df[symbol.name] = choices_enumerated[:, i]
    for key in compiled_df:
        df[key] = compiled_df[key](*choices_enumerated.T)

    for n in job.mapping.nodes:
        if not isinstance(n, Loop) or not n._fused:
            continue
        stride = n.tile_pattern.stride
        n_iterations = str(n.tile_pattern.calculated_n_iterations)
        if n_iterations in df:
            continue
        outer = get_tiles(stride, what_tiles_symbol)
        a = df[outer.name] if isinstance(outer, Symbol) else outer
        b = df[stride.name] if isinstance(stride, Symbol) else stride
        df[n_iterations] = np.round(a / b)

    df = pd.DataFrame(df, columns=df.keys())
    assert not df.isna().any().any()
    job.n_valid_pmappings = job.n_total_pmappings * prod(
        job.pmapping_keep_rates.values()
    )
    return df


def explore_tile_shapes(job: "Job"):
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
        return _explore_tile_shapes(job)
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


class GivenShape(int):
    def __str__(self):
        return f"GivenShape({super().__repr__()})"

    def __repr__(self):
        return f"GivenShape({super().__repr__()})"


@dataclass(repr=True)
class NumLoops:
    n_loops: int = 0
    initial_delta_choices: dict[int, set[int]] = field(default_factory=dict)


class TilingSegment:
    def __init__(self, full_shape, max_fused_loops: int):
        self.data: list[GivenShape | NumLoops] = [GivenShape(full_shape), NumLoops(0)]
        self.indices: list[int] = []
        self.is_symbol: list[bool] = []
        self.max_fused_loops: int = max_fused_loops
        self.is_fused: list[bool] = []

    def add_symbol(self, loop_idx: int, fused: bool):
        self.data[-1].n_loops += 1
        self.indices.append(loop_idx)
        self.is_symbol.append(True)
        self.is_fused.append(fused)

    def add_pattern(self, loop_idx: int, initial_delta_choices, fused: bool):
        last_data = self.data[-1]
        last_data.initial_delta_choices[last_data.n_loops] = initial_delta_choices
        last_data.n_loops += 1
        self.indices.append(loop_idx)
        self.indices.append(loop_idx + 1)
        self.is_symbol.append(True)
        self.is_symbol.append(True)
        self.is_fused.append(fused)

    def add_tile_shape(self, shape, loop_idx: int, fused: bool):
        self.data.append(GivenShape(shape))
        self.data.append(NumLoops(1))
        self.data.append(GivenShape(shape))
        self.data.append(NumLoops(0))
        self.indices.append(loop_idx)
        self.is_symbol.append(False)
        self.is_fused.append(fused)

    def finish(self):
        if isinstance(self.data[-1], NumLoops) and self.data[-1].n_loops == 0:
            self.data.pop()
        assert self.data[-1] == GivenShape(1)

    def iterate_segments(self):
        """Returns iterator over tuples (n_loops, max_shape, min_shape)."""
        for i in range(0, len(self.data) - 1, 2):
            if self.data[i + 1].n_loops == 0:
                continue
            max_shape = self.data[i]
            n_loops = self.data[i + 1].n_loops
            initial_delta_choices = self.data[i + 1].initial_delta_choices
            min_shape = self.data[i + 2]
            yield (n_loops, initial_delta_choices, max_shape, min_shape)


def _explore_tile_shapes_old(job: "Job"):
    pmapping = job.mapping
    constraints = job.constraints
    spec = job.spec

    constraints.set_loop_indices(pmapping.nodes)

    set_last_tile_shape_to_one(pmapping)

    symbols, symbolic_df, per_memory_usage_df, utilization_df = run_model(job)

    try:
        compiled_df = compile_dict(symbols, symbolic_df)
        compiled_per_memory_usage_df = compile_dict(symbols, per_memory_usage_df)
        compiled_utilization_df = compile_dict(symbols, utilization_df)
    except Exception as e:
        print("Compilation failed for this mapping:")
        for node in pmapping.nodes:
            if hasattr(node, "compact_str"):
                print(node.compact_str())
        print(symbolic_df)
        raise RuntimeError("Compilation failed") from e

    tile_shapes, is_symbol = generate_tile_shapes(
        pmapping,
        constraints,
        compiled_per_memory_usage_df,
        compiled_utilization_df,
        spec,
        job,
    )

    df = {}
    for i in range(tile_shapes.shape[1]):
        df[f"tile_shape{i}"] = tile_shapes[:, i]

    tile_shapes = tile_shapes[:, is_symbol]
    tile_shapes = [tile_shapes[:, i] for i in range(tile_shapes.shape[1])]

    for key in compiled_df:
        df[key] = compiled_df[key](*tile_shapes)

    df = pd.DataFrame(df, columns=df.keys())
    assert not df.isna().any().any()
    return df


def generate_tile_shapes(
    pmapping: list[MappingNode],
    constraints: MappingConstraints,
    usage_df: dict[str, Callable],
    utilization_df: dict[str, Callable],
    spec: Spec,
    job: Job,
):
    pmapping = pmapping.nodes
    workload = spec.workload

    initial_delta_choices = get_initial_delta_choices(pmapping[-1].einsum, workload)

    shape = get_rank_variable_bounds(workload, pmapping[-1].einsum)

    rank_var_to_tiling_segments = collect_tiling_segments(
        pmapping, shape, spec, initial_delta_choices
    )

    def update_mask(prev_mask, new_mask, cause: str, track_masked: bool = True):
        prev_sum = prev_mask.sum()
        new_mask = new_mask & prev_mask
        new_sum = new_mask.sum()
        if track_masked and prev_sum != 0 and prev_sum != new_sum:
            # print(f"Mask {cause}: {new_sum / prev_sum}")
            job.log_porp_pmappings_kept(cause, new_sum / prev_sum, out_of=prev_sum)
        return new_mask

    def listify_choices(choices):
        return list(choices[:, i] for i in range(choices.shape[1]))

    def get_corrected_choices(
        combined_choices, indices, is_symbols, other_rank_var_and_choices
    ):
        indices = indices.copy()
        is_symbols = is_symbols.copy()

        # Insert ones
        combined_choices_with_ones = combined_choices
        combined_choices_with_largest = combined_choices
        complete_indices = indices.copy()

        for (
            other_ranks,
            other_indices,
            other_is_symbol,
            other_choices,
        ) in other_rank_var_and_choices:
            indices.extend(other_indices)
            is_symbols.extend(other_is_symbol)
            complete_indices.extend(
                [i for i, s in zip(other_indices, other_is_symbol) if not s]
            )

            other_n_loops = len(other_indices)
            combined_choices_with_ones = np.concatenate(
                (
                    combined_choices_with_ones,
                    np.ones((combined_choices.shape[0], other_n_loops), dtype=np.int64),
                ),
                axis=1,
            )

            largest_other_choices = np.max(
                other_choices
            )  # np.max(other_choices, axis=0, keepdims=True)
            combined_choices_with_largest = np.concatenate(
                (
                    combined_choices_with_largest,
                    np.ones((combined_choices.shape[0], other_n_loops), dtype=np.int64)
                    * largest_other_choices,
                ),
                axis=1,
            )

        sorted_indices = np.asarray(invert_indices(indices))
        is_symbols = np.asarray(is_symbols)[sorted_indices]
        corrected_choices = combined_choices_with_ones[:, sorted_indices]
        corrected_choices = corrected_choices[:, is_symbols]
        corrected_choices_with_largest = combined_choices_with_largest[
            :, sorted_indices
        ]
        corrected_choices_with_largest = corrected_choices_with_largest[:, is_symbols]
        return (
            corrected_choices,
            corrected_choices_with_largest,
            complete_indices,
            indices,
            is_symbols,
        )

    def check_valid_tile_shape(
        combined_choices,
        is_symbols,
        other_rank_var_and_choices,
        indices,
        ranks,
        shape,
        track_masked=True,
    ):
        # print(f'\t Combined rank {rank_a} and {rank_b}: {choices_a.shape[0]} x {choices_b.shape[0]} -> {combined_choices.shape[0]}')
        if combined_choices.shape[0] == 0:
            return combined_choices

        # Insert ones
        (
            corrected_choices,
            corrected_choices_with_largest,
            complete_indices,
            indices,
            is_symbols,
        ) = get_corrected_choices(
            combined_choices, indices, is_symbols, other_rank_var_and_choices
        )
        corrected_choices_2 = corrected_choices_with_largest

        # TODO: there may be a more efficient order
        mask = np.ones(corrected_choices.shape[0], dtype=np.bool)
        mask = update_mask(
            mask,
            constraints.check_tile_shape_constraints(
                corrected_choices, complete_indices
            ),
            "tile shape constraints",
            track_masked=track_masked,
        )

        listified_choices = listify_choices(corrected_choices)
        listified_choices_2 = listify_choices(corrected_choices_2)

        # Check if capacity is overused
        for memory, usage_model in usage_df.items():
            usage = usage_model(*listified_choices)
            mask = update_mask(
                mask, usage <= 1.0, f"{memory} usage", track_masked=track_masked
            )

        # Calculate utilization
        utilization = {}
        utilization2 = {}
        for component_dim, utilization_model in utilization_df.items():
            _, component, dim = component_dim.split("<SEP>")
            utilization[(component, dim)] = utilization_model(*listified_choices)
            utilization2[(component, dim)] = utilization_model(*listified_choices_2)
            util = np.minimum(
                utilization[(component, dim)], utilization2[(component, dim)]
            )
            mask = update_mask(
                mask,
                util <= 1.0,
                f"{component} {dim} utilization",
                track_masked=track_masked,
            )
            util = util * mask
            mask = update_mask(
                mask,
                constraints.check_min_utilization_constraints(
                    component, dim, util * mask, complete_indices
                ),
                f"{component} {dim} minimum utilization",
                track_masked=track_masked,
            )

        good_choices = combined_choices[mask, :]

        # TODO: Bring back loop limit
        # # Check that we're less than the n_loops limit
        # n_loops = np.zeros(good_choices.shape[0], dtype=np.int64)
        # for rank in ranks:
        #     tiling_segments = rank_var_to_tiling_segments[rank]

        #     # Previous size = the full rank variable size. Initialize a vector with that size
        #     cur_size = np.zeros(good_choices.shape[0], dtype=np.int64) + shape[rank]

        #     for i in [indices.index(i) for i in tiling_segments.indices]:
        #         n_loops += cur_size != good_choices[:,i]
        #         cur_size = good_choices[:,i]
        # max_loops = min(
        #     spec.mapper.ffm.max_loops,
        #     spec.mapper.ffm.max_loops_minus_ranks + len(ranks)
        # )
        # mask = np.ones(good_choices.shape[0], dtype=np.bool)
        # mask = update_mask(mask, n_loops <= max_loops, "max loops", track_masked=track_masked)
        # good_choices = good_choices[mask,:]

        return good_choices

    rank_var_and_choices: list[
        tuple[frozenset[str], list[int], list[bool], np.array]
    ] = []
    rv2choices = {
        rank_var: make_shapes_for_one_rank(tiling_segments)
        for rank_var, tiling_segments in rank_var_to_tiling_segments.items()
    }

    nominal_n_mappings = np.prod([x.shape[0] for x in rv2choices.values()])

    for rank_var, tiling_segments in rank_var_to_tiling_segments.items():
        choices = rv2choices.pop(rank_var)

        good_choices = choices
        is_symbols = tiling_segments.is_symbol.copy()

        # Insert ones
        indices = tiling_segments.indices.copy()
        complete_indices = indices.copy()
        is_symbols = tiling_segments.is_symbol.copy()
        choices2 = choices.copy()
        for other_rank, other_segments in rank_var_to_tiling_segments.items():
            if rank_var == other_rank:
                continue
            indices.extend(other_segments.indices)
            is_symbols.extend(other_segments.is_symbol)
            complete_indices.extend(
                [
                    i
                    for i, s in zip(other_segments.indices, other_segments.is_symbol)
                    if not s
                ]
            )
            choices = np.concatenate(
                (
                    choices,
                    np.repeat(
                        make_ones_for_one_rank(other_segments),
                        repeats=choices.shape[0],
                        axis=0,
                    ),
                ),
                axis=1,
            )
            choices2 = np.concatenate(
                (
                    choices2,
                    np.repeat(
                        make_biggest_for_one_rank(other_segments),
                        repeats=choices2.shape[0],
                        axis=0,
                    ),
                ),
                axis=1,
            )

        job.set_total_pmappings(choices.shape[0])

        # # TODO: there may be a more efficient order
        corrected_indices = np.asarray(invert_indices(indices))
        corrected_choices = choices[:, corrected_indices]
        corrected_choices2 = choices2[:, corrected_indices]

        mask = constraints.check_tile_shape_constraints(corrected_choices, indices)
        # if mask.sum() == 0:
        #     print(f'No valid tile shapes for {rank_var}')
        is_symbols = np.asarray(is_symbols)[corrected_indices]
        corrected_choices = corrected_choices[:, is_symbols]
        corrected_choices2 = corrected_choices2[:, is_symbols]
        listified_choices = listify_choices(corrected_choices)
        listified_choices2 = listify_choices(corrected_choices2)

        # Check if capacity is overused
        for memory, usage_model in usage_df.items():
            usage = usage_model(*listified_choices)
            mask = update_mask(mask, usage <= 1.0, f"{memory} usage")

        # if mask.sum() == 0:
        #     print(f'No valid memory usage for {rank_var}')

        utilization = {}
        utilization2 = {}
        for component_dim, utilization_model in utilization_df.items():
            _, component, dim = component_dim.split("<SEP>")
            utilization[(component, dim)] = utilization_model(*listified_choices)
            utilization2[(component, dim)] = utilization_model(*listified_choices2)
            util = np.minimum(
                utilization[(component, dim)], utilization2[(component, dim)]
            )
            mask = update_mask(mask, util <= 1.0, f"{component} {dim} utilization")
            util = util * mask
            mask = update_mask(
                mask,
                constraints.check_min_utilization_constraints(
                    component, dim, util, complete_indices
                ),
                f"{component} {dim} minimum utilization",
            )

        good_choices = choices[mask, :]

        job.log_message(
            f"Rank {rank_var} has {good_choices.shape[0]} valid tile shape choices."
        )

        rank_var_and_choices.append(
            (
                frozenset((rank_var,)),
                tiling_segments.indices.copy(),
                tiling_segments.is_symbol.copy(),
                good_choices,
            )
        )

    prev_rank_var_and_choices = rank_var_and_choices
    rank2prev_rank_var_and_choices = {
        next(iter(x[0])): x for x in prev_rank_var_and_choices
    }
    rank_var_and_choices = []
    for i, rank_var_and_choices_a in enumerate(prev_rank_var_and_choices):
        rank_a, index_a, is_symbol_a, choices_a = rank_var_and_choices_a
        for j, index in enumerate(index_a):
            rank_var_and_choices.append(
                (
                    rank_a,
                    [index],
                    [is_symbol_a[j]],
                    np.unique(choices_a[:, j]).reshape(-1, 1),
                )
            )

    def get_combined_choices(
        rank_var_and_choices_a,
        rank_var_and_choices_b,
        other_rank_var_and_choices,
        shape,
        tile_shape=512,
        track_masked=True,
    ):
        rank_a, index_a, is_symbol_a, choices_a = rank_var_and_choices_a
        rank_b, index_b, is_symbol_b, choices_b = rank_var_and_choices_b

        new_rank = rank_a | rank_b
        new_index = index_a + index_b
        new_is_symbol = is_symbol_a + is_symbol_b
        if any(
            x.shape[0] == 0
            for x in [choices_a, choices_b]
            + [x[-1] for x in other_rank_var_and_choices]
        ):
            return (
                new_rank,
                new_index,
                new_is_symbol,
                np.empty((0, len(new_index)), dtype=np.int64),
            )

        combined_choices = []

        # Now bin the choices of a by the choices for loops in rank_b
        bin_a = {}
        prev = rank2prev_rank_var_and_choices[next(iter(rank_b))]
        _, prev_index, _, prev_choices = prev

        a_index_prev = [i for i in index_a if i in prev_index]
        b_index_prev = [i for i in index_b if i in prev_index]
        prev_options = set(
            [
                tuple(row)
                for row in prev_choices[
                    :, [prev_index.index(i) for i in a_index_prev + b_index_prev]
                ]
            ]
        )

        b_loops = choices_a[:, [index_a.index(i) for i in a_index_prev]]
        for i in range(choices_a.shape[0]):
            bin_a.setdefault(tuple(b_loops[i]), []).append(i)

        b_option_to_valid_a_indices = {}
        for b_option in choices_b.flatten():
            b_option_to_valid_a_indices[b_option] = []
            for k, v in bin_a.items():
                if tuple(k) + (b_option,) in prev_options:
                    b_option_to_valid_a_indices[b_option].extend(v)

        if track_masked:
            n = sum(len(v) for v in b_option_to_valid_a_indices.values())
            job.log_message(
                f"Combining {choices_a.shape[0]} choices for loops {index_a}, {choices_b.shape[0]} choices for loops {index_b}. N valid combinations: {n}"
            )
            job.set_total_pmappings(n)

        assert choices_b.shape[1] == 1
        for b_option, valid_a_indices in b_option_to_valid_a_indices.items():
            cur_choices_a = choices_a[valid_a_indices]

            choices = np.concatenate(
                (
                    cur_choices_a,
                    np.repeat(
                        np.array([b_option]), repeats=cur_choices_a.shape[0], axis=0
                    ).reshape(-1, 1),
                ),
                axis=1,
            )

            good_choices = check_valid_tile_shape(
                choices,
                is_symbol_a + is_symbol_b,
                other_rank_var_and_choices,
                index_a + index_b,
                rank_a | rank_b,
                shape,
                track_masked=track_masked,
            )
            combined_choices.append(good_choices)

        good_choices = np.concatenate(combined_choices, axis=0)
        good_choices = check_valid_tile_shape(
            good_choices,
            is_symbol_a + is_symbol_b,
            other_rank_var_and_choices,
            index_a + index_b,
            rank_a | rank_b,
            shape,
            track_masked=track_masked,
        )
        return (new_rank, new_index, new_is_symbol, good_choices)

    # def _greedily_maximize_reuse(rank_var_and_choices_a, other_rank_var_and_choices):
    #     rank_a, index_a, is_symbol_a, choices_a = rank_var_and_choices_a

    #     # Check if completed loops are:
    #     # - Above any storage node
    #     # - Directly beneath beneath the last storage node for a memory

    #     completed_loops = index_a
    #     outermost_completed_loop = min(completed_loops)
    #     n = -1
    #     nodes = job.mapping.nodes
    #     for i, node in enumerate(job.mapping.nodes):
    #         if not isinstance(node, Loop):
    #             continue
    #         n += 1

    #         if n != outermost_completed_loop:
    #             continue

    #         if i == 0:  # Outermost loop!
    #             break

    #         next_tensor_holders = [
    #             n for n in nodes[i:] if isinstance(n, mapping.Reservation)
    #         ]
    #         next_names = set([n.resource for n in next_tensor_holders])

    #         if (
    #             isinstance(nodes[i - 1], mapping.Reservation)
    #             and next_names
    #             and nodes[i - 1].resource not in next_names
    #         ):
    #             break
    #         else:
    #             return rank_var_and_choices_a
    #     else:
    #         raise RuntimeError("BUG")

    #     (
    #         corrected_choices,
    #         corrected_choices_with_largest,
    #         complete_indices,
    #         indices,
    #         is_symbols,
    #     ) = get_corrected_choices(
    #         choices_a, index_a, is_symbol_a, other_rank_var_and_choices
    #     )
    #     corrected_choices_2 = corrected_choices.copy()

    #     listified_choices = sort_choices(corrected_choices, indices, is_symbols)
    #     listified_choices_2 = sort_choices(corrected_choices_2, indices, is_symbols)

    #     # Get the spatial utilizations
    #     df = {}
    #     for i in range(choices_a.shape[1]):
    #         df[i] = choices_a[:, i]

    #     for component_dim, utilization_model in utilization_df.items():
    #         _, component, dim = component_dim.split("<SEP>")
    #         utilization[(component, dim)] = utilization_model(*listified_choices)
    #         utilization2[(component, dim)] = utilization_model(*listified_choices_2)
    #         util = np.minimum(
    #             utilization[(component, dim)], utilization2[(component, dim)]
    #         )
    #         df[(component, dim)] = util

    #     import paretoset

    #     return (
    #         rank_a,
    #         index_a,
    #         is_symbol_a,
    #         choices_a[
    #             paretoset.paretoset(
    #                 np.concatenate([x.reshape(-1, 1) for x in df.values()], axis=1),
    #                 sense=["max"] * len(df),
    #             ),
    #             :,
    #         ],
    #     )

    # First, combine spatial loops
    # loops = [x for x in job.mapping.nodes if isinstance(x, (Temporal, Spatial))]
    # while True:
    #     def get_spatial():
    #         for i, rank_var_and_choices_a in enumerate(rank_var_and_choices):
    #             rank_a, index_a, is_symbol_a, choices_a = rank_var_and_choices_a
    #             spatial_a = any(isinstance(loops[i], Spatial) for i in index_a)
    #             if spatial_a:
    #                 return rank_var_and_choices.pop(i)
    #         return None

    #     spatial_a = get_spatial()
    #     spatial_b = get_spatial()
    #     if spatial_b is None:
    #         if spatial_a is not None:
    #             rank_var_and_choices.append(spatial_a)
    #         break
    #     rank_var_and_choices.insert(0, get_combined_choices(spatial_a, spatial_b, rank_var_and_choices, shape))

    def get_best_reduction(
        a, rank_var_and_choices, shape, lookahead=1, _recursed=False
    ):
        if lookahead == 0 or not rank_var_and_choices:
            return 1 if _recursed else 0

        best_reduction, best_index = 2, 0
        i = 0
        while i < len(rank_var_and_choices):
            b = rank_var_and_choices[i]
            choices_b = b[-1]
            if choices_b.shape[0] == 1:
                best_index = i
                break

            total_shape = max((choices_a.shape[0] * choices_b.shape[0]), 1)
            if total_shape > 10000:
                continue

            other_rank_var_and_choices = [
                x for k, x in enumerate(rank_var_and_choices) if k != i
            ]
            combined = get_combined_choices(
                a, b, other_rank_var_and_choices, shape, track_masked=False
            )
            reduction = (
                combined[-1].shape[0]
                / total_shape
                * get_best_reduction(
                    combined,
                    other_rank_var_and_choices,
                    shape,
                    lookahead - 1,
                    _recursed=True,
                )
            )

            if reduction < best_reduction:
                best_reduction = reduction
                best_index = i

            i += 1

        return best_reduction if _recursed else best_index

    if True:  # not spec.mapper.ffm._greedily_maximize_reuse:
        # Start combining from the loop with the fewest choices
        _, fewest_index = min(
            ((x[-1].shape[0], i) for i, x in enumerate(rank_var_and_choices))
        )
        rank_var_and_choices.insert(0, rank_var_and_choices.pop(fewest_index))

        # # Then, combine the loops that lead to the most reduction in the number of choices
        while len(rank_var_and_choices) > 1:
            #     best_reduction = 2
            #     best_index = None
            a = rank_var_and_choices.pop(0)

            #     choices_a = a[-1]
            #     for i, b in enumerate(rank_var_and_choices):
            #         choices_b = b[-1]
            #         if choices_b.shape[0] == 1:
            #             best_index = i
            #             break

            #         # If we're going to have too many choices, skip
            #         total_shape = max((choices_a.shape[0] * choices_b.shape[0]), 1)
            #         if total_shape > 100000:
            #             continue

            #         other_rank_var_and_choices = [x for k, x in enumerate(rank_var_and_choices) if k != i]
            #         combined = get_combined_choices(a, b, other_rank_var_and_choices, `shape`, track_masked=False)
            #         choices_combined = combined[-1]
            #         reduction = choices_combined.shape[0] / total_shape

            #         if reduction < best_reduction:
            #             best_reduction = reduction
            #             best_index = i

            #     if best_index is None:
            #         rank_var_and_choices.insert(0, a)
            #         break
            best_index = get_best_reduction(a, rank_var_and_choices, shape)
            b = rank_var_and_choices.pop(best_index)
            # print(f"{a[-1].shape, b[-1].shape, get_combined_choices(a, b, rank_var_and_choices, shape)[-1].shape}")
            rank_var_and_choices.insert(
                0, get_combined_choices(a, b, rank_var_and_choices, shape)
            )

        # If we still have loops to combine, just combine them all
        while len(rank_var_and_choices) > 1:
            a, b = rank_var_and_choices.pop(0), rank_var_and_choices.pop(0)
            rank_var_and_choices.insert(
                0, get_combined_choices(a, b, rank_var_and_choices, shape)
            )
    # else:
    #     assert (
    #         spec.mapper.ffm.force_memory_hierarchy_order
    #     ), "Maximizing memory usage requires force_memory_hierarchy_order to be set"
    #     # Start combining from the outside in
    #     while len(rank_var_and_choices) > 1:
    #         a, b = rank_var_and_choices.pop(-1), rank_var_and_choices.pop(-1)
    #         rank_var_and_choices.append(
    #             get_combined_choices(a, b, rank_var_and_choices, shape)
    #         )

    #         # If we've just finished processing all loops that affect a memory, then
    #         # save only the ones with the pareto-largest tile shapes and spatial utilization
    #         rank_var_and_choices.append(
    #             _greedily_maximize_reuse(
    #                 rank_var_and_choices.pop(-1), rank_var_and_choices
    #             )
    #         )

    combined_choices = rank_var_and_choices[0][-1]
    is_symbol = rank_var_and_choices[0][2]
    indices = rank_var_and_choices[0][1]
    ranks = rank_var_and_choices[0][0]
    other_rank_var_and_choices = (
        []
    )  # All rank variables have been combined, so no others remain
    good_choices = check_valid_tile_shape(
        combined_choices, is_symbol, other_rank_var_and_choices, indices, ranks, shape
    )

    _, inverted_indices, is_symbol, choices = rank_var_and_choices[0]
    is_symbol = np.asarray(is_symbol)

    # Invert indices
    indices = invert_indices(inverted_indices)

    job.log_message(f"Found {choices[:,indices].shape[0]} valid tile shapes")
    job.log_message(f"Nominal n mappings: {nominal_n_mappings}")
    job.log_message(f"Actual n mappings: {choices[:,indices].shape[0]}")
    job.log_message(f"Ratio: {choices[:,indices].shape[0] / nominal_n_mappings}")
    job.n_total_pmappings = nominal_n_mappings
    job.n_valid_pmappings = choices.shape[0]

    return choices[:, indices], is_symbol[indices]


def invert_indices(inverted_indices):
    return np.argsort(inverted_indices)


def collect_tiling_segments(
    pmapping,
    rank_shape: dict,
    spec: "Spec",
    initial_delta_choices: dict[str, set[int]] = {},
) -> dict[str, TilingSegment]:
    rank_var_to_tiling_segments = {}
    loop_idx = 0
    for node in pmapping:
        if isinstance(node, Temporal) or isinstance(node, Spatial):
            rank_var = node.rank_variable
            tile_shape = node.tile_shape
            tile_pattern = node.tile_pattern

            tiling_segment = rank_var_to_tiling_segments.setdefault(
                rank_var,
                TilingSegment(
                    rank_shape[rank_var],
                    spec.mapper.ffm.max_fused_loops_per_rank_variable,
                ),
            )

            if tile_shape == "symbol" or isinstance(tile_shape, sympy.Symbol):
                tiling_segment.add_symbol(loop_idx, node._fused)
            elif isinstance(tile_shape, int):
                tiling_segment.add_tile_shape(tile_shape, loop_idx, node._fused)
            elif isinstance(tile_pattern, TilePattern):
                stride = tile_pattern.stride
                initial_tile_shape = tile_pattern.initial_tile_shape
                pattern_shape = tile_pattern.tile_shape
                if pattern_shape is not None:
                    raise ValueError("Recomputation not yet supported")
                assert stride is not None and initial_tile_shape is not None
                if isinstance(stride, int):
                    tiling_segment.add_tile_shape(stride, loop_idx, node._fused)
                elif isinstance(stride, sympy.Symbol):
                    tiling_segment.add_pattern(
                        loop_idx, initial_delta_choices[rank_var], node._fused
                    )
                else:
                    raise RuntimeError("BUG")
            else:
                raise NotImplementedError(f"Unsupported tile shape {tile_shape}")

            loop_idx += 1

    for rank_var, tiling_segment in rank_var_to_tiling_segments.items():
        tiling_segment.finish()

    return rank_var_to_tiling_segments


def make_shapes_for_one_rank(tiling_segments: TilingSegment):
    all_tile_shapes = None
    total_loops = 0
    for (
        n_loops,
        initial_delta_choices,
        max_shape,
        min_shape,
    ) in tiling_segments.iterate_segments():
        total_loops += n_loops

        factors = integer_factorizations_to_n_parts(max_shape, n_loops + 1)
        factors = np.asarray(list(factors), dtype=np.int64)[:, :-1]
        tile_shape = max_shape // np.cumprod(factors, axis=1)
        tile_shape = tile_shape.astype(np.int64)
        tile_shape = tile_shape[np.all(tile_shape >= min_shape, axis=1), :]

        for i in sorted(initial_delta_choices, reverse=True):
            choices = (
                np.array(list(initial_delta_choices[i])).reshape(-1, 1).astype(np.int64)
            )
            tile_shape = np.concatenate(
                (
                    np.tile(tile_shape[:, : i + 1], (choices.shape[0], 1)),
                    np.repeat(choices, repeats=tile_shape.shape[0], axis=0),
                    np.tile(tile_shape[:, i + 1 :], (choices.shape[0], 1)),
                ),
                axis=1,
            )
            tile_shape[:, i + 1] += tile_shape[:, i]

            if i >= 1:
                mask_full_tile = tile_shape[:, i] == tile_shape[:, i - 1]
            else:
                mask_full_tile = tile_shape[:, i] == max_shape
            tile_shape[mask_full_tile, i + 1] = tile_shape[mask_full_tile, i]

            if i >= 1:
                mask_valid_initial = tile_shape[:, i + 1] <= tile_shape[:, i - 1]
            else:
                mask_valid_initial = tile_shape[:, i + 1] <= max_shape
            tile_shape = tile_shape[mask_valid_initial, :]

        if all_tile_shapes is None:
            all_tile_shapes = tile_shape
        else:
            all_tile_shapes_n_rows = all_tile_shapes.shape[0]
            all_tile_shapes = np.tile(all_tile_shapes, (tile_shape.shape[0], 1))
            tile_shape = np.repeat(tile_shape, repeats=all_tile_shapes_n_rows, axis=0)
            all_tile_shapes = np.concatenate((all_tile_shapes, tile_shape), axis=1)

    # Max fused loops check
    n_loops = np.zeros(all_tile_shapes.shape[0], dtype=np.int64)
    for i in range(len(tiling_segments.is_fused)):
        if tiling_segments.is_fused[i]:
            prev = all_tile_shapes[:, i - 1] if i > 0 else max_shape
            n_loops += all_tile_shapes[:, i] != prev
    all_tile_shapes = all_tile_shapes[n_loops <= tiling_segments.max_fused_loops, :]

    return all_tile_shapes


def make_ones_for_one_rank(tiling_segments: TilingSegment):
    total_cols = 0
    for (
        n_loops,
        initial_delta_choices,
        max_shape,
        min_shape,
    ) in tiling_segments.iterate_segments():
        total_cols += n_loops + len(initial_delta_choices.keys())
    return np.ones((1, total_cols))


def make_biggest_for_one_rank(tiling_segments: TilingSegment):
    total_cols = 0
    max_max_shape = 0
    for (
        n_loops,
        initial_delta_choices,
        max_shape,
        min_shape,
    ) in tiling_segments.iterate_segments():
        total_cols += n_loops + len(initial_delta_choices.keys())
        max_max_shape = max(max_max_shape, max_shape)
    return np.ones((1, total_cols)) * max_max_shape


def set_last_tile_shape_to_one(pmapping):
    pmapping = pmapping.nodes

    rank_var_to_last_node = {}
    for node in pmapping:
        if isinstance(node, Temporal) or isinstance(node, Spatial):
            rank_var_to_last_node[node.rank_variable] = node

    for last_node in rank_var_to_last_node.values():
        last_node.initial_tile_shape = None
        last_node.stride = 1
