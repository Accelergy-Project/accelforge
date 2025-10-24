from functools import lru_cache
from math import ceil, prod
import copy
import re
import resource
from typing import Callable, Optional, Union
from sympy import Expr, Symbol, lambdify
from fastfusion.accelerated_imports import np
from fastfusion.accelerated_imports import pd
import fastfusion.frontend.arch as arch
from fastfusion.frontend.workload._isl import get_rank_variable_bounds
from fastfusion.frontend.workload._symbolic import get_projection_expr
from fastfusion.frontend.workload import Einsum
from fastfusion.frontend.mapping import (
    Iteration,
    Mapping,
    Temporal,
    Spatial,
    TensorHolder,
)
from fastfusion.mapper.FFM._make_pmappings.mapper_one_einsum.mapper_job import Job
from fastfusion.mapper.FFM._pmapping_group.df_convention import stride2col, initial2col, iterations2col
from fastfusion.model.looptree.reuse.summarized.symbolic import (
    analyze_reuse_and_add_reservations_to_mapping,
)
from fastfusion.model.looptree.energy import compute_energy_from_actions, gather_actions
from fastfusion.model.looptree.latency import get_latency
from fastfusion.model.looptree.latency.memory import component_latency
from fastfusion.mapper.FFM._pmapping_group import (
    nameloop2col,
    tensor2col,
    firstlatency2col,
)
from fastfusion.frontend.mapper.metrics import Metrics
from fastfusion.util.util import fzs
import math
import sympy
import numpy as np
import paretoset
from numbers import Number

from fastfusion.mapper.FFM._make_pmappings.mapper_one_einsum.symbol_value_relations import (
    SymbolValueRelations,
)
from fastfusion.util.sympy.broadcast_max import Max


def run_model(
    job: Job,
) -> tuple[list[Symbol], dict[str, float], dict[str, float], dict[str, float]]:
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
    # overall_latency, comp_latency, mem_latency = get_latency(
    #     reuse, pmapping, workload, job.flattened_arch
    # )

    latency = component_latency(reuse, job.flattened_arch, pmapping, spec)
    try:
        overall_latency = Max(*latency.values()) if latency else 0
    except Exception as e:
        for k, v in latency.items():
            if not isinstance(v, (Number, sympy.Symbol, sympy.Expr)):
                raise ValueError(
                    f"Invalid type for latency: {k}: {type(v)} {str(v).strip()}"
                )
        
        raise ValueError(
            f"Error calculating latency for {job.einsum_name}. Could not calculate "
            f"a symbolic max of the following latencies:\n\t" + "\n\t".join(
                [f"{k}: {type(v)} {str(v).strip()}" for k, v in latency.items()]
            )
        )

    actions = gather_actions(reuse, None, use_name=True)
    energy = compute_energy_from_actions(actions, ert, overall_latency)

    fusable_tensors = workload.fusable_tensor_names
    tensor_to_backing = {}
    for node in pmapping.nodes:
        if isinstance(node, TensorHolder):
            for tensor in node.tensors:
                if tensor not in tensor_to_backing and tensor in fusable_tensors:
                    tensor_to_backing[tensor] = node.component

    total_occupancy = {}
    compute_unit = pmapping.nodes[-1].compute

    n_instances = workload.n_instances * workload.einsums[job.einsum_name].n_instances

    n_loop_options = set()
    for buffet, stats in reuse.buffet_stats.items():
        if buffet.level == compute_unit:
            continue

        occupancy = stats.max_occupancy

        if occupancy == 0:
            continue
        if stats.persistent:
            occupancy *= n_instances

        for tensor, backing in tensor_to_backing.items():
            if (is_copy_op or buffet.tensor == tensor) and buffet.level == backing:
                df[tensor2col(tensor)] = occupancy

        total_occupancy.setdefault(buffet.level, {}).setdefault(stats.n_loops_above, 0)
        total_occupancy[buffet.level][stats.n_loops_above] += occupancy
        n_loop_options.add(stats.n_loops_above)

    for memory, occupancies in total_occupancy.items():
        if memory not in job.memories_track_all:
            continue
        running_total = 0
        for n_loop in n_loop_options:
            if n_loop in occupancies:
                running_total += occupancies[n_loop]
                df[nameloop2col(memory, n_loop)] = running_total

    if metrics & Metrics.LATENCY:
        df[f"Total<SEP>latency"] = overall_latency * n_instances
        # df[f"latency<SEP>compute"] = comp_latency * n_instances
        # For first latency, we'll follow the convention of treating compute
        # as a component, similarly to memory (see below).
        for compute_level, stats in reuse.compute_stats.items():  # FIRST LATENCY
            for idx, max_first_latency in stats.max_first_latency.items():
                df[firstlatency2col(compute_level.level, idx)] = (
                    max_first_latency * n_instances
                )
        for component, latency in latency.items():
            df[f"latency<SEP>{component}"] = latency * n_instances

    if metrics & Metrics.ENERGY:
        df[f"Total<SEP>energy"] = sum(energy.values()) * n_instances
        for (component, action), energy in energy.items():
            df[f"energy<SEP>{component}<SEP>{action}"] = energy * n_instances

    # if metrics & Metrics.RESERVATIONS:
    #     for memory, occupancies in total_occupancy.items():
    #         df[f"reservations<SEP>{memory}"] = sum(occupancies.values())

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
        x = sympy.lambdify(symbols, value)
        # x._loops = [int(loop) for loop in re.findall(r'loop(\d+)_', key)]
        return x

    return {k: lambdify(symbols, v) for k, v in dictionary.items()}


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


class Objective:
    def __init__(
        self,
        name: str,
        formula: Expr | Number,
        max_value: float = None,
        symbols: list[str] = None,
        only_care_if_valid: bool = False,
        min_value: float = None,
    ):
        if isinstance(formula, Number):
            formula = sympy.Number(formula)
        self.name: str = name
        self.formula: Expr = formula.simplify()
        self._symbols: list[str] = symbols
        self.max_value: float = max_value
        self.only_care_if_valid: bool = only_care_if_valid
        if only_care_if_valid:
            assert self.max_value is not None
        self.min_value: float = min_value


def is_constant(f: Expr) -> bool:
    try:
        return f.is_constant()
    except ValueError:
        return all(is_constant(arg) for arg in f.args)


@lru_cache(maxsize=10000)
def _try_replace_single_term(t: Expr, symbols_enumerated: fzs[Symbol]):
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
    f: Expr,
    symbols_enumerated: set[Symbol],
) -> dict[Symbol, Goal]:
    goals: dict[Symbol, Goal] = {}

    def update_goal(symbol: Symbol, goal: str, **kwargs):
        goals[symbol] = Goal(goal) | goals.get(symbol, Goal())

    negate = False

    if not f.free_symbols & symbols_enumerated:
        return goals

    def _try_replace_unknowns(t: Expr):
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

    def _recombine_terms(terms: list[Expr]):
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
    f: Expr,
    symbols_enumerated: set[Symbol],
) -> dict[Symbol, Goal]:
    return _partition_formula(f, fzs(symbols_enumerated & f.free_symbols))


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
    what_tiles_symbol: SymbolValueRelations,
    minimize_formula: Expr = None,
):
    choices_padded = {}
    ones = np.ones(choices_enumerated.shape[0])
    for symbol in symbols_enumerated:
        choices_padded[symbol] = choices_enumerated[:, symbols_enumerated.index(symbol)]
    for symbol in symbols_non_enumerated_set:
        choices_padded[symbol] = ones
        if minimize_formula is not None:
            if sympy.diff(minimize_formula, symbol) < 0:
                choices_padded[symbol] = ones * what_tiles_symbol.get_max_size(symbol)
            elif sympy.diff(minimize_formula, symbol) > 0:
                pass
            elif sympy.diff(minimize_formula, symbol) == 0:
                pass
            else:
                raise ValueError(f"Can't tell if {symbol} is increasing or decreasing")

    return choices_padded


def check_max_fused_loops_per_rank(
    symbols_enumerated: list[Symbol],
    choices_enumerated: np.ndarray,
    max_fused_loop_check_groups: list[tuple[Number, list[Symbol]]],
    what_tiles_symbol: SymbolValueRelations,
):
    def get_size(x: Union[Symbol, int]):
        if isinstance(x, Symbol) and x in symbols_enumerated:
            return choices_enumerated[:, symbols_enumerated.index(x)]
        elif isinstance(x, Symbol):
            return what_tiles_symbol.get_max_size(x)
        else:
            return x

    def has_fanout(x: Union[Symbol, int]):
        outer = get_size(what_tiles_symbol.get_inner_tiles(x))  # TODO: is this a bug?
        inner = get_size(x)
        return outer != inner

    def can_check(x: Union[Symbol, int]):
        if isinstance(x, Symbol) and x not in symbols_enumerated:
            return False
        # tiles = what_tiles_symbol.get_outer_tiles(x, none_if_fail=True)
        # if tiles is not None and isinstance(tiles, Symbol) and tiles not in symbols_enumerated:
        #     return False
        return True

    for limit, group in max_fused_loop_check_groups:
        if len(group) <= limit:
            continue

        n = 0
        for g in group:
            if can_check(g):
                n += has_fanout(g)
        if isinstance(n, np.ndarray):
            choices_enumerated = choices_enumerated[n <= limit]
        elif n > limit:
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
                formula, new_goal = _try_replace_single_term(
                    formula, fzs(symbols_enumerated)
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
                    try:
                        diffed = sympy.diff(formula, s)
                        if diffed < 0:
                            this_goal = (~goal).goal

                        elif diffed > 0:
                            this_goal = (~goal).goal
                        else:
                            raise TypeError  # Can't tell if increasing or decreasing

                        if g != this_goal:
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
    what_tiles_symbol: SymbolValueRelations,
    job: "Job",
    keep_symbols: list[Symbol] = (),
    max_fused_loop_check_groups: list[tuple[Number, list[Symbol]]] = (),
):
    objectives = [copy.deepcopy(o) for o in objectives]

    import time

    objectives = objectives.copy()

    symbols_enumerated: list[Symbol] = []
    choices_enumerated: np.ndarray = None

    symbols_remaining = list(symbols)

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

    def log_message(message: str, *args: str):
        t = time.time() - prev_time
        s = "**" if t > 1 else ""
        job.log_message(f"{s}{t:.2f}s: {message} {' '.join(args)}")
        # print(f"{time.time() - prev_time:.2f}s: {message} {' '.join(args)}")
        time_end(message)

    log_message("init")

    def eval_objective(
        formula: Expr | Objective, choices: np.ndarray, minimize_formula: Expr = None
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
        return sympy.lambdify(symbols, formula)(
            **{str(k): v for k, v in padded_choices.items()},
        )

    def grab_symbol(prev_symbol: Symbol = None):
        # Continue with a symbol representing the parent tile of the last symbol
        # if possible. Otherwise (see return), just grab any symbol.
        choice = what_tiles_symbol.get_outer_tiles(prev_symbol, none_if_fail=True)
        if choice is not None and choice in symbols_remaining:
            symbols_remaining.remove(choice)
            return choice

        # TODO: Maybe start with a symbol that would result in more pruning up front?
        # Maximize the # of choices that can be resolved easily
        return symbols_remaining.pop()

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

            if inner_tiles_type in {"set", "unknown"} and outer_tiles_type in {
                "set",
                "unknown",
            }:
                choices.append(
                    append_vector(
                        choices_enumerated,
                        (
                            np.array(
                                list(
                                    get_possible_factor_sizes(
                                        math.ceil(outer_size / inner_size), imperfect
                                    )
                                )
                            )
                            * inner_size
                        ),
                    )
                )
            elif inner_tiles_type == "enumerated":
                assert isinstance(outer_size, int)
                i = symbols_enumerated.index(inner_tiles)
                for inner_choice in np.unique(choices_enumerated[:, i]):
                    partition = choices_enumerated[
                        np.where(choices_enumerated[:, i] == inner_choice)
                    ]
                    choices.append(
                        append_vector(
                            partition,
                            (
                                np.array(
                                    list(
                                        get_possible_factor_sizes(
                                            math.ceil(outer_size / inner_choice),
                                            imperfect,
                                        )
                                    )
                                )
                                * inner_choice
                            ),
                        )
                    )
            else:
                assert outer_tiles_type == "enumerated"
                assert isinstance(inner_size, int)
                i = symbols_enumerated.index(outer_tiles)
                for outer_choice in np.unique(choices_enumerated[:, i]):
                    partition = choices_enumerated[
                        np.where(choices_enumerated[:, i] == outer_choice)
                    ]
                    choices.append(
                        append_vector(
                            partition,
                            (
                                np.array(
                                    list(
                                        get_possible_factor_sizes(
                                            math.ceil(outer_choice / inner_size),
                                            imperfect,
                                        )
                                    )
                                )
                                * inner_size
                            ),
                        )
                    )
        elif what_tiles_symbol.is_initial_tile_shape(symbol):
            stride = what_tiles_symbol.get_stride(symbol)
            delta_choices = np.array(list(what_tiles_symbol.get_delta_choices(symbol)))

            outer_stride = what_tiles_symbol.get_outer_tiles(stride, none_if_fail=True)
            assert outer_stride is None or isinstance(outer_stride, int), f"outer stride is symbol {outer_stride}"
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
                choices.append(
                    append_vector(choices_enumerated, initial_choices)
                )
            else:
                i = symbols_enumerated.index(stride)
                for stride_choice in np.unique(choices_enumerated[:, i]):
                    partition = choices_enumerated[
                        np.where(choices_enumerated[:, i] == stride_choice)
                    ]
                    initial_choices = delta_choices + stride_choice
                    initial_choices = initial_choices[initial_choices <= outer_size]
                    choices.append(
                        append_vector(partition, initial_choices)
                    )
        else:
            raise RuntimeError(
                f"BUG: symbol {symbol} is neither stride nor initial tile shape"
            )

        # if not partitions:
        #     return np.array([]).reshape(-1, len(symbols))

        prev_size = choices_enumerated.shape[0] if choices_enumerated is not None else 1
        choices_enumerated = np.concatenate(choices, axis=0)
        job.total_pmappings *= choices_enumerated.shape[0] / prev_size
        symbols_enumerated.append(symbol)
        log_message("enumerate", f"{symbol}", f"size={choices_enumerated.shape[0]}")

        # ==============================================================================
        # Max fused loops per rank check
        # ==============================================================================

        prev_size = choices_enumerated.shape[0]
        choices_enumerated = check_max_fused_loops_per_rank(
            symbols_enumerated,
            choices_enumerated,
            max_fused_loop_check_groups,
            what_tiles_symbol,
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

        # If we're a symbol and a non-enumerated outer loop depends on us, then we need
        # to track this loop. Minimize it if we're imperfect (giving the outer the most
        # choices possible), or diff if we're perfect (since perfect constrains choices
        # so we can't just min).
        for s in symbols_enumerated:
            tiles = what_tiles_symbol.get_outer_tiles(s, none_if_fail=True)
            if isinstance(tiles, Symbol) and tiles not in symbols_enumerated:
                update_symbol2goal(s, Goal("min" if imperfect else "diff"))

        # Same for inner loops depending on us, but maximize if we're imperfect
        for s in symbols_enumerated:
            tiled_by = what_tiles_symbol.get_inner_tiles(s, none_if_fail=True)
            if isinstance(tiled_by, Symbol) and tiled_by not in symbols_enumerated:
                update_symbol2goal(tiled_by, Goal("max" if imperfect else "diff"))

        # If we need to keep this symbol, must preserve all choices for it
        for s in set(symbols_enumerated) & set(keep_symbols):
            update_symbol2goal(s, Goal("diff"))

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
                    # minimize_for_objective may raise a TypeError if there's unknown
                    # symbols
                    result = eval_objective(
                        objective.formula,
                        choices_enumerated,
                        minimize_formula=objective.formula,
                    )
                    complete = objective.formula.free_symbols.issubset(
                        sym_enumerated_set
                    )
                    valid = result <= objective.max_value
                    if sum(valid) == 0:
                        # print(f'No valid pmappings. Previous: {prev.sum()}. Best: {result[valid].max()}')
                        eval_objective(
                            objective.formula,
                            choices_enumerated,
                            minimize_formula=objective.formula,
                        )
                    if not isinstance(valid, np.ndarray):
                        valid = (
                            np.zeros(choices_enumerated.shape[0], dtype=bool) + valid
                        )
                    if objective.min_value is not None and valid.any() and complete:
                        above_min = valid & (result >= objective.min_value)
                        if not above_min.any():
                            best = result[valid].max()
                            above_min = valid & (result == best)
                        valid = valid & above_min
                        if not valid.any():
                            prev = result <= objective.max_value
                            print(
                                f"No valid pmappings. Previous: {prev.sum()}. Best: {result[valid].max()}"
                            )

                    porp = sum(valid) / max(1, choices_enumerated.shape[0])
                    job.log_porp_pmappings_kept(
                        f"{objective.name}",
                        sum(valid) / max(1, choices_enumerated.shape[0]),
                    )
                    choices_enumerated = choices_enumerated[valid]
                    log_message(f"Valid check", f"{objective.name}", f"porp={porp:.2%}")
                    if complete:
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

        job.evaluated_pmappings += choices_enumerated.shape[0]
        if not choices_enumerated.shape[0]:
            return np.array([]).reshape(-1, len(symbols))

        if choices_enumerated.shape[0] < 100:
            continue

        # ==============================================================================
        # Coalesce symbols. This simplifies our tracked goals. It also breaks down
        # partially-unknown goals into fully-known and/or fully-unknown goals.
        # ==============================================================================
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
            job.log_message(
                "Skipping Pareto because we've already found a Pareto with these objectives."
            )
            continue
        paretoed_by.append(paretoed_by_key)

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
            job.log_porp_pmappings_kept(
                f"Pareto", sum(keep) / choices_enumerated.shape[0]
            )
            log_message("pareto", f"size {prev_size} -> {choices_enumerated.shape[0]}")

    # ==================================================================================
    # Return the choices
    # ==================================================================================
    t = time.time() - start_time
    if t > 60:
        a = [
            f"Total time: {t:.2f}s",
            f"Pmapping: {job.mapping.compact_str()}",
        ]
        print("\n\t" + f"\n\t".join(a + job.messages))

    # Rearrange in tile shape order
    if choices_enumerated is None:
        return np.array([])
    return choices_enumerated[:, [symbols_enumerated.index(s) for s in symbols]]


def makesymbol(name: str):
    # TODO: Do the solve() calls work with integer=True?
    return Symbol(name, positive=True, integer=True)


def make_keep_symbols(pmapping: Mapping) -> set[Symbol]:
    keep_symbols = set()
    for node in pmapping.nodes:
        if isinstance(node, Iteration) and node._fused:
            if isinstance(node.initial_tile_shape, Symbol):
                keep_symbols.add(node.initial_tile_shape)
            if isinstance(node.stride, Symbol):
                keep_symbols.add(node.stride)
    return keep_symbols


def get_rank_var_to_fused_loops(
    pmapping: Mapping, shape: dict[str, int]
) -> dict[str, list[Symbol]]:
    rank_var_to_fused_loops: dict[str, list[Symbol]] = {}
    for node in [n for n in pmapping.nodes if isinstance(n, Iteration) and n._fused]:
        rank_var_to_fused_loops.setdefault(node.rank_variable, []).append(node.stride)
    return rank_var_to_fused_loops


def set_last_tile_shape_to_one(pmapping):
    pmapping = pmapping.nodes

    rank_var_to_last_node = {}
    for node in pmapping:
        if isinstance(node, Temporal) or isinstance(node, Spatial):
            rank_var_to_last_node[node.rank_variable] = node

    for last_node in rank_var_to_last_node.values():
        last_node.initial_tile_shape = None
        last_node.stride = 1


def _explore_tile_shapes_new(job: "Job"):
    # We're going to convert the job into a list of symbols and objectives
    pmapping = job.mapping
    constraints = job.constraints
    constraints.set_loop_indices(pmapping.nodes)
    set_last_tile_shape_to_one(pmapping)
    symbols, symbolic_df, per_memory_usage_df, utilization_df = run_model(job)
    shape = get_rank_variable_bounds(job.spec.workload, pmapping.nodes[-1].einsum)
    what_tiles_symbol = SymbolValueRelations.from_pmapping_and_shape(
        pmapping, shape, job.spec.workload
    )
    keep_symbols = make_keep_symbols(pmapping)
    rank_var_to_fused_loops = get_rank_var_to_fused_loops(pmapping, shape)
    all_fused_loops = set(sum(rank_var_to_fused_loops.values(), []))

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

    max_fused_loop_check_groups = [
        (job.spec.mapper.ffm.max_fused_loops, all_fused_loops),
        *[
            (job.spec.mapper.ffm.max_fused_loops_per_rank_variable, x)
            for x in rank_var_to_fused_loops.values()
        ],
    ]

    choices_enumerated = get_tile_shape_choices(
        objectives=objectives,
        symbols=symbols,
        what_tiles_symbol=what_tiles_symbol,
        job=job,
        keep_symbols=keep_symbols,
        max_fused_loop_check_groups=max_fused_loop_check_groups,
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

    # Some initial tile shapes are invalid
    for nloops, n in enumerate(node for node in job.mapping.nodes
                               if isinstance(node, Iteration) and node._fused):
        stride = n.tile_pattern.stride
        initial = n.tile_pattern.initial_tile_shape if n.tile_pattern.initial_tile_shape is not None else stride
        outer_stride = what_tiles_symbol.get_outer_tiles(stride)
        outer_initial = what_tiles_symbol.get_initial(outer_stride, none_if_fail=True)
        outer_stride = df[outer_stride.name] if isinstance(outer_stride, Symbol) else outer_stride

        outer_initial = df[outer_initial.name] if isinstance(outer_initial, Symbol) else outer_stride

        rank_var_stride = df[stride.name] if isinstance(stride, Symbol) else stride
        rank_var_initial = df[initial.name] if isinstance(initial, Symbol) else initial

        # NOTE: The concept of having one "n_iterations" is precarious when imperfect factorization in involved
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

                rank_stride = expr.coeff(n.rank_variable)*rank_var_stride

                args = []
                for free_rank_var in free_symbols:
                    if free_rank_var.name == n.rank_variable:
                        args.append(rank_var_initial)
                    else:
                        args.append(shape[free_rank_var.name])
                rank_initial = lambdify(free_symbols, expr)(*args)

                df[stride2col(rank, nloops)] = rank_stride
                df[initial2col(rank, nloops)] = rank_initial

    try:
        df = pd.DataFrame(df, columns=df.keys())
    except ValueError as e:
        df = pd.DataFrame(df, columns=df.keys(), index=[0])
    assert not df.isna().any().any()

    energy_cols = [c for c in df.columns if "Total<SEP>energy" in c]
    if (df[energy_cols] < 0).any(axis=None):
        mapping_with_negative_energy = df[(df[energy_cols] < 0).any(axis=1)]
        print(df.columns)
        msg = ''
        for _, row in mapping_with_negative_energy.iterrows():
            for k, v in row.items():
                msg += f"{k}: {v}\n"
            msg += '\n'
        raise RuntimeError(
            f'negative energy:\n{msg}'
        )

    job.valid_pmappings = job.total_pmappings * prod(job.pmapping_keep_rates.values())
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


EXPERIMENTAL_TILE_SHAPE_EXPLORATION = True


def _explore_tile_shapes(job: "Job"):
    if EXPERIMENTAL_TILE_SHAPE_EXPLORATION:
        return _explore_tile_shapes_new(job)
    from fastfusion.mapper.FFM.deprecate_maybe.tile_shape_exploration_old import (
        _explore_tile_shapes_old,
    )

    return _explore_tile_shapes_old(job)
