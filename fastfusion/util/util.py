import itertools
from numbers import Number
import tempfile
from typing import Generic, TypeVar

import pydot
import sympy

from joblib import Parallel, delayed
import joblib
import sys
import os
from tqdm import tqdm
import numpy as np

PARALLELIZE = True
N_PARALLEL_PROCESSES = os.cpu_count()

NUMPY_FLOAT_TYPE = np.float32


def lambdify_type_check(*args, **kwargs):
    f = sympy.lambdify(*args, **kwargs)

    def f_type_checked(*args, **kwargs):
        for a in args:
            if isinstance(a, np.ndarray):
                if a.dtype != NUMPY_FLOAT_TYPE:
                    raise ValueError(f"Expected {NUMPY_FLOAT_TYPE}, got {a.dtype}")
            elif not isinstance(a, Number):
                raise ValueError(f"Expected {NUMPY_FLOAT_TYPE}, got {type(a)}")
        for v in kwargs.values():
            if isinstance(v, np.ndarray):
                if v.dtype != NUMPY_FLOAT_TYPE:
                    raise ValueError(f"Expected {NUMPY_FLOAT_TYPE}, got {v.dtype}")
            elif not isinstance(v, Number):
                raise ValueError(f"Expected {NUMPY_FLOAT_TYPE}, got {type(v)}")
        return f(*args, **kwargs)

    return f_type_checked


def set_n_parallel_jobs(n_jobs: int, print_message: bool = False):
    global N_PARALLEL_PROCESSES
    N_PARALLEL_PROCESSES = n_jobs
    global PARALLELIZE
    PARALLELIZE = n_jobs > 1
    if print_message:
        print(f"Using {n_jobs} parallel job{'s' if n_jobs > 1 else ''}")


def using_parallel_processing():
    return PARALLELIZE and N_PARALLEL_PROCESSES > 1


T = TypeVar("T")


class fzs(frozenset[T], Generic[T]):
    def __repr__(self):
        return f"{{{', '.join(sorted(x.__repr__() for x in self))}}}"

    def __str__(self):
        return self.__repr__()

    def __or__(self, other: "fzs[T]") -> "fzs[T]":
        return fzs(super().__or__(other))

    def __and__(self, other: "fzs[T]") -> "fzs[T]":
        return fzs(super().__and__(other))

    def __sub__(self, other: "fzs[T]") -> "fzs[T]":
        return fzs(super().__sub__(other))

    def __xor__(self, other: "fzs[T]") -> "fzs[T]":
        return fzs(super().__xor__(other))

    def __lt__(self, other: "fzs[T]") -> bool:
        return sorted(self) < sorted(other)

    def __le__(self, other: "fzs[T]") -> bool:
        return sorted(self) <= sorted(other)

    def __gt__(self, other: "fzs[T]") -> bool:
        return sorted(self) > sorted(other)

    def __ge__(self, other: "fzs[T]") -> bool:
        return sorted(self) >= sorted(other)


def defaultintersection(*args) -> set:
    allargs = []
    for arg in args:
        if isinstance(arg, set):
            allargs.append(arg)
        else:
            allargs.extend(arg)
    return set.intersection(*allargs) if allargs else set()


def expfmt(x):
    if isinstance(x, Number):
        x = round(x)
        if x < 10000:
            return f"{x}"
        x = f"{x:.2e}"
    else:
        x = str(x)
    if "e+00" in x:
        x = x.replace("e+00", "")
    x = x.replace("e+", "e")
    return x


def fakeparallel(**kwargs):
    if (
        "return_as" in kwargs
        and kwargs["return_as"] == "generator"
        or kwargs["return_as"] == "generator_unordered"
    ):

        def fake_parallel_generator(jobs):
            for j in jobs:
                yield j[0](*j[1], **j[2])

        return fake_parallel_generator
    return lambda jobs: [j[0](*j[1], **j[2]) for j in jobs]


def parallel(
    jobs,
    n_jobs: int = None,
    pbar: str = None,
    return_as: str = None,
):
    jobs = list(jobs)

    args = {}
    if return_as is not None:
        args["return_as"] = return_as

    if n_jobs is None:
        n_jobs = N_PARALLEL_PROCESSES

    if isinstance(jobs, dict):
        assert return_as == None, "return_as is not supported for dict jobs"
        r = zip(
            jobs.keys(),
            parallel(
                jobs.values(),
                pbar=pbar,
            ),
        )
        return {k: v for k, v in r}

    if n_jobs == 1 or len(jobs) == 1:
        if pbar:
            jobs = tqdm(jobs, total=len(jobs), desc=pbar, leave=True)
        return [j[0](*j[1], **j[2]) for j in jobs]

    total_jobs = len(jobs)

    pbar = tqdm(total=total_jobs, desc=pbar, leave=True) if pbar else None

    def yield_results():
        for result in Parallel(n_jobs=n_jobs, **args)(jobs):
            if pbar:
                pbar.update(1)
            yield result
        if pbar:
            pbar.close()

    if return_as in ["generator", "generator_unordered"]:
        return yield_results()

    return list(yield_results())


def symbol2str(x: str | sympy.Symbol) -> str:
    return x.name if isinstance(x, sympy.Symbol) else x


def pydot_graph() -> pydot.Dot:
    graph = pydot.Dot(graph_type="digraph", rankdir="TD", ranksep=0.2)
    graph.set_node_defaults(shape="box", fontname="Arial", fontsize="12")
    graph.set_edge_defaults(fontname="Arial", fontsize="10")
    return graph


import cProfile
import io
import pstats


class ProfilePrint:
    def __init__(self):
        self.profiler = cProfile.Profile()

    def __enter__(self):
        self.profiler.enable()
        self.n_jobs = N_PARALLEL_PROCESSES
        set_n_parallel_jobs(1)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.profiler.disable()
        s = io.StringIO()
        stats = pstats.Stats(self.profiler, stream=s).sort_stats("cumulative")
        stats.print_stats(20)
        print("\n===== Profiling Results (sorted by total time) =====")
        print(s.getvalue())
        # set_n_parallel_jobs(self.n_jobs)


class SVGJupyterRender(str):
    def _repr_svg_(self):
        return self


def memmap_read(x):
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
    joblib.dump(x, f.name)
    return joblib.load(f.name, mmap_mode="r")
