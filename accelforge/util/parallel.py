import itertools
from numbers import Number
import tempfile
from typing import Any, Callable, Generic, TypeVar, Generator

import pydot
import sympy

from joblib import Parallel, delayed
import joblib
import sys
import os
from tqdm import tqdm
import numpy as np

__all__ = [
    "set_n_parallel_jobs",
    "get_n_parallel_jobs",
    "is_using_parallel_processing",
    "parallel",
    "delayed",
]

PARALLELIZE = True
N_PARALLEL_PROCESSES = os.cpu_count()

NUMPY_FLOAT_TYPE = np.float32


def _lambdify_type_check(*args, **kwargs):
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


def set_n_parallel_jobs(n_jobs: int, print_message: bool = False) -> None:
    """
    Set the number of parallel jobs to use.

    Parameters
    ----------
    n_jobs : int
        The number of parallel jobs to use.
    print_message : bool, optional
        Whether to print a message when the number of parallel jobs is set.
    """
    global N_PARALLEL_PROCESSES
    N_PARALLEL_PROCESSES = n_jobs
    global PARALLELIZE
    PARALLELIZE = n_jobs > 1
    if print_message:
        print(f"Using {n_jobs} parallel job{'s' if n_jobs > 1 else ''}")


def get_n_parallel_jobs() -> int:
    """
    Returns the number of parallel jobs being used. If parallel processing is not
    enabled, returns 1.
    """
    return N_PARALLEL_PROCESSES if is_using_parallel_processing() else 1


def is_using_parallel_processing() -> bool:
    """Returns True if parallel processing is enabled."""
    return PARALLELIZE and N_PARALLEL_PROCESSES > 1


def _expfmt(x):
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


def _dict_job(key, f):
    r = f[0](*f[1], **f[2])
    return key, r


def parallel(
    jobs: list[tuple[Callable, tuple, dict]],
    n_jobs: int = None,
    pbar: str = None,
    pbar_position: int = 0,
    return_as: str = None,
) -> list[Any] | Generator[Any, None, None] | dict[Any, Any]:
    """
    Parallelizes a list of jobs.

    Parameters
    ----------
    jobs : list[tuple[Callable, tuple, dict]]
        The jobs to parallelize. The first element of each tuple is a function, the
        second is a tuple of arguments, and the third is a dictionary of keyword
        arguments.
    n_jobs : int, optional
        The number of jobs to run in parallel. If not provided, the number of parallel
        jobs is set to the number of CPU cores.
    pbar : str, optional
        A label for a progress bar. If not provided, no progress bar is shown.
    pbar_position : int, optional
        The position of the progress bar. If not provided, the progress bar is shown at
        the beginning of the output.
    return_as : Literal["list", "generator", "generator_unordered"], optional
        The type of return value. If not provided, the return value is a list.

    Returns
    -------
    list[Any] | Generator[Any, None, None] | dict[Any, Any]
        The result of the parallelized jobs.
    """
    args = {}
    if return_as is not None:
        args["return_as"] = return_as

    if n_jobs is None:
        n_jobs = N_PARALLEL_PROCESSES

    if isinstance(jobs, dict):
        assert return_as is None, "return_as is not supported for dict jobs"
        result = {
            k: v
            for k, v in parallel(
                [delayed(_dict_job)(k, v) for k, v in jobs.items()],
                pbar=pbar,
                return_as="generator_unordered",
                n_jobs=n_jobs,
                pbar_position=pbar_position,
            )
        }
        return {k: result[k] for k in jobs}

    jobs = list(jobs)

    if n_jobs == 1 or len(jobs) == 1:
        if pbar:
            jobs = tqdm(
                jobs, total=len(jobs), desc=pbar, leave=True, position=pbar_position
            )
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

    # Coerce into generator_unordered so that the progress bar won't hang on one slow
    # job.
    def f(i, job):
        return i, job[0](*job[1], **job[2])

    jobs = [delayed(f)(i, job) for i, job in enumerate(jobs)]
    results = [None] * total_jobs
    args["return_as"] = "generator_unordered"
    for i, result in yield_results():
        results[i] = result
    return results


# import cProfile
# import io
# import pstats


# class ProfilePrint:
#     def __init__(self):
#         self.profiler = cProfile.Profile()

#     def __enter__(self):
#         self.profiler.enable()
#         self.n_jobs = N_PARALLEL_PROCESSES
#         set_n_parallel_jobs(1)
#         return self

#     def __exit__(self, exc_type, exc_value, traceback):
#         self.profiler.disable()
#         s = io.StringIO()
#         stats = pstats.Stats(self.profiler, stream=s).sort_stats("cumulative")
#         stats.print_stats(20)
#         print("\n===== Profiling Results (sorted by total time) =====")
#         print(s.getvalue())
#         # set_n_parallel_jobs(self.n_jobs)


class _SVGJupyterRender(str):
    def _repr_svg_(self):
        return self


def _memmap_read(x):
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
    joblib.dump(x, f.name)
    return joblib.load(f.name, mmap_mode="r")
