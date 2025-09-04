import itertools
from numbers import Number

# import joblib.externals.loky
# joblib.externals.loky.process_executor._MAX_MEMORY_LEAK_SIZE = int(3e9)  # 3GB

from joblib import Parallel, delayed
import sys

from tqdm import tqdm

PARALLELIZE = True
N_PARALLEL_PROCESSES = 24


def using_parallel_processing():
    return PARALLELIZE and N_PARALLEL_PROCESSES > 1


class fzs(frozenset):
    def __repr__(self):
        return f"{{{', '.join(sorted(x.__repr__() for x in self))}}}"

    def __str__(self):
        return self.__repr__()

    def __or__(self, other):
        return fzs(super().__or__(other))

    def __and__(self, other):
        return fzs(super().__and__(other))

    def __sub__(self, other):
        return fzs(super().__sub__(other))

    def __xor__(self, other):
        return fzs(super().__xor__(other))

    def __lt__(self, other):
        return sorted(self) < sorted(other)

    def __le__(self, other):
        return sorted(self) <= sorted(other)

    def __gt__(self, other):
        return sorted(self) > sorted(other)

    def __ge__(self, other):
        return sorted(self) >= sorted(other)

def defaultintersection(*args) -> set:
    allargs = []
    for arg in args:
        if isinstance(arg, set):
            allargs.append(arg)
        else:
            allargs.extend(arg)
    return set.intersection(*allargs) if allargs else set()


def debugger_active():
    return not PARALLELIZE
    return 'pydevd' in sys.modules or sys.gettrace() is not None


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
    one_job_if_debugging: bool = True,
    pbar: str = None,
    return_as: str = None,
):
    jobs = list(jobs)
    
    args = {}
    if return_as is not None:
        args["return_as"] = return_as

    if n_jobs is None:
        n_jobs = N_PARALLEL_PROCESSES

    if one_job_if_debugging and debugger_active():
        n_jobs = 1
        
    if isinstance(jobs, dict):
        assert return_as == None, "return_as is not supported for dict jobs"
        r = zip(
            jobs.keys(),
            parallel(
                jobs.values(), pbar=pbar, one_job_if_debugging=one_job_if_debugging
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
