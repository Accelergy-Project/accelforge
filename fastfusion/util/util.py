import itertools
from numbers import Number

# import joblib.externals.loky
# joblib.externals.loky.process_executor._MAX_MEMORY_LEAK_SIZE = int(3e9)  # 3GB

from joblib import Parallel, delayed
import sys

from tqdm import tqdm

PARALLELIZE = True
N_PARALLEL_THREADS = 16


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
    chunk: bool = False,
    delete_job_after: bool = False,
):
    jobs = list(jobs)
    
    args = {}
    if return_as is not None:
        args["return_as"] = return_as

    if n_jobs is None:
        n_jobs = N_PARALLEL_THREADS

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

    # We were getting a lot of the runtime in parallelizing overhead. What this
    # does is chunks jobs into larger groups. The last n_jobs groups are 1 job,
    # the previous n_jobs groups are 2 jobs, the previous n_jobs groups are 4
    # jobs, and so on. Jobs get smaller near the end to reduce the impact of
    # long pole jobs.

    if chunk and not delete_job_after:

        def job_chunk(chunk_of_jobs):
            # print(list(j[0](*j[1], **j[2]) for j in chunk_of_jobs[::-1])[0][0])
            return list(j[0](*j[1], **j[2]) for j in chunk_of_jobs[::-1])

        jobs = list(reversed(jobs))
        new_jobs = []
        i = 0
        chunksize = 1
        while i < len(jobs):
            stop = min(i + n_jobs * chunksize, len(jobs))
            new_jobs += [
                delayed(job_chunk)(jobs[j : j + chunksize])
                for j in range(i, stop, chunksize)
            ]
            i = stop
            chunksize *= 2
        jobs = list(reversed(new_jobs))
        # jobs = [delayed(job_chunk)([j]) for j in jobs]
        if pbar:
            jobs = tqdm(jobs, total=len(jobs), desc=pbar, leave=True)
        if return_as == "generator" or return_as == "generator_unordered":

            def yield_jobs():
                for job in Parallel(n_jobs=n_jobs, **args)(jobs):
                    yield from job

            return yield_jobs()
        return list(itertools.chain(*Parallel(n_jobs=n_jobs, **args)(jobs)))

    total_jobs = len(jobs)

    if delete_job_after:

        def job_delete_iterator(jobs):
            jobs = list(reversed(jobs))
            while jobs:
                yield jobs.pop()

        jobs = job_delete_iterator(jobs)

    if pbar:
        return Parallel(n_jobs=n_jobs, **args)(
            tqdm(jobs, total=total_jobs, desc=pbar, leave=True)
        )
    return Parallel(n_jobs=n_jobs, **args)(jobs)


if __name__ == "__main__":

    def test_job(x):
        print(f"Called with {x}")
        return (x, 1)

    jobs = [delayed(test_job)(i) for i in range(100)]
    for j in parallel(jobs, pbar="test", return_as="generator"):
        print(j)
