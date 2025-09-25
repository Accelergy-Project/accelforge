import pickle
from pathlib import Path
from typing import Callable, TypeVar


T = TypeVar("T")


class PickleCache:
    def __init__(self, fname: Path):
        self.fname = fname

    def get(self, cache_miss_thunk: Callable[[], T]) -> T:
        """
        Return loaded pickle at `self.fname` if exists;
        otherwise, calls `cache_miss_thunk` and stores result in `self.fname`.
        """
        print(self.fname)
        if self.fname.exists():
            with open(self.fname, "rb") as f:
                return pickle.load(f)
        else:
            result = cache_miss_thunk()
            with open(self.fname, "wb") as f:
                pickle.dump(result, f)
            return result

    def set(self, data: T) -> None:
        """Set data at `self.fname`."""
        with open(self.fname, "wb") as f:
            pickle.dump(data, f)
        return data
