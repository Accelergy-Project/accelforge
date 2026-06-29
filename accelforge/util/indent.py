import builtins
import contextlib
import threading

import tqdm as _tqdm

INDENT = "\t"

_real_print = builtins.print
_state = threading.local()


def _depth():
    return getattr(_state, "depth", 0)


def _prefix():
    return INDENT * _depth()


class _Indent:
    @contextlib.contextmanager
    def indent(self):
        _state.depth = _depth() + 1
        try:
            yield
        finally:
            _state.depth = _depth() - 1


indent = _Indent()


def print(*args, sep=" ", end="\n", file=None, flush=False):
    prefix = _prefix()
    if not prefix:
        return _real_print(*args, sep=sep, end=end, file=file, flush=flush)
    msg = sep.join(str(a) for a in args)
    msg = prefix + msg.replace("\n", "\n" + prefix)
    _real_print(msg, end=end, file=file, flush=flush)


def tqdm(*args, **kwargs):
    if kwargs.get("desc"):
        kwargs["desc"] = _prefix() + kwargs["desc"]
    return _tqdm.tqdm(*args, **kwargs)
