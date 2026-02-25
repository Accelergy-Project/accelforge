import inspect
import linecache
import os
import textwrap
from pathlib import Path
from collections.abc import Callable as AbcCallable
from typing import Any, Callable as TypingCallable, List, Optional, Tuple, Union, get_origin
from types import GenericAlias
from ruamel.yaml.compat import StringIO

import ruamel.yaml
import logging, sys
import numpy as np

logger = logging.getLogger("pytimeloop")
formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def show_config(*paths):
    """Print YAML configuration files from provided paths.

    Args:
        *paths: Files or directories. Directories are scanned for `*.yaml`.

    Returns:
        None.
    """
    total = ""
    for path in paths:
        if isinstance(path, str):
            path = Path(path)

        if path.is_dir():
            for p in path.glob("*.yaml"):
                with p.open() as f:
                    total += f.read() + "\n"
        else:
            with path.open() as f:
                total += f.read() + "\n"
    print(total)
    # return total


def check_type(context, a, t):
    """Validate a value against the required type spec.

    Args:
        context: Label used in error messages.
        a: Value to validate.
        t: Required type, tuple of types, or list spec.

    Returns:
        None.
    """
    pref = f"For {context}, expected "
    if isinstance(t, list):
        check_type(context, a, list)
        assert len(a) == len(
            t
        ), f"{pref} a tuple of length {len(t)}, but got a tuple of length {len(a)}"
        for i, (ai, ti) in enumerate(zip(a, t)):
            check_type(f"{context}[{i}]", ai, ti)
    else:
        assert (
            not isinstance(a, str) or a != "FILL ME"
        ), f"For {context}, expected an answer. Please fill in the answer."
        if not isinstance(t, tuple):
            t = (t,)
        if any(_is_callable_type(t0) for t0 in t) and callable(a):
            return
        if isinstance(t, tuple) and any(a == ti for ti in t):
            return
        tn = tuple(
            AbcCallable if _is_callable_type(t0) else _without_parameters(t0)
            for t0 in t
        )
        assert isinstance(a, tn), f"{pref} {tn}, but got {a} of type {type(a).__name__}"


def _without_parameters(t):
    """Return type without parameters, e.g., set[int] becomes set."""
    if isinstance(t, GenericAlias):
        return get_origin(t)
    else:
        return t


def _is_callable_type(t):
    """Return True when `t` represents a callable type.

    Args:
        t: Type annotation or sentinel used in `required_type`.

    Returns:
        True if `t` is a callable marker/type, otherwise False.
    """
    if t is callable or t is AbcCallable or t is TypingCallable:
        return True
    return get_origin(t) is AbcCallable


def _requires_callable(t):
    """Check if a required_type spec includes a callable requirement.

    Args:
        t: A type, tuple of types, or list spec passed to `answer`.

    Returns:
        True when a callable requirement is present.
    """
    if isinstance(t, list):
        return any(_requires_callable(t0) for t0 in t)
    if not isinstance(t, tuple):
        t = (t,)
    return any(_is_callable_type(t0) for t0 in t)


_REF_MODULE = None


def _load_ref_module():
    """Load the ref module containing callable format tests.

    Returns:
        The imported module if found, otherwise None.
    """
    global _REF_MODULE
    if _REF_MODULE is not None:
        return _REF_MODULE

    try:
        import ref  # type: ignore

        _REF_MODULE = ref
        return _REF_MODULE
    except Exception:
        pass

    import importlib.util

    this_dir = Path(__file__).resolve().parent
    candidates = [
        this_dir / "ref.py",
        this_dir.parent / "ref.py",
        this_dir.parent / "labs" / "ref.py",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        spec = importlib.util.spec_from_file_location("lab_ref", candidate)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _REF_MODULE = module
            return _REF_MODULE
    return None


def _run_callable_format_test(answer):
    """Run a format-only test function from ref.py if it exists.

    Args:
        answer: Callable submitted by the student.

    Returns:
        None.
    """
    ref = _load_ref_module()
    if ref is None:
        return
    name = getattr(answer, "__name__", None)
    if not name:
        return
    tester = getattr(ref, f"test_{name}", None)
    if tester is None:
        return
    tester(answer)


def _callable_source(answer):
    """Return source code for a callable to embed in answers.yaml.

    Args:
        answer: Callable submitted by the student.

    Returns:
        The source text for the callable.

    Raises:
        AssertionError: If the source cannot be retrieved from a notebook or .py file.
    """
    try:
        source = inspect.getsource(answer)
    except (OSError, TypeError) as exc:
        filename = getattr(answer, "__code__", None)
        filename = filename.co_filename if filename else None
        lines = linecache.getlines(filename) if filename else []
        # Fallback to reading from disk when linecache misses (e.g., notebooks).
        if not lines and filename:
            path = Path(filename)
            if path.exists():
                lines = path.read_text().splitlines(True)
        if lines:
            start = max(getattr(answer.__code__, "co_firstlineno", 1) - 1, 0)
            source = "".join(inspect.getblock(lines[start:]))
        else:
            raise AssertionError(
                "Could not retrieve source for callable answer. "
                "Please define the function in the notebook (not dynamically) "
                "or in a saved .py file so it can be graded."
            ) from exc
    return textwrap.dedent(source).strip()


def check_string(context, a):
    """Validate a short string answer.

    Args:
        context: Label used in error messages.
        a: String value to validate.

    Returns:
        None.
    """
    check_type(context, a, str)
    assert (
        len(a) < 120
    ), f"For {context}, expected a string of length < 120, but got a string of length {len(a)}"


def answer(
    question: str,
    subquestion: str,
    answer: Any,
    required_type: Union[type, Tuple[type]],
    assumptions: Optional[List[str]] = None,
    weight: Optional[float] = 1,
):
    """Record and persist an answer to answers.yaml.

    Args:
        question: Question identifier (e.g., "1.1").
        subquestion: Prompt text for the subquestion.
        answer: Student answer value or callable.
        required_type: Expected type or type spec.
        assumptions: Optional list of assumptions.
        weight: Optional scoring weight.

    Returns:
        None.
    """
    if isinstance(answer, np.ndarray):
        answer = answer.tolist()
    check_type("answer", answer, required_type)
    stored_answer = answer
    if _requires_callable(required_type):
        # Validate format first, then persist both name and source for autograding.
        _run_callable_format_test(answer)
        stored_answer = {
            "callable": getattr(answer, "__name__", "<callable>"),
            "source": _callable_source(answer),
        }

    if not assumptions:
        assumptions = []

    check_type("assumptions", assumptions, list)
    check_type("assumptions", assumptions, [str] * len(assumptions))

    for i, a in enumerate(assumptions):
        check_string(f"assumptions[{i}]", a)

    this_dir = os.path.dirname(os.path.realpath(__file__))
    answer_path = os.path.join(this_dir, "answers.yaml")

    yaml = ruamel.yaml.YAML(typ="rt")
    yaml.width = 9999999999999999

    answers = {}
    if os.path.exists(answer_path):
        with open(answer_path, "r") as f:
            answers = yaml.load(f.read())

    answers.setdefault(question, {})
    answers[question][subquestion] = {
        "answer": stored_answer,
        "assumptions": assumptions,
        "weight": weight,
    }

    # Sort the answers
    answers = dict(sorted(answers.items(), key=lambda x: x[0]))
    answers[question] = dict(sorted(answers[question].items(), key=lambda x: x[0]))

    s = StringIO()
    yaml.dump(answers, s)

    with open(answer_path, "w") as f:
        f.write(s.getvalue())

    print("\n\t".join([f"{question}: {subquestion}", f"{stored_answer}"] + assumptions))


def AnswerMe():
    raise ValueError("You missed a question!")
