from inspect import signature
import copy
from importlib.machinery import SourceFileLoader
import logging
import math
import re
import threading
import traceback
from typing import Any, Callable, Union
from numbers import Number
from .yaml import load_yaml, SCRIPTS_FROM
import ruamel.yaml
import os
import keyword

MATH_FUNCS = {
    "ceil": math.ceil,
    "comb": math.comb,
    "copysign": math.copysign,
    "fabs": math.fabs,
    "factorial": math.factorial,
    "floor": math.floor,
    "fmod": math.fmod,
    "frexp": math.frexp,
    "fsum": math.fsum,
    "gcd": math.gcd,
    "isclose": math.isclose,
    "isfinite": math.isfinite,
    "isinf": math.isinf,
    "isnan": math.isnan,
    "isqrt": math.isqrt,
    "ldexp": math.ldexp,
    "modf": math.modf,
    "perm": math.perm,
    "prod": math.prod,
    "remainder": math.remainder,
    "trunc": math.trunc,
    "exp": math.exp,
    "expm1": math.expm1,
    "log": math.log,
    "log1p": math.log1p,
    "log2": math.log2,
    "log10": math.log10,
    "pow": math.pow,
    "sqrt": math.sqrt,
    "acos": math.acos,
    "asin": math.asin,
    "atan": math.atan,
    "atan2": math.atan2,
    "cos": math.cos,
    "dist": math.dist,
    "hypot": math.hypot,
    "sin": math.sin,
    "tan": math.tan,
    "degrees": math.degrees,
    "radians": math.radians,
    "acosh": math.acosh,
    "asinh": math.asinh,
    "atanh": math.atanh,
    "cosh": math.cosh,
    "sinh": math.sinh,
    "tanh": math.tanh,
    "erf": math.erf,
    "erfc": math.erfc,
    "gamma": math.gamma,
    "lgamma": math.lgamma,
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
    "nan": math.nan,
    "abs": abs,
    "round": round,
    "pow": pow,
    "sum": sum,
    "range": range,
    "len": len,
    "min": min,
    "max": max,
    "float": float,
    "int": int,
    "str": str,
    "bool": bool,
    "list": list,
    "tuple": tuple,
    "enumerate": enumerate,
    "getcwd": os.getcwd,
    "map": map,
}
SCRIPT_FUNCS = {}

parse_expressions_local = threading.local()


class OwnedLock:
    def __init__(self):
        super().__init__()
        self._owner = None
        self.lock = threading.Lock()

    def acquire(self, *args, **kwargs):
        result = self.lock.acquire(*args, **kwargs)
        if result:
            self._owner = threading.get_ident()
        return result

    def release(self, *args, **kwargs):
        self._owner = None
        self.lock.release(*args, **kwargs)

    def is_locked_by_current_thread(self):
        return self._owner == threading.get_ident() and self.lock.locked()


parse_expression_thread_lock = OwnedLock()


class ParseExpressionsContext:
    def __init__(self, spec: "Specification"):
        self.spec = spec
        self.grabbed_lock = False

    def __enter__(self):
        if parse_expression_thread_lock.is_locked_by_current_thread():
            return
        parse_expression_thread_lock.acquire()
        parse_expressions_local.script_funcs = {}
        for p in self.spec.config.expression_custom_functions:
            parse_expressions_local.script_funcs.update(load_functions_from_file(p))
        self.grabbed_lock = True

    def __exit__(self, exc_type, exc_value, traceback):
        if self.grabbed_lock:
            self.spec = None
            del parse_expressions_local.script_funcs
            parse_expression_thread_lock.release()


def cast_to_numeric(x: Any) -> Union[int, float, bool]:
    if str(x).lower() == "true":
        return True
    if str(x).lower() == "false":
        return False
    if float(x) == int(x):
        return int(x)
    return float(x)


def is_quoted_string(expression):
    return isinstance(
        expression, ruamel.yaml.scalarstring.DoubleQuotedScalarString
    ) or isinstance(expression, ruamel.yaml.scalarstring.SingleQuotedScalarString)


def get_callable_lambda(func, expression):
    l = lambda *args, **kwargs: func(*args, **kwargs)
    l.__name__ = func.__name__
    l.__doc__ = func.__doc__
    l._original_expression = expression
    l._func = func
    return l


def parse_expression(
    expression,
    binding_dictionary,
    location: str,
    strings_allowed: bool = True,
    use_bindings_after: str = None,
):
    if strings_allowed and is_quoted_string(expression):
        return expression

    try:
        return cast_to_numeric(expression)
    except:
        pass

    if not isinstance(expression, str):
        return expression

    if use_bindings_after is not None:
        keys = list(binding_dictionary.keys())
        index = keys.index(use_bindings_after)
        if index != -1:
            keys = keys[:index]
            binding_dictionary = {k: binding_dictionary[k] for k in keys}

    FUNCTION_BINDINGS = {}
    FUNCTION_BINDINGS["__builtins__"] = None  # Safety
    FUNCTION_BINDINGS.update(parse_expressions_local.script_funcs)
    FUNCTION_BINDINGS.update(MATH_FUNCS)

    try:
        v = eval(expression, FUNCTION_BINDINGS, binding_dictionary)
        infostr = f'Calculated {location} as "{expression}" = {v}.'
        if isinstance(v, str):
            v = ruamel.yaml.scalarstring.DoubleQuotedScalarString(v)
        if isinstance(v, Callable):
            v = get_callable_lambda(v, expression)
        success = True
    except Exception as e:
        errstr = f"Failed to evaluate: {expression}\n"
        errstr += f"Location: {location}\n"
        if (
            isinstance(expression, str)
            and expression.isidentifier()
            and expression not in binding_dictionary
            and expression not in FUNCTION_BINDINGS
        ):
            e = NameError(f"Name '{expression}' is not defined.")
        errstr += f"Problem encountered: {e.__class__.__name__}: {e}\n"
        err = errstr
        errstr += f"Available bindings: "
        bindings = {}
        bindings.update(binding_dictionary)
        bindings.update(parse_expressions_local.script_funcs)
        extras = []
        for k, v in bindings.items():
            if isinstance(v, Callable):
                bindings[k] = f"{k}{signature(getattr(v, '_func', v))}"
            else:
                extras.append(f"\n    {k} = {v}")
        errstr += "".join(f"\n\t{k} = {v}" for k, v in bindings.items())
        errstr += "\n\n" + err
        errstr += (
            f"Please ensure that the expression used is a valid Python expression.\n"
        )
        possibly_used = {
            k: bindings.get(k, FUNCTION_BINDINGS.get(k, "UNDEFINED"))
            for k in re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)", expression)
            if k not in keyword.kwlist
        }
        if possibly_used:
            errstr += f"The following may have been used in the expression:"
            errstr += "".join(f"\n\t{k} = {v}" for k, v in possibly_used.items())
            errstr += "\n"
        if strings_allowed:
            errstr += "Strings are allowed here. If you meant to enter a string, please wrap the\n"
            errstr += "expression in single or double quotes:\n"
            errstr += f"    Found expression: {expression}\n"
            errstr += f'    Expression as valid string: "{expression}"\n'
        success = False

    if not success:
        raise ArithmeticError(f"{errstr}\n")

    logging.info(infostr)
    return v


def load_functions_from_file(path: str):
    path = path.strip()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find math function file {path}.")
    python_module = SourceFileLoader("python_plug_in", path).load_module()
    funcs = {}
    defined_funcs = [
        f for f in dir(python_module) if isinstance(getattr(python_module, f), Callable)
    ]
    for func in defined_funcs:
        logging.info(f"Adding function {func} from {path} to the script library.")
        funcs[func] = getattr(python_module, func)
    return funcs
