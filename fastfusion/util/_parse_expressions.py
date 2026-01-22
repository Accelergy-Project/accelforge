import functools
from inspect import signature
from importlib.machinery import SourceFileLoader
import logging
import math
import re
import threading
from typing import Any, Callable
from ._yaml import load_yaml, SCRIPTS_FROM
from ruamel.yaml.scalarstring import DoubleQuotedScalarString, SingleQuotedScalarString
import os
import keyword


class ParseError(Exception):
    def __init__(self, *args, source_field: Any = None, message: str = None, **kwargs):
        self._fields = [source_field] if source_field is not None else []
        if message is None and len(args) > 0:
            message = args[0]
        self.message = message
        super().__init__(*args, **kwargs)

    def add_field(self, field: Any):
        self._fields.append(field)

    def __str__(self) -> str:
        s = f"{self.__class__.__name__} in {'.'.join(str(field) for field in self._fields[::-1])}"
        if self.message is not None:
            s += f": {self.message}"
        if getattr(self, "__notes__", None):
            s += f"\n\n{'\n\n'.join(self.__notes__)}"
        return s


class LiteralString(str):
    """
    A string literal that should not be parsed.
    """
    pass


def is_literal_string(value: Any) -> bool:
    return isinstance(
        value, (DoubleQuotedScalarString, SingleQuotedScalarString, LiteralString)
    )


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
    def __init__(self, spec: "Spec"):
        self.spec = spec
        self.grabbed_lock = False

    def __enter__(self):
        if parse_expression_thread_lock.is_locked_by_current_thread():
            return
        parse_expression_thread_lock.acquire()
        parse_expressions_local.script_funcs = {}
        for p in self.spec.config.expression_custom_functions:
            if isinstance(p, str):
                parse_expressions_local.script_funcs.update(load_functions_from_file(p))
            elif isinstance(p, Callable):
                parse_expressions_local.script_funcs[p.__name__] = p
            else:
                raise ValueError(f"Invalid expression custom function: {p}")
        self.grabbed_lock = True

    def __exit__(self, exc_type, exc_value, traceback):
        if self.grabbed_lock:
            self.spec = None
            del parse_expressions_local.script_funcs
            parse_expression_thread_lock.release()


def cast_to_numeric(x: Any) -> int | float | bool:
    if str(x).lower() == "true":
        return True
    if str(x).lower() == "false":
        return False
    if float(x) == int(x):
        return int(x)
    return float(x)


class CallableLambda:
    def __init__(self, func, expression):
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
        self._original_expression = expression
        self._func = func

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)


@functools.lru_cache(maxsize=1000)
def infostr_log_cache(infostr: str):
    logging.info(infostr)


def parse_expression(
    expression, symbol_table, attr_name: str = None, location: str = None
):
    try:
        return cast_to_numeric(expression)
    except:
        pass

    if not isinstance(expression, str):
        return expression

    if expression in symbol_table:
        result = symbol_table[expression]
        if isinstance(result, str):
            result = LiteralString(result)
        return result

    FUNCTION_BINDINGS = {}
    FUNCTION_BINDINGS["__builtins__"] = None  # Safety
    if hasattr(parse_expressions_local, "script_funcs"):
        FUNCTION_BINDINGS.update(parse_expressions_local.script_funcs)
    FUNCTION_BINDINGS.update(MATH_FUNCS)

    try:
        v = eval(expression, FUNCTION_BINDINGS, symbol_table)
        infostr = f'Calculated "{expression}" = {v}.'
        if isinstance(v, str):
            v = LiteralString(v)
        if isinstance(v, Callable):
            v = CallableLambda(v, expression)
        success = True
    except Exception as e:
        errstr = f"Failed to evaluate: {expression}\n"
        if (
            isinstance(expression, str)
            and expression.isidentifier()
            and expression not in symbol_table
            and expression not in FUNCTION_BINDINGS
        ):
            e = NameError(f"Name '{expression}' is not defined.")
        extra = ""
        if attr_name and location:
            extra = f" while parsing {location}.{attr_name}"
        elif attr_name:
            extra = f" while parsing {attr_name}"
        elif location:
            extra = f" while parsing {location}"
        errstr += f"Problem encountered{extra}: {e.__class__.__name__}: {e}\n"
        err = errstr
        errstr += f"Symbol table: "
        bindings = {}
        bindings.update(symbol_table)
        bindings.update(getattr(parse_expressions_local, "script_funcs", {}))
        extras = []
        for k, v in bindings.items():
            if isinstance(v, Callable):
                bindings[k] = f"{k}{signature(getattr(v, '_func', v))}"
            else:
                extras.append(f"\n    {k} = {v}")
        for k, v in bindings.items():
            bindings[k] = str(v).replace("\n", "\\n")
            if len(bindings[k]) > 100:
                bindings[k] = bindings[k][:100] + "..."
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
            errstr += (
                f"The following may have been used in the expression:\n\t"
                + "\n\t".join(f"{k} = {v}" for k, v in possibly_used.items())
            )
        errstr += (
            "\n\nIf you meant to enter a string in a YAML file, please wrap the\n"
            "expression in single or double quotes. If you meant to enter a raw \n"
            "string, cast it to a fastfusion.LiteralString object."
        )
        success = False

    if not success:
        raise ParseError(errstr)

    infostr_log_cache(infostr)

    return v


class PicklingSafeCallable:
    def __init__(self, func: Callable, path: str):
        self.func = func
        self.__name__ = func.__name__
        self.path = path

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __getstate__(self):
        return {
            "func": self.func.__name__,
            "module": self.func.__module__,
            "path": self.path,
        }

    def __setstate__(self, state):
        # Restore required attributes so subsequent pickling works
        self.path = state.get("path")
        func_name = state["func"]
        self.func = load_functions_from_file(self.path)[func_name]
        self.__name__ = func_name

    def __copy__(self):
        return PicklingSafeCallable(self.func, self.path)

    def __deepcopy__(self, memo):
        return PicklingSafeCallable(self.func, self.path)


@functools.lru_cache(maxsize=100)
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
        funcs[func] = PicklingSafeCallable(getattr(python_module, func), path)
    return funcs
