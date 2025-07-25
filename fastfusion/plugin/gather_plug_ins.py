import glob
from importlib.machinery import SourceFileLoader
from types import ModuleType
from typing import List, Set
from .estimator_wrapper import EnergyAreaEstimator, EnergyAreaEstimatorWrapper
import inspect
import logging
import copy
import sys
import os


def get_all_estimators_in_module(
    module: ModuleType, plug_in_ids: Set
) -> List[EnergyAreaEstimator]:
    try:
        classes = getattr(module, "get_plug_in_classes")()
    except AttributeError:
        classes = [
            (x, name)
            for name in dir(module)
            if inspect.isclass(x := getattr(module, name))
        ]
    classes = [(x, name) for x, name in classes if x.__module__ == module.__name__]
    found = []
    for x, name in classes:
        if (
            issubclass(x, EnergyAreaEstimator)
            and not x is EnergyAreaEstimator
            and id(x) not in plug_in_ids
        ):
            plug_in_ids.add(id(x))
            found.append(EnergyAreaEstimatorWrapper(x, name))
    return found


def gather_models(paths: list):
    """
    instantiate a list of estimator plug-in objects for later queries estimator
    plug-in paths are specified in config file
    """
    paths_globbed = []
    for p in paths:
        if os.path.isfile(p):
            assert p.endswith(".py"), f"Plug-in {p} is not a Python file"
            paths_globbed.append(p)
        else:
            paths_globbed += list(glob.glob(os.path.join(p, "**"), recursive=True))

    # Remove any trailing "/" from file names
    paths_globbed = [p.rstrip("/") for p in paths_globbed]
    # Filter out setup.py files
    paths_globbed = [
        p for p in paths_globbed if p.endswith(".py") and not p.endswith("setup.py")
    ]

    # Load Python plug-ins
    models = []
    plug_in_ids = set()
    n_plugins = 0
    for path in paths_globbed:
        logging.info(
            f"Loading plug-ins from {path}. Errors below are likely due to the plug-in."
        )
        prev_sys_path = copy.deepcopy(sys.path)
        sys.path.append(os.path.dirname(os.path.abspath(path)))
        python_module = SourceFileLoader(f"plug_in{n_plugins}", path).load_module()
        models += get_all_estimators_in_module(python_module, plug_in_ids)
        sys.path = prev_sys_path
        n_plugins += 1
    return models
