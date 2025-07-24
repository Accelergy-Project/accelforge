from abc import ABC, abstractmethod
from logging import info
from numbers import Number
import os
from typing import Callable, List, Union
from fastfusion.plugin.logging import ListLoggable
from fastfusion.util.yaml import load_yaml, write_yaml_file


def actionDynamicEnergy(func: Callable) -> Callable:
    """
    Decorator that adds an action to an energy/area estimator. Actions are
    expected to return an energy value in Juoles or an Estimation object with
    the energy and units.
    """
    func._is_component_energy_action = True
    return func


class EnergyAreaEstimator(ListLoggable, ABC):
    """
    EnergyAreaEstimator base class. EnergyAreaEstimator class must have "name"
    attribute, "percent_accuracy_0_to_100" attribute, and "get_area" method.
    EnergyAreaEstimators may have any number of methods that are decorated with
    @.
    """

    name: Union[str, List[str]] = None
    percent_accuracy_0_to_100: Number = None

    def __init__(self, name: str = None):
        super().__init__(name=name)

    @abstractmethod
    def get_area(self) -> Number:
        """Returns the area in m^2 or an Estimation object with the area and
        units."""
        pass

    @abstractmethod
    def leak(self, global_cycle_seconds: float) -> Number:
        """Returns the leakage energy per global cycle or an Estimation object
        with the leakage energy and units."""
        pass


def install(path: str, add_full_dir: bool = False):
    """Adds a path to the list of Python plug-in paths in the config file."""
    print(
        f"Warning: This method of installation is for testing purposes only. For public "
        f"plug-ins, use the pip install instructions in the tutorial exercises."
    )
    cfg_yaml = get_config_file_path()
    path = os.path.abspath(path)
    if add_full_dir:
        path = os.path.dirname(path)

    cfg = load_yaml(cfg_yaml)
    python_paths = cfg.get("python_models", [])

    # Update the list of paths
    if path in python_paths:
        info(f"Path {path} already in the list of python paths.")
    else:
        cfg.update({"python_models": python_paths + [path]})
        info(f"Added path {path} to the list of python paths.")
        write_yaml_file(cfg_yaml, cfg)


def uninstall(path: str, remove_full_dir: bool = False):
    """Removes a path from the list of Python plug-in paths in the config file."""
    cfg_yaml = get_config_file_path()
    path = os.path.abspath(path)
    if remove_full_dir:
        path = os.path.dirname(path)

    cfg = load_yaml(cfg_yaml)
    python_paths = cfg.get("python_models", [])

    # Update the list of paths
    if path not in python_paths:
        info(f"Path {path} not in the list of python paths.")
    else:
        cfg.update({"python_models": [p for p in python_paths if p != path]})
        info(f"Removed path {path} from the list of python paths.")
        write_yaml_file(cfg_yaml, cfg)


def add_estimator_path(*args, **kwargs):
    """Alias for install"""
    install(*args, **kwargs)


def remove_estimator_path(*args, **kwargs):
    """Alias for uninstall"""
    uninstall(*args, **kwargs)
