from typing import Annotated, Callable, Optional

from pydantic import ConfigDict
from hwcomponents import ComponentModel
from accelforge.util._basetypes import EvalableDict, EvalableList, EvalableModel
from platformdirs import user_config_dir
import logging
import os
import sys


class Config(EvalableModel):
    expression_custom_functions: EvalableList[str | Callable] = EvalableList()
    """
    A list of functions to use while parsing expressions. These can either be functions
    or paths to Python files that contain the functions. If a path is provided, then all
    functions in the file will be added to the evaluator.
    """
    component_models: EvalableList[str | ComponentModel] = EvalableList()
    """
    A list of hwcomponents models to use for the energy and area calculations. These can
    either be paths to Python files that contain the models, or `hwcomponents`
    :py:class:`~hwcomponents.ComponentModel` objects.
    """
    use_installed_component_models: Optional[bool] = True
    """
    If True, then the `hwcomponents` library will find all installed models. If False,
    then only the models specified in `component_models` will be used.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_yaml(cls, f: str) -> "Config":
        from accelforge.util import _yaml

        data = _yaml.load_yaml(f)
        return cls(**data)

    def add_component_models(self, *args):
        """
        Adds component models to ``component_models``. Each argument may be a
        :py:class:`~hwcomponents.ComponentModel` class, a path to a Python file
        containing models, a module (all non-underscore ComponentModel subclasses in the
        module are added), or a list/tuple of any of these.
        
        Parameters
        ----------
        *args : list of str | ComponentModel | module | list | tuple
            The component models to add. Each argument may be a
            :py:class:`~hwcomponents.ComponentModel` class, a path to a Python file
            containing models, a module (all non-underscore ComponentModel subclasses in
            the module are added), or a list/tuple of any of these.

        Returns
        -------
        None

        """
        from types import ModuleType
        from hwcomponents.find_models import get_models_in_module

        model_ids = set()
        to_add = list(args)
        while to_add:
            arg = to_add.pop(0)
            if isinstance(arg, (list, tuple)):
                to_add[:0] = list(arg)
            elif isinstance(arg, ModuleType):
                self.component_models.extend(get_models_in_module(arg, model_ids))
            else:
                self.component_models.append(arg)
