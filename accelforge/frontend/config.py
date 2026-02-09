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
