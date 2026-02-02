from typing import Annotated, Callable, Optional

from pydantic import ConfigDict
from hwcomponents import ComponentModel
from accelforge.util._basetypes import EvalableDict, EvalableList, EvalableModel
from platformdirs import user_config_dir
import logging
import os
import sys

USER_CUSTOM_CONFIG_PATH_VAR = "ACCELFORGE_CONFIG_PATH"


def get_config() -> "Config":
    if USER_CUSTOM_CONFIG_PATH_VAR in os.environ:
        f = os.environ[USER_CUSTOM_CONFIG_PATH_VAR]
    elif hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    ):
        f = os.path.join(sys.prefix, "accelforge", "config.yaml")
    else:
        f = os.path.join(user_config_dir("accelforge"), "config.yaml")

    if not os.path.exists(f):
        logging.warning(f"No configuration file found. Creating config file at {f}.")
        os.makedirs(os.path.dirname(f), exist_ok=True)
        config = Config()
        config.to_yaml(f)

    logging.warning(f"Loading configuration file from {f}")
    return Config.from_yaml(f)


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
