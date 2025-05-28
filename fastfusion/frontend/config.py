from typing import List, Dict, Annotated
from fastfusion.util.basetypes import ParsableDict, ParsableList, ParsableModel
from fastfusion.version import assert_version, __version__
from platformdirs import user_config_dir
import logging
import os
import sys
from pathlib import Path

def get_config():
    if hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    ):
        f = os.path.join(sys.prefix, "fastfusion", "config.yaml")
    else:
        f = os.path.join(user_config_dir("fastfusion"), "config.yaml")

    if not os.path.exists(f):
        from fastfusion.util import yaml

        logging.warning(f"No configuration file found. Creating config file at {f}.")
        os.makedirs(os.path.dirname(f), exist_ok=True)
        config = Config()
        config.to_yaml(f)

    logging.warning(f"Loading configuration file from {f}")
    return Config.from_yaml(f)


class Config(ParsableModel):
    version: Annotated[str, assert_version] = __version__
    environment_variables: ParsableDict[str, str] = ParsableDict()
    expression_custom_functions: ParsableList[str] = ParsableList()
    component_plug_ins: ParsableList[str] = ParsableList()

    @classmethod
    def from_yaml(cls, f: str) -> "Config":
        from fastfusion.util import yaml
        data = yaml.load_yaml(f)
        return cls(**data)
