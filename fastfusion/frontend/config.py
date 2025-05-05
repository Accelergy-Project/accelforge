from fastfusion.yamlparse.nodes import CombinableListNode, DictNode
from .version import assert_version
from platformdirs import user_config_dir
import logging


def get_config():
    import os
    import sys
    from pathlib import Path
    
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        f = os.path.join(sys.prefix, "fastfusion", "config.yaml")
    else:
        f = os.path.join(user_config_dir("fastfusion"), "config.yaml")
        
    if not os.path.exists(f):
        from fastfusion.yamlparse import yaml
        logging.warning(f"No configuration file found. Creating config file at {f}.")
        os.makedirs(os.path.dirname(f), exist_ok=True)
        config = Config()
        config.to_yaml(f)

    logging.warning(f"Loading configuration file from {f}")
    return Config.from_yaml(f)


class Config(DictNode):
    """
    Top-level Config key.

    Attributes:
        version (str): Version of the Timeloop file.
        environment_variables (EnvironmentVariables): Environment variables to be used.
        expression_custom_functions (ExpressionCustomFunctions): Paths to Python files containing functions to be used in expressions.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("version", default="0.5", callfunc=assert_version)
        super().add_attr("environment_variables", EnvironmentVariables, [])
        super().add_attr("expression_custom_functions", ExpressionCustomFunctions, [])
        super().add_attr("component_plug_ins", ComponentPlugIns, [])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.version: str = self["version"]
        self.environment_variables: EnvironmentVariables = self["environment_variables"]
        self.expression_custom_functions: ExpressionCustomFunctions = self[
            "expression_custom_functions"
        ]
        self.component_plug_ins: ComponentPlugIns = self["component_plug_ins"]


class EnvironmentVariables(DictNode):
    """
    Dictionary of environment variables.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        cls.recognize_all()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ExpressionCustomFunctions(CombinableListNode):
    """
    A list of paths to Python files containing functions to be used in expressions.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ComponentPlugIns(CombinableListNode):
    """
    A list of paths to Python files containing component plug-ins.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
