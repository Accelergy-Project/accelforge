from typing import Annotated

from pydantic import ConfigDict
from accelforge.util._basetypes import EvalableModel, EvalExtras, EvalsTo


class Variables(EvalExtras):
    """
    Variables that can be used in parsing. All variables defined here can be referenced
    elsewhere in any of the Spec's evaluated expressions.
    """
