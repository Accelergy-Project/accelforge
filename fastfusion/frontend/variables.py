from typing import Annotated

from pydantic import ConfigDict
from fastfusion.util._basetypes import ParsableModel, ParseExtras, ParsesTo
from fastfusion._version import assert_version, __version__


class Variables(ParseExtras):
    """
    Variables that can be used in parsing. All variables defined here can be referenced
    elsewhere in any of the Spec's parsed expressions.
    """
