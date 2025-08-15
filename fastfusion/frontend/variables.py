from typing import Dict, Any, Annotated, Union

from pydantic import ConfigDict
from fastfusion.util.basetypes import ParsableModel, ParseExtras, ParsesTo
from fastfusion.version import assert_version, __version__


class Variables(ParseExtras):
    version: Annotated[str, assert_version] = __version__
    model_config = ConfigDict(extra="allow")

    global_cycle_period: ParsesTo[Union[int, float]] = \
        '"Set me with Specification().variables.global_cycle_period = [value]"'