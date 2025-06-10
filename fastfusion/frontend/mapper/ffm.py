from typing import Dict, Any, Annotated

from pydantic import ConfigDict
from fastfusion.util.basetypes import ParsableModel, ParseExtras
from fastfusion.version import assert_version, __version__


class MapperFFM(ParsableModel, ParseExtras):
    version: Annotated[str, assert_version] = __version__
    timeloop_style_even: bool = False
    force_memory_hierarchy_order: bool = True
    
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
