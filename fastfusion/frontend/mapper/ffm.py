from typing import Dict, Any, Annotated

from pydantic import ConfigDict
from fastfusion.mapper.metrics import Metrics
from fastfusion.util.basetypes import ParsableModel, ParseExtras
from fastfusion.version import assert_version, __version__


class MapperFFM(ParsableModel, ParseExtras):
    version: Annotated[str, assert_version] = __version__
    timeloop_style_even: bool = False
    force_memory_hierarchy_order: bool = True
    max_fused_loops_per_rank: int = 1
    metrics: Metrics = Metrics.ENERGY
    
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
