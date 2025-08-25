from typing import Any, Annotated

from fastfusion.mapper.metrics import Metrics
from fastfusion.util.basetypes import ParsableModel
from fastfusion.version import assert_version, __version__


class FFM(ParsableModel):
    version: Annotated[str, assert_version] = __version__
    timeloop_style_even: bool = False
    force_memory_hierarchy_order: bool = True
    max_fused_loops_per_rank: int = 1
    max_fused_loops: float | int = float('inf')
    max_loops: float | int = float('inf')
    max_loops_minus_ranks: float | int = float('inf')
    max_explored_tile_shapes_per_bypass_choice: float | int = float('inf')
    metrics: Metrics = Metrics.ENERGY
    
    memory_limit: float | int = float('inf')
    memory_limit_per_process: float | int = float('inf')
    time_limit: float | int = float('inf')
    time_limit_per_bypass_choice: float | int = float('inf')
    
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
