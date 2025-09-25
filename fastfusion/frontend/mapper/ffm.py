from typing import Any, Annotated

from fastfusion.frontend.mapper.metrics import Metrics
from fastfusion.util.basetypes import ParsableModel
from fastfusion.version import assert_version, __version__


class FFM(ParsableModel):
    """Configuration for the Fast and Fusiest Mapper."""

    version: Annotated[str, assert_version] = __version__
    """ Version """

    metrics: Metrics = Metrics.ENERGY
    """ Metrics used to optimize mappings. """

    timeloop_style_even: bool = False
    """ Timeloop-style even mappings must have, for each memory, at most two
        locations where storage nodes may be placed. """

    force_memory_hierarchy_order: bool = True
    """ If set to true, storage nodes for lower-level memories must be placed below
        storage nodes for higher-level memories. Note that this applies across different
        tensors, and the storage nodes for the same tensor always obey this ordering.
        """

    max_fused_loops_per_rank: int = 1
    """ The maximum number of fused loops in a pmapping for a given rank. """

    max_fused_loops: float | int = float("inf")
    """ The maximum total number of fused loops in a pmapping. """

    max_loops: float | int = float("inf")
    """ The maximum total loops in a pmapping. """

    max_loops_minus_ranks: float | int = float("inf")
    """ The maximum total loops in a pmapping minus the number of ranks. For example,
        3 means that the number of loops can be up to (the number of ranks + 3). """

    memory_limit: float | int = float("inf")
    """ The maximum memory limit for the mapper. """

    memory_limit_per_process: float | int = float("inf")
    """ The maximum memory limit per process for one of the mapper's processes. """

    time_limit: float | int = float("inf")
    """ The maximum time limit for the mapper. """

    time_limit_per_pmapping_template: float | int = float("inf")
    """ The maximum time limit per pmapping template. """
