from typing import Any, Annotated, Literal

from fastfusion.frontend.mapper.metrics import Metrics
from fastfusion.util._basetypes import ParsableModel
from fastfusion._version import assert_version, __version__


class FFM(ParsableModel):
    """Configuration for the Fast and Fusiest Mapper."""

    # version: Annotated[str, assert_version] = __version__
    # """ Version """

    metrics: Metrics = Metrics.ENERGY
    """ Metrics used to optimize mappings. """

    info_metrics: Metrics = Metrics.all_metrics()
    """Metrics to be reported for final mappings."""

    _timeloop_style_even: bool = False
    """ Timeloop-style even mappings must have, for each memory, at most two
        locations where storage nodes may be placed. """

    force_memory_hierarchy_order: bool = True
    """
    If set to true, storage nodes for lower-level memories must be placed below storage
    nodes for higher-level memories. For example, all MainMemory storage nodes must go
    above all LocalBuffer storage nodes.

    This constraint always applies to same-tensor storage nodes (e.g., MainMemory
    reusing Output must go above LocalBuffer reusing Output); turning it off will
    permit things like MainMemory reusing Output going above LocalBuffer reusing
    Input.
    """

    max_fused_loops_per_rank_variable: int = 1
    """ The maximum number of fused loops in a pmapping for a given rank variable. """

    max_fused_loops: float | int = float("inf")
    """ The maximum total number of fused loops in a pmapping. """

    max_loops: float | int = float("inf")
    """ The maximum total loops in a pmapping. """

    max_loops_minus_ranks: float | int = float("inf")
    """
    The maximum total loops in a pmapping minus the number of ranks. For example,
    3 means that the number of loops can be up to (the number of ranks + 3).
    """

    _can_lower_outermost_memory: bool = False
    """
    Whether the storage node of outermost memory can be lowered. If set to True, the
    mapper may exchange tiles of tensors via the outermost memory, instead of storing
    full tensors. Set this to True to explore reducing outermost memory usage.

    TODO: Also need to explore putting loops above the outermost memory then. This is
    currently private because we may want to have a catch-all term like
    "save_outermost_memory_usage".

    """

    memory_limit: float | int = float("inf")
    """ The maximum memory limit for the mapper. """

    memory_limit_per_process: float | int = float("inf")
    """ The maximum memory limit per process for one of the mapper's processes. """

    time_limit: float | int = float("inf")
    """ The maximum time limit for the mapper. """

    time_limit_per_pmapping_template: float | int = float("inf")
    """ The maximum time limit per pmapping template. """

    max_pmapping_templates_per_einsum: float | int = float("inf")
    """
    The maximum number of pmapping templates per Einsum. Once this many templates are
    generated, the mapper will stop generating more. This is useful for debugging (why
    are so many templates being generated?).
    """

    _count_option_for_mapsapce_size_evaluation: tuple[
        Literal[
            "redundant_loop_orders",
            "non_helpful_loops_for_loop_orders",
            "non_helpful_tile_shapes",
            "redundant_dataplacements",
        ]
    ] = ()
