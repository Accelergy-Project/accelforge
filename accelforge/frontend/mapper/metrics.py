from enum import auto, Flag

from functools import reduce
from operator import or_


class Metrics(Flag):
    """
    Metrics used to optimize mappings or reported by model.
    """

    LATENCY = auto()
    """The amount of time taken to execute the workload. """

    ENERGY = auto()
    """The amount of energy consumed by the workload. """

    DYNAMIC_ENERGY = auto()
    """The amount of dynamic energy consumed by the workload. """

    LEAK_ENERGY = auto()
    """The amount of leak energy consumed by the workload. """

    RESOURCE_USAGE = auto()
    """
    The amount of resources used by the workload.

    When used as a mapper objective, this objective is multivariate, and must
    consider every resource available to the hardware.
    """

    ACTIONS = auto()
    """Action counts."""

    DETAILED_MEMORY_USAGE = auto()
    """
    Memory usage broken down by tensor and Einsum.
    """

    @classmethod
    def all_metrics(cls):
        return reduce(or_, iter(cls), cls.LATENCY)

    def includes_leak_energy(self) -> bool:
        """Returns True if the metrics include leak energy, either alone or as part of
        total energy. False otherwise."""
        return self & (Metrics.ENERGY | Metrics.LEAK_ENERGY)

    def includes_dynamic_energy(self) -> bool:
        """Returns True if the metrics include dynamic energy, either alone or as part
        of total energy. False otherwise."""
        return self & (Metrics.ENERGY | Metrics.DYNAMIC_ENERGY)
