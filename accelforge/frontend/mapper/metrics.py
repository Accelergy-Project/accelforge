from enum import auto, Flag

from functools import reduce
from operator import or_


class Metrics(Flag):
    """
    Metrics used to optimize mappings.
    """

    LATENCY = auto()
    """ Latency. Minimize the amount of time taken to execute the workload. """

    ENERGY = auto()
    """ Energy. Minimize the amount of energy consumed by the workload. """

    DYNAMIC_ENERGY = auto()
    """ Dynamic energy. Minimize the amount of dynamic energy consumed by the workload. """

    LEAK_ENERGY = auto()
    """ Leak energy. Minimize the amount of leak energy consumed by the workload. """

    RESOURCE_USAGE = auto()
    """
    Resource usage. Minimize the amount of resources used by the workload. This
    objective is multivariate, and must consider every resource available to the
    hardware.
    """

    ACTIONS = auto()
    """Action counts."""

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
