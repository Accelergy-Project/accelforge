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
