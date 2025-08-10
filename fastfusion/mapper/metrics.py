from enum import auto, Flag

from functools import reduce
from operator import or_


class Metrics(Flag):
    LATENCY = auto()
    ENERGY = auto()
    RESERVATIONS = auto()
    RESOURCE_USAGE = auto()

    @classmethod
    def all_metrics(cls):
        return reduce(or_, iter(cls), cls.LATENCY)

