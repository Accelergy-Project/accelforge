from enum import auto, Flag

from functools import reduce
from operator import or_


class Metrics(Flag):
    LATENCY = auto()
    ENERGY = auto()
    RESERVATIONS = auto()
    # OCCUPANCY = auto()
    OP_INTENSITY = auto()
    DEBUG = auto()
    PER_COMPONENT_ACCESSES_ENERGY = auto()
    MAPPING = auto()
    RESOURCE_USAGE = auto()

    @classmethod
    def all_metrics(cls):
        return reduce(or_, iter(cls), cls.LATENCY)

