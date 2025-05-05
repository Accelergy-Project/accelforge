from enum import auto, Flag

from functools import reduce
from operator import or_


class Metrics(Flag):
    LATENCY = auto()
    ENERGY = auto()
    # OCCUPANCY = auto()
    OFF_CHIP_ACCESSES = auto()
    OP_INTENSITY = auto()
    DEBUG = auto()
    VALID = auto()
    PER_COMPONENT_ACCESSES_ENERGY = auto()
    MAPPING = auto()

    @classmethod
    def all_metrics(cls):
        return reduce(or_, iter(cls), cls.LATENCY) ^ Metrics.OP_INTENSITY

