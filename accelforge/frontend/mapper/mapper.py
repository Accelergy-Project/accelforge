from accelforge.util._basetypes import EvalableModel
from accelforge.frontend.mapper.ffm import FFM


class Mapper(EvalableModel):
    ffm: FFM = FFM()
    """ Fast and Fusiest Mapper configuration. Currently the only supported mapper. """
