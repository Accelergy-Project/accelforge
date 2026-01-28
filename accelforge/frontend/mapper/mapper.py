from accelforge.util._basetypes import ParsableModel
from accelforge.frontend.mapper.ffm import FFM


class Mapper(ParsableModel):
    ffm: FFM = FFM()
    """ Fast and Fusiest Mapper configuration. Currently the only supported mapper. """
