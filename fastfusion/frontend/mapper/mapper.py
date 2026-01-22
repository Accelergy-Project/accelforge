from fastfusion.util._basetypes import ParsableModel
from fastfusion.frontend.mapper.ffm import FFM


class Mapper(ParsableModel):
    ffm: FFM = FFM()
    """ Fast and Fusiest Mapper configuration. Currently the only supported mapper. """
