from fastfusion.util.basetypes import ParsableModel
from fastfusion.frontend.mapper.ffm import FFM


class Mapper(ParsableModel):
    ffm: FFM = FFM()
