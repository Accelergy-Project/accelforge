from dataclasses import dataclass
from fastfusion.frontend.mapping import TensorName

@dataclass(eq=True, frozen=True)
class Buffet:
    tensor: TensorName
    einsum: str
    level: str