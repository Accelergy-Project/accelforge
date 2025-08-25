from dataclasses import dataclass

from 
@dataclass(eq=True, frozen=True)
class Buffet:
    tensor: TensorName
    einsum: str
    level: str