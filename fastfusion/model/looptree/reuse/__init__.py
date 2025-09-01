from dataclasses import dataclass
from fastfusion.frontend.mapping import TensorName
from fastfusion.frontend.workload.workload import EinsumName

@dataclass(eq=True, frozen=True)
class Buffet:
    """
    A logical buffer that stores a tensor, an einsum operating on it, and the
    level the buffer exists on in hardware.

    :param tensor:  The tensor held by the buffet.
    :param einsum:  An einsum operating on the tensor.
    :param level:   The abstract hardware level the buffet resides in.

    :type tensor:   TensorName
    :type einsum:   EinsumName
    :type level:    str
    """
    tensor: TensorName
    einsum: EinsumName
    level: str