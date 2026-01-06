"""
Contains shared classes of analysis.
"""

from dataclasses import dataclass
from typing import TypeAlias

from fastfusion.frontend.mapping import TensorName
from fastfusion.frontend.workload import EinsumName

ComponentName: TypeAlias = str


@dataclass(eq=True, frozen=True)
class Buffet:
    """
    A logical buffer that stores a tensor, an einsum operating on it, and the
    level the buffer exists on in hardware.
    """

    tensor: TensorName
    "The tensor held by the buffet."
    einsum: EinsumName
    "An einsum operating on the tensor."
    level: ComponentName
    "The abstract hardware level the buffet resides in."
