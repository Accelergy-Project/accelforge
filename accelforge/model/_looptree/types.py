"""
Contains shared classes of analysis.
"""

from dataclasses import dataclass
from typing import TypeAlias

from accelforge.frontend.mapping import TensorName
from accelforge.frontend.workload import EinsumName

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


@dataclass(eq=True, frozen=True)
class Compute:
    einsum: str
    level: str


@dataclass(eq=True, frozen=True)
class Network:
    """
    A logical network that delivers a tensor.
    """

    tensor: TensorName
    "The tensor held by the buffet."
    einsum: EinsumName
    "An einsum operating on the tensor."
    source: ComponentName
    "The source of data."
    destination: ComponentName
    "The destination of data."
    component: ComponentName = None
    "The network component used to deliver the data."
