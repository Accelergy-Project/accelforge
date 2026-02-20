from accelforge.frontend.mapping import (
    Mapping,
)

from accelforge.frontend.mapping import (
    Mapping,
    TensorHolder,
)
from accelforge.frontend.workload import (
    TensorName,
)
from accelforge.model._looptree.types import Buffet, Compute


class DataMovementConnections:
    """
    For querying connection between Buffet or Compute
    (e.g., which Buffet supplies this Compute).
    """
    def __init__(self):
        self.src_to_dst = {}
        self.dst_to_src = {}

    def get_src(self, dst: Buffet | Compute) -> Buffet | None:
        return self.dst_to_src[dst]

    def get_dst(self, src: TensorHolder) -> Buffet | Compute:
        return self.src_to_dst[src]

    @property
    def sources(self) -> list[TensorHolder]:
        return list(self.src_to_dst.keys())

    @property
    def destinations(self) -> list[TensorHolder]:
        return list(self.dst_to_src.keys())

    @classmethod
    def from_pmapping(cls, pmapping):
        einsum_name = pmapping[-1].einsum
        src_to_dst = {}
        dst_to_src = {}
        tensor_to_last_src = {}
        for node in pmapping:
            if isinstance(node, TensorHolder):
                for tensor in node.tensors:
                    buffet = Buffet(tensor, einsum_name, node.component)
                    if tensor in tensor_to_last_src:
                        src = tensor_to_last_src[tensor]
                    else:
                        src = None
                    tensor_to_last_src[tensor] = buffet
                    dst_to_src[buffet] = src
                    if src is not None:
                        src_to_dst[src] = buffet
                    src_to_dst[buffet] = None
        result = DataMovementConnections()
        result.src_to_dst = src_to_dst
        result.dst_to_src = dst_to_src
        return result


def get_tensor_to_backer_id(mapping: Mapping):
    tensor_to_ids: dict[TensorName, set[int]] = {}
    for node in mapping:
        if isinstance(node, TensorHolder):
            for tensor in node.tensors:
                if tensor in tensor_to_ids:
                    continue
                tensor_to_ids[tensor] = id(node)
    return tensor_to_ids

