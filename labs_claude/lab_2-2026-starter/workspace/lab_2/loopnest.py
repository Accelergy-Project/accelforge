"""
Python-based loop nest and memory hierarchy simulation for educational purposes.

Examples
--------
```python
# Create memory levels
DRAM = MemoryLevel("DRAM")
GLB = MemoryLevel("GLB")

# Starting tensor (a tile in DRAM that holds everything)
A = TileInMemory(DRAM, "A", shape=((0, 64), (0, 64)))
B = TileInMemory(DRAM, "B", shape=((0, 64), (0, 64)))
Z = TileInMemory(DRAM, "Z", shape=((0, 64), (0, 64)), is_output=True)

# A loop nest that performs Z = A x B

for m in range(64):
    for n in range(64):
        for k in range(64):
            with GLB.allocate(A[m, k]) as A_in_GLB:
                with GLB.allocate(B[k, n]) as B_in_GLB:
                    with GLB.allocate(Z[m, n]) as Z_in_GLB:
                        Z_in_GLB[0,0] += A_in_GLB[0,0] * B_in_GLB[0,0]
```

Now with tiling:

```python
# Create memory levels
DRAM = MemoryLevel("DRAM")
GLB = MemoryLevel("GLB")

# Starting tensor (a tile in DRAM that holds everything)
A = TileInMemory(DRAM, "A", shape=((0, 64), (0, 64)))
B = TileInMemory(DRAM, "B", shape=((0, 64), (0, 64)))
Z = TileInMemory(DRAM, "Z", shape=((0, 64), (0, 64)), is_output=True)

# A loop nest that performs Z = A x B

for m1 in range(16):
    for n1 in range(16):
        for k1 in range(16):
            with GLB.allocate(A[4*m1:4*(m1+1), 4*k1:4*(k1+1)]) as A_in_GLB:
                with GLB.allocate(B[4*k1:4*(k1+1), 4*n1:4*(n1+1)]) as B_in_GLB:
                    with GLB.allocate(Z[4*m1:4*(m1+1), 4*n1:4*(n1+1)]) as Z_in_GLB:
                        for m0 in range(4):
                            for n0 in range(4):
                                for k0 in range(4):
                                    Z_in_GLB[m0,n0] += A_in_GLB[m0,k0] * B_in_GLB[k0,n0]
```
"""

from collections import defaultdict
from collections.abc import Sequence
from functools import reduce
from operator import mul

import numpy as np


class Bounded:
    def __init__(self, shape: tuple[int]):
        self.shape = shape
        self._size = None
    
    @property
    def size(self) -> int:
        if self._size is None:
            self._size = reduce(mul, self.shape, 1)
        return self._size
    
    def _set_size(self, size: int):
        self._size = size
    
    def within_shape(self, indices: Sequence[int | slice]) -> bool:
        if len(indices) != len(self.shape):
            return False

        for idx, upper in zip(indices, self.shape):
            if isinstance(idx, slice):
                start = idx.start or 0
                stop = idx.stop or start + 1
                if start < 0 or stop > upper:
                    return False
            elif isinstance(idx, int):
                if idx < 0 or idx >= upper:
                    return False
            else:
                return False

        return True


class MemoryLevel:
    """A memory level in the memory hierarchy."""

    def __init__(self, name: str):
        """
        Create a memory level.
        
        Parameters
        ----------
        name : str
            The name of the memory level.
        """
        self.name = name
        self.reads = defaultdict(lambda: 0)
        self.writes = defaultdict(lambda: 0)

        self.usage = defaultdict(lambda: 0)

    def allocate(self, tile) -> "TileInMemory":
        """Allocate a tile in this memory level."""
        if isinstance(tile, TileInMemory):
            return Allocation(
                tile.memory_level,
                TileInMemory(self, tile.tensor_name, tile.shape, tile.is_output)
            )
        elif isinstance(tile, Tensor):
            return Allocation(
                None,
                TileInMemory(self, tile.tensor_name, tile.shape, tile.is_output)
            )


class Allocation:
    def __init__(
        self,
        parent_memory_level: MemoryLevel | None,
        tile: "TileInMemory"
    ):
        self.parent_memory_level = parent_memory_level
        self.tile = tile

    def __enter__(self):
        self.tile.memory_level.usage[self.tile.tensor_name] = max(
            self.tile.memory_level.usage[self.tile.tensor_name],
            self.tile.size
        )

        if self.parent_memory_level:
            # Write into this memory level
            self.tile.memory_level.writes[self.tile.tensor_name] += self.tile.nonzero_count()
        return self.tile
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.tile.is_output and self.parent_memory_level:
            # Read from tile memory level...
            self.tile.memory_level.reads[self.tile.tensor_name] += self.tile.nonzero_count()
            # ... and write back to parent memory level
            self.parent_memory_level.writes[self.tile.tensor_name] += self.tile.nonzero_count()


class Tensor(Bounded):
    """A tensor in the computation."""

    def __init__(
        self,
        tensor_name: str,
        shape: tuple[int],
        is_output: bool = False
    ):
        """
        Create a tensor.

        Parameters
        ----------
        tensor_name : str
            The name of the tensor.
        shape: tuple[int]
            The maximum (exclusive) coordinate of the tensor in each rank.
        is_output : bool, optional
            Whether this tensor is an output in the Einsum, by default False.
        """
        super().__init__(shape)
        self.tensor_name = tensor_name
        self.is_output = is_output

    def nonzero_count(self):
        return self.size


class TileInMemory(Bounded):
    """A tile of a tensor stored in a specific memory level."""

    def __init__(
        self,
        memory_level: MemoryLevel,
        tensor_name: str,
        shape: tuple[int],
        is_output: bool = False,
        should_count_read = None
    ):
        """
        A tile of a tensor stored in a specific memory level.

        Parameters
        ----------
        memory_level : MemoryLevel
            The memory level where this tile is stored.
        tensor_name : str
            The name of the tensor.
        shape: tuple[tuple[int, int]]
            The maximum (exclusive) coordinate of the tile in each rank.
        is_output : bool, optional
            Whether this tensor is an output in the Einsum, by default False.
        """
        super().__init__(shape)
        self.memory_level = memory_level
        self.tensor_name = tensor_name
        self.is_output = is_output
        if should_count_read is not None:
            self.should_count_read = should_count_read
        elif is_output:
            self.should_count_read = np.zeros(shape)
        else:
            self.should_count_read = np.ones(shape)

    def __getitem__(self, indices: Sequence[int | slice]):
        if not self.within_shape(indices):
            raise IndexError("Indices out of shape for this tile.")

        size = 1
        subtile_shape = []
        for idx, upper in zip(indices, self.shape):
            if isinstance(idx, slice):
                start = idx.start or 0
                stop = idx.stop or upper
                subtile_shape.append(stop-start)
                size *= (stop - start)
            elif isinstance(idx, int):
                subtile_shape.append(1)
            else:
                raise TypeError("Indices must be int or slice.")

        # simulate read
        self.memory_level.reads[self.tensor_name] += self.should_count_read[indices].sum().item()

        tile = TileInMemory(
            self.memory_level,
            tensor_name=self.tensor_name,
            shape=tuple(subtile_shape),
            is_output=self.is_output,
            should_count_read=self.should_count_read[indices]
        )
        tile._set_size(size)
        return tile

    def __setitem__(self, indices: Sequence[int | slice], value):
        if not self.within_shape(indices):
            raise IndexError("Indices out of shape for this tile.")

        size = 1
        subtile_shape = []
        for idx in indices:
            if isinstance(idx, slice):
                start = idx.start or 0
                stop = idx.stop or start + 1
                subtile_shape.append((start, stop))
                size *= (stop - start)
            elif isinstance(idx, int):
                subtile_shape.append((idx, idx + 1))
            else:
                raise TypeError("Indices must be int or slice.")

        self.should_count_read[indices] = 1

        # simulate write
        self.memory_level.writes[self.tensor_name] += size

    def __mul__(self, other):
        pass

    def __rmul__(self, other):
        pass

    def __add__(self, other):
        pass

    def __radd__(self, other):
        pass

    def nonzero_count(self):
        return self.should_count_read.sum().item()


def get_actual_throughput(bandwidth, throughput, memory, ops):
    accesses = sum(memory.reads.values()) + sum(memory.writes.values())
    overall_latency = max(
        accesses / bandwidth,
        ops/throughput,
    )
    return ops/overall_latency
