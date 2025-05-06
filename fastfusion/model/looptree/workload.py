from collections.abc import Iterable


class Einsum:
    def __init__(self):
        self.rank_variables: list[str] = []


class Tensor:
    def __init__(self):
        self.ranks: list[str] = []
        pass


class LooptreeWorkload:
    def __init__(self):
        self.einsums: dict[str, Einsum] = {}
        self.tensors: dict[str, Tensor] = {}

        self._tensors_read_by_einsum: dict[str, set[str]] = {}
        self._tensors_written_by_einsum: dict[str, set[str]] = {}

        self._einsums_reading_tensor: dict[str, set[str]] = {}
        self._einsums_writing_tensor: dict[str, set[str]] = {}

    def add_einsum(self, einsum_name: str):
        if einsum_name in self.einsums:
            raise KeyError(f'{einsum_name} already exists')
        self.einsums[einsum_name] = Einsum()
        self._tensors_written_by_einsum[einsum_name] = set()
        self._tensors_read_by_einsum[einsum_name] = set()

    def set_einsum_rank_variables(self,
                                  einsum_name: str,
                                  rank_variables: Iterable[str]):
        try:
            self.einsums[einsum_name].rank_variables = list(rank_variables)
        except KeyError:
            raise KeyError(f'Einsum {einsum_name} not in workload')

    def set_einsum_shape(self, einsum_name: str, shape):
        raise NotImplementedError()

    def add_tensor(self, tensor_name: str):
        if tensor_name in self.einsums:
            raise KeyError(f'{tensor_name} already exists')
        self.tensors[tensor_name] = Tensor()
        self._einsums_reading_tensor[tensor_name] = set()
        self._einsums_writing_tensor[tensor_name] = set()

    def set_tensor_ranks(self, tensor_name: str, ranks: Iterable[str]):
        try:
            self.tensors[tensor_name].ranks = list(ranks)
        except KeyError:
            raise KeyError(f'Tensor {tensor_name} not in workload')

    def set_tensor_shape(self, tensor_name: str, shape):
        raise NotImplementedError()

    def set_projection(self,
                       einsum_name: str,
                       tensor_name: str,
                       projection,
                       is_output: bool):
        if is_output:
            self._tensors_written_by_einsum[einsum_name].add(tensor_name)
            self._einsums_writing_tensor[tensor_name].add(einsum_name)
        else:
            self._tensors_read_by_einsum[einsum_name].add(tensor_name)
            self._einsums_reading_tensor[tensor_name].add(einsum_name)

        raise NotImplementedError()

    def get_einsum_with_name(self, einsum_name: str):
        return self.einsums[einsum_name]

    def get_tensor_with_name(self, tensor_name: str):
        return self.tensors[tensor_name]
