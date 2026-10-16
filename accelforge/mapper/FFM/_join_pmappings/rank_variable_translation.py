from collections import defaultdict

from accelforge.frontend.renames import EinsumName, Rank, RankVariable, TensorName
from accelforge.frontend.workload import Workload
from accelforge.mapper.FFM._join_pmappings.compatibility import Compatibility


class RankVariableTranslation:
    """
    A loop at an index indexes into tensors below it. For each fused tensor, a loop is
    created that indexes all Einsums from the first to the last that has that tensor.

    If multiple tensors cause double indexing of a given tensor through the same loop in
    some other Einsum, then there is no way for the later compatiblity to match with
    this compatibility.

    SIMPLE RANKS ONLY. Can't block any invalid non-simple ranks
    """

    def __init__(self, workload: Workload, einsum_names: list[EinsumName]):
        self.tensor2last_einsum_using_it: dict[TensorName, int] = {}

        # Per Einsum for simple ranks only
        self.tensor_rank2rank_var: list[dict[tuple[TensorName, Rank], RankVariable]] = (
            []
        )

        self.rank_var2tensor_ranks: list[
            dict[RankVariable, list[tuple[TensorName, Rank]]]
        ] = []
        for i, name in enumerate(einsum_names):
            tensor_rank2rank_var: dict[tuple[TensorName, Rank], RankVariable] = {}
            rank_var2tensor_ranks: dict[RankVariable, list[tuple[TensorName, Rank]]] = (
                defaultdict(list)
            )
            for t in workload.einsums[name].tensor_accesses:
                self.tensor2last_einsum_using_it[t.name] = i
                for rank, rank_var in t.projection.items():
                    if rank_var in t.directly_indexing_rank_variables:
                        tensor_rank2rank_var[t.name, rank] = rank_var
                        rank_var2tensor_ranks[rank_var].append((t.name, rank))
            self.tensor_rank2rank_var.append(tensor_rank2rank_var)
            self.rank_var2tensor_ranks.append(dict(rank_var2tensor_ranks))

        self._loop_ranks_cache: dict[
            tuple[Compatibility, int], dict[tuple[int, TensorName], set[Rank]] | None
        ] = {}

    def compatible(
        self, left: Compatibility, right_index: int, right: Compatibility
    ) -> bool:
        """
        Returns True if left and right can join without failing at a later join.
        right_index is the index of right in the list of Einsums.
        """
        if (left, right_index) not in self._loop_ranks_cache:
            self._loop_ranks_cache[left, right_index] = self._loop_ranks(
                left, right_index
            )
        loop_tensor2ranks = self._loop_ranks_cache[left, right_index]
        if loop_tensor2ranks is None:
            return False
        for t in right.tensors:
            for i, loop in enumerate(t.loops):
                rank: Rank = str(loop.rank_name)
                if loop_tensor2ranks.get((i, t.name), {rank}) != {rank}:
                    return False
        return True

    def _loop_ranks(
        self, left: Compatibility, right_index: int
    ) -> dict[tuple[int, TensorName], set[Rank]] | None:
        """
        Returns {(loop index, tensor): the ranks of the tensor that the loop indexes}.
        
        Checks Einsums starting from right_index to the last one above any left tensor
        (e.g., for a fused loop, if left has one tensor stored under this loop, we check
        until the last consumer of this tensor).
        """
        loop_tensor2rank: dict[tuple[int, TensorName], Rank] = {}
        loop2last_use: dict[int, int] = {}
        for t in left.tensors:
            for i, loop in enumerate(t.loops):
                loop_tensor2rank[i, t.name] = str(loop.rank_name)
                loop2last_use[i] = max(
                    loop2last_use.get(i, -1), self.tensor2last_einsum_using_it.get(t.name, -1)
                )

        loop_tensor2ranks: dict[tuple[int, TensorName], set[Rank]] = {
            k: {rank} for k, rank in loop_tensor2rank.items()
        }
        for e in range(right_index, len(self.tensor_rank2rank_var)):
            for i, last_use in loop2last_use.items():
                if e > last_use:
                    continue
                rank_vars: set[RankVariable] = {
                    self.tensor_rank2rank_var[e][t, rank]
                    for (j, t), rank in loop_tensor2rank.items()
                    if j == i and (t, rank) in self.tensor_rank2rank_var[e]
                }
                if len(rank_vars) > 1:
                    return None
                for rank_var in rank_vars:
                    for t, rank in self.rank_var2tensor_ranks[e][rank_var]:
                        loop_tensor2ranks.setdefault((i, t), set()).add(rank)

        for (i, t), rank in loop_tensor2rank.items():
            if loop_tensor2ranks[i, t] != {rank}:
                return None
        return loop_tensor2ranks
