from bindings.looptree import (LooptreeWorkload,
                               LooptreeWorkloadDependencyAnalyzer)


class EquivalentGroups:
    def __init__(self):
        self.group_id_to_ranks = {}
        self.rank_to_group_id = {}

    @staticmethod
    def from_workload(workload: LooptreeWorkload,
                      analyzer: LooptreeWorkloadDependencyAnalyzer):
        einsum_id_to_name = workload.einsum_id_to_name()

        groups = EquivalentGroups()

        seen_ranks = set()
        for einsum_id in einsum_id_to_name:
            for rank_id in workload.einsum_ospace_dimensions(einsum_id):
                equiv_ranks = analyzer.equivalent_dimensions(einsum_id,
                                                             rank_id)
                equiv_ranks = frozenset(equiv_ranks)
                if equiv_ranks not in seen_ranks:
                    seen_ranks.add(equiv_ranks)
                    group_id = len(groups.group_id_to_ranks)
                    groups.group_id_to_ranks[group_id] = equiv_ranks
                    for r in equiv_ranks:
                        groups.rank_to_group_id[r] = group_id

        return groups