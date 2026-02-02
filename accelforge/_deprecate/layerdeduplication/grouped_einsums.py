from collections.abc import Iterable

from bindings._looptree import LooptreeWorkload


type Id = int
type Name = str


class GroupOfSimilarEinsums[IdOrName: Id | Name]:
    def __init__(self, reference_einsum: Id, workload: LooptreeWorkload):
        self.reference_einsum = reference_einsum
        self.workload = workload
        self.similar_einsums_to_renaming = {}
        self.in_id = True

    def add_similar_einsum(
        self,
        similar_einsum: IdOrName,
        rank_renaming: IdOrName,
        tensor_renaming: IdOrName,
    ):
        self.similar_einsums_to_renaming[similar_einsum] = (
            rank_renaming,
            tensor_renaming,
        )

    @property
    def similar_einsums(self) -> Iterable[IdOrName]:
        return self.similar_einsums_to_renaming.keys()

    @property
    def get_renaming(
        self, other_einsum: IdOrName
    ) -> tuple[dict[IdOrName, IdOrName], dict[IdOrName, IdOrName]]:
        """Returns iterable over tuple `(rank_renaming, tensor_renaming)`"""
        try:
            return self.similar_einsums_to_renaming[other_einsum]
        except Exception as e:
            e.add_note(f"{other_einsum} not in group of similar Einsums.")
            raise

    @property
    def similar_einsums_and_renamings(
        self,
    ) -> Iterable[
        tuple[IdOrName, tuple[dict[IdOrName, IdOrName], dict[IdOrName, IdOrName]]]
    ]:
        """
        Returns iterable over tuple `(similar_einsum, renaming)`
        where `renaming` itself is `(rank_renaming, tensor_renaming).
        """
        return self.similar_einsums_and_renamings.items()

    def convert_id_to_name(self) -> "GroupOfSimilarEinsums[Name]":
        einsum_id_to_name = self.workload.EinsumIdToName()
        tensor_id_to_name = self.workload.DataSpaceIdToName()
        rank_id_to_name = self.workload.DimensionIdToName()

        grouped_einsums_in_name = GroupOfSimilarEinsums(
            einsum_id_to_name[self.reference_einsum], self.workload
        )
        self.in_id = False

        similar_einsums_to_renamings = self.get_einsums_similar_to_reference(
            self.reference_einsum
        )
        for einsum_id, renaming in similar_einsums_to_renamings.items():
            rank_renaming, tensor_renaming = renaming
            rank_renaming_in_names = {
                rank_id_to_name[k]: rank_id_to_name[v] for k, v in rank_renaming.items()
            }
            tensor_renaming_in_names = {
                tensor_id_to_name[k]: tensor_id_to_name[v]
                for k, v in tensor_renaming.items()
            }
            grouped_einsums_in_name.add_einsum_similar_to_reference(
                einsum_id_to_name[self.reference_einsum],
                einsum_id_to_name[einsum_id],
                rank_renaming_in_names,
                tensor_renaming_in_names,
            )

        return grouped_einsums_in_name
