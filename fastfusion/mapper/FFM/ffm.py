from typing import Callable
from uuid import UUID
from fastfusion.frontend import architecture
from fastfusion.frontend.specification import Specification
from fastfusion.mapper.FFM.compress_pmappings import compress_einsum2pmappings, decompress_pmappings
from fastfusion.mapper.FFM.exploration.mapper_multi_einsum import get_sims
from fastfusion.mapper.FFM.joining.sim import SIM
from fastfusion.frontend.workload import EinsumName
from fastfusion.frontend.mapping import Mapping
from fastfusion.mapper.FFM.joining.simexplore import join_sims
from fastfusion.mapper.FFM.pareto.df_convention import MAPPING_COLUMN
from fastfusion.mapper.FFM.pareto.partial_mappings import PartialMappings, row2pmappings
from fastfusion.mapper.FFM.exploration.mapper_multi_einsum import get_rank_variable_bounds_for_all_einsums
from fastfusion.accelerated_imports import pd

class MultiEinsumPmappings:
    def __init__(
        self,
        einsum2pmappings: dict[EinsumName, list[SIM]],
        pmapping_objects: dict[EinsumName, dict[UUID, Mapping]],
        resource2capacity: dict[str, int],
    ):
        self.einsum2pmappings: dict[EinsumName, list[SIM]] = einsum2pmappings
        self.pmapping_objects: dict[EinsumName, dict[UUID, Mapping]] = pmapping_objects
        self.resource2capacity = resource2capacity

    def __or__(self, other: "MultiEinsumPmappings"):
        for einsum_name, pmappings in other.einsum2pmappings.items():
            self.einsum2pmappings.setdefault(einsum_name, []).extend(pmappings)
        for resource, capacity in other.resource2capacity.items():
            if resource not in self.resource2capacity:
                self.resource2capacity[resource] = capacity
            if self.resource2capacity[resource] != other.resource2capacity[resource]:
                raise ValueError(
                    f"Resource {resource} has different capacities in different "
                    f"specifications: {self.resource2capacity[resource]} and "
                    f"{other.resource2capacity[resource]}."
                )
        self.pmapping_objects.update(other.pmapping_objects)
        return self
    
    def filter(self, filter_lambda: Callable[[SIM], bool], einsum_names: list[EinsumName] | None = None):
        if einsum_names is None:
            einsum_names = list(self.einsum2pmappings.keys())
        for einsum_name in einsum_names:
            self.einsum2pmappings[einsum_name] = [
                pm for pm in self.einsum2pmappings[einsum_name]
                if filter_lambda(pm)
            ]
            
    def drop_einsums(self, *einsum_names: EinsumName):
        for einsum_name in einsum_names:
            del self.einsum2pmappings[einsum_name]
            del self.pmapping_objects[einsum_name]


class MappingFromRow:
    def __init__(self, row: pd.Series, spec: Specification, rank_variable_bounds: dict[str, dict[str, int]]):
        self.row = row
        self.spec = spec
        self.rank_variable_bounds = rank_variable_bounds

    def __call__(self) -> Mapping:
        return row2mapping(self.row, self.spec, self.rank_variable_bounds)

    def render(self) -> str:
        return self().render()


def make_pmappings(
    spec: Specification, einsum_names: list[EinsumName] | None = None, tagger = None,
) -> MultiEinsumPmappings:
    flattened_arch = spec.get_flattened_architecture()
    sims, pmapping_objects = get_sims(
        spec,
        flattened_arch,
        tagger=tagger,
        metrics=spec.mapper_ffm.metrics,
        einsum_names=einsum_names
    )
    resource2capacity = {}
    for l in flattened_arch:
        if isinstance(l, architecture.Memory):
            resource2capacity[l.name] = l.attributes.size
    return MultiEinsumPmappings(sims, pmapping_objects, resource2capacity)

def row2mapping(row: pd.Series, spec: Specification, rank_variable_bounds: dict[str, dict[str, int]]) -> Mapping:
    return Mapping.from_pmappings(row2pmappings(row, spec.workload.einsum_names, rank_variable_bounds), rank_variable_bounds=rank_variable_bounds)


def join_pmappings(spec: Specification, pmappings: MultiEinsumPmappings) -> PartialMappings:
    compressed, decompress_data = compress_einsum2pmappings(pmappings.einsum2pmappings)
    joined = join_sims(
        compressed,
        spec,
        pmappings.resource2capacity
    )
    joined = decompress_pmappings(joined, decompress_data)

    for einsum_name in pmappings.einsum2pmappings:
        col = f"{einsum_name}_{MAPPING_COLUMN}"
        joined.data[col] = joined.data[col].apply(
            lambda x: pmappings.pmapping_objects[einsum_name][x]
        )

    rank_variable_bounds = get_rank_variable_bounds_for_all_einsums(spec)
    joined.data[MAPPING_COLUMN] = joined.data.apply(lambda row: MappingFromRow(row, spec, rank_variable_bounds), axis=1)
    return joined