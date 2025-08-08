from typing import Callable
from uuid import UUID
from fastfusion.mapper.FFM.joining.sim import SIM
from fastfusion.frontend.workload import EinsumName
from fastfusion.frontend.mapping import Mapping

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
