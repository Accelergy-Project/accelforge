from typing import Callable
from uuid import UUID
from fastfusion.mapper.FFM._join_pmappings.sim import SIM
from fastfusion.frontend.workload import EinsumName
from fastfusion.frontend.mapping import Mapping
from fastfusion.mapper.FFM._make_pmappings.mapper_one_einsum.mapper_job import Job


class MultiEinsumPmappings:
    def __init__(
        self,
        einsum2pmappings: dict[EinsumName, list[SIM]],
        pmapping_objects: dict[EinsumName, dict[UUID, Mapping]],
        resource2capacity: dict[str, int],
        einsum2jobs: dict[EinsumName, list[Job]],
        can_combine_multiple_runs: bool,
    ):
        self.einsum2pmappings: dict[EinsumName, list[SIM]] = einsum2pmappings
        self.pmapping_objects: dict[EinsumName, dict[UUID, Mapping]] = pmapping_objects
        self.resource2capacity = resource2capacity
        self.mapper_jobs = einsum2jobs
        self.can_combine_multiple_runs = can_combine_multiple_runs

    def __or__(self, other: "MultiEinsumPmappings"):
        if not self.can_combine_multiple_runs or not other.can_combine_multiple_runs:
            raise ValueError(
                "Must call make_pmappings with can_combine_multiple_runs=True to combine pmappings "
                "from multiple runs."
            )
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
        for einsum_name, jobs in other.mapper_jobs.items():
            self.mapper_jobs.setdefault(einsum_name, []).extend(jobs)
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

    def pmapping_keep_rates(self, per_einsum: bool = False) -> dict[EinsumName, dict[str, float]] | dict[str, float]:
        result = {}
        einsum2npmappings = self.total_pmappings(per_einsum=True)

        for einsum_name, jobs in self.mapper_jobs.items():
            cur_result = result.setdefault(einsum_name, {})
            for job in jobs:
                for cause, keep_rate in job.pmapping_keep_rates.items():
                    cur_result.setdefault(cause, 0)
                    cur_result[cause] += job.total_pmappings * keep_rate

        if per_einsum:
            for einsum_name, npmappings in einsum2npmappings.items():
                for cause, keep_rate in result[einsum_name].items():
                    result[einsum_name][cause] = keep_rate / npmappings
        else:
            new_result = {}
            total_pmappings = sum(einsum2npmappings.values())
            for einsum_name, keep_rates in result.items():
                for cause, keep_rate in keep_rates.items():
                    new_result.setdefault(cause, 0)
                    new_result[cause] += keep_rate / total_pmappings
            result = new_result

        return result

    def total_pmappings(self, per_einsum: bool = False) -> int | dict[EinsumName, int]:
        result = {
            einsum_name: sum(job.total_pmappings for job in jobs)
            for einsum_name, jobs in self.mapper_jobs.items()
        }
        if per_einsum:
            return result
        return sum(result.values())

    def valid_pmappings(self, per_einsum: bool = False) -> int | dict[EinsumName, int]:
        result = {
            einsum_name: sum(job.valid_pmappings for job in jobs)
            for einsum_name, jobs in self.mapper_jobs.items()
        }
        if per_einsum:
            return result
        return sum(result.values())

    def pareto_optimal_pmappings(self, per_einsum: bool = False) -> int | dict[EinsumName, int]:
        result = {
            einsum_name: sum(len(p.mappings.data) for p in pmappings)
            for einsum_name, pmappings in self.einsum2pmappings.items()
        }
        if per_einsum:
            return result
        return sum(result.values())

    def evaluated_pmappings(self, per_einsum: bool = False) -> int | dict[EinsumName, int]:
        result = {
            einsum_name: sum(job.evaluated_pmappings for job in jobs)
            for einsum_name, jobs in self.mapper_jobs.items()
        }
        if per_einsum:
            return result
        return sum(result.values())