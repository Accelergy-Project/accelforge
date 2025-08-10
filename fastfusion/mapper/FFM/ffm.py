from fastfusion.frontend import architecture
from fastfusion.frontend.specification import Specification
from fastfusion.mapper.FFM.interface.pmappings import MultiEinsumPmappings
from fastfusion.mapper.FFM.interface.mappings import Mappings
from fastfusion.mapper.FFM.joining.compress_pmappings import compress_einsum2pmappings, decompress_pmappings
from fastfusion.mapper.FFM.exploration.mapper_multi_einsum import get_sims
from fastfusion.frontend.workload import EinsumName
from fastfusion.frontend.mapping import Mapping
from fastfusion.mapper.FFM.joining.simexplore import join_sims
from fastfusion.mapper.FFM.pareto.df_convention import MAPPING_COLUMN
from fastfusion.mapper.FFM.pareto.partial_mappings import PartialMappings, row2pmappings
from fastfusion.mapper.FFM.exploration.mapper_multi_einsum import get_rank_variable_bounds_for_all_einsums
from fastfusion.accelerated_imports import pd


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
        metrics=spec.mapper.ffm.metrics,
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
        col = f"{einsum_name}\0{MAPPING_COLUMN}"
        joined.data[col] = joined.data[col].apply(
            lambda x: pmappings.pmapping_objects[einsum_name][x]
        )

    rank_variable_bounds = get_rank_variable_bounds_for_all_einsums(spec)
    joined.data[f"Total\0{MAPPING_COLUMN}"] = joined.data.apply(lambda row: MappingFromRow(row, spec, rank_variable_bounds), axis=1)
    # Fill nans with 0. We might get missing columns for some mapping entries if there
    # are energy entries for some pmappings but not others (e.g., one pmapping accesses
    # DRAM while another doesn't.)
    joined._data = joined.data.fillna(0)
    return Mappings(spec, list(pmappings.einsum2pmappings.keys()), joined.data)