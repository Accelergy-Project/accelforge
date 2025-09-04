from fastfusion.frontend.mapping import Mapping
from fastfusion.mapper.FFM._pmapping_group import row2pmappings
from fastfusion.frontend.mapping import Mapping# NOFILL:, Fill

def make_mapping(row, einsum_names, rank_variable_bounds):
    pmappings = row2pmappings(row, einsum_names, rank_variable_bounds)
    newmapping = Mapping.from_pmappings(pmappings, rank_variable_bounds=rank_variable_bounds)
    # NOFILL: newmapping.clear_nodes_of_type(Fill)
    return newmapping