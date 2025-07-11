import copy
import re
from typing import Optional
from fastfusion.frontend.mapping import Iteration, Mapping, Nested, Sequential, Storage
from fastfusion.mapper.FFM.pareto import MAPPING_COLUMN

from fastfusion.mapper.FFM.pareto import row2pmappings
# importlib.reload(fastfusion.visualization.interactive)
# importlib.reload(fastfusion.frontend.mapping)
from fastfusion.frontend.mapping import Mapping, Fill, Reservation

def make_mapping(row, einsum_names, rank_variable_bounds):
    pmappings = row2pmappings(row, einsum_names, rank_variable_bounds)
    newmapping = Mapping.from_pmappings(pmappings, rank_variable_bounds=rank_variable_bounds)
    newmapping.clear_nodes_of_type(Fill)
    return newmapping