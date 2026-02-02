"""Applies the binding layer into one that can be used for later analysis,"""
from accelforge.frontend.binding import Binding
from accelforge.frontend.mapping import Mapping
from accelforge.frontend.workload import Workload

from accelforge.model._looptree.reuse.isl.mapping_to_isl.analyze_mapping import (
    occupancies_from_mapping,
    MappingAnalysisResult
)



def apply_binding(binding: Binding, mapping: Mapping, workload: Workload):
    """
    Given a mapping, apply the mapping from logical components onto physical
    components.
    """
    map_analysis: MappingAnalysisResult = occupancies_from_mapping(
        mapping, workload
    )
    for buffet, occ in map_analysis.buffet_to_occupancy.items():
        print(occ)
    print(binding)
