from accelforge import Spec
from accelforge.frontend.mapper.metrics import Metrics
from accelforge.mapper.FFM.pmappings import MultiEinsumPmappings
from accelforge.mapper.FFM.mappings import Mappings
import accelforge.mapper.FFM._make_pmappings.make_pmappings as pmapper
from accelforge.frontend.workload import EinsumName
from accelforge.util._frozenset import oset


def make_copy_adapter(spec: Spec) -> MultiEinsumPmappings:
    """
    Return a MultiEinsumPmappings that simply allows two pmappings to be
    compatible iff they are already compatible.
    """
    return MultiEinsumPmappings(
        spec=spec,
        einsum2pmappings={},
        pmapping_objects={},
        einsum2jobs={},
        can_combine_multiple_runs=True,
        einsums_with_pmappings_generated=oset(),
        flattened_arches={},
        evaluated_specs={},
    )
