import accelforge as af
from accelforge.tracegen.tracemapping import trace_iterations


class TestUnfused:
    def test_unfused_matvecs_to_simple(self):
        spec = af.Spec.from_yaml(
            af.examples.arches.simple,
            af.examples.workloads.matvecs,
            af.examples.mappings.unfused_matvecs_to_simple,
            jinja_parse_data={"N_EINSUMS": 2, "KN": 4, "MainMemoryEnergy": 10}
        )
        result = spec.evaluate_mapping()
        trace_iterations(result.mapping(), spec)
