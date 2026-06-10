from math import ceil, log2
from hwcomponents import ComponentModel, action, ActionCost
from hwcomponents.scaling import linear, quadratic, reciprocal
from hwcomponents_library.library.aladdin import AladdinComparator, AladdinCounter
from hwcomponents_neurosim import FlipFlop


class ZeroComparator(ComponentModel):
    """
    Counts the number of zeros in a list of values. Includes a flag for each zero.

    Based on the zero gating logic in the paper: A Programmable Heterogeneous
    Microprocessor Based on Bit-Scalable In-Memory Computing, by Hongyang Jia, Hossein
    Valavi, Yinqi Tang, Jintao Zhang, and Naveen Verma, JSSC 2020
    10.1109/JSSC.2020.2987714

    Parameters
    ----------
    n_comparators: int
        The number of comparators to include.
    n_bits: int
        The number of bits of each comparator.
    tech_node: str
        The technology node of the comparators.
    voltage: float
        The voltage of the comparators.
    """

    priority = 0.5

    def __init__(
        self,
        n_comparators: int,
        n_bits: int,
        tech_node: str,
        voltage: float = 0.85,
        cycle_period: float = 1e-9,
    ):
        self.n_comparators = n_comparators
        self.n_bits = n_bits

        # Scale up the comparator to handle all the comparators
        self.comparator = AladdinComparator(
            width=n_bits,
            tech_node=tech_node,
        )
        self.comparator.energy_scale *= n_comparators
        self.comparator.area_scale *= n_comparators

        # Flip flops are used one bit at a time, so we only make one bit and scale the
        # energy and latency
        self.flip_flop = FlipFlop(
            n_bits=1,
            tech_node=tech_node,
            cycle_period=cycle_period,
        )
        self.flip_flop.energy_scale *= n_bits
        self.flip_flop.latency_scale *= n_bits
        self.flip_flop.throughput_scale /= n_bits

        # Zero counter is shared between all the comparators, so scale the energy and
        # latency to activate with each one
        self.zeros_counter = AladdinCounter(
            width=ceil(log2(n_comparators)),
            tech_node=tech_node,
        )
        self.zeros_counter.energy_scale *= n_comparators
        self.zeros_counter.latency_scale *= n_comparators
        self.zeros_counter.throughput_scale /= n_comparators

        super().__init__(
            subcomponents=[
                self.comparator,
                self.flip_flop,
                self.zeros_counter,
            ],
        )

        for subcomponent in self.subcomponents:
            subcomponent.scale(
                "voltage",
                voltage,
                0.85,
                area_scale_function=linear,
                energy_scale_function=quadratic,
                latency_scale_function=reciprocal,
                throughput_scale_function=linear,
                leak_power_scale_function=linear,
            )
            subcomponent.leak_power_scale *= 0.02  # Low-leakage technology

    @action
    def read(self) -> ActionCost:
        self.comparator.read()
        self.flip_flop.read()
        self.zeros_counter.read()
