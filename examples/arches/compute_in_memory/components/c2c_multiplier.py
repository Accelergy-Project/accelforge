from hwcomponents.scaling import linear
from hwcomponents_neurosim import NOTGate
from hwcomponents import ComponentModel, action
from misc import Capacitor


class C2CMultiplier(ComponentModel):
    """
    The C2C multiplier looks like the following:

    - For operand A as an analog voltage
    - Operand B is a binary digital value with bits B0, B1, B2... from least to most
      significant

    The circuit looks like:

         2C      2C         2C         2C         2C         2C
      G──||───┰──||──────┰──||──────┰──||──────┰──||──────┰──||──── -> OUT
              = C        = C        = C        = C        = C
              │          │          │          │          │
               ╲─── B0    ╲─── B1    ╲─── B2    ╲─── B3    ╲─── B4
             │  G       │  G       │  G       │  G       │  G
      A──────┴──────────┴──────────┴──────────┴──────────┴─────────

    Energy is consumed when: 1. A increases, and all the B capacitors are charged 2. Any
    B bit goes 0->1, and the corresponding capacitor is charged

    USAGE: In your architecture, initialize a both a C2CMultiplier and a
    C2CMultiplierPortB. Have the "a" port process the analog operand and have the "b"
    port process the digital operand.

    The C2CMultiplier component has area accounted for. The C2CMultiplierPortB component
    does not have any area!

    Parameters
    ----------
    resolution: int
        The resolution of the multiplier.
    voltage: float
        The voltage of the multiplier in volts.
    unit_capacitance: float
        The unit capacitance of the multiplier in Farads.
    a_hist: list[float]
        The histogram of the analog operand's values. This is a histogram of the values,
        assumed to be spaced between 0 and voltage, inclusive.
    b_bit_distribution: list[float]
        The distribution of the binary operand's bits. Each is a probability of a given
        bit being 1.
    tech_node: str
        The tech node of the multiplier in meters.
    """

    priority = 0.5

    def __init__(
        self,
        resolution: int,
        voltage: float,
        unit_capacitance: float,
        a_hist: list[float],
        b_bit_distribution: list[float],
        tech_node: str,
    ):
        self.voltage = voltage
        self.unit_capacitance = unit_capacitance
        self.a_hist = a_hist
        self.b_bit_distribution = b_bit_distribution
        self.tech_node = tech_node

        self.unit_cap = Capacitor(
            capacitance=unit_capacitance,
            voltage=voltage,
            tech_node=tech_node,
        )
        self.unit2_cap = Capacitor(
            capacitance=unit_capacitance * 2,
            voltage=voltage,
            tech_node=tech_node,
        )
        self.inverter = NOTGate(tech_node=self.tech_node, cycle_period=1e-9)

        a_rms = (sum(i**2 * p for i, p in enumerate(a_hist)) / sum(a_hist)) ** 0.5
        self.a_rms = a_rms * voltage / (len(a_hist) - 1)

        if not all(0 <= p <= 1 for p in b_bit_distribution):
            raise ValueError("Bit probabilities must be between 0 and 1")
        self.b_lo2hi_probability = sum(p * (1 - p) for p in b_bit_distribution) / len(
            b_bit_distribution
        )

        # Pass gates are 2 transistors, 100F^2 each
        control_pass_gate_area = 2 * self.tech_node**2 * 100
        cap_area = self.unit_cap.area + self.unit2_cap.area
        inverter_area = self.inverter.area

        # Assume pass gates don't leak
        inverter_leak = self.inverter.leak_power
        cap_leak = self.unit_cap.leak_power + self.unit2_cap.leak_power

        super().__init__(
            area=cap_area + inverter_area + control_pass_gate_area,
            leak_power=cap_leak + inverter_leak,
        )

        self.resolution: float = self.scale(
            "resolution",
            resolution,
            1,
            area_scale_function=linear,
            energy_scale_function=linear,
            latency_scale_function=None,
            leak_power_scale_function=linear,
        )

    @action
    def switch_a(self):
        """
        Charge all capacitors to the values in a_hist.
        """
        # Count energy by just charging one of the capacitors and multiplying by the
        # number of bits.
        energy_latency = self.unit_cap.switch(self.a_hist)
        energy_latency += self.unit2_cap.switch(self.a_hist)

        # The reference node sees a cap of unit_capacitance * 1.67 / resolution per bit
        # on average assuming a uniform-ish distribution of bits
        energy, latency = energy_latency
        energy *= 1.67 / self.resolution

        return energy, latency

    @action
    def switch_b(self):
        """
        Connect capacitors to A with probability b_lo2hi_probability.
        """
        energy_latency = self.unit_cap.raise_voltage_to(self.a_rms)
        energy_latency += self.unit2_cap.raise_voltage_to(self.a_rms)
        energy, latency = energy_latency
        energy *= self.b_lo2hi_probability
        return energy, latency

    @action
    def read(self):
        """
        Returns the energy and latency to send a value through the multiplier's analog
        port. If you are only using the read() action, then also initialize a
        C2CMultiplierPortB to have it process the digital operand with the read()
        action.
        """
        return self.switch_a()


class C2CMultiplierPortB(C2CMultiplier):
    def __init__(
        self,
        resolution: int,
        voltage: float,
        unit_capacitance: float,
        a_hist: list[float],
        b_bit_distribution: list[float],
        tech_node: str,
    ):
        super().__init__(
            resolution=resolution,
            voltage=voltage,
            unit_capacitance=unit_capacitance,
            a_hist=a_hist,
            b_bit_distribution=b_bit_distribution,
            tech_node=tech_node,
        )
        self.area_scale = 0

    @action
    def read(self):
        """
        Returns the energy and latency to send a value through the multiplier's digital
        port. If you are only using the read() action, then also initialize a
        C2CMultiplier to have it process the analog operand with the read() action.
        """
        return self.switch_b()
