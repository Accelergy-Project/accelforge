import math

from hwcomponents.scaling import linear
from hwcomponents_neurosim import FlipFlop
from misc import Capacitor
from typing import List

from hwcomponents import ComponentModel, action


def value2bits(value: int, resolution: int) -> List[int]:
    """Converts a value to a list of bits."""
    return [int(i) for i in bin(value)[2:].zfill(resolution)]


class _X2XLadderDAC(ComponentModel):
    """
    Base class for X2X Ladder DACs. This class is not intended to be instantiated
    directly. Use a subclass.
    """

    priority = 0.5

    def __init__(
        self,
        resolution: int,
        voltage: float,
        unit_x: float,
        tech_node: int,
        area: float,
        leak_power: float,
        kind: str,
        hist: List[float] = None,
        load_resistance: float = 0,
        load_capacitance: float = 0,
        cycle_period: float = 1e-9,
    ):
        self.resolution = resolution
        self.voltage = voltage
        self._unit_x = unit_x
        self.tech_node = tech_node
        self._max_value = 2**resolution - 1
        self.hist = hist
        self.load_resistance = load_resistance
        self.load_capacitance = load_capacitance
        self._kind = kind
        self.cycle_period = cycle_period
        self._output_cap = Capacitor(
            voltage=voltage, capacitance=load_capacitance, tech_node=tech_node
        )
        self._flip_flops = FlipFlop(
            tech_node=tech_node, cycle_period=cycle_period, n_bits=resolution
        )
        super().__init__(
            area=self._flip_flops.area + area,
            leak_power=self._flip_flops.leak_power + leak_power,
        )

    def _get_latency(
        self,
        load_capacitance: float,
        load_resistance: float,
        lsbs_expected_to_change: float = None,
        lsbs_allowed_incorrect: float = 0,
        porp_charge_loss_to_overcome: float = 0,
    ) -> float:
        """
        Returns the latency in seconds to convert the input value to an analog voltage
        and charge a load. If the latency is greater than the cycle period, the DAC will
        be run more slowly in order to meet the required hold time and settle to <=0.5
        LSB error. If the latency is less than the cycle period, the DAC will hold the
        value for the full cycle period, and may consume more power than necessary. Note
        that hold time will be longer if load_capacitance is increased or
        load_resistance is decreased.

        Parameters
        ----------
        load_capacitance: float
            Load capacitance in Farads
        load_resistance: float
            Load resistance in Ohms
        lsbs_expected_to_change: float, optional
            Expected values. Lower values mean a lower-swing output. Defaults to None.
        lsbs_allowed_incorrect: float, optional
            Number of LSBs allowed to be incorrect in the final converted value.
            Defaults to 0.
        porp_charge_loss_to_overcome: float, optional
            Porportion (0 to 1) of the charge on the load capacitor that is leaked
            between cycles amd must be recharged. Defaults to 0.

        Returns
        -------
        float: Latency in seconds

        """
        # Instant convergence if there's no load resistance
        if self.load_resistance + load_resistance == 0:
            return self.cycle_period

        if lsbs_expected_to_change is None:
            lsbs_expected_to_change = self.resolution

        lsbs_allowed_incorrect += 0.5

        # Time it takes for the output to converge within lsb_error_allowed LSB
        r = self.load_resistance + load_resistance
        c = self.load_capacitance + load_capacitance

        delta = 2**lsbs_expected_to_change / self._max_value
        t0 = -math.log(delta + porp_charge_loss_to_overcome) * r * c
        t1 = -math.log(2**lsbs_allowed_incorrect / self._max_value) * r * c

        hold_time = t1 - t0
        if hold_time > self.cycle_period:
            self.logger.warning(
                f"Required hold time {hold_time} is greater than the cycle period "
                f"{self.cycle_period}. DAC will be run more slowly in order to meet "
                f"the required hold time and settle to <=0.5 LSB error. Note that hold "
                f"time will be longer if load_capacitance is increased or "
                f"load_resistance is decreased."
            )
        if hold_time < self.cycle_period:
            self.logger.warning(
                f"Cycle period {self.cycle_period} is greater than the required hold "
                f"time {hold_time}. DAC will hold the value for the full cycle period, "
                f"and may therefore consume more power than necessary. Note that hold "
                f"time will be longer if load_capacitance is increased or "
                f"load_resistance is decreased."
            )

        return max(hold_time, self.cycle_period)

    def solve_for_voltage_at_each_node(self, input_value: int) -> list:
        """
        Solves the matrix:
                         [ 4 -2  0....         ][V_0]
        input_voltages = [-2  5 -2  0 .....    ][V_1]
                         [ 0 -2  5 -2  0 ....  ][V_2]
                         [ 0  0 -2  5 -2 0 ....][V_3]
                         [ ................... ][V_4]
                         [ 0  0  0  0  0  -2  3][V_{n-1}]


        This matrix arises in C-2C and R-2R ladders. To solve, we set up the equation
        for each node: 0 = (V_{i} - V_{i-1}) + (V_{i} - V_{i+1}) + (V_{i} - Input_{i}) /
        2 The first node has an additional connector, and the last node does not

        Parameters
        ----------
        input_value: int
            Input value to be converted. Must be between 0 and 2^resolution - 1.

        Returns
        -------
        list[float]:
            The voltage at each node in the ladder. The first element is the voltage at
            the voltage farthest from the output, and the last element is the voltage
            immediately before the output.
        """
        # Reverse input_value_bits to get the MSB on the right side of the circuit,
        # closest to the output
        input_value_bits = value2bits(input_value, self.resolution)[::-1]

        lhs = [i * self.voltage for i in input_value_bits]
        matrix_values = [[0, 4, -2]]
        for i in range(len(lhs) - 2):
            matrix_values.append([-2, 5, -2])
        matrix_values.append([-2, 3, 0])

        for i in range(len(lhs) - 2, -1, -1):
            mult = matrix_values[i][2] / matrix_values[i + 1][1]
            matrix_values[i][1] -= matrix_values[i + 1][0] * mult
            matrix_values[i][2] -= matrix_values[i + 1][1] * mult
            lhs[i] -= lhs[i + 1] * mult
            lhs[i] /= matrix_values[i][1]
            matrix_values[i] = [j / matrix_values[i][1] for j in matrix_values[i]]

        for i in range(0, len(lhs) - 1):
            mult = matrix_values[i + 1][0] / matrix_values[i][1]
            matrix_values[i + 1][0] -= matrix_values[i][1] * mult
            matrix_values[i + 1][1] -= matrix_values[i][2] * mult
            lhs[i + 1] -= lhs[i] * mult

        lhs = [l / matrix_values[i][1] for i, l in enumerate(lhs)]

        # Un-reverse the bits
        return lhs[::-1]

    def _input_value_to_analog_energy_or_power(self, input_value: int) -> float:
        """
        Returns the energy or power to convert the input value to an analog voltage,
        depending on whether unit_x is a capacitance or a resistance.

        Parameters
        ----------
        input_value: int
            Input value to be converted. Must be between 0 and 2^resolution - 1.

        Returns
        -------
        float: Energy in Joules
        """
        input_value_bits = value2bits(input_value, self.resolution)
        node_voltages = self.solve_for_voltage_at_each_node(input_value)
        current = 0
        for i, bit in enumerate(input_value_bits):
            current += (self.voltage - node_voltages[i]) * (bit != 0)
        energy = current * self.voltage * self._unit_x
        assert energy >= 0
        return energy

    def _convert_energy(
        self,
        latency: float | None = None,
    ):
        """
        Returns the average energy and latency to convert the input value to an analog
        voltage

        Parameters
        ----------
        latency: float | None, optional
            If this is a resistive DAC, energy scales with the latency for which the
            value is held. If no latency is given, energy is returned without scaling.
        """
        energy = 0

        # This code resizes the histogram into the full distribution of values
        # that this DAC can produce. It also makes sure to map the 0
        # probability exactly.
        newhist = [0] * 2**self.resolution
        idx0 = len(self.hist) // 2
        new_idx0 = len(newhist) // 2
        width_scale = len(self.hist) / len(newhist)
        prunedhist = [i for i in self.hist]
        for exact_maps in [(0, 0), (idx0, new_idx0)]:
            newhist[exact_maps[1]] = prunedhist[exact_maps[0]] / min(width_scale, 1)
            prunedhist[exact_maps[0]] = 0

        for i in range(len(newhist)):
            if i == new_idx0 or i == 0:
                continue
            loc = i / len(newhist) * (len(prunedhist) - 1)
            if width_scale > 1:
                start = math.floor(loc)
                end = min(math.ceil(start + width_scale), len(prunedhist) - 1)
                for j in range(start, end):
                    newhist[i] += prunedhist[j]
            else:
                porp = loc - math.floor(loc)
                newhist[i] += prunedhist[math.floor(loc)] * (1 - porp)
                newhist[i] += prunedhist[math.ceil(loc)] * porp

        sum_newhist = sum(newhist)
        newhist = [n / sum_newhist for n in newhist]

        # Calculate the energy
        for i, p in enumerate(newhist):
            energy += self._input_value_to_analog_energy_or_power(i) * p
        energy /= sum(newhist)
        assert energy >= 0

        # Latency not none ->
        if self._kind == "R2R":
            assert latency is not None
            energy *= latency

        # Output cap energy
        cap_e, _ = self._output_cap.switch(newhist, zero_between_values=False)
        self.logger.info(f"Output capacitance consumed {cap_e}J")
        assert cap_e >= 0

        # Flip flop energy
        flip_flops_e, _ = self._flip_flops.read()
        probabilities = [0] * self.resolution
        for value, probability in enumerate(newhist):
            bits = value2bits(value, self.resolution)
            for i, b in enumerate(bits):
                probabilities[i] += probability * b
        lo2hi_probability = sum(p * (1 - p) for p in probabilities)
        flip_flops_e *= lo2hi_probability
        assert flip_flops_e >= 0

        self.logger.info(f"Flip-flops consumed {flip_flops_e}J")
        return energy + self._get_controller_energy() + cap_e + flip_flops_e

    def _get_controller_energy(self):
        # 0.08pJ/bit at 22nm 1.0V
        return (self.voltage * self.tech_node) ** 2 * self.resolution * 8e-8

    @action
    def convert(self):
        """
        Returns the energy and latency to convert the input value to an analog voltage.

        Returns
        -------
        float: Energy in Joules float: Latency in seconds
        """
        min_latency = self._get_latency(
            load_capacitance=self.load_capacitance, load_resistance=self.load_resistance
        )
        energy = self._convert_energy(latency=min_latency)
        controller_energy = self._get_controller_energy()
        self.logger.info(
            f"{self._kind} DAC consumes {energy}J over{min_latency} seconds. "
            f"Resistors consume {energy - controller_energy}J, "
            f"Controller energy: {controller_energy}J"
        )
        return energy, min_latency

    @action
    def read(self):
        """
        Returns the energy and latency to convert the input value to an analog voltage.

        Returns
        -------
        float: Energy in Joules float: Latency in seconds
        """
        return self.convert()


class C2CLadderDAC(_X2XLadderDAC):
    """
    C-2C ladder DAC.

    R-2R ladder and C-2C ladder DACs model those described in the paper: A Charge Domain
    SRAM Compute-in-Memory Macro With C-2C Ladder-Based 8b MAC Unit in 22-nm FinFET
    Process for Edge Inference Wang, Hechen and Liu, Renzhi and Dorrance, Richard and
    Dasalukunte, Deepak and Lake, Dan and Carlton, Brent 10.1109/JSSC.2022.3232601

    Parameters
    ----------
    resolution: int
        The resolution of the DAC.
    voltage: float
        The voltage of the DAC.
    unit_resistance: float
        The unit resistance of the DAC.
    tech_node: int
        The tech node of the DAC.
    cycle_period: float, optional
        The cycle period of the clock driving the DAC, and also the minimum time the DAC
        must hold any output values (longer if it takes longer to settle). Defaults to
        1e-9.
    hist: List[float], optional
        The histogram of the DAC's input values. Defaults to None. This should be a list
        of probabilities, the first item being the probability of the minimum value, the
        last item being the probability of the maximum value, and the rest being the
        probabilities of the intermediate values.
    load_resistance: float
        The load resistance on the DAC's output. Defaults to 0.
    load_capacitance: float
        The load capacitance on the DAC's output. Defaults to 0.

    Attributes
    ----------
    resolution: int
        The resolution of the DAC.
    voltage: float
        The voltage of the DAC.
    unit_capacitance: float
        The unit capacitance of the DAC.
    tech_node: int
        The tech node of the DAC.
    cycle_period: float, optional
        The cycle period of the clock driving the DAC, and also the minimum time the DAC
        must hold any output values (longer if it takes longer to settle). Defaults to
        1e-9.
    hist: List[float], optional
        The histogram of the DAC's input values. Defaults to None. This should be a list
        of probabilities, the first item being the probability of the minimum value, the
        last item being the probability of the maximum value, and the rest being the
        probabilities of the intermediate values.
    load_resistance: float
        The load resistance on the DAC's output. Defaults to 0.
    load_capacitance: float
        The load capacitance on the DAC's output. Defaults to 0.

    """

    component_name: list[str] = ["C2CLadderDAC", "C2CDAC"]

    def __init__(
        self,
        resolution: int,
        voltage: float,
        unit_capacitance: float,
        tech_node: int,
        hist: List[float] = None,
        capacitors_consume_area: bool = True,
        cycle_period: float = 1e-9,
        load_capacitance: float = 0,
        load_resistance: float = 0,
    ):
        self.unit_capacitance = unit_capacitance

        self._unit_cap = Capacitor(
            voltage=self.voltage, capacitance=unit_capacitance, tech_node=tech_node
        )
        self._unit2_cap = Capacitor(
            voltage=self.voltage, capacitance=unit_capacitance * 2, tech_node=tech_node
        )
        if not capacitors_consume_area:
            self._unit_cap.area_scale *= 0
            self._unit2_cap.area_scale *= 0

        super().__init__(
            resolution=resolution,
            voltage=voltage,
            unit_x=unit_capacitance,
            tech_node=tech_node,
            hist=hist,
            load_capacitance=unit_capacitance * 2 + load_capacitance,
            area=self._unit_cap.area + self._unit2_cap.area,
            leak_power=self._unit_cap.leak_power + self._unit2_cap.leak_power,
            kind="C2C",
            load_resistance=load_resistance,
            cycle_period=cycle_period,
        )
        self.scale(
            "resolution",
            resolution,
            1,
            area_scale_function=linear,
            leak_power_scale_function=linear,
        )


class R2RLadderDAC(_X2XLadderDAC):
    """
    R-2R ladder DAC.

    R-2R ladder and C-2C ladder DACs model those described in the paper: A Charge Domain
    SRAM Compute-in-Memory Macro With C-2C Ladder-Based 8b MAC Unit in 22-nm FinFET
    Process for Edge Inference Wang, Hechen and Liu, Renzhi and Dorrance, Richard and
    Dasalukunte, Deepak and Lake, Dan and Carlton, Brent 10.1109/JSSC.2022.3232601

    Parameters
    ----------
    resolution: int
        The resolution of the DAC.
    voltage: float
        The voltage of the DAC.
    unit_resistance: float
        The unit resistance of the DAC.
    tech_node: int
        The tech node of the DAC.
    cycle_period: float, optional
        The cycle period of the clock driving the DAC, and also the minimum time the DAC
        must hold any output values (longer if it takes longer to settle). Defaults to
        1e-9.
    hist: List[float], optional
        The histogram of the DAC's input values. Defaults to None. This should be a list
        of probabilities, the first item being the probability of the minimum value, the
        last item being the probability of the maximum value, and the rest being the
        probabilities of the intermediate values.
    load_resistance: float
        The load resistance on the DAC's output. Defaults to 0.
    load_capacitance: float
        The load capacitance on the DAC's output. Defaults to 0.

    Attributes
    ----------
    resolution: int
        The resolution of the DAC.
    voltage: float
        The voltage of the DAC.
    unit_resistance: float
        The unit resistance of the DAC.
    tech_node: int
        The tech node of the DAC.
    cycle_period: float, optional
        The cycle period of the clock driving the DAC, and also the minimum time the DAC
        must hold any output values (longer if it takes longer to settle). Defaults to
        1e-9.
    hist: List[float], optional
        The histogram of the DAC's input values. Defaults to None. This should be a list
        of probabilities, the first item being the probability of the minimum value, the
        last item being the probability of the maximum value, and the rest being the
        probabilities of the intermediate values.
    load_resistance: float
        The load resistance on the DAC's output. Defaults to 0.
    load_capacitance: float
        The load capacitance on the DAC's output. Defaults to 0.
    """

    component_name: list[str] = ["R2RLadderDAC", "R2RDAC"]

    def __init__(
        self,
        resolution: int,
        voltage: float,
        unit_resistance: float,
        tech_node: int,
        cycle_period: float = 1e-9,
        hist: List[float] = None,
        load_resistance: float = 0,
        load_capacitance: float = 0,
    ):
        self.unit_resistance = unit_resistance
        self._m2_chip_area_per_ohm = 2.35e-15 * (tech_node / 22e-9) ** 2

        super().__init__(
            resolution=resolution,
            voltage=voltage,
            unit_x=1 / unit_resistance / 2,
            tech_node=tech_node,
            hist=hist,
            area=self._m2_chip_area_per_ohm * resolution * unit_resistance * 3,
            leak_power=0,
            cycle_period=cycle_period,
            kind="R2R",
            load_resistance=load_resistance,
            load_capacitance=load_capacitance,
        )


class DualSidedR2RLadderDAC(R2RLadderDAC):
    """
    Dual-sided R-2R ladder DAC. This DAC has two sides, one for the positive half of the
    voltage range and one for the negative half.

    R-2R ladder and C-2C ladder DACs model those described in the paper: A Charge Domain
    SRAM Compute-in-Memory Macro With C-2C Ladder-Based 8b MAC Unit in 22-nm FinFET
    Process for Edge Inference Wang, Hechen and Liu, Renzhi and Dorrance, Richard and
    Dasalukunte, Deepak and Lake, Dan and Carlton, Brent 10.1109/JSSC.2022.3232601

    Parameters
    ----------
    resolution: int
        The resolution of the DAC.
    voltage: float
        The voltage of the DAC.
    unit_resistance: float
        The unit resistance of the DAC.
    tech_node: int
        The tech node of the DAC.
    cycle_period: float, optional
        The cycle period of the clock driving the DAC, and also the minimum time the DAC
        must hold any output values (longer if it takes longer to settle). Defaults to
        1e-9.
    hist: List[float], optional
        The histogram of the DAC's input values. Defaults to None. This should be a list
        of probabilities, the first item being the probability of the minimum value, the
        last item being the probability of the maximum value, and the rest being the
        probabilities of the intermediate values.
    load_resistance: float
        The load resistance on the DAC's output. Defaults to 0.
    load_capacitance: float
        The load capacitance on the DAC's output. Defaults to 0.

    Attributes
    ----------
    resolution: int
        The resolution of the DAC.
    voltage: float
        The voltage of the DAC.
    unit_resistance: float
        The unit resistance of the DAC.
    tech_node: int
        The tech node of the DAC.
    cycle_period: float, optional
        The cycle period of the clock driving the DAC, and also the minimum time the DAC
        must hold any output values (longer if it takes longer to settle). Defaults to
        1e-9.
    hist: List[float], optional
        The histogram of the DAC's input values. Defaults to None. This should be a list
        of probabilities, the first item being the probability of the minimum value, the
        last item being the probability of the maximum value, and the rest being the
        probabilities of the intermediate values.
    load_resistance: float
        The load resistance on the DAC's output. Defaults to 0.
    load_capacitance: float
        The load capacitance on the DAC's output. Defaults to 0.
    """

    component_name: list[str] = ["DualSidedR2RLadderDAC", "DualSidedR2RDAC"]

    def __init__(
        self,
        resolution: int,
        voltage: float,
        unit_resistance: float,
        tech_node: int,
        cycle_period: float = 1e-9,
        hist: List[float] = None,
        load_resistance: float = 0,
        load_capacitance: float = 0,
    ):
        super().__init__(
            resolution=resolution,
            voltage=voltage,
            unit_resistance=unit_resistance,
            tech_node=tech_node,
            cycle_period=cycle_period,
            hist=hist,
            load_resistance=load_resistance,
            load_capacitance=load_capacitance,
        )
        self.leak_power_scale *= 2  # Two sides
        self.area_scale *= 2  # Two sides
        self.energy_scale *= 2  # Two sides
        # Voltage is still the full range because we're pulling from a VDD supply