from numbers import Number
from typing import Optional, List
from hwcomponents.scaling import tech_node_area
from util.bit_functions import rescale_sum_to_1
from hwcomponents import ComponentModel, action


class Capacitor(ComponentModel):
    """
    A capacitor.

    Parameters
    ----------
    capacitance: float
        The capacitance of this capacitor in Farads.
    tech_node: int
        The tech_node node in meters.
    voltage: float
        The supply voltage in volts.
    cap_per_m2: float
        The capacitance per square meter in Farads per square meter.
    border_area: float
        The border area around the capacitor in square meters.
    voltage_raise_threshold: float
        Latency is calculated as the time it takes to raise voltage to this proportion
        of the target voltage.
    supply_resistance: float
        The supply resistance in ohms. If 0, then voltage is assumed to converge
        instantly.

    Attributes
    ----------
    capacitance: float
        The capacitance of this capacitor in Farads.
    tech_node: int
        The tech_node node in meters.
    voltage: float
        The supply voltage in volts.
    cap_per_m2: float
        The capacitance per square meter in Farads per square meter.
    border_area: float
        The border area around the capacitor in square meters.
    voltage_raise_threshold: float
        Latency is calculated as the time it takes to raise voltage to this proportion
        of the target voltage.
    supply_resistance: float
        The supply resistance in ohms. If 0, then voltage is assumed to converge
        instantly.
    """

    priority = 0.5
    """
    Priority determines which model is used when multiple models are available for a
    given component. Higher priority models are used first. Must be a number between 0
    and 1.
    """

    def __init__(
        self,
        capacitance: Number,
        tech_node: float,
        voltage: Number = 0.7,
        cap_per_m2: Optional[Number] = "1e-3 scaled by tech node",
        border_area: Optional[Number] = "1e-12 scaled by tech node",
    ):
        self.capacitance = capacitance
        self.voltage = voltage

        if cap_per_m2 == "1e-3 scaled by tech node":
            cap_per_m2 = 2.3e-3 * tech_node_area(tech_node, 22e-9)
        if border_area == "1e-12 scaled by tech node":
            border_area = 1e-12 * tech_node_area(tech_node, 22e-9)

        self.cap_per_m2 = cap_per_m2
        self.border_area = border_area

        super().__init__(
            area=self.capacitance / self.cap_per_m2 + self.border_area, leak_power=0
        )

    @action
    def raise_voltage_to(
        self,
        target_voltage: float,
        supply_voltage: float = None,
    ) -> float:
        """
        Raise the voltage to the target voltage using the supply voltage as a supply.

        Parameters
        ----------
        target_voltage: float
            The target voltage to raise the voltage to.
        supply_voltage: float
            The supply voltage to use as a supply. If None, then the supply voltage is
            assumed to be the voltage set in the attributes of this capacitor.

        Returns
        -------
        energy, latency: tuple
            The energy required to raise the voltage to the target voltage. Latency is
            0.
        """
        if supply_voltage is None:
            supply_voltage = self.voltage
        assert target_voltage <= supply_voltage, (
            f"Can not raise voltage to {target_voltage} when supply voltage "
            f"is {supply_voltage}."
        )
        return self.capacitance * target_voltage * supply_voltage, 0

    @action
    def switch(
        self,
        value_probabilities: List[Number],
        zero_between_values: bool = True,
        supply_voltage: float = None,
    ) -> float:
        """
        Calculates the expected energy to switch voltage to the values in
        value_probabilities.

        Parameters
        ----------
        value_probabilities: List[Number]
            The probabilities of the values to switch to. This is a histogram, assumed
            to be spaced between 0 and supply_voltage, inclusive.
        zero_between_values: bool
            Whether to zero the voltage between values.
        supply_voltage: float
            The supply voltage to use as a supply. If None, then the supply voltage is
            assumed to be the voltage set in the attributes of this capacitor.

        Returns
        -------
        energy, latency: tuple
            The energy required to switch the voltage to the values in
            value_probabilities. Latency is 0.
        """
        supply_voltage = self.voltage if supply_voltage is None else supply_voltage
        expected_energy = 0
        value_probabilities = rescale_sum_to_1(value_probabilities)
        for v0, p0 in enumerate(value_probabilities):
            for v1, p1 in enumerate(value_probabilities):
                v0 = 0 if zero_between_values else v0
                if v1 < v0:
                    continue
                e0 = self.raise_voltage_to(
                    v0 / (len(value_probabilities) - 1) * self.voltage, supply_voltage
                )[0]
                e1 = self.raise_voltage_to(
                    v1 / (len(value_probabilities) - 1) * self.voltage, supply_voltage
                )[0]
                expected_energy += (e1 - e0) * p0 * p1
        return expected_energy, 0


class Wire(Capacitor):
    """
    A wire.

    Parameters
    ----------
    length: Number
        The length of the wire in meters.
    capacitance_per_m: Number
        The capacitance per meter in Farads per meter.
    voltage: Number
        The supply voltage of the wire in volts.

    Attributes
    ----------
    length: Number
        The length of the wire in meters.
    capacitance_per_m: Number
        The capacitance per meter in Farads per meter.
    voltage: Number
        The supply voltage of the wire in volts.
    """

    def __init__(
        self,
        length: Number,
        capacitance_per_m: Number = 2e-10,
        voltage: Number = 0.7,
        **kwargs,
    ):
        super().__init__(
            capacitance=length * capacitance_per_m,
            voltage=voltage,
        )
        self.length = length
        self.capacitance_per_m = capacitance_per_m
        self.voltage = voltage
        self.area_scale = 0
