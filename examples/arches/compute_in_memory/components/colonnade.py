from hwcomponents.model import ComponentModel, action
from hwcomponents import ActionCost
from hwcomponents_neurosim.main import (
    Adder as _NeuroSimAdder,
    FlipFlop as _NeuroSimFlipFlop,
)


class ColonnadeCimLogic(ComponentModel):
    """
    The CiM logic for Colonnade includes, for each memory cell, two MUXes for selection,
    an XNOR for multiplication, and a full adder for accumulation.

    Parameters
    ----------
    tech_node: float
        The technology node in meters.
    width: int
        The number of bits of CiM logic. Equal to the width of the adder and the number
        of memory cells.
    switching_activity: float
        The switching activity of the logic. The number of switches per read action.
    voltage_energy_scale: float
        An energy scaling factor used to modify energy based on the supply voltage.
    cycle_period: float
        The cycle period in seconds.


    """

    priority = 0.8

    def __init__(
        self,
        tech_node: float,
        width: int = 1,
        switching_activity: float = 0.5,
        voltage_energy_scale: float = 1.0,
        cycle_period: float = 10e-9,
    ):
        # There's also two MUXes and an XNOR, but we approximate everything using
        # a calibrated adder as a proxy for the entire logic
        self.adder = _NeuroSimAdder(
            tech_node=tech_node, cycle_period=cycle_period, n_bits=width
        )
        super().__init__(subcomponents=[self.adder])
        self.scale_tech_node(tech_node, 65e-9)
        self.adder.area_scale *= 0.9  # Calibrate to Colonnade
        self.adder.energy_scale *= voltage_energy_scale * 0.0285 * switching_activity
        self.adder.latency_scale = 0  # Ignore adder latency
        self.adder.throughput_scale = float("inf")
        self.adder.leak_power_scale *= 15
        self.cycle_period = cycle_period
        self.width = width

    @action()
    def read(self) -> tuple:
        """
        Returns the cost consumed by one read of the CiM logic.

        Returns
        -------
        ActionCost: The cost of this action
        """
        self.adder.read()
        return ActionCost(energy=0, throughput=float("inf"), latency=0)


class ColonnadeCimLogicInputPort(ComponentModel):
    """
    The input port for Colonnade CiM logic. This input port captures the energy of the
    one-bit input that is applied to all XNORs in the row (i.e., the one-bit operand in
    the 1-bit by n-bit multiplication).

    Note that we do not charge for area or latency in this component, as these should be
    handled in the CiM logic component.

    Parameters
    ----------
    tech_node: float
        The technology node in meters.
    switching_activity: float
        The switching activity of the input port. The number of switches per read
        action.
    voltage: float
        The voltage of the input signal. This is unused; fixed at 0.8V because we
        account for voltage differences in voltage_energy_scale.
    voltage_energy_scale: float
        An energy scaling factor used to modify energy based on the supply voltage.
    n_instances: int
        The number of instances of this input port. This is used to scale energy, since
        one input port drives n XNORs in the row.
    cycle_period: float
        The cycle period in seconds.
    """

    priority = 0.8

    def __init__(
        self,
        tech_node: float,
        switching_activity: float = 0.5,
        voltage: float = 0.8,  # KEEP FIXED AT 0.8 BECAUSE WE ALREADY ACCOUNTED FOR THIS IN VOLTAGE_ENERGY_SCALE
        voltage_energy_scale: float = 1.0,
        n_instances: int = 1,
        cycle_period: float = 10e-9,
    ):
        self.adder = _NeuroSimAdder(
            tech_node=tech_node, cycle_period=cycle_period, n_bits=1
        )
        super().__init__(subcomponents=[self.adder])
        self._wire_cap = 1e-15
        self.switching_activity = switching_activity
        self.voltage_energy_scale = voltage_energy_scale
        self._n_instances_scale = n_instances

        # Asserting one row activates ALL input ports on that row, so scale by n_instances
        self.adder.energy_scale *= (
            n_instances * voltage_energy_scale * switching_activity * switching_activity
        )
        self.adder.latency_scale = 0  # Ignore adder latency
        self.adder.throughput_scale = float("inf")

        self.area_scale *= 0  # Account for area in ColonnadeCimLogic
        self.leak_power_scale *= 0
        self.adder.area_scale *= 0
        self.adder.leak_power_scale *= 0

    @action(bits_per_action=1)
    def read(self) -> tuple:
        """
        Returns the cost consumed by one read of the CiM logic input port.

        Returns
        -------
        ActionCost: The cost of this action
        """
        self.adder.read()
        p_switch = self.switching_activity * self.switching_activity
        wire_energy = (
            self._wire_cap
            * (
                0.8**2
            )  # KEEP FIXED AT 0.8 BECAUSE WE ALREADY ACCOUNTED FOR THIS IN VOLTAGE_ENERGY_SCALE
            * self._n_instances_scale
            * self.voltage_energy_scale
            * p_switch
        )
        return ActionCost(energy=wire_energy, throughput=float("inf"), latency=0)


class ColonnadeRegister(ComponentModel):
    """
    Registers that are inserted into Colonnade's CiM logic to allow for pipelining CiM
    array operations.

    Parameters
    ----------
    tech_node: float
        The technology node in meters.
    width: int
        The number of bits in the register.
    voltage_energy_scale: float
        An energy scaling factor used to modify energy based on the supply voltage.
    cycle_period: float
        The cycle period in seconds.
    """

    priority = 0.8

    def __init__(
        self,
        tech_node: float,
        width: int = 1,
        voltage_energy_scale: float = 1.0,
        cycle_period: float = 10e-9,
    ):
        self.ff = _NeuroSimFlipFlop(
            tech_node=tech_node, cycle_period=cycle_period, n_bits=1
        )
        super().__init__(subcomponents=[self.ff])
        self.scale_tech_node(tech_node, 65e-9)
        self.ff.area_scale *= 1.3
        self.ff.energy_scale *= voltage_energy_scale * 4.5
        self.width = width

    @action(bits_per_action="width")
    def read(self) -> tuple:
        self.ff.read()

    @action(bits_per_action="width")
    def write(self) -> tuple:
        self.ff.write()
