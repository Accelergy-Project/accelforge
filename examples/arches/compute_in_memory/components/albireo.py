"""
Components for the Albireo photonic CiM accelerator.

@INPROCEEDINGS{9499746,
  author={Shiflett, Kyle and Karanth, Avinash and Bunescu, Razvan and Louri, Ahmed},
  booktitle={2021 ACM/IEEE 48th Annual International Symposium on Computer Architecture (ISCA)},
  title={Albireo: Energy-Efficient Acceleration of Convolutional Neural Networks via Silicon Photonics},
  year={2021},
  volume={},
  number={},
  pages={860-873},
  doi={10.1109/ISCA52012.2021.00072}}

@INPROCEEDINGS{10590061,
  author={Andrulis, Tanner and Chaudhry, Gohar Irfan and Suriyakumar, Vinith M. and Emer, Joel S. and Sze, Vivienne},
  booktitle={2024 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)},
  title={Architecture-Level Modeling of Photonic Deep Neural Network Accelerators},
  year={2024},
  volume={},
  number={},
  pages={307-309},
  doi={10.1109/ISPASS61541.2024.00040}}


The 'scaling' parameter selects between three technology projections:
  - 'conservative': pessimistic scaling
  - 'moderate': moderate scaling
  - 'aggressive': optimistic scaling
"""

from hwcomponents import ComponentModel, action
from hwcomponents.scaling import tech_node_area


def _scale_val(scaling, conservative, moderate, aggressive):
    return {
        "conservative": conservative,
        "moderate": moderate,
        "aggressive": aggressive,
    }[scaling]


class AlbireoTIA(ComponentModel):
    """
    Transimpedance amplifier (TIA) from the Albireo photonic accelerator. Converts
    analog photocurrent from photodiodes to voltage that can be read by an ADC.

    Parameters
    ----------
    tech_node : float
        Technology node in meters.
    scaling : str
        Scaling scenario: 'conservative', 'moderate', or 'aggressive'.
    cycle_period : float, optional
        Cycle period in seconds.
    """

    component_name = ["AlbireoTIA", "albireo_tia"]
    priority = 0.5

    def __init__(
        self,
        tech_node: float,
        scaling: str = "conservative",
        cycle_period: float = 1e-9,
    ):
        self._energy = _scale_val(scaling, 0.625e-12, 0.3125e-12, 0.045e-12)
        self.cycle_period = float(cycle_period)
        area_scale = tech_node_area(tech_node, 7e-9)
        super().__init__(area=2769e-12 * area_scale, leak_power=0)

    @action
    def read(self) -> tuple[float, float]:
        return self._energy, self.cycle_period


class AlbireoDAC(ComponentModel):
    """
    Digital-to-analog converter (DAC) from the Albireo photonic accelerator.

    Converts digital input/weight values to analog electrical signals.

    Parameters
    ----------
    tech_node : float
        Technology node in meters.
    scaling : str
        Scaling scenario: 'conservative', 'moderate', or 'aggressive'.
    """

    component_name = ["AlbireoDAC", "dac_albireo"]
    priority = 0.5

    def __init__(self, tech_node: float, scaling: str = "conservative"):
        self._energy = _scale_val(scaling, 5.2e-12, 2.6e-12, 0.325e-12)
        area_scale = tech_node_area(tech_node, 7e-9)
        super().__init__(area=153e-12 * area_scale, leak_power=0)

    @action
    def read(self) -> tuple[float, float]:
        return self._energy, 0


class AlbireoMachZehnderModulator(ComponentModel):
    """
    Mach-Zehnder modulator (MZM) from the Albireo photonic accelerator. Modulates laser
    light with an electrical input/weight signal. Zero active energy; consumes static
    (leak) power for thermal control.

    Parameters
    ----------
    tech_node : float
        Technology node in meters.
    scaling : str
        Scaling scenario: 'conservative', 'moderate', or 'aggressive'.
    """

    component_name = ["AlbireoMachZehnderModulator", "albireo_mzm"]
    priority = 0.5

    def __init__(self, tech_node: float, scaling: str = "conservative"):
        leak = _scale_val(scaling, 11.3e-3, 1.41e-3, 0.565e-3)
        area_scale = tech_node_area(tech_node, 7e-9)
        super().__init__(area=15066e-12 * area_scale, leak_power=leak)

    @action
    def read(self) -> tuple[float, float]:
        return 0, 0


class AlbireoMicroRingResonator(ComponentModel):
    """
    Micro-ring resonator (MRR) from the Albireo photonic accelerator. Routes specific
    wavelengths of light based from one path to another. Zero active energy; consumes
    static (leak) power for thermal wavelength selection.

    Parameters
    ----------
    tech_node : float
        Technology node in meters.
    scaling : str
        Scaling scenario: 'conservative', 'moderate', or 'aggressive'.
    """

    component_name = ["AlbireoMicroRingResonator", "albireo_mrr"]
    priority = 0.5

    def __init__(self, tech_node: float, scaling: str = "conservative"):
        leak = _scale_val(scaling, 3.1e-3, 0.388e-3, 0.155e-3)
        area_scale = tech_node_area(tech_node, 7e-9)
        super().__init__(area=410e-12 * area_scale, leak_power=leak)

    @action
    def read(self) -> tuple[float, float]:
        return 0, 0


class AlbireoDoubleMicroRingResonator(ComponentModel):
    """
    Double micro-ring resonator from the Albireo photonic accelerator. Each MAC site
    uses two MRRs (one for positive and one for negative). Area and leak power are both
    2x a single MRR.

    Parameters
    ----------
    tech_node : float
        Technology node in meters.
    scaling : str
        Scaling scenario: 'conservative', 'moderate', or 'aggressive'.
    """

    component_name = ["AlbireoDoubleMicroRingResonator", "albireo_double_mrr"]
    priority = 0.5

    def __init__(self, tech_node: float, scaling: str = "conservative"):
        leak = 2 * _scale_val(scaling, 3.1e-3, 0.388e-3, 0.155e-3)
        area_scale = tech_node_area(tech_node, 7e-9)
        super().__init__(area=820e-12 * area_scale, leak_power=leak)

    @action
    def read(self) -> tuple[float, float]:
        return 0, 0


class AlbireoPhotodiode(ComponentModel):
    """
    Photodiode from the Albireo photonic accelerator.

    Converts optical power to analog electrical current.
    Zero energy and zero leak power (passive component).

    CSV reference (7nm):
      read/write/update/leak = 0 pJ, area = 9230 um^2

    Parameters
    ----------
    tech_node : float
        Technology node in meters.
    """

    component_name = ["AlbireoPhotodiode", "albireo_photodiode"]
    priority = 0.5

    def __init__(
        self,
        tech_node: float,
        cycle_period: float = 1e-9,
    ):
        self.cycle_period = float(cycle_period)
        area_scale = tech_node_area(tech_node, 7e-9)
        super().__init__(area=1.846e-9 * area_scale, leak_power=0)

    @action
    def read(self) -> tuple[float, float]:
        return 0, self.cycle_period


class AlbireoArrayedWaveguideGrating(ComponentModel):
    """
    Arrayed waveguide grating (AWG) from the Albireo photonic accelerator.

    Demultiplexes wavelength-division-multiplexed (WDM) input light, routing
    specific wavelengths to each photonic locally connected unit (PLCU).
    Zero energy; area-only component.

    CSV reference (7nm):
      read/compute = 0 pJ, area = 1108089 um^2

    Parameters
    ----------
    tech_node : float
        Technology node in meters.
    """

    component_name = ["AlbireoArrayedWaveguideGrating", "albireo_awg"]
    priority = 0.5

    def __init__(self, tech_node: float):
        area_scale = tech_node_area(tech_node, 7e-9)
        super().__init__(area=1108089e-12 * area_scale, leak_power=0)

    @action
    def read(self) -> tuple[float, float]:
        return 0, 0


class AlbireoStarCoupler(ComponentModel):
    """
    Star coupler from the Albireo photonic accelerator.

    An any-to-any optical interconnect that distributes input signals
    from multiple rows to multiple columns within a star-coupled group.
    Zero energy; area-only component.

    CSV reference (7nm):
      read/compute = 0 pJ, area = 262500 um^2

    Parameters
    ----------
    tech_node : float
        Technology node in meters.
    """

    component_name = ["AlbireoStarCoupler", "albireo_star_coupler"]
    priority = 0.5

    def __init__(self, tech_node: float):
        area_scale = tech_node_area(tech_node, 7e-9)
        super().__init__(area=262500e-12 * area_scale, leak_power=0)

    @action
    def read(self) -> tuple[float, float]:
        return 0, 0


class AlbireoLaser(ComponentModel):
    """
    Laser (driver) from the Albireo photonic accelerator.

    Provides the optical carrier signal. Consumes static (leak) power proportional to
    the number of endpoints it feeds.

    CSV reference (7nm):

    - conservative: leak = 1.94 mW, area = 6153 um^2
    - moderate:     leak = 0.0713 mW
    - aggressive:   leak = 0.0713 mW

    Parameters
    ----------
    tech_node : float
        Technology node in meters.
    scaling : str
        Scaling scenario: 'conservative', 'moderate', or 'aggressive'.
    """

    component_name = ["AlbireoLaser", "albireo_laser"]
    priority = 0.5

    def __init__(self, tech_node: float, scaling: str = "conservative"):
        leak = _scale_val(scaling, 1.94e-3, 0.0743e-3, 0.0614e-3)
        area_scale = tech_node_area(tech_node, 7e-9)
        super().__init__(area=6153e-12 * area_scale, leak_power=leak)

    @action
    def read(self) -> tuple[float, float]:
        return 0, 0
