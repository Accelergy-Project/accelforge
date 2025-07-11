from fastfusion.frontend.mapper import MapperFFM
from fastfusion.frontend.renames import Renames
from fastfusion.util.parse_expressions import ParseExpressionsContext
from fastfusion.frontend.architecture import Leaf
from fastfusion.frontend.architecture import Architecture
from fastfusion.frontend.constraints import Constraints
from fastfusion.frontend.workload import Workload
from fastfusion.frontend.variables import Variables
from fastfusion.frontend.component_classes import Components
from fastfusion.frontend.config import Config, get_config
from fastfusion.frontend.component_area import ComponentArea, AreaEntry
from fastfusion.frontend.component_energy import ComponentEnergy, EnergyEntry
from fastfusion.frontend.mapping import Mapping

from typing import Any, Dict, List, Optional, Union
from fastfusion.util.basetypes import ParsableModel

from ..plugin.gather_plug_ins import gather_plug_ins


class Specification(ParsableModel):
    architecture: Architecture = Architecture()
    component_classes: Components = Components()
    constraints: Constraints = Constraints()
    mapping: Mapping = Mapping()
    workload: Workload = Workload()
    variables: Variables = Variables()
    config: Config = None
    component_energy: ComponentEnergy = ComponentEnergy()
    component_area: ComponentArea = ComponentArea()
    renames: Renames = Renames()
    mapper_ffm: MapperFFM = MapperFFM()

    def __init__(self, **data):
        if data.get("config") is None:
            data["config"] = get_config()
        super().__init__(**data)

    def parse_expressions(
        self,
        symbol_table: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> tuple["Specification", dict[str, Any]]:
        symbol_table = {} if symbol_table is None else symbol_table.copy()
        symbol_table["spec"] = self
        with ParseExpressionsContext(self):
            parsed_variables, _ = self.variables.parse_expressions(symbol_table, **kwargs)
            symbol_table.update(parsed_variables)
            symbol_table["variables"] = parsed_variables
            return super().parse_expressions(symbol_table, **kwargs)

    def estimate_energy_area(self):
        plug_ins = gather_plug_ins(self.config.component_plug_ins)
        with ParseExpressionsContext(self):
            processed, _ = self.parse_expressions()
            components = processed.architecture._flatten(processed.variables)
            area = ComponentArea()
            energy = ComponentEnergy()
            for component in components:
                area.entries.append(
                    AreaEntry.from_plug_ins(
                        component.component_class,
                        component.attributes,
                        processed,
                        plug_ins,
                        name=component.name,
                    )
                )

                action_names = [action.name for action in component.actions]
                action_args = [action.arguments for action in component.actions]
                energy.entries.append(
                    EnergyEntry.from_plug_ins(
                        component.component_class,
                        component.attributes,
                        action_args,
                        action_names,
                        processed,
                        plug_ins,
                        name=component.name,
                    )
                )
            self.component_area = area
            self.component_energy = energy
            
    def get_flattened_architecture(self) -> list[Leaf]:
        with ParseExpressionsContext(self):
            processed, _ = self.parse_expressions()
            return processed.architecture._flatten(processed.variables)