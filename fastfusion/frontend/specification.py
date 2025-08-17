from fastfusion.frontend.mapper.mapper import Mapper
from fastfusion.frontend.renames import Renames
from fastfusion.util.parse_expressions import ParseError, ParseExpressionsContext
from fastfusion.frontend.architecture import Compute, Leaf
from fastfusion.frontend.architecture import Architecture
from fastfusion.frontend.constraints import Constraints
from fastfusion.frontend.workload import Workload
from fastfusion.frontend.variables import Variables
from fastfusion.frontend.component_classes import Components
from fastfusion.frontend.config import Config, get_config
from fastfusion.frontend.component_area import ComponentArea, AreaEntry
from fastfusion.frontend.component_energy import ComponentEnergy, EnergyEntry
from fastfusion.frontend.mapping import Mapping
import hwcomponents

from typing import Any, Dict, Optional, Union
from fastfusion.util.basetypes import ParsableModel


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
    mapper: Mapper = Mapper()

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
            try:
                parsed_variables, _ = self.variables.parse_expressions(symbol_table, **kwargs)
            except ParseError as e:
                e.add_field("Specification().variables")
                raise e
            symbol_table.update(parsed_variables)
            symbol_table["variables"] = parsed_variables
            return super().parse_expressions(symbol_table, **kwargs)
        
    def calculate_component_energy_area(self, energy: bool = True, area: bool = True):
        self.component_energy = ComponentEnergy() if energy else self.component_energy
        self.component_area = ComponentArea() if area else self.component_area
        models = hwcomponents.get_models(
            self.config.component_models, 
            include_installed=self.config.use_installed_component_models
        )
        
        components = set()
        arches, processed = self.get_flattened_architecture(return_processed=True)
        for arch in arches:
            for component in arch:
                if component.name in components:
                    continue
                components.add(component.name)
                self.component_area.entries.append(
                    AreaEntry.from_models(
                        component.get_component_class(),
                        component.attributes,
                        processed,
                        models,
                        name=component.name,
                    )
                )
                action_names = [action.name for action in component.actions]
                action_args = [action.arguments for action in component.actions]
                self.component_energy.entries.append(
                    EnergyEntry.from_models(
                        component.get_component_class(),
                        component.attributes,
                        action_args,
                        action_names,
                        processed,
                        models,
                        name=component.name,
                    )
                )
            
    def get_flattened_architecture(self, compute_node: Union[str, Compute] = None, return_processed: bool=False) -> list[list[Leaf]] | list[Leaf]:
        all_leaves = self.architecture.get_instances_of_type(Leaf)
        found_names = set()
        for leaf in all_leaves:
            if leaf.name in found_names:
                raise ParseError(f"Duplicate name in architecture: {leaf.name}")
            found_names.add(leaf.name)

        found = []
        with ParseExpressionsContext(self):
            processed, _ = self.parse_expressions()
        
            if compute_node is None:
                compute_nodes = [c.name for c in self.architecture.get_instances_of_type(Compute)]
            else:
                compute_nodes = [compute_node.name if isinstance(compute_node, Compute) else compute_node]
            
            for c in compute_nodes:
                found.append(processed.architecture._flatten(processed.variables, c))
                if found[-1][-1].name != c:
                    raise ParseError(f"Compute node {c} not found in architecture")

        found = found if compute_node is None else [found[0]]
        return (found, processed) if return_processed else found
