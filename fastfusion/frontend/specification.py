from fastfusion.frontend.mapper.mapper import Mapper
from fastfusion.frontend.renames import Renames
from fastfusion.util.parse_expressions import ParseError, ParseExpressionsContext
from fastfusion.frontend.arch import Compute, Leaf, Component, Arch
from fastfusion.frontend.constraints import Constraints
from fastfusion.frontend.workload import Workload
from fastfusion.frontend.variables import Variables
from fastfusion.frontend.components import Components
from fastfusion.frontend.config import Config, get_config
from fastfusion.frontend.component_area import ComponentArea, AreaEntry
from fastfusion.frontend.component_energy import ComponentEnergy, EnergyEntry
from fastfusion.frontend.mapping import Mapping
import hwcomponents

from typing import Any, Dict, Optional, Union
from fastfusion.util.basetypes import ParsableModel


class Specification(ParsableModel):
    arch: Arch = Arch()
    components: Components = Components()
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
        if not getattr(self, "_parsed", False):
            self, _ = self.parse_expressions()
        for arch in self.get_flattened_architecture():
            for component in arch:
                if component.name in components:
                    continue
                assert isinstance(component, Component)
                components.add(component.name)
                if area:
                    self.component_area.entries.append(
                        AreaEntry.from_models(
                            component.get_component_class,
                            component.attributes,
                            self,
                            models,
                            name=component.name,
                        )
                    )
                if energy:
                    self.component_energy.entries.append(
                        EnergyEntry.from_models(
                            component.get_component_class,
                            component.attributes,
                            [action.arguments for action in component.actions],
                            [action.name for action in component.actions],
                            self,
                            models,
                            name=component.name,
                        )
                    )

    def get_flattened_architecture(self, compute_node: Union[str, Compute] = None) -> list[list[Leaf]] | list[Leaf]:
        # Assert that we've been parsed
        assert getattr(self, "_parsed", False), "Specification must be parsed before getting flattened architecture"
        all_leaves = self.arch.get_instances_of_type(Leaf)
        found_names = set()
        for leaf in all_leaves:
            if leaf.name in found_names:
                raise ParseError(f"Duplicate name in architecture: {leaf.name}")
            found_names.add(leaf.name)

        found = []
        if compute_node is None:
            compute_nodes = [c.name for c in self.arch.get_instances_of_type(Compute)]
        else:
            compute_nodes = [compute_node.name if isinstance(compute_node, Compute) else compute_node]
        
        for c in compute_nodes:
            found.append(self.arch._flatten(self.variables, c))
            if found[-1][-1].name != c:
                raise ParseError(f"Compute node {c} not found in architecture")

        return found if compute_node is None else [found[0]]
