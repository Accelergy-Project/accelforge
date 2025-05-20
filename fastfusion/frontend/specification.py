import copy
from fastfusion.frontend import renames
from fastfusion.frontend._refs2copies import refs2copies_fast
from fastfusion.frontend.renames import Renames
from fastfusion.yamlparse.parse_expressions import ParseExpressionsContext
from . import arch, constraints, variables, workload
from fastfusion.yamlparse.nodes import ListNode
from .arch import Architecture
from .constraints import Constraints, ConstraintsList
from .workload import Workload
from .variables import Variables
from .components import Components
from .config import Config, get_config
from .area_table import ComponentArea, AreaEntry
from .energy_table import ComponentEnergy, EnergyEntry

from typing import Any, Dict, List, Optional, Union
from fastfusion.yamlparse.nodes import DictNode

from ..plugin.gather_plug_ins import gather_plug_ins


class Specification(DictNode):
    """
    A top-level class for the Timeloop specification.

    Attributes:
        architecture: The top-level architecture description.
        components: List of compound components.
        constraints: Additional constraints on the architecture and mapping.
        mapping: Additional constraints on the architecture and mapping.
        workload: The workload specification.
        variables: Variables to be used in parsing.
        mapper: Directives to control the mapping process.
        sparse_optimizations: Additional sparse optimizations available to the architecture.
        mapspace: The top-level mapspace description.
        config: Configuration of extra parsing functions and environment variables.

    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("architecture", Architecture)
        super().add_attr(
            "components", Components, {"version": 0.5}, part_name_match=True
        )
        super().add_attr(
            "constraints", Constraints, {"version": 0.5}, part_name_match=True
        )
        super().add_attr("mapping", ConstraintsList, [], part_name_match=True)
        super().add_attr("variables", Variables, {"version": 0.5})
        super().add_attr("workload", Workload, {"version": 0.5})
        super().add_attr("config", Config, {"version": 0.5}, part_name_match=True)
        super().add_attr(
            "component_energy", ComponentEnergy, {"version": 0.5, "tables": []}
        )
        super().add_attr(
            "component_area", ComponentArea, {"version": 0.5, "tables": []}
        )
        super().add_attr("renames", Renames, {"version": 0.5})

    def __init__(self, *args, **kwargs):
        i = 0
        while f"config{i}" in kwargs:
            i += 1
        kwargs[f"config{i}"] = get_config()
        super().__init__(*args, **kwargs)
        self.architecture: arch.Architecture = self["architecture"]
        self.constraints: constraints.Constraints = self["constraints"]
        self.mapping: constraints.Constraints = self["mapping"]
        self.workload: workload.Workload = self["workload"]
        self.variables: variables.Variables = self["variables"]
        self.components: ListNode = self["components"]
        self.config: Config = self["config"]
        self.renames: renames.Renames = self["renames"]

    def parse_expressions(
        self,
        symbol_table: Optional[Dict[str, Any]] = None,
        parsed_ids: Optional[set] = None,
    ) -> "Specification":
        self = copy.deepcopy(self)
        self = refs2copies_fast(self, self)

        symbol_table = {} if symbol_table is None else symbol_table.copy()
        parsed_ids = set() if parsed_ids is None else parsed_ids
        parsed_ids.add(id(self))
        parsed_ids.add(id(self.variables))
        symbol_table["spec"] = self
        with ParseExpressionsContext(self):
            parsed_variables = self.variables.parse_expressions(
                symbol_table, parsed_ids
            )
            symbol_table.update(parsed_variables)
            symbol_table["variables"] = parsed_variables
            super().parse_expressions(symbol_table, parsed_ids)
            
        return self

    def to_diagram(
        self,
        container_names: Union[str, List[str]] = (),
        ignore_containers: Union[str, List[str]] = (),
    ) -> "pydot.Graph":
        from .processors.to_diagram_processor import ToDiagramProcessor

        s = self._process()
        proc = ToDiagramProcessor(container_names, ignore_containers, spec=s)
        return proc.process(s)

    def estimate_energy_area(self):
        plug_ins = gather_plug_ins(self.config.component_plug_ins)
        with ParseExpressionsContext(self):
            processed = self.parse_expressions()
            components = processed.architecture._flatten(processed.variables)
            area = ComponentArea()
            energy = ComponentEnergy()
            for component in components:
                area.tables.append(
                    AreaEntry.from_plug_ins(
                        component._class,
                        component.attributes,
                        processed,
                        plug_ins,
                        name=component.name,
                    )
                )

                action_names = [action.name for action in component.actions]
                action_args = [action.arguments for action in component.actions]
                energy.tables.append(
                    EnergyEntry.from_plug_ins(
                        component._class,
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

    def get_flattened_architecture(self) -> list[arch.Leaf]:
        with ParseExpressionsContext(self):
            processed = self.parse_expressions()
            return processed.architecture._flatten(processed.variables)