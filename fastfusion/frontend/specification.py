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

from typing import Any, Dict, Optional, Union, Self
from fastfusion.util.basetypes import ParsableModel
from pydantic import Field


class Specification(ParsableModel):
    """Top-level specification class."""

    arch: Arch = Arch()
    """ The hardware being used. """

    components: Components = Components()
    """ Component classes that may be instantiated in the architecture. Component
    classes include compound components only; primitive components may be used directly
    in the architecture without being defined here. """

    constraints: Constraints = Constraints()
    """ Constrains how the workload is mapped onto the architecture. May be
    defined here or directly in the architecture. """

    mapping: Mapping = Mapping()
    """ How the workload is programmed onto the architecture. """

    workload: Workload = Workload()
    """ The program to be run on the architecture. """

    variables: Variables = Variables()
    """ Top-level variables that can be referenced in other places in the spec. """

    config: Config = Field(default_factory=get_config)
    """ Top-level configuration settings. """

    component_energy: ComponentEnergy = ComponentEnergy()
    """ To be deprecated. """

    component_area: ComponentArea = ComponentArea()
    """ To be deprecated. """

    renames: Renames = Renames()
    """ Aliases for tensors in the workload so that they can be called
    by canonical names when writing architecture constraints. For example, workload
    tensors may be renamed to "input", "output", and "weight"."""

    mapper: Mapper = Mapper()
    """ Configures the mapper used to map the workload onto the architecture. """

    def _parse_expressions(
        self,
        symbol_table: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> tuple[Self, dict[str, Any]]:
        """
        Parse all string expressions in the specification into concrete values.

        :param symbol_table: Optional pre-populated symbols to seed parsing; a
            shallow copy is made and augmented with ``spec`` and ``variables``.
        :param kwargs: Additional keyword arguments forwarded to the base
            ``ParsableModel._parse_expressions``.
        :returns: A tuple of ``(parsed_specification, final_symbol_table)``.
        :raises ParseError: If any field fails to parse; the error is annotated
            with the field path.
        """
        symbol_table = {} if symbol_table is None else symbol_table.copy()
        symbol_table["spec"] = self
        with ParseExpressionsContext(self):
            try:
                parsed_variables, _ = self.variables._parse_expressions(
                    symbol_table, **kwargs
                )
            except ParseError as e:
                e.add_field("Specification().variables")
                raise e
            symbol_table.update(parsed_variables)
            symbol_table["variables"] = parsed_variables
            return super()._parse_expressions(symbol_table, **kwargs)

    def calculate_component_energy_area(
        self, energy: bool = True, area: bool = True
    ) -> None:
        """
        Populate per-component area and/or energy entries using installed
        component models.

        Ensures the specification is parsed before evaluation.

        :param energy: Whether to compute and populate energy entries.
        :param area: Whether to compute and populate area entries.
        :returns: None. Updates ``component_energy`` and/or ``component_area`` in
            place.
        :raises ParseError: If parsing fails or evaluation detects invalid
            component references while flattening the architecture.
        """
        self.component_energy = ComponentEnergy() if energy else self.component_energy
        self.component_area = ComponentArea() if area else self.component_area
        models = hwcomponents.get_models(
            self.config.component_models,
            include_installed=self.config.use_installed_component_models,
        )

        components = set()
        if not getattr(self, "_parsed", False):
            self, _ = self._parse_expressions()
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

    def get_flattened_architecture(
        self, compute_node: Union[str, Compute] = None
    ) -> list[list[Leaf]] | list[Leaf]:
        """
        Return the architecture as paths of ``Leaf`` instances from each
        ``Compute`` node to its leaves.

        :param compute_node: Optional compute node (name or ``Compute``) to
            restrict results to a single compute node.
        :returns:
            - If ``compute_node`` is ``None``: list of lists of ``Leaf`` for all
              compute nodes.
            - Otherwise: a single-item list containing the list of ``Leaf`` for
              the requested node.
        :raises AssertionError: If the specification has not been parsed.
        :raises ParseError: If there are duplicate names or the requested compute
            node cannot be found.
        """
        # Assert that we've been parsed
        assert getattr(
            self, "_parsed", False
        ), "Specification must be parsed before getting flattened architecture"
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
            compute_nodes = [
                compute_node.name if isinstance(compute_node, Compute) else compute_node
            ]

        for c in compute_nodes:
            found.append(self.arch._flatten(self.variables, c))
            if found[-1][-1].name != c:
                raise ParseError(f"Compute node {c} not found in architecture")

        return found if compute_node is None else [found[0]]


Spec = Specification
