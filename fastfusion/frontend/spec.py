from fastfusion.frontend.mapper.mapper import Mapper
from fastfusion.frontend.renames import Renames
from fastfusion.util._parse_expressions import ParseError, ParseExpressionsContext
from fastfusion.frontend.arch import Compute, Leaf, Component, Arch, Fanout

from fastfusion.frontend.workload import Workload
from fastfusion.frontend.variables import Variables
from fastfusion.frontend.config import Config, get_config
from fastfusion.frontend.mapping import Mapping
from fastfusion.frontend.model import Model
import hwcomponents

from typing import Any, Dict, Optional, Self
from fastfusion.util._basetypes import ParsableModel
from pydantic import Field


class Spec(ParsableModel):
    """The top-level spec of all of the inputs to this package."""

    arch: Arch = Arch()
    """ The hardware architecture being used. """

    mapping: Mapping = Mapping()
    """ How the workload is programmed onto the architecture. Do not specify this if
    you'd like the mapper to generate a mapping for you. """

    workload: Workload = Workload()
    """ The program to be run on the arch. """

    variables: Variables = Variables()
    """ Variables that can be referenced in other places in the spec. """

    config: Config = Field(default_factory=get_config)
    """ Configuration settings. """

    renames: Renames = Renames()
    """ Aliases for tensors in the workload so that they can be called
    by canonical names when writing architecture constraints. For example, workload
    tensors may be renamed to "input", "output", and "weight"."""

    mapper: Mapper = Mapper()
    """ Configures the mapper used to map the workload onto the architecture. """

    model: Model = Model()
    """Configures the model used to evaluate mappings."""

    def _parse_expressions(
        self,
        symbol_table: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> tuple[Self, dict[str, Any]]:
        """
        Parse all string expressions in the spec into concrete values.

        Parameters
        ----------
        symbol_table : dict, optional
            Optional pre-populated symbols to seed parsing; a shallow copy is made and
            augmented with ``spec`` and ``variables``.
        kwargs : dict, optional
            Additional keyword arguments forwarded to the base
            ``ParsableModel._parse_expressions``.

        Returns
        -------
        tuple[Self, dict[str, Any]]
            A tuple of ``(parsed_specification, final_symbol_table)``.

        Raises
        ------ParseError
            If any field fails to parse; the error is annotated with the field path.
        """
        symbol_table = {} if symbol_table is None else symbol_table.copy()
        symbol_table["spec"] = self
        with ParseExpressionsContext(self):
            try:
                parsed_variables, _ = self.variables._parse_expressions(
                    symbol_table, **kwargs
                )
            except ParseError as e:
                e.add_field("Spec().variables")
                raise e
            symbol_table.update(parsed_variables)
            symbol_table["variables"] = parsed_variables
            return super()._parse_expressions(symbol_table, **kwargs)

    def calculate_component_area_energy_latency_leak(
        self,
        area: bool = True,
        energy: bool = True,
        latency: bool = True,
        leak: bool = True,
    ) -> "Spec":
        """
        Populates per-component area, energy, latency, and/or leak power. For each
        component, populates the ``attributes.area``, ``attributes.total_area``,
        ``attributes.leak_power`` and ``attributes.total_leak_power``. Additionally, for
        each action of each component, populates the ``arguments.energy`` and
        ``arguments.latency`` fields. Extends the ``component_modeling_log`` field with
        log messages. Also populates the ``component_model`` attribute for each
        component if not already set.

        Parameters
        ----------
        area : bool, optional
            Whether to compute and populate area entries.
        energy : bool, optional
            Whether to compute and populate energy entries.
        latency : bool, optional
            Whether to compute and populate latency entries.
        leak : bool, optional
            Whether to compute and populate leak power entries.
        """
        if not area and not energy and not latency and not leak:
            return self

        models = hwcomponents.get_models(
            self.config.component_models,
            include_installed=self.config.use_installed_component_models,
        )

        components = set()
        if not getattr(self, "_parsed", False):
            self, _ = self._parse_expressions()
        else:
            self = self.copy()

        for arch in self.get_flattened_architecture():
            fanout = 1
            for component in arch:
                fanout *= component.get_fanout()
                if component.name in components or isinstance(component, Fanout):
                    continue
                assert isinstance(component, Component)
                components.add(component.name)
                orig: Component = self.arch.find(component.name)
                if area:
                    c = component.calculate_area(models)
                    orig.attributes.area = c.attributes.area
                    orig.attributes.total_area = c.attributes.area * fanout
                if energy:
                    c = component.calculate_action_energy(models)
                    for a in c.actions:
                        orig_action = orig.actions[a.name]
                        orig_action.arguments.energy = a.arguments.energy
                if latency:
                    c = component.calculate_action_latency(models)
                    for a in c.actions:
                        orig_action = orig.actions[a.name]
                        orig_action.arguments.latency = a.arguments.latency
                if leak:
                    c = component.calculate_leak_power(models)
                    orig.attributes.leak_power = c.attributes.leak_power
                    orig.attributes.total_leak_power = c.attributes.leak_power * fanout
                orig.component_modeling_log.extend(c.component_modeling_log)
                orig.component_model = c.component_model

        return self

    def get_flattened_architecture(
        self, compute_node: str | Compute | None = None
    ) -> list[list[Leaf]] | list[Leaf]:
        """
        Return the architecture as paths of ``Leaf`` instances from the highest-level
        node to each ``Compute`` node. Parses arithmetic expressions in the
        architecture for each one.

        Parameters
        ----------
        compute_node : str or Compute, optional
            Optional compute node (name or ``Compute``) to restrict results to a single
            compute node.

        Returns
        -------
            - If ``compute_node`` is ``None``: list of lists of ``Leaf`` for all compute
              nodes.
            - Otherwise: a single-item list containing the list of ``Leaf`` for the
              requested node.

        Raises
        ------
        AssertionError
            If the spec has not been parsed.
        ParseError
            If there are duplicate names or the requested compute node cannot be found.
        """
        # Assert that we've been parsed
        assert getattr(
            self, "_parsed", False
        ), "Spec must be parsed before getting flattened architecture"
        all_leaves = self.arch.get_nodes_of_type(Leaf)
        found_names = set()
        for leaf in all_leaves:
            if leaf.name in found_names:
                raise ParseError(f"Duplicate name in architecture: {leaf.name}")
            found_names.add(leaf.name)

        found = []
        if compute_node is None:
            compute_nodes = [c.name for c in self.arch.get_nodes_of_type(Compute)]
        else:
            compute_nodes = [
                compute_node.name if isinstance(compute_node, Compute) else compute_node
            ]

        for c in compute_nodes:
            found.append(self.arch._flatten(self.variables, c))
            if found[-1][-1].name != c:
                raise ParseError(f"Compute node {c} not found in architecture")

        return found if compute_node is None else [found[0]]


Specification = Spec
