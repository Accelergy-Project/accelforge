from fastfusion.frontend.mapper.mapper import Mapper
from fastfusion.frontend.renames import EinsumName, Renames
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
        einsum_name: EinsumName | None = None,
        **kwargs,
    ) -> tuple[Self, dict[str, Any]]:
        raise NotImplementedError("Call _spec_parse_expressions instead.")

    def _spec_parse_expressions(
        self,
        einsum_name: EinsumName | None = None,
        _parse_arch: bool = True,
        _parse_non_arch: bool = True,
    ) -> Self:
        """
        Parse all string expressions in the spec into concrete values.

        Parameters
        ----------
        einsum_name: EinsumName | None = None
            Optional Einsum name to populate symbols with the Einsum's symbols from the
            workload. If None, no symbols are populated from the workload.

        _parse_arch: bool = True
            Whether to parse the architecture.

        _parse_non_arch: bool = True
            Whether to parse the non-architecture fields.

        Returns
        -------
        Self
            The parsed specification.

        Raises
        ------
        ParseError
            If any field fails to parse; the error is annotated with the field path.
        """
        st = {}
        st["spec"] = self
        with ParseExpressionsContext(self):
            already_parsed = {}

            parsed_variables = self.variables
            if _parse_non_arch:
                try:
                    parsed_variables, st = self.variables._parse_expressions(st)
                except ParseError as e:
                    e.add_field("Spec().variables")
                    raise e
            already_parsed["variables"] = parsed_variables
            st.update(parsed_variables)
            st["variables"] = parsed_variables

            parsed_renames = self.renames
            if _parse_non_arch:
                try:
                    parsed_renames, st = self.renames._parse_expressions(st)
                except ParseError as e:
                    e.add_field("Spec().renames")
                    raise e
            already_parsed["renames"] = parsed_renames
            st["renames"] = parsed_renames

            parsed_workload = self.workload
            if _parse_non_arch:
                try:
                    parsed_workload, st = self.workload._parse_expressions(st, renames=parsed_renames)
                except ParseError as e:
                    e.add_field("Spec().workload")
                    raise e
            already_parsed["workload"] = parsed_workload
            st["workload"] = parsed_workload

            if einsum_name is not None:
                renames = parsed_workload.einsums[einsum_name].renames
                st.update(**{k.name: k.source for k in renames})

            if _parse_arch:
                parsed_arch, st = self.arch._parse_expressions(st)
            else:
                parsed_arch = self.arch
            st["arch"] = parsed_arch
            already_parsed["arch"] = parsed_arch

            parsed_spec, _ = super()._parse_expressions(
                st,
                already_parsed=already_parsed,
            )
            parsed_spec._parsed = True
            return parsed_spec

    def calculate_component_area_energy_latency_leak(
        self,
        area: bool = True,
        energy: bool = True,
        latency: bool = True,
        leak: bool = True,
        einsum_name: EinsumName | None = None,
    ) -> "Spec":
        """
        Populates per-component area, energy, latency, and/or leak power. For each
        component, populates the ``attributes.area``, ``attributes.total_area``,
        ``attributes.leak_power`` and ``attributes.total_leak_power``. Additionally, for
        each action of each component, populates the ``arguments.energy`` and
        ``arguments.latency`` fields. Extends the ``component_modeling_log`` field with
        log messages. Also populates the ``component_model`` attribute for each
        component if not already set.

        Some architectures' attributes may depend on the workload. In that case, an
        Einsum name can be provided to populate those symbols with the Einsum's symbols
        from the workload.

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
        einsum_name : EinsumName | None, optional
            Optional Einsum name to populate symbols with the Einsum's symbols from the
            workload. If None, no symbols are populated from the workload.
        """
        if not area and not energy and not latency and not leak:
            return self

        models = hwcomponents.get_models(
            self.config.component_models,
            include_installed=self.config.use_installed_component_models,
        )

        components = set()
        try:
            if not getattr(self, "_parsed", False):
                self = self._spec_parse_expressions(einsum_name=einsum_name)
            else:
                self = self.copy()
        except ParseError as e:
            if "arch" in e.message:
                e.add_note(
                    "If this error seems to be caused by a missing symbol that depends on \n"
                    "the workload, you may need to provide an appropriate einsum_name to \n"
                    "calculate_component_area_energy_latency_leak. This may occur if the \n"
                    "architecture depends on something in the workload.\n"
                )
            raise

        for arch in self._get_flattened_architecture():
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

    def _get_flattened_architecture(
        self,
        compute_node: str | Compute | None = None,
    ) -> list[list[Leaf]] | list[Leaf]:
        """
        Return the architecture as paths of ``Leaf`` instances from the highest-level
        node to each ``Compute`` node. Parses arithmetic expressions in the
        architecture for each one. If a symbol table is provided, it will be used to
        parse the expressions.

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
            found.append(self.arch._flatten(c))
            if found[-1][-1].name != c:
                raise ParseError(f"Compute node {c} not found in architecture")

        return found if compute_node is None else found[0]


Specification = Spec
