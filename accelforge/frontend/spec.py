from __future__ import annotations

from accelforge.frontend.mapper import FFM
from accelforge.frontend.renames import EinsumName, Renames
from accelforge.util._eval_expressions import EvaluationError, ParseExpressionsContext
from accelforge.frontend.arch import Compute, Leaf, Component, Arch, Container

from accelforge.frontend.workload import Workload
from accelforge.frontend.variables import Variables
from accelforge.frontend.config import Config
from accelforge.frontend.mapping import Mapping
from accelforge.frontend.model import Model
import hwcomponents

from accelforge._accelerated_imports import pd
from typing import Any, Callable, Dict, Optional, Self, TYPE_CHECKING
from accelforge.util._basetypes import EvalableModel
from pydantic import Field

if TYPE_CHECKING:
    from accelforge.mapper.FFM.mappings import Mappings


class Spec(EvalableModel):
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

    config: Config = Config()
    """ Configuration settings. """

    renames: Renames = Renames()
    """ Aliases for tensors in the workload so that they can be called
    by canonical names when writing architecture constraints. For example, workload
    tensors may be renamed to "input", "output", and "weight"."""

    mapper: FFM = FFM()
    """ Configures the mapper used to map the workload onto the architecture. """

    model: Model = Model()
    """Configures the model used to evaluate mappings."""

    def _eval_expressions(
        self,
        einsum_name: EinsumName | None = None,
        **kwargs,
    ) -> tuple[Self, dict[str, Any]]:
        raise NotImplementedError("Call _spec_eval_expressions instead.")

    def _spec_eval_expressions(
        self,
        einsum_name: EinsumName | None = None,
        eval_arch: bool = True,
        eval_non_arch: bool = True,
    ) -> Self:
        """
        Parse all string expressions in the spec into concrete values.

        Parameters
        ----------
        einsum_name: EinsumName | None = None
            Optional Einsum name to populate symbols with the Einsum's symbols from the
            workload. If None, only some symbols may be populated from the workload.

        eval_arch: bool = True
            Whether to evaluate expressions in the the architecture.

        eval_non_arch: bool = True
            Whether to evaluate expressions in the non-architecture fields.

        Returns
        -------
        Self
            The evaluated specification.

        Raises
        ------
        EvaluationError
            If any field fails to evaluate; the error is annotated with the field path.
        """
        st = {}
        st["spec"] = self
        with ParseExpressionsContext(self):
            already_evaluated = {}

            evaluated_variables = self.variables
            if eval_non_arch:
                try:
                    evaluated_variables, st = self.variables._eval_expressions(st)
                except EvaluationError as e:
                    e.add_field("Spec().variables")
                    raise e
            already_evaluated["variables"] = evaluated_variables
            st.update(evaluated_variables)
            st["variables"] = evaluated_variables

            evaluated_renames = self.renames
            if eval_non_arch:
                try:
                    evaluated_renames, st = self.renames._eval_expressions(st)
                except EvaluationError as e:
                    e.add_field("Spec().renames")
                    raise e
            already_evaluated["renames"] = evaluated_renames
            st["renames"] = evaluated_renames

            evaluated_workload = self.workload
            if eval_non_arch:
                try:
                    evaluated_workload, st = self.workload._eval_expressions(
                        st, renames=evaluated_renames
                    )
                except EvaluationError as e:
                    e.add_field("Spec().workload")
                    raise e
            already_evaluated["workload"] = evaluated_workload
            st["workload"] = evaluated_workload

            if einsum_name is not None:
                renames = evaluated_workload.einsums[einsum_name].renames
                st.update(**{k.name: k.source for k in renames})
            else:
                st.update(evaluated_workload.empty_renames())

            if eval_arch:
                evaluated_arch, st = self.arch._eval_expressions(st)
            else:
                evaluated_arch = self.arch
            st["arch"] = evaluated_arch
            already_evaluated["arch"] = evaluated_arch

            evaluated_spec, _ = super()._eval_expressions(
                st,
                already_evaluated=already_evaluated,
            )
            evaluated_spec._evaluated = True
            return evaluated_spec

    def calculate_component_area_energy_latency_leak(
        self,
        einsum_name: EinsumName | None = None,
        area: bool = True,
        energy: bool = True,
        latency: bool = True,
        leak: bool = True,
    ) -> "Spec":
        """
        Populates per-component area, energy, latency, and/or leak power. For each
        component, populates the ``area``, ``total_area``, ``leak_power`` and
        ``total_leak_power``. Additionally, for each action of each component, populates
        the ``<action>.energy`` and ``<action>.latency`` fields. Extends the
        ``component_modeling_log`` field with log messages. Also populates the
        ``component_model`` attribute for each component if not already set.

        Some architectures' attributes may depend on the workload. In that case, an
        Einsum name can be provided to populate those symbols with the Einsum's symbols
        from the workload.

        Parameters
        ----------
        einsum_name: EinsumName | None = None
            Optional Einsum name to populate symbols with the Einsum's symbols from the
            workload. If None, and there are Einsums in the workload, the first Einsum
            is used. If None and there are no Einsums in the workload, then no symbols
            are populated from the workload.
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

        if einsum_name is None and len(self.workload.einsums) > 0:
            einsum_name = self.workload.einsums[0].name

        components = set()
        try:
            if not getattr(self, "_evaluated", False):
                self = self._spec_eval_expressions(einsum_name=einsum_name)
            else:
                self = self.model_copy()
                self.arch = self.arch.model_copy(deep=True)
        except EvaluationError as e:
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
                if component.name in components or isinstance(component, Container):
                    continue
                assert isinstance(component, Component)
                components.add(component.name)
                orig: Component = self.arch.find(component.name)
                c = component
                if area:
                    c = c.calculate_area(models)
                    orig.area = c.area
                    orig.total_area = c.area * fanout
                if energy:
                    c = c.calculate_action_energy(models)
                    for a in c.actions:
                        orig_action = orig.actions[a.name]
                        orig_action.energy = a.energy
                if latency:
                    c = c.calculate_action_latency(models)
                    for a in c.actions:
                        orig_action = orig.actions[a.name]
                        orig_action.latency = a.latency
                if leak:
                    c = c.calculate_leak_power(models)
                    orig.leak_power = c.leak_power
                    orig.total_leak_power = c.leak_power * fanout
                orig.component_modeling_log.extend(c.component_modeling_log)
                orig.component_model = c.component_model

        return self

    def _get_flattened_architecture(
        self,
        compute_node: str | Compute | None = None,
        einsum_name: EinsumName | None = None,
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

        einsum_name: EinsumName | None = None
            Optional Einsum name to populate symbols with the Einsum's symbols from the
            workload. If None, no symbols are populated from the workload.

        Returns
        -------
            - If ``compute_node`` is ``None``: list of lists of ``Leaf`` for all compute
              nodes.
            - Otherwise: a single-item list containing the list of ``Leaf`` for the
              requested node.

        Raises
        ------
        AssertionError
            If the spec has not been evaluated.
        EvaluationError
            If there are duplicate names or the requested compute node cannot be found.
        """
        # Assert that we've been evaluated
        assert getattr(
            self, "_evaluated", False
        ), "Spec must be evaluated before getting flattened architecture"
        if einsum_name is not None:
            self = self._spec_eval_expressions(einsum_name=einsum_name)

        all_leaves = self.arch.get_nodes_of_type(Leaf)
        found_names = set()
        for leaf in all_leaves:
            if leaf.name in found_names:
                raise EvaluationError(f"Duplicate name in architecture: {leaf.name}")
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
                raise EvaluationError(f"Compute node {c} not found in architecture")

        # These can't be pickled if they use dynamically-loaded code
        for f in found:
            for n in f:
                if hasattr(n, "component_model"):
                    n.component_model = None

        return found if compute_node is None else found[0]

    def evaluate_mapping(self) -> Mappings:
        """
        Evaluate the mapping in the spec.
        """
        from accelforge.model import evaluate_mapping

        return evaluate_mapping(self)

    def map_workload_to_arch(
        self,
        einsum_names: list[EinsumName] | None = None,
        one_pbar_only: bool = False,
        print_progress: bool = True,
        print_number_of_pmappings: bool = True,
        _pmapping_row_filter_function: Callable[[pd.Series], bool] | None = None,
    ) -> Mappings:
        """
        Maps the workload to the architecture using the AccelForge Fast and Fusiest
        Mapper (FFM).

        Parameters
        ----------
        spec:
            The Spec to map.
        einsum_names:
            The einsum names to map. If None, all einsums will be mapped.
        can_combine_multiple_runs: Whether we would like to be able to combine multiple
            make_pmappings runs. Having this as True allows you to do things like
            `pmappings = make_pmappings(*args_a) | make_pmappings(*args_b)` but slows
            down execution.
        cache_dir:
            The directory to cache pmappings in. If None, no caching will be done.
        one_pbar_only:
            Whether to only print only a single progress bar. If this is True, then only
            a progress bar will be created for making tile shapes, which is generally
            the longest-running part of the mapping process.
        print_progress:
            Whether to print progress of the mapping process, including progress bars.
        print_number_of_pmappings:
            Whether to print the number of pmappings for each einsum.
        _pmapping_row_filter_function:
            A function that takes in a row of the pmapping dataframe and returns True if
            the row should be included in the final mappings, and False otherwise. If
            None, all rows will be included.

        Returns
        -------
        Mappings
            The mappings of the workload to the architecture.
        """
        from accelforge.mapper.FFM.main import map_workload_to_arch

        return map_workload_to_arch(
            self,
            einsum_names=einsum_names,
            one_pbar_only=one_pbar_only,
            print_progress=print_progress,
            print_number_of_pmappings=print_number_of_pmappings,
            _pmapping_row_filter_function=_pmapping_row_filter_function,
        )


Specification = Spec
