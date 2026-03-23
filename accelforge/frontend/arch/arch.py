from typing import (
    Any,
)

from accelforge.util._basetypes import (
    EvalExtras,
    _PostCall,
)

from accelforge.frontend.arch.structure import Leaf, Branch, Hierarchical
from accelforge.frontend.arch.components import Component
from accelforge.frontend.arch.spatialable import Spatialable, Spatial


class Arch(Hierarchical):
    """
    Top-level architecture specification.

    All attributes in the architecture can refrence variables in the spec-level
    `variables` field as well as symbols from the individual Einsum being processed.
    """

    variables: EvalExtras = EvalExtras()
    """
    Like the spec-level `variables` field, this field is evaluated first and its contents
    can be referenced elsewhere in the architecture. Unlike the spec-level `variables`
    field, this, like ther rest of the architecture, is evaluated per-Einsum and can
    reference Einsum-specific symbols.
    """

    extra_attributes_for_all_component_models: EvalExtras = EvalExtras()
    """
    Extra attributes to pass to all component models. This can be used to pass global
    attributes, such as technology node or clock period, to every component model.
    """

    @property
    def total_area(self) -> float:
        """
        Returns the total area of the architecture in m^2.

        Returns
        -------
        float
            The total area of the architecture in m^2.
        """
        return sum(self.per_component_total_area.values())

    @property
    def total_leak_power(self) -> float:
        """
        Returns the total leak power of the architecture in W.

        Returns
        -------
        float
            The total leak power of the architecture in W.
        """
        return sum(self.per_component_total_leak_power.values())

    @property
    def per_component_total_area(self) -> dict[str, float]:
        """
        Returns the total area used by each component in the architecture in m^2.

        Returns
        -------
        dict[str, float]
            A dictionary of component names to their total area in m^2.
        """
        area = {
            node.name: node.total_area for node in self.get_nodes_of_type(Component)
        }
        for k, v in area.items():
            if v is None:
                raise ValueError(
                    f"Area of {k} is not set. Please call the Spec's "
                    "`calculate_component_area_energy_latency_leak` method before accessing this "
                    "property."
                )
        return area

    @property
    def per_component_total_leak_power(self) -> dict[str, float]:
        """
        Returns the total leak power of each component in the architecture in W.

        Returns
        -------
        dict[str, float]
            A dictionary of component names to their total leak power in W.
        """
        leak_power = {
            node.name: node.total_leak_power
            for node in self.get_nodes_of_type(Component)
        }
        for k, v in leak_power.items():
            if v is None:
                raise ValueError(
                    f"Leak power of {k} is not set. Please call the Spec's "
                    "`calculate_component_area_energy_latency_leak` method before accessing this "
                    "property."
                )
        return leak_power

    def _eval_expressions(self, symbol_table: dict[str, Any], *args, **kwargs):
        outer_st = symbol_table

        class PostCallArch(_PostCall):
            def __call__(self, field, value, evaluated, symbol_table):
                if field == "variables":
                    # We're going to override the spec-level "variables", so make sure
                    # we copy over all the symbols from the spec-level "variables".
                    evaluated_dump = evaluated.shallow_model_dump()
                    for k, v in symbol_table.get("variables", {}).items():
                        if k not in evaluated_dump:
                            evaluated_dump[k] = v
                    symbol_table.update(evaluated_dump)
                    symbol_table["variables"] = evaluated_dump
                if field == "extra_attributes_for_all_component_models":
                    evaluated_dump = evaluated.shallow_model_dump()
                    symbol_table["arch_extra_attributes_for_all_component_models"] = (
                        evaluated_dump
                    )
                return evaluated

        cur_st = dict(symbol_table)

        for node in self.get_nodes_of_type(Leaf):
            cur_st[node.name] = node

        evaluated, _ = super()._eval_expressions(
            cur_st,
            *args,
            **kwargs,
            post_calls=(PostCallArch(),),
            order=(
                "variables",
                "extra_attributes_for_all_component_models",
            ),
        )
        return evaluated, symbol_table

    def __getitem__(self, name: str) -> Leaf:
        attr_error = None
        find_error = None
        try:
            return super().__getitem__(name)
        except KeyError as e1:
            attr_error = e1
        try:
            return self.find(name)
        except ValueError as e2:
            find_error = e2

        e1_str = "\n\t" + str(attr_error).replace("\n", "\n\t")
        e2_str = "\n\t" + str(find_error).replace("\n", "\n\t")

        errstr = (
            f"Could not access {name} in {self}, either as an attribute or as the name "
            f"of a leaf node. Error for accessing as an attribute:{e1_str}\n"
            f"Error for accessing as the name of a leaf node:{e2_str}"
        )

        raise KeyError(errstr)

    def find_spatial(
        self, name: str, return_spatialable: bool = False
    ) -> Spatial | tuple[Spatialable, Spatial]:
        """
        Find a spatial dimension by name. Raises an error if zero or more than one
        spatial dimension has the given name.

        Parameters
        ----------
        name: str
            The name of the spatial dimension to find.
        return_spatialable: bool
            Whether to return the node that the spatial dimension is associated with.

        Returns
        -------
        Spatial
            The spatial dimension with the given name if `return_spatialable` is False.
        tuple[Spatialable, Spatial]
            The node and spatial dimension with the given name if `return_spatialable`
            is True.

        Raises
        ------
        ValueError
            If no spatial dimension with the given name exists, or if more than one
            spatial dimension with the given name exists.
        """

        matches = []
        for node in self.get_nodes_of_type(Spatialable):
            for s in node.spatial:
                if s.name == name:
                    matches.append((node, s))

        if len(matches) == 0:
            raise ValueError(
                f"No spatial dimension with name '{name}' found in the architecture"
            )
        if len(matches) > 1:
            node_names = [getattr(n, "name", type(n).__name__) for n, _ in matches]
            raise ValueError(
                f"Multiple spatials with name '{name}' found in the architecture "
                f"(in nodes: {node_names}). Expected exactly one."
            )
        if return_spatialable:
            return matches[0]
        return matches[0][1]

    def model_post_init(self, __context__=None) -> None:
        # Make sure all leaf names are unique
        leaves = {}
        for l in self.get_nodes_of_type(Leaf):
            n = l.name
            leaves.setdefault(n, l)
            assert l is leaves[n], f"Duplicate name {n} found in architecture"
