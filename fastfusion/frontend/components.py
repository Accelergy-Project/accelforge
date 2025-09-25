from typing import Annotated, Any, Dict, Iterator, Optional, Union

from pydantic import ConfigDict
from fastfusion.util.basetypes import (
    NonParsableModel,
    ParsableList,
    ParsableModel,
    ParsesTo,
    ParseExtras,
)
from fastfusion.version import assert_version, __version__


class ComponentAttributes(ParseExtras):
    global_cycle_period: ParsesTo[Union[int, float]] = "REQUIRED"
    n_instances: ParsesTo[Union[int, float]] = 1
    energy_scale: ParsesTo[Union[int, float]] = 1
    area_scale: ParsesTo[Union[int, float]] = 1
    energy: ParsesTo[Union[int, float, None]] = None
    area: ParsesTo[Union[int, float, None]] = None
    model_config = ConfigDict(extra="allow")

    def parse_expressions(
        self,
        symbol_table: Optional[Dict[str, Any]] = None,
        inherit_all: bool = False,
        multiply_multipliers: bool = False,
        **kwargs,
    ) -> tuple[Any, dict[str, Any]]:
        new_self, new_symbol_table = super().parse_expressions(
            symbol_table, **kwargs, multiply_multipliers=multiply_multipliers
        )

        if multiply_multipliers:
            if "n_instances" in symbol_table:
                new_self.n_instances *= symbol_table["n_instances"]
            if "energy_scale" in symbol_table:
                new_self.energy_scale *= symbol_table["energy_scale"]
            if "area_scale" in symbol_table:
                new_self.area_scale *= symbol_table["area_scale"]

        if inherit_all:
            for key, value in symbol_table.items():
                if isinstance(key, str) and not hasattr(new_self, key):
                    setattr(new_self, key, value)
        else:
            for key, value in symbol_table.items():
                if isinstance(key, str) and getattr(new_self, key, None) is None:
                    setattr(new_self, key, value)

        return new_self, new_symbol_table


class Subcomponent(ParsableModel):
    name: str
    component_class: str
    attributes: ComponentAttributes = ComponentAttributes()


class SubcomponentAction(ParsableModel):
    name: str
    arguments: ComponentAttributes = ComponentAttributes()


class SubcomponentActionGroup(ParsableModel):
    name: str
    actions: ParsableList[SubcomponentAction]


class Action(ParsableModel):
    name: str
    subcomponents: ParsableList[SubcomponentActionGroup]


class CompoundComponent(ParsableModel):
    name: str
    attributes: ComponentAttributes
    subcomponents: ParsableList[Subcomponent]
    actions: ParsableList[Action]

    def get_subcomponent_actions(
        self,
        action_name: str,
        attributes: ComponentAttributes,
        arguments: ComponentAttributes,
    ) -> Iterator[tuple[str, ComponentAttributes, ComponentAttributes, str]]:
        try:
            action: Action = self.actions[action_name]
        except KeyError:
            raise KeyError(f"Action {action_name} not found in {self.name}") from None

        for subcomponent in action.subcomponents:
            try:
                component: Subcomponent = self.subcomponents[subcomponent.name]
            except KeyError:
                raise KeyError(
                    f"Subcomponent {subcomponent.name} referenced in action {action_name} of {self.name} not found"
                ) from None
            component_attributes = component.attributes.parse_expressions(
                attributes.model_dump_non_none(), multiply_multipliers=True
            )[0]
            arguments = arguments.parse_expressions(
                component_attributes.model_dump_non_none(), multiply_multipliers=False
            )[0]
            for subaction in subcomponent.actions:
                subaction_args = subaction.arguments.parse_expressions(
                    arguments.model_dump_non_none(), multiply_multipliers=True
                )[0]
                yield (
                    component.get_component_class(),
                    component_attributes,
                    subaction_args,
                    subaction.name,
                )


# Components are only instantiated when they are called in the arch. The
# top-level components are non-parsable.
class Components(NonParsableModel):
    version: Annotated[str, assert_version] = __version__
    components: ParsableList[CompoundComponent] = ParsableList()
