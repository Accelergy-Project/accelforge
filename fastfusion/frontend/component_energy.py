import copy
from typing import Callable, Union, Annotated
from fastfusion.frontend.components import ComponentAttributes
from fastfusion.version import assert_version, __version__
from fastfusion.util.basetypes import ParsableModel, ParsableList, ParsesTo
from hwcomponents import get_energy

class Subaction(ParsableModel):
    name: str
    attributes: ComponentAttributes = ComponentAttributes()
    arguments: ComponentAttributes = ComponentAttributes()
    energy: ParsesTo[Union[int, float]]
    model_name: Union[str, None] = None
    messages: list[str] = []

class Action(ParsableModel):
    name: str
    arguments: ComponentAttributes = ComponentAttributes()
    energy: ParsesTo[Union[int, float]]
    subactions: ParsableList['Subaction']

    @staticmethod
    def from_models(
        class_name: str | Callable[[], str],
        attributes: dict,
        arguments: dict,
        action_name: str,
        spec: "Specification",
        models: list,
        return_subactions: bool = False,
    ) -> Union["EnergyEntry", list["Subaction"]]:
        attributes, arguments = copy.copy((attributes, arguments))
        entries = []

        definition = None
        try:
            definition = spec.components.components[class_name]
        except KeyError:
            pass
                
        if arguments.energy is not None:
            entries = [
                Subaction(
                    name=action_name,
                    attributes=attributes,
                    arguments=arguments,
                    energy=arguments.energy
                    * attributes.energy_scale
                    * arguments.energy_scale,
                    messages=["Using predefined energy value"],
                )
            ]
        elif attributes.energy is not None:
            entries = [
                Subaction(
                    name=action_name,
                    attributes=attributes,
                    arguments=arguments,
                    energy=attributes.energy
                    * attributes.energy_scale
                    * arguments.energy_scale,
                    messages=["Using predefined energy value"],
                )
            ]
        else:
            class_name = class_name if isinstance(class_name, str) else class_name()
            try:
                definition = spec.components.components[class_name]
            except KeyError:
                pass
            if definition is not None:
                for (
                    component,
                    component_attributes,
                    sub_arguments,
                    subaction_name,
                ) in definition.get_subcomponent_actions(
                    action_name, attributes, arguments
                ):
                    entries.extend(
                        Action.from_models(
                            component,
                            component_attributes,
                            sub_arguments,
                            subaction_name,
                            spec,
                            models,
                            return_subactions=True,
                        )
                    )
            else:
                estimation = get_energy(
                    component_name=class_name,
                    component_attributes=attributes.model_dump(),
                    action_name=action_name,
                    action_arguments=arguments.model_dump(),
                    models=models,
                )
                energy = estimation.value
                entries.append(
                    Subaction(
                        name=class_name,
                        attributes=attributes,
                        arguments=arguments,
                        energy=energy * attributes.energy_scale * arguments.energy_scale,
                        model_name=estimation.model_name,
                        messages=estimation.messages,
                    )
                )

        if return_subactions:
            return entries

        energy = sum(subaction.energy for subaction in entries)
        return Action(
            name=action_name,
            subactions=entries,
            energy=energy,
        )

class EnergyEntry(ParsableModel):
    name: str
    attributes: ComponentAttributes = ComponentAttributes()
    actions: ParsableList[Action]

    def find_action(self, name: str):
        for action in self.actions:
            if name == action.name:
                return action

    def __hash__(self) -> int:
        return id(self)

    @staticmethod
    def from_models(
        class_name: str | Callable[[], str],
        attributes: dict,
        arguments: list[dict],
        action_names: list[str],
        spec: "Specification",
        models: list,
        name: str,
    ):
        actions = []
        for action_name, action_arguments in zip(action_names, arguments):
            actions.append(
                Action.from_models(
                    class_name,
                    attributes,
                    action_arguments,
                    action_name,
                    spec,
                    models,
                )
            )
        return EnergyEntry(name=name, actions=actions, attributes=attributes)

class ComponentEnergy(ParsableModel):
    version: Annotated[str, assert_version] = __version__
    entries: ParsableList[EnergyEntry] = ParsableList()

    def isempty(self) -> bool:
        return len(self.entries) == 0

    def to_dict(self):
        r = {}
        for t in self.entries:
            r[t.name] = {}
            for a in t.actions:
                r[(t.name, a.name)] = a.energy
        return r

    def find_action(self, component: str, action: str):
        try:
            return self.entries[component].find_action(action)
        except KeyError:
            pass
        raise KeyError(
            f"Could not find energy for component {component} action {action}. Try "
            f"running calculate_component_energy_area() on the Specification first."
        )