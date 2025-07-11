import copy
from typing import Union, Annotated
from fastfusion.frontend.component_classes import ComponentAttributes
from fastfusion.version import assert_version, __version__
from fastfusion.util.basetypes import ParsableDict, ParsableModel, ParsableList, ParsesTo
from numbers import Number
from fastfusion.plugin.query_plug_ins import EnergyAreaQuery
from fastfusion.plugin.query_plug_ins import get_best_estimate

class Subaction(ParsableModel):
    name: str
    attributes: ComponentAttributes = ComponentAttributes()
    arguments: ComponentAttributes = ComponentAttributes()
    energy: ParsesTo[Union[int, float]]
    estimator: Union[str, None] = None
    messages: list[str] = []

class Action(ParsableModel):
    name: str
    arguments: ComponentAttributes = ComponentAttributes()
    energy: ParsesTo[Union[int, float]]
    subactions: ParsableList['Subaction']

    @staticmethod
    def from_plug_ins(
        class_name: str,
        attributes: dict,
        arguments: dict,
        action_name: str,
        spec: "Specification",
        plug_ins: list,
        return_subactions: bool = False,
    ) -> Union["EnergyEntry", list["Subaction"]]:
        attributes, arguments = copy.deepcopy((attributes, arguments))
        entries = []

        definition = None
        try:
            definition = spec.component_classes.component_classes[class_name]
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
        elif definition is not None:
            for (
                component,
                component_attributes,
                sub_arguments,
                subaction_name,
            ) in definition.get_subcomponent_actions(
                action_name, attributes, arguments
            ):
                entries.extend(
                    Action.from_plug_ins(
                        component,
                        component_attributes,
                        sub_arguments,
                        subaction_name,
                        spec,
                        plug_ins,
                        return_subactions=True,
                    )
                )
        else:
            query = EnergyAreaQuery(class_name, attributes.model_dump(), action_name, arguments.model_dump())
            estimation = get_best_estimate(plug_ins, query, True)
            energy = estimation.value
            entries.append(
                Subaction(
                    name=class_name,
                    attributes=attributes,
                    arguments=arguments,
                    energy=energy * attributes.energy_scale * arguments.energy_scale,
                    estimator=estimation.estimator_name,
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
    def from_plug_ins(
        class_name: str,
        attributes: dict,
        arguments: list[dict],
        action_names: list[str],
        spec: "Specification",
        plug_ins: list,
        name: str,
    ):
        actions = []
        for action_name, action_arguments in zip(action_names, arguments):
            actions.append(
                Action.from_plug_ins(
                    class_name,
                    attributes,
                    action_arguments,
                    action_name,
                    spec,
                    plug_ins,
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

    