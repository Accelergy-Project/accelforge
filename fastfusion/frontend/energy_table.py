import copy
from typing import Union
from fastfusion.frontend.version import assert_version
from fastfusion.yamlparse.nodes import DictNode, ListNode
from numbers import Number
from fastfusion.plugin.query_plug_ins import EnergyAreaQuery
from fastfusion.plugin.query_plug_ins import get_best_estimate


class ComponentEnergy(DictNode):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("version", default="0.5", callfunc=assert_version)
        super().add_attr("tables", EnergyTable, [])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.version: str = self["version"]
        self.tables: EnergyTable = self["tables"]

    def find_component(self, name: str):
        for table in self.tables:
            # Component names in ERT are compound, e.g.,
            # Parent1.Parent0[0..NumOfParent].Component[0..NumOfComponent]
            full_name = table.name
            last_component_in_compound_name: str = full_name.split(".")[-1]
            if name == last_component_in_compound_name:
                return table
        raise KeyError(f"Could not find component {name}")

    def isempty(self) -> bool:
        return self.tables.isempty()

    def to_dict(self):
        r = {}
        for t in self.tables:
            r[t.name] = {}
            for a in t.actions:
                r[(t.name, a.name)] = a.energy
        return r


class EnergyTable(ListNode):
    """
    A table of component energy values.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", EnergyEntry)


class EnergyEntry(DictNode):
    """
    Table of actions and their energy values.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name: str = self["name"]
        self.actions: Actions = self["actions"]

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("name", str)
        super().add_attr("actions", Actions)

    # @functools.cache
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
        return EnergyEntry(name=name, actions=actions)


class Actions(ListNode):
    """
    List of actions.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", Action)


class Action(DictNode):
    """
    Action with arguments and energy value.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("name", str)
        super().add_attr("arguments", ActionArguments, {})
        super().add_attr("energy", Number)
        super().add_attr("subactions", Subactions, [])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name: str = self["name"]
        self.arguments: ActionArguments = self["arguments"]
        self.energy: Number = self["energy"]
        self.subactions: Subactions = self["subactions"]

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
            definition = spec.components.classes[class_name]
        except KeyError:
            pass

        predefined_energy = None
        if attributes.get("energy", None) is not None:
            predefined_energy = attributes["energy"]
        elif arguments.get("energy", None) is not None:
            predefined_energy = arguments["energy"]

        if predefined_energy is not None:
            entries = [
                Subaction(
                    name=action_name,
                    attributes=attributes,
                    arguments=arguments,
                    energy=predefined_energy
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
            query = EnergyAreaQuery(class_name, attributes, action_name, arguments)
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


class ActionArguments(DictNode):
    """
    Arguments for an action.
    """

    pass


class Subactions(ListNode):
    """
    List of subactions.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", Subaction)


class Subaction(DictNode):
    """
    Subaction with arguments and energy value.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("name", str)
        super().add_attr("arguments", ActionArguments)
        super().add_attr("energy", Number)
        super().add_attr("estimator", str, "")
        super().add_attr("attributes", dict, {})
        super().add_attr("messages", list, [])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name: str = self["name"]
        self.arguments: ActionArguments = self["arguments"]
        self.energy: Number = self["energy"]
        self.estimator: str = self["estimator"]
        self.attributes: dict = self["attributes"]
        self.messages: list = self["messages"]
