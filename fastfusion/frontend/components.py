from numbers import Number
from typing import Any, Dict, Optional
from fastfusion.yamlparse.nodes import DictNode, ListNode, FlatteningListNode
from .version import assert_version
from fastfusion.yamlparse.parse_expressions import parse_expression
from ruamel.yaml.scalarstring import DoubleQuotedScalarString


class Components(DictNode):
    """
    A collection of components.

    Attributes:
        version (str): The version of the components.
        classes (FlatteningListNode): The list of classes associated with the components.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("version", default="0.5", callfunc=assert_version)
        super().add_attr("classes", ComponentsList, [])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.version: str = self["version"]
        self.classes: FlatteningListNode = self["classes"]

    def to_component_dict(self):
        d = {}
        for c in self.classes:
            if c.name not in d:
                d[c.name] = c
            else:
                raise KeyError(f"Duplicate component name: {c.name}")
        return d


class ComponentsList(FlatteningListNode):
    """
    A list of components.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, *kwargs)
        super().add_attr("", CompoundComponent)


class CompoundComponent(DictNode):
    """
    Represents a compound component.

    Attributes:
        name (str): The name of the compound component.
        attributes (ComponentAttributes): The attributes of the compound component.
        subcomponents (SubcomponentList): The list of subcomponents.
        actions (ActionsList): The list of actions.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("name", str)
        super().add_attr("attributes", ComponentAttributes, {})
        super().add_attr("subcomponents", SubcomponentList, [])
        super().add_attr("actions", ActionsList, [])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name: str = self["name"]
        self.attributes: ComponentAttributes = self["attributes"]
        self.subcomponents: SubcomponentList = self["subcomponents"]
        self.actions: ActionsList = self["actions"]

    def get_subcomponent_actions(
        self, action_name: str, attributes: dict, arguments: dict
    ):
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
            component_attributes = component.attributes.parse(attributes)
            for subaction in subcomponent.actions:
                subaction_args = subaction.arguments.parse(
                    arguments, multiply_multipliers=False
                )
                yield component._class, component_attributes, subaction_args, subaction.name


class ComponentEnergyAreaDictNode(DictNode):
    def declare_attrs(self, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr(
            "technology", str, "<SPECIFY ME>. technology must be specified in a parent."
        )
        super().add_attr(
            "global_cycle_seconds",
            (Number, str),
            "<SPECIFY ME>. global_cycle_seconds must be specified in a parent.",
        )
        super().add_attr("n_instances", (Number, str), 1)
        super().add_attr("energy_scale", (Number, str), 1)
        super().add_attr("area_scale", (Number, str), 1)
        super().add_attr("energy", (Number, str, None), None)
        super().add_attr("area", (Number, str, None), None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_cycle_seconds: float = self["global_cycle_seconds"]
        self.technology: str = self["technology"]
        self.n_instances: int = self["n_instances"]
        self.energy_scale: float = self["energy_scale"]
        self.area_scale: float = self["area_scale"]
        self.energy: Optional[float] = self["energy"]
        self.area: Optional[float] = self["area"]

    def parse(
        self,
        symbol_table: dict,
        location: str = "",
        inherit_all: bool = False,
        multiply_multipliers: bool = True,
    ):
        parsed = DictNode()

        for key, value in self.items():
            parsed[key] = parse_expression(
                value,
                symbol_table,
                f'{self.get_name()}["{key}"] in {location}',
                strings_allowed=True,
            )
            if isinstance(parsed[key], str):
                parsed[key] = DoubleQuotedScalarString(parsed[key])

        if "global_cycle_seconds" not in symbol_table:
            raise KeyError(
                f'Required key "global_cycle_seconds" not found in {self.get_name()}["{location}"]'
            )
        if "technology" not in symbol_table:
            raise KeyError(
                f'Required key "technology" not found in {self.get_name()}["{location}"]'
            )

        def copy_attr_if_needed(attr_name):
            if attr_name in symbol_table and parsed.get(attr_name, None) is None:
                parsed[attr_name] = symbol_table[attr_name]

        copy_attr_if_needed("global_cycle_seconds")
        copy_attr_if_needed("technology")
        # copy_attr_if_needed("energy")
        # copy_attr_if_needed("area")
        parsed.setdefault("n_instances", 1)
        parsed.setdefault("energy_scale", 1)
        parsed.setdefault("area_scale", 1)

        if multiply_multipliers:
            if "n_instances" in symbol_table:
                parsed["n_instances"] *= symbol_table["n_instances"]
            if "energy_scale" in symbol_table:
                parsed["energy_scale"] *= symbol_table["energy_scale"]
            if "area_scale" in symbol_table:
                parsed["area_scale"] *= symbol_table["area_scale"]

        if inherit_all:
            for key, value in symbol_table.items():
                parsed.setdefault(key, value)

        return ComponentEnergyAreaDictNode(parsed)

    def parse_expressions(
        self,
        symbol_table: Optional[Dict[str, Any]] = None,
        parsed_ids: Optional[set] = None,
        *args,
        **kwargs,
    ):
        def update_attr_if_needed(attr_name):
            attr_value = getattr(self, attr_name)
            if (
                attr_name in symbol_table
                and isinstance(attr_value, str)
                and "<SPECIFY ME>" in attr_value
            ):
                setattr(self, attr_name, symbol_table[attr_name])

        if symbol_table:
            update_attr_if_needed("technology")
            update_attr_if_needed("global_cycle_seconds")
        super().parse_expressions(**kwargs)


class SubcomponentList(ListNode):
    """
    A list of subcomponents.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", Subcomponent, part_name_match=True, no_change_key=True)


class Subcomponent(DictNode):
    """
    A subcomponent.

    Attributes:
        name (str): The name of the subcomponent.
        attributes (ComponentAttributes): The attributes of the subcomponent.
        area_scale (Union[Number, str]): The area scale of the subcomponent.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("name", str)
        super().add_attr("class", str)
        super().add_attr("attributes", ComponentAttributes, {})

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name: str = self["name"]
        self._class: str = self["class"]
        self.attributes: ComponentAttributes = self["attributes"]


class ComponentAttributes(ComponentEnergyAreaDictNode):
    """
    Represents the attributes of a component.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", part_name_match=True, no_change_key=True)


class ActionsList(ListNode):
    """
    A list of actions for a component.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", Action, part_name_match=True, no_change_key=True)


class Action(DictNode):
    """
    Represents an action that a component can perform.

    Attributes:
        name (str): The name of the action.
        arguments (DictNode): The arguments of the action.
        subcomponents (ActionSubcomponentsList): The subcomponents of the action.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("name", str)
        super().add_attr("subcomponents", ActionSubcomponentsList, [])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name: str = self["name"]
        self.subcomponents: ActionSubcomponentsList = self["subcomponents"]


class ActionSubcomponentsList(ListNode):
    """
    A list of actions that may be taken as subactions of an action.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr(
            "",
            SubcomponentActionGroup,
            part_name_match=True,
            no_change_key=True,
        )


class SubcomponentActionGroup(DictNode):
    """
    A group of subactions taken by a particular subcomponent.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("name", str)
        super().add_attr("actions", SubcomponentActionList, [])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name: str = self["name"]
        self.actions: SubcomponentActionList = self["actions"]


class SubcomponentActionList(ListNode):
    """
    A list of subcomponent actions.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", SubcomponentAction)


class SubcomponentAction(DictNode):
    """
    A subcomponent action.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("name", str)
        super().add_attr("arguments", ComponentEnergyAreaDictNode, {})

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name: str = self["name"]
        self.arguments: DictNode = self["arguments"]
