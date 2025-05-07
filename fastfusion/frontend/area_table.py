import copy
from typing import Union
from fastfusion.frontend.version import assert_version
from fastfusion.yamlparse.nodes import DictNode, ListNode
from numbers import Number
from fastfusion.plugin.query_plug_ins import EnergyAreaQuery
from fastfusion.plugin.query_plug_ins import get_best_estimate


class ComponentArea(DictNode):
    """
    A the table of component areas.

    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("version", default="0.5", callfunc=assert_version)
        super().add_attr("tables", AreaTable, [])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.version: str = self["version"]
        self.tables: AreaTable = self["tables"]

    def isempty(self) -> bool:
        return self.tables.isempty()


class AreaTable(ListNode):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", AreaEntry)


class AreaEntry(DictNode):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("name", str)
        super().add_attr("area", Number)
        super().add_attr("subcomponents", AreaSubcomponents, [])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name: str = self["name"]
        self.area: Number = self["area"]
        self.subcomponents: AreaSubcomponents = self["subcomponents"]

    @staticmethod
    def from_plug_ins(
        class_name: str,
        attributes: dict,
        spec: "Specification",
        plug_ins: list,
        return_subcomponents: bool = False,
        name: str = None,
    ) -> Union["AreaEntry", list["AreaSubcomponent"]]:
        attributes = copy.deepcopy(attributes)
        entries = []
        definition = None
        try:
            definition = spec.components.classes[class_name]
        except KeyError:
            pass
        predefined_area = attributes.get("area", None)

        if predefined_area is not None:
            entries = [
                AreaSubcomponent(
                    name=name,
                    attributes=attributes,
                    area=predefined_area * attributes.area_scale,
                    estimator=None,
                    messages=["Using predefined area value"],
                )
            ]
        elif definition is not None:
            for component in definition.subcomponents:
                component_attributes = component.attributes.parse(attributes)
                entries.extend(
                    AreaEntry.from_plug_ins(
                        component._class,
                        component_attributes,
                        spec,
                        plug_ins,
                        return_subcomponents=True,
                    )
                )
        else:
            query = EnergyAreaQuery(class_name, attributes)
            estimation = get_best_estimate(plug_ins, query, False)
            area = estimation.value
            entries.append(
                AreaSubcomponent(
                    name=class_name,
                    attributes=attributes,
                    area=area * attributes.area_scale,
                    estimator=estimation.estimator_name,
                    messages=estimation.messages,
                )
            )

        if return_subcomponents:
            return entries

        area = sum(subcomponent.area for subcomponent in entries)
        assert name is not None, f"Name is required for AreaEntry"
        return AreaEntry(name=name, area=area, subcomponents=entries)


class AreaSubcomponents(ListNode):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", AreaSubcomponent)


class AreaSubcomponent(DictNode):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("name", str)
        super().add_attr("area", Number)
        super().add_attr("estimator", str)
        super().add_attr("messages", list, [])
        super().add_attr("attributes", dict, {})

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name: str = self["name"]
        self.area: Number = self["area"]
        self.estimator: str = self["estimator"]
        self.messages: list = self["messages"]
        self.attributes: dict = self["attributes"]