import copy
from typing import Annotated, Any, Callable, Optional, Union
from fastfusion.version import assert_version, __version__
from fastfusion.util.basetypes import (
    ParsableDict,
    ParsableList,
    ParsableModel,
    ParsesTo,
    ParsableDict,
)
from hwcomponents import get_area


class AreaSubcomponent(ParsableModel):
    name: str
    area: ParsesTo[Union[int, float]]
    model_name: Optional[str] = None
    messages: list[str] = []
    attributes: ParsableDict[str, ParsesTo[Any]]


class AreaEntry(ParsableModel):
    name: str
    area: ParsesTo[Union[int, float]]
    subcomponents: ParsableList[AreaSubcomponent]

    @staticmethod
    def from_models(
        class_name: str | Callable[[], str],
        attributes: dict,
        spec: "Specification",
        models: list,
        return_subcomponents: bool = False,
        name: str = None,
    ) -> Union["AreaEntry", list["AreaSubcomponent"]]:
        attributes = copy.copy(attributes)
        entries = []
        definition = None

        from fastfusion import Specification

        spec: Specification = spec

        if attributes.area is not None:
            entries = [
                AreaSubcomponent(
                    name=name,
                    attributes=attributes.model_dump(),
                    area=attributes.area * attributes.area_scale,
                    messages=["Using predefined area value"],
                )
            ]
        else:
            class_name = class_name if isinstance(class_name, str) else class_name()
            try:
                definition = spec.components.components[class_name]
            except KeyError:
                pass

            if definition is not None:
                for component in definition.subcomponents:
                    component_attributes = component.attributes.parse_expressions(
                        attributes.model_dump()
                    )[0]
                    entries.extend(
                        AreaEntry.from_models(
                            component.get_component_class(),
                            component_attributes,
                            spec,
                            models,
                            return_subcomponents=True,
                        )
                    )
            else:
                estimation = get_area(
                    component_name=class_name,
                    component_attributes=attributes.model_dump(),
                    models=models,
                )
                area = estimation.value
                entries.append(
                    AreaSubcomponent(
                        name=class_name,
                        attributes=attributes.model_dump(),
                        area=area * attributes.area_scale,
                        model_name=estimation.model_name,
                        messages=estimation.messages,
                    )
                )

        if return_subcomponents:
            return entries

        area = sum(subcomponent.area for subcomponent in entries)
        assert name is not None, f"Name is required for AreaEntry"
        return AreaEntry(name=name, area=area, subcomponents=entries)


class ComponentArea(ParsableModel):
    version: Annotated[str, assert_version] = __version__
    entries: ParsableList[AreaEntry] = ParsableList()
