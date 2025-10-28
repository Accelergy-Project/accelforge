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
from hwcomponents import get_leak_power
from fastfusion.frontend.components import ComponentAttributes


class LeakSubcomponent(ParsableModel):
    name: str
    leak_power: ParsesTo[Union[int, float]]
    model_name: Optional[str] = None
    messages: list[str] = []
    attributes: ParsableDict[str, ParsesTo[Any]]


class LeakEntry(ParsableModel):
    name: str
    attributes: ComponentAttributes = ComponentAttributes()
    leak_power: ParsesTo[Union[int, float]]
    subcomponents: ParsableList[LeakSubcomponent]
    messages: list[str]

    @staticmethod
    def from_models(
        class_name: str | Callable[[], str],
        attributes: dict,
        spec: "Specification",
        models: list,
        return_subcomponents: bool = False,
        name: str = None,
    ) -> Union["LeakEntry", list["LeakSubcomponent"]]:
        try:
            return LeakEntry._from_models(
                class_name,
                attributes,
                spec,
                models,
                return_subcomponents,
                name,
            )
        except Exception as e:
            e.add_note(
                'If you\'d like to use a predefined leak power value, set the '
                '"leak_power" attribute of the component.'
            )
            raise

    @staticmethod
    def _from_models(
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

        if attributes.leak_power is not None:
            entries = [
                LeakSubcomponent(
                    name=name,
                    attributes=attributes.model_dump(),
                    leak_power=attributes.leak_power * attributes.leak_power_scale,
                    messages=["Using predefined leak power value"],
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
                    component_attributes = component.attributes._parse_expressions(
                        attributes.model_dump()
                    )[0]
                    entries.extend(
                        LeakEntry.from_models(
                            component.get_component_class(),
                            component_attributes,
                            spec,
                            models,
                            return_subcomponents=True,
                        )
                    )
            else:
                estimation = get_leak_power(
                    component_name=class_name,
                    component_attributes=attributes.model_dump(),
                    models=models,
                )
                leak_power = estimation.value
                entries.append(
                    LeakSubcomponent(
                        name=class_name,
                        attributes=attributes.model_dump(),
                        leak_power=leak_power * attributes.leak_power_scale,
                        model_name=estimation.model_name,
                        messages=estimation.messages,
                    )
                )

        if return_subcomponents:
            return entries

        leak_power = sum(subcomponent.leak_power for subcomponent in entries)
        assert name is not None, f"Name is required for LeakEntry"
        return LeakEntry(
            name=name,
            leak_power=leak_power,
            subcomponents=entries,
            attributes=attributes,
            messages=[m for e in entries for m in e.messages],
        )


class ComponentLeak(ParsableModel):
    version: Annotated[str, assert_version] = __version__
    entries: ParsableList[LeakEntry] = ParsableList()
