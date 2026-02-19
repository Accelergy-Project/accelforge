import copy
import itertools
import logging
from typing import (
    Any,
    Literal,
    Optional,
    Self,
    TypeVar,
)
from pydantic import ConfigDict
from hwcomponents import (
    ComponentModel,
    get_models,
    get_model,
)

from accelforge.util._basetypes import (
    EvalableModel,
    EvalableList,
    EvalExtras,
    EvalsTo,
    TryEvalTo,
    _PostCall,
)

from accelforge.util.exceptions import EvaluationError
from accelforge.util._eval_expressions import eval_expression
from accelforge.util._setexpressions import InvertibleSet, eval_set_expression
from accelforge.frontend.renames import TensorName
from accelforge.frontend.arch.constraints import Comparison
from accelforge.frontend.arch.structure import ArchNode, Leaf
from accelforge.frontend.arch.spatialable import Spatialable

from accelforge.util._basetypes import _uninstantiable

T = TypeVar("T", bound="ArchNode")


class Action(EvalableModel):
    """
    An action that may be performed by a component.
    """

    name: str
    """ The name of this action. """

    energy: EvalsTo[int | float | None] = None
    """
    Dynamic energy of this action. Per-action energy is multiplied by the component's
    energy_scale and the action's energy_scale.
    """

    energy_scale: EvalsTo[int | float] = 1
    """
    The scale factor for dynamic energy of this action. Multiplies this action's energy
    by this value.
    """

    latency: EvalsTo[int | float | None] = None
    """
    Latency of this action. Per-action latency is multiplied by the component's
    latency_scale and the action's latency_scale.
    """

    latency_scale: EvalsTo[int | float] = 1
    """
    The scale factor for dynamic latency of this action. Multiplies this action's
    latency by this value.
    """

    extra_attributes_for_component_model: EvalExtras = EvalExtras()
    """ Extra attributes to pass to the component model. In addition to all attributes
    of this action, any extra attributes will be passed to the component model as
    arguments to the component model's action. This can be used to define attributes
    that are known to the component model, but not accelforge, such as clock
    frequency."""

    def _attributes_for_component_model(self) -> dict[str, Any]:
        return {
            **self.shallow_model_dump(),
            **self.extra_attributes_for_component_model.shallow_model_dump(),
        }


class TensorHolderAction(Action):
    bits_per_action: EvalsTo[int | float] = (
        "1 if bits_per_action is None else bits_per_action"
    )
    """ The number of bits accessed in this action. For example, setting bits_per_action
    to 16 means that each call to this action yields 16 bits. """


_COMPONENT_MODEL_CACHE: dict[tuple, "Component"] = {}


def _set_component_model_cache(key: tuple, value: "Component"):
    while len(_COMPONENT_MODEL_CACHE) > 1000:
        _COMPONENT_MODEL_CACHE.popitem(last=False)
    _COMPONENT_MODEL_CACHE[key] = value


class _ExtraAttrs(EvalExtras):
    def _eval_expressions(self, symbol_table: dict[str, Any], *args, **kwargs):
        if getattr(self, "_evaluated", False):
            return super()._eval_expressions(symbol_table, *args, **kwargs)

        orig_symbol_table = dict(symbol_table)
        if "arch_extra_attributes_for_all_component_models" not in orig_symbol_table:
            raise EvaluationError(
                "arch_extra_attributes_for_all_component_models is not set in the symbol "
                "table. Was parsing called from the architecture on top?"
            )

        for k, v in orig_symbol_table[
            "arch_extra_attributes_for_all_component_models"
        ].items():
            if getattr(self, k, None) is None:
                setattr(self, k, v)

        return super()._eval_expressions(symbol_table, *args, **kwargs)


@_uninstantiable
class Component(Spatialable):
    """A component object in the architecture. This is overridden by different
    component types, such as `Memory` and `Compute`."""

    name: str
    """ The name of this `Component`. """

    component_class: Optional[str] = None
    """ The class of this `Component`. Used if an energy or area model needs to be
    called for this `Component`. """

    component_model: ComponentModel | None = None
    """ The model to use for this `Component`. If not set, the model will be found with
    `hwcomponents.get_models()`. If set, the `component_class` will be ignored. """

    component_modeling_log: list[str] = []
    """ A log of the energy and area calculations for this `Component`. """

    actions: EvalableList[Action]
    """ The actions that this `Component` can perform. """

    enabled: TryEvalTo[bool] = True
    """ Whether this component is enabled. If the expression resolves to False, then
    the component is disabled. This is evaluated per-pmapping-template, so it is a function
    of the tensors in the current Einsum. For example, you may say `len(All) >= 3` and
    the component will only be enabled with Einsums with three or more tensors.
    """

    area: EvalsTo[int | float | None] = None
    """
    The area of a single instance of this component in m^2. If set, area calculations
    will use this value.
    """

    total_area: EvalsTo[int | float | None] = None
    """
    The total area of all instances of this component in m^2. Do not set this value. It
    is calculated when the architecture's area is calculated.
    """

    area_scale: EvalsTo[int | float] = 1
    """
    The scale factor for the area of this comxponent. This is used to scale the area of
    this component. For example, if the area is 1 m^2 and the scale factor is 2, then
    the area is 2 m^2.
    """

    leak_power: EvalsTo[int | float | None] = None
    """
    The leak power of a single instance of this component in W. If set, leak power
    calculations will use this value.
    """

    total_leak_power: EvalsTo[int | float | None] = None
    """
    The total leak power of all instances of this component in W. Do not set this value.
    It is calculated when the architecture's leak power is calculated. If instances are
    power gated, actual leak power may be less than this value.
    """

    leak_power_scale: EvalsTo[int | float] = 1
    """
    The scale factor for the leak power of this component. This is used to scale the
    leak power of this component. For example, if the leak power is 1 W and the scale
    factor is 2, then the leak power is 2 W.
    """

    energy_scale: EvalsTo[int | float] = 1
    """
    The scale factor for dynamic energy of this component. For each action, multiplies
    this action's energy. Multiplies the calculated energy of each action.
    """

    total_latency: str | int | float = "sum(*action2latency.values())"
    """
    An expression representing the total latency of this component in seconds. This is
    used to calculate the latency of a given Einsum. Special variables available are the
    following:

    - `min`: The minimum value of all arguments to the expression.
    - `max`: The maximum value of all arguments to the expression.
    - `sum`: The sum of all arguments to the expression.
    - `X_actions`: The number of times action `X` is performed. For example,
      `read_actions` is the number of times the read action is performed.
    - `X_latency`: The total latency of all actions of type `X`. For example,
      `read_latency` is the total latency of all read actions. It is equal to the
      per-read latency multiplied by the number of read actions.
    - `action2latency`: A dictionary of action names to their latency.

    Additionally, all component attributes are availble as variables, and all other
    functions generally available in parsing. Note this expression is evaluated after other
    component attributes are evaluated.

    For example, the following expression calculates latency assuming that each read or
    write action takes 1ns: ``1e-9 * (read_actions + write_actions)``.
    """

    latency_scale: EvalsTo[int | float] = 1
    """
    The scale factor for the latency of this component. This is used to scale the
    latency of this component. For example, if the latency is 1 ns and the scale factor
    is 2, then the latency is 2 ns. Multiplies the calculated latency of each action.
    """

    n_parallel_instances: EvalsTo[int | float] = 1
    """
    The number of parallel instances of this component. Increasing parallel instances
    will proportionally increase area and leakage, while reducing latency (unless
    latency calculation is overridden).
    """

    extra_attributes_for_component_model: _ExtraAttrs = _ExtraAttrs()
    """ Extra attributes to pass to the component model. In addition to all attributes
    of this component, any extra attributes will be passed to the component model. This
    can be used to define attributes that are known to the component model, but not
    accelforge, such as the technology node."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _update_actions(self, new_actions: EvalableList[Action]):
        has_actions = set(x.name for x in self.actions)
        for action in new_actions:
            if action.name not in has_actions:
                self.actions.append(action)

    def get_component_class(self, trying_to_calculate: str = None) -> str:
        """Returns the class of this Component.

        Parameters
        ----------
        trying_toeval : str, optional
            What was trying to be calculated using this component. If provided, the
            error message will be more specific.

        Raises
        ------
        EvaluationError
            If the component_class is not set.
        """
        extra_info = ""
        if trying_to_calculate is not None:
            extra_info = f" Occurred while trying to calculate {trying_to_calculate}."

        if self.component_class is None:
            raise EvaluationError(
                f"component_class must be set to a valid string. "
                f"Got {self.component_class}. This occurred because the model tried to "
                "talk to hwcomponents, but was missing necessary attributes. If you do "
                "not want to use hwcomponents component_models, ensure that area and "
                "leak_power are set, as well as, for each action, "
                f"energy and latency are set.{extra_info}",
                source_field=f"{self.name}.component_class",
            )
        return self.component_class

    def _is_dummy(self) -> bool:
        return (
            self.component_model is None
            and self.component_class is not None
            and self.component_class.lower() == "dummy"
        )

    def populate_component_model(
        self: T,
        component_models: list[ComponentModel] | None = None,
        in_place: bool = False,
        trying_to_calculate: str = None,
    ) -> T:
        """
        Populates the ``component_model`` attribute with the model for this component.
        Extends the ``component_modeling_log`` field with log messages. Uses the
        ``component_class`` attribute to find the model and populate the
        ``component_model`` attribute. Uses the ``hwcomponents.get_model()`` function to
        find the model.

        Parameters
        ----------
        component_models : list[hwcomponents.ComponentModel] | None
            The models to use for energy calculation. If not provided, the models will
            be found with `hwcomponents.get_models()`.
        in_place : bool
            If True, the component will be modified in place. Otherwise, a copy will be
            returned.
        trying_to_calculate : str, optional
            What was trying to be calculated using this component. If provided, the
            error messages for missing component_class will be more specific.

        Returns
        -------
        T
            A copy of the component with the populated ``component_model`` attribute.
        """
        if not in_place:
            self: Self = self._copy_for_component_modeling()

        if self._is_dummy():
            return self

        if self.component_model is None:
            if component_models is None:
                component_models = get_models()
            estimation = get_model(
                self.get_component_class(trying_to_calculate=trying_to_calculate),
                self._attributes_for_component_model(),
                required_actions=list(x.name for x in self.actions),
                models=component_models,
                _return_estimation_object=True,
            )
            self.component_model = estimation.value
            self.component_modeling_log.extend(estimation.messages)
        return self

    def calculate_action_energy(
        self,
        component_models: list[ComponentModel] | None = None,
        in_place: bool = False,
    ) -> Self:
        """
        Calculates energy for each action of this component. If energy is set in the
        action or component (with action taking precedence), that value will be used.
        Otherwise, the energy will be calculated using hwcomponents. Populates, for each
        action, the ``<action>.energy`` and field. Extends the
        ``component_modeling_log`` field with log messages.

        Uses the ``component_model`` attribute, or, if not set, the ``component_class``
        attribute to find the model and populate the ``component_model`` attribute.

        Note that these methods will be called by the Spec when calculating energy and
        area. If you call them yourself, note that string expressions may not be evaluated
        because they need the Spec's global scope. If you are sure that all necessary
        values are present and not a result of an expression, you can call these
        directly. Otherwise, you can call the
        ``Spec.calculate_component_area_energy_latency_leak`` and then grab components
        from the returned ``Spec``.

        Parameters
        ----------
        component_models : list[hwcomponents.ComponentModel] | None
            The models to use for energy calculation. If not provided, the models will
            be found with `hwcomponents.get_models()`.
        in_place : bool
            If True, the component will be modified in place. Otherwise, a copy will be
            returned.

        Returns
        -------
        T
            A copy of the component with the calculated energy.
        """
        if not in_place:
            self: Component = self._copy_for_component_modeling()

        messages = self.component_modeling_log

        for action in self.actions:
            messages.append(f"Calculating energy for {self.name} action {action.name}.")
            if action.energy is not None:
                energy = action.energy
                messages.append(f"Setting {self.name} energy to {action.energy=}")
            elif self._is_dummy():
                energy = 0
                messages.append('Component is "Dummy". Setting energy to 0.')
            else:
                self.populate_component_model(
                    component_models,
                    in_place=True,
                    trying_to_calculate=f"energy for action {action.name}",
                )
                energy = self.component_model.try_call_arbitrary_action(
                    action_name=action.name,
                    _return_estimation_object=True,
                    **{
                        **self._attributes_for_component_model(),
                        **action._attributes_for_component_model(),
                    },
                )
                messages.extend(energy.messages)
                energy = energy.value[0]
            if self.energy_scale != 1:
                energy *= self.energy_scale
                messages.append(f"Scaling {self.name} energy by {self.energy_scale=}")
            if action.energy_scale != 1:
                energy *= action.energy_scale
                messages.append(f"Scaling {self.name} energy by {action.energy_scale=}")
            action.energy = energy
            if action.energy < 0:
                logging.warning(
                    f"Component {self.name} action {action.name} has negative energy: "
                    f"{action.energy=}"
                )
        return self

    def calculate_leak_power(
        self,
        component_models: list[ComponentModel] | None = None,
        in_place: bool = False,
    ) -> Self:
        """
        Calculates the leak power for this component. If leak power is set in the
        component, that value will be used. Otherwise, the leak power will be calculated
        using hwcomponents. Populates ``leak_power`` field. Extends the
        ``component_modeling_log`` field with log messages.

        Uses the ``component_model`` attribute, or, if not set, the ``component_class``
        attribute to find the model and populate the ``component_model`` attribute.

        Note that these methods will be called by the Spec when calculating energy and
        area. If you call them yourself, note that string expressions may not be evaluated
        because they need the Spec's global scope. If you are sure that all necessary
        values are present and not a result of an expression, you can call these
        directly. Otherwise, you can call the
        ``Spec.calculate_component_area_energy_latency_leak`` and then grab components
        from the returned ``Spec``.

        Parameters
        ----------
        component_models : list[hwcomponents.ComponentModel] | None
            The models to use for energy calculation. If not provided, the models will
            be found with `hwcomponents.get_models()`.
        in_place : bool
            If True, the component will be modified in place. Otherwise, a copy will be
            returned.

        Returns
        -------
        Self
            A copy of the component with the calculated energy.
        """
        if not in_place:
            self: Self = self._copy_for_component_modeling()

        messages = self.component_modeling_log
        if self.leak_power is not None:
            leak_power = self.leak_power
            messages.append(f"Using predefined leak power value {self.leak_power=}")
        elif self._is_dummy():
            leak_power = 0
            messages.append("Component is dummy. Setting leak power to 0.")
        else:
            self.populate_component_model(
                component_models,
                in_place=True,
                trying_to_calculate="leak_power",
            )
            leak_power = self.component_model.leak_power
        if self.leak_power_scale != 1:
            leak_power *= self.leak_power_scale
            messages.append(f"Scaling leak power by {self.leak_power_scale=}")
        if self.n_parallel_instances != 1:
            leak_power *= self.n_parallel_instances
            messages.append(f"Scaling leak power by {self.n_parallel_instances=}")
        self.leak_power = leak_power
        if self.leak_power < 0:
            logging.warning(
                f"Component {self.name} has negative leak power: {self.leak_power}"
            )
        return self

    def calculate_area(
        self,
        component_models: list[ComponentModel] | None = None,
        in_place: bool = False,
    ) -> Self:
        """
        Calculates the area for this component. If area is set in the component, that
        value will be used. Otherwise, the area will be calculated using the
        hwcomponents library. Populates ``area`` field. Extends the
        ``component_modeling_log`` field with log messages.

        Uses the ``component_model`` attribute, or, if not set, the ``component_class``
        attribute to find the model and populate the ``component_model`` attribute.

        Note that these methods will be called by the Spec when calculating energy and
        area. If you call them yourself, note that string expressions may not be evaluated
        because they need the Spec's global scope. If you are sure that all necessary
        values are present and not a result of an expression, you can call these
        directly. Otherwise, you can call the
        ``Spec.calculate_component_area_energy_latency_leak`` and then grab components
        from the returned ``Spec``.

        Parameters
        ----------
        component_models : list[hwcomponents.ComponentModel] | None
            The models to use for area calculation. If not provided, the models will be
            found with `hwcomponents.get_models()`.
        in_place : bool
            If True, the component will be modified in place. Otherwise, a copy will be
            returned.

        Returns
        -------
        Self
            A copy of the component with the calculated area.
        """
        if not in_place:
            self: Self = self._copy_for_component_modeling()

        messages = self.component_modeling_log
        if self.area is not None:
            area = self.area
            messages.append(f"Using predefined area value {self.area=}")
        elif self._is_dummy():
            area = 0
            messages.append("Component is dummy. Setting area to 0.")
        else:
            self.populate_component_model(
                component_models,
                in_place=True,
                trying_to_calculate="area",
            )
            area = self.component_model.area
        if self.area_scale != 1:
            area *= self.area_scale
            messages.append(f"Scaling area by {self.area_scale=}")
        if self.n_parallel_instances != 1:
            area *= self.n_parallel_instances
            messages.append(f"Scaling area by {self.n_parallel_instances=}")
        self.area = area
        if self.area < 0:
            logging.warning(f"Component {self.name} has negative area: {self.area}")
        return self

    def calculate_action_latency(
        self,
        component_models: list[ComponentModel] | None = None,
        in_place: bool = False,
    ) -> Self:
        """
        Calculates the latency for each action by this component. Populates the
        ``<action>.latency`` field. Extends the ``component_modeling_log`` field with
        log messages.

        Parameters
        ----------
        component_models : list[hwcomponents.ComponentModel] | None
            The models to use for latency calculation. If not provided, the models will be
            found with `hwcomponents.get_models()`.
        in_place : bool
            If True, the component will be modified in place. Otherwise, a copy will be
            returned.

        Returns
        -------
        Self
            A copy of the component with the calculated latency for each action.
        """
        if not in_place:
            self: Self = self._copy_for_component_modeling()

        messages = self.component_modeling_log

        for action in self.actions:
            messages.append(
                f"Calculating latency for {self.name} action {action.name}."
            )
            if action.latency is not None:
                latency = action.latency
                messages.append(f"Setting {self.name} latency to {action.latency=}")
            elif self._is_dummy():
                latency = 0
                messages.append("Component is dummy. Setting latency to 0.")
            else:
                self.populate_component_model(
                    component_models,
                    in_place=True,
                    trying_to_calculate=f"latency for action {action.name}",
                )
                latency = self.component_model.try_call_arbitrary_action(
                    action_name=action.name,
                    _return_estimation_object=True,
                    **{
                        **self._attributes_for_component_model(),
                        **action._attributes_for_component_model(),
                    },
                )
                messages.extend(latency.messages)
                latency = latency.value[1]
            if self.latency_scale != 1:
                latency *= self.latency_scale
                messages.append(f"Scaling {self.name} latency by {self.latency_scale=}")
            if action.latency_scale != 1:
                latency *= action.latency_scale
                messages.append(
                    f"Scaling {self.name} latency by {action.latency_scale=}"
                )
            if self.n_parallel_instances != 1:
                latency /= self.n_parallel_instances
                messages.append(
                    f"Dividing {self.name} latency by {self.n_parallel_instances=}"
                )
            action.latency = latency
            if action.latency < 0:
                logging.warning(
                    f"Component {self.name} action {action.name} has negative latency: "
                    f"{action.latency}"
                )
        return self

    def calculate_area_energy_latency_leak(
        self,
        component_models: list[ComponentModel] | None = None,
        in_place: bool = False,
        _use_cache: bool = False,
    ) -> Self:
        """
        Calculates the area, energy, latency, and leak power for this component.
        Populates the ``area``, ``total_area``, ``leak_power``, ``total_leak_power``,
        ``total_latency``, and ``component_modeling_log`` fields of this component.
        Additionally, for each action, populates the ``<action>.area``,
        ``<action>.energy``, ``<action>.latency``, and ``<action>.leak_power`` fields.
        Extends the ``component_modeling_log`` field with log messages.

        Note that these methods will be called by the Spec when calculating energy and
        area. If you call them yourself, note that string expressions may not be evaluated
        because they need the Spec's global scope. If you are sure that all necessary
        values are present and not a result of an expression, you can call these
        directly. Otherwise, you can call the
        ``Spec.calculate_component_area_energy_latency_leak`` and then grab components
        from the returned ``Spec``.

        Parameters
        ----------
        component_models : list[hwcomponents.ComponentModel] | None
            The models to use for energy calculation. If not provided, the models will
            be found with `hwcomponents.get_models()`.
        in_place : bool
            If True, the component will be modified in place. Otherwise, a copy will be
            returned.
        _use_cache : bool
            If True, the component model will be cached and reused if the same component
            class, attributes, and actions are provided. Note that this may return
            copies of the same object across multiple calls.

        Returns
        -------
        Self
            The component with the calculated energy, area, and leak power.
        """
        if not in_place:
            self: Self = self._copy_for_component_modeling()

        if _use_cache and self.component_model is None:
            cachekey = (
                self.component_class,
                self._attributes_for_component_model(),
                self.actions,
            )
            if cachekey in _COMPONENT_MODEL_CACHE:
                component = _COMPONENT_MODEL_CACHE[cachekey]
                self.component_model = component.component_model
                self.component_modeling_log = component.component_modeling_log
                self.actions = component.actions
                return self

        self.calculate_area(component_models, in_place=True)
        self.calculate_action_energy(component_models, in_place=True)
        self.calculate_action_latency(component_models, in_place=True)
        self.calculate_leak_power(component_models, in_place=True)
        if _use_cache:
            _set_component_model_cache(cachekey, self)
        return self

    def _attributes_for_component_model(self) -> dict[str, Any]:
        return {
            **self.shallow_model_dump(),
            **self.extra_attributes_for_component_model.shallow_model_dump(),
        }

    def _copy_for_component_modeling(self) -> Self:
        self: Component = self.model_copy()
        self.extra_attributes_for_component_model = (
            self.extra_attributes_for_component_model.model_copy()
        )
        self.actions = type(self.actions)([a.model_copy() for a in self.actions])
        for action in self.actions:
            action.extra_attributes_for_component_model = (
                action.extra_attributes_for_component_model.model_copy()
            )
        return self


MEMORY_ACTIONS = EvalableList[TensorHolderAction](
    [
        TensorHolderAction(name="read"),
        TensorHolderAction(name="write"),
    ]
)


PROCESSING_STAGE_ACTIONS = EvalableList[TensorHolderAction](
    [
        TensorHolderAction(name="read"),
    ]
)

COMPUTE_ACTIONS = EvalableList(
    [
        Action(name="compute"),
    ]
)


def _eval_tensor2bits(
    toeval: dict[str, Any], location: str, symbol_table: dict[str, Any]
) -> dict[str, Any]:
    result = {}
    for key, value in toeval.items():
        key_evaluated = eval_set_expression(
            expression=key,
            symbol_table=symbol_table,
            expected_space=TensorName,
            location=f"{location} key {key}",
        ).instance
        result[key_evaluated] = eval_expression(
            expression=value,
            symbol_table=symbol_table,
            attr_name=key,
            location=location,
        )

    all = symbol_table["All"].instance
    for k in result:
        all -= k

    if all:
        raise EvaluationError(f"Missing bits_per_value_scale for {all}")

    for a, b in itertools.combinations(result.keys(), 2):
        if a & b:
            raise EvaluationError(f"bits_per_value_scale for {a} and {b} overlap")

    return {k2: v for k, v in result.items() for k2 in k}


class Tensors(EvalableModel):
    """
    Fields that control which tensor(s) are kept in a :py:class:`~.TensorHolder` and in
    what order their nodes may appear in the mapping.
    """

    keep: TryEvalTo[InvertibleSet[TensorName]] = "<Defaults to Nothing>"
    """
    A set expression describing which tensors must be kept in this
    :class:`accelforge.frontend.arch.TensorHolder`. If this is not defined, then all
    tensors must be kept. Any tensors that are in ``back`` will also be added to
    ``keep``.
    """

    may_keep: TryEvalTo[InvertibleSet[TensorName]] = (
        "<Nothing if keep is defined, else All>"
    )
    """
    A set expression describing which tensors may optionally be kept in this
    :class:`accelforge.frontend.arch.TensorHolder`. The mapper will explore both keeping
    and not keeping each of these tensors. If this is not defined, then all tensors may
    be kept.
    """

    back: TryEvalTo[InvertibleSet[TensorName]] = "Nothing"
    """
    A set expression describing which tensors must be backed by this
    :class:`accelforge.frontend.arch.TensorHolder`. If this is not defined, then no
    tensors must be backed.
    """

    tile_shape: EvalableList[Comparison] = []
    """
    The tile shape for each rank variable. This is given as a list of
    :class:`~.Comparison` objects, where each comparison must evaluate to True for a
    valid mapping.
    """

    no_refetch_from_above: TryEvalTo[InvertibleSet[TensorName]] = "~All"
    """
    The tensors that are not allowed to be refetched from above. This is given as a set
    of :class:`~.TensorName` objects or a set expression that resolves to them. These
    tensors must be fetched at most one time from above memories, and may not be
    refetched across any temporal or spatial loop iterations. Tensors may be fetched in
    pieces (if they do not cause re-fetches of any piece).
    """

    tensor_order_options: EvalableList[
        EvalableList[TryEvalTo[InvertibleSet[TensorName]]]
    ] = EvalableList()
    """
    Options for the order of tensor storage nodes in the mapping. This is given as a
    list-of-lists-of-sets. Each list-of-sets is a valid order of tensor storage nodes.
    Order is given from highest in the mapping to lowest.

    For example, an option could be [input | output, weight], which means that there is
    no relative ordering required between input and output, but weight must be below
    both.
    """

    force_memory_hierarchy_order: bool = True
    """
    If set to true, storage nodes for lower-level memories must be placed below storage
    nodes for higher-level memories. For example, all MainMemory storage nodes must go
    above all LocalBuffer storage nodes.

    This constraint always applies to same-tensor storage nodes (e.g., MainMemory
    reusing Output must go above LocalBuffer reusing Output); turning it off will permit
    things like MainMemory reusing Output going above LocalBuffer reusing Input.

    This is identical to the `force_memory_hierarchy_order` field in the `FFM` class,
    but only applies to this tensor holder.
    """

    def _eval_expressions(self, *args, **kwargs):
        self = copy.copy(self)
        keep, may_keep = self.keep, self.may_keep
        if may_keep == "<Nothing if keep is defined, else All>":
            self.may_keep = "All" if keep == "<Defaults to Nothing>" else "~All"
        if keep == "<Defaults to Nothing>":
            self.keep = "Nothing"
        evaluated, symbol_table = super(self.__class__, self)._eval_expressions(
            *args, **kwargs
        )
        if isinstance(evaluated.back, InvertibleSet):
            if isinstance(evaluated.keep, InvertibleSet):
                evaluated.keep |= evaluated.back
            if isinstance(evaluated.may_keep, InvertibleSet):
                evaluated.may_keep -= evaluated.back

        # Assert that there are no intersecting sets
        for order in evaluated.tensor_order_options:
            for i, s0 in enumerate(order):
                for j, s1 in enumerate(order):
                    if i == j:
                        continue
                    if not isinstance(s0, InvertibleSet) or not isinstance(
                        s1, InvertibleSet
                    ):
                        continue
                    if s0 & s1:
                        raise ValueError(
                            f"Intersecting entries in tensor_order_options: {s0} and {s1}"
                        )

        return evaluated, symbol_table


@_uninstantiable
class TensorHolder(Component, Leaf):
    """A TensorHolder is a component that holds tensors. These are usually Memories,
    but can also be Tolls."""

    actions: EvalableList[TensorHolderAction] = MEMORY_ACTIONS
    """ The actions that this `TensorHolder` can perform. """

    tensors: Tensors = Tensors()
    """
    Fields that control which tensor(s) are kept in this `TensorHolder` and in what
    order their nodes may appear in the mapping.
    """

    bits_per_value_scale: EvalsTo[dict] = {"All": 1}
    """
    A scaling factor for the bits per value of the tensors in this `TensorHolder`. If
    this is a dictionary, keys in the dictionary are evaluated as expressions and may
    reference one or more tensors.
    """

    bits_per_action: EvalsTo[int | float | None] = None
    """
    The number of bits accessed in each of this component's actions. Overridden by
    bits_per_action in any action of this component. If set here, acts as a default
    value for the bits_per_action of all actions of this component.
    """

    def model_post_init(self, __context__=None) -> None:
        self._update_actions(MEMORY_ACTIONS)

    def _eval_expressions(self, *args, **kwargs):
        # Sometimes the same component object may appear in the mapping and the
        # architecture, in which case we don't want parsing to happen twice.
        if getattr(self, "_evaluated", False):
            return super()._eval_expressions(*args, **kwargs)

        class MyPostCall(_PostCall):
            def __call__(self, field, value, evaluated, symbol_table):
                if field == "bits_per_value_scale":
                    evaluated = _eval_tensor2bits(
                        evaluated,
                        location="bits_per_value_scale",
                        symbol_table=symbol_table,
                    )
                return evaluated

        return super()._eval_expressions(*args, **kwargs, post_calls=(MyPostCall(),))


class Container(Leaf, Spatialable):
    """
    Creates a container, used to conveniently define spatial arrays, and doesn't do
    anything else.
    """

    def _render_node_color(self) -> str:
        return "#FCC2FC"


class Memory(TensorHolder):
    """A `Memory` is a `TensorHolder` that stores data over time, allowing for temporal
    reuse."""

    size: EvalsTo[int | float]
    """ The size of this `Memory` in bits. """

    actions: EvalableList[TensorHolderAction] = MEMORY_ACTIONS
    """ The actions that this `Memory` can perform. """

    def _render_node_shape(self) -> str:
        return "cylinder"

    def _render_node_color(self) -> str:
        return "#D7FCD7"

    def __str__(self) -> str:
        """String representation of the Memory node."""
        return f"{self.name} (size: {self.size})" + self._spatial_str(
            include_newline=False
        )

    def _render_node_label(self) -> str:
        """The label for a Pydot node."""
        return f"{self.name} with size {self.size}" + self._spatial_str()


class Toll(TensorHolder):
    """A `Toll` is a `TensorHolder` that does not store data over time, and
    therefore does not allow for temporal reuse. Use this as a toll that charges reads
    and writes every time a piece of data moves through it.

    Every write to a `Toll` is immediately written to the next `Memory`
    (which may be above or below depending on where the write came from), and same for
    reads.

    The access counts of a `Toll` are only included in the "read" action.
    Each traversal through the `Toll` is counted as a read. Writes are always
    zero.
    """

    direction: Literal["up", "down", "up_and_down"]
    """
    The direction in which data flows through this `Toll`. If "up", then data
    flows from below `TensorHolder`, through this `Toll` (plus paying
    associated costs), and then to the next `TensorHolder` above it. Other data
    movements are assumed to avoid this Toll.
    """

    actions: EvalableList[TensorHolderAction] = PROCESSING_STAGE_ACTIONS
    """ The actions that this `Toll` can perform. """

    def model_post_init(self, __context__=None) -> None:
        self._update_actions(PROCESSING_STAGE_ACTIONS)

    def _render_node_shape(self) -> str:
        return "rarrow"

    def _render_node_color(self) -> str:
        return "#FFCC99"


class Compute(Component, Leaf):
    actions: EvalableList[Action] = COMPUTE_ACTIONS
    """ The actions that this `Compute` can perform. """

    def model_post_init(self, __context__=None) -> None:
        self._update_actions(COMPUTE_ACTIONS)

    def _render_node_shape(self) -> str:
        return "ellipse"

    def _render_node_color(self) -> str:
        return "#E0EEFF"
