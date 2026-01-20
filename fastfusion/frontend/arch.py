import copy
import math
from numbers import Number
import re
from typing import (
    Any,
    Callable,
    Iterator,
    List,
    Literal,
    Optional,
    TypeVar,
    Annotated,
    Type,
    Union,
)
from pydantic import ConfigDict, Tag
import pydantic
from hwcomponents import (
    ComponentModel,
    get_models,
    get_model,
)
import pydot

from fastfusion.util._basetypes import (
    ParsableModel,
    ParsableList,
    ParseExtras,
    ParsesTo,
    _PostCall,
    _get_tag,
)
import numpy as np

from fastfusion.util._parse_expressions import ParseError, parse_expression
from fastfusion.util._setexpressions import InvertibleSet, eval_set_expression
from fastfusion.frontend.renames import RankVariable, TensorName

from fastfusion._version import assert_version, __version__
from pydantic import Discriminator
from fastfusion.util._basetypes import _uninstantiable
from fastfusion.util.parallel import _SVGJupyterRender
from fastfusion.util._visualization import _pydot_graph

T = TypeVar("T", bound="ArchNode")


class ArchNode(ParsableModel):
    """A node in the architecture."""

    def find(self, name: str) -> "Leaf":
        """
        Finds a `Leaf` node with the given name.
        :raises ValueError: If the `Leaf` node with the given name is not found.
        """
        if isinstance(self, Leaf) and getattr(self, "name", None) == name:
            return self

        if isinstance(self, Branch):
            for element in self.nodes:
                try:
                    return element.find(name)
                except (AttributeError, ValueError):
                    pass
        raise ValueError(f"Leaf {name} not found in {self}")


class ArchNodes(ParsableList):
    """A list of `ArchNode`s."""

    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()})"

    def _parse_expressions(self, *args, **kwargs):
        class PostCallArchNode(_PostCall):
            def __call__(self, field, value, parsed, symbol_table):
                if isinstance(parsed, Container):
                    symbol_table.update(parsed.attributes)
                return parsed

        return super()._parse_expressions(
            *args, **kwargs, post_calls=(PostCallArchNode(),)
        )


class Comparison(ParsableModel):
    """
    A comparison between a rank variable's bound and a value. A comparison is performed
    for each rank variable.

    The LHS of each comparison is the loop bound of a loop that affects this rank
    variable. The RHS is the given value.

    For example, if the expression resolves to [a, b], the operator is "<=", and the
    value is 10, and we have loops "for a0 in [0..A0)" and "for b0 in [0..B0)", then a
    mapping is only valid if A0 <= 10 and B0 <= 10.
    """

    expression: str | InvertibleSet[RankVariable] | set[RankVariable]
    """ The expression to compare. This expression should resolve to a set of rank
    variables. A comparison is performed for each rank variable independently, and the
    result passes if and only if all comparisons pass. The LHS of each comparison is the
    loop bound of a loop that affects this rank variable. The RHS is the given value.
    """

    operator: str
    """ The operator to use for the comparison. Supported operators are:
    - == (equal to)
    - <= (less than or equal to)
    - >= (greater than or equal to)
    - < (less than)
    - > (greater than)
    - product== (product of all loop bounds is equal to)
    - product<= (product of all loop bounds is less than or equal to)
    - product>= (product of all loop bounds is greater than or equal to)
    - product< (product of all loop bounds is less than)
    - product> (product of all loop bounds is greater than)
    """

    value: ParsesTo[int]
    """ The value to compare against. """

    def _parse(self, symbol_table: dict[str, Any], location: str):
        # if len(self) != 3:
        #     raise ValueError(f"Comparison can only have 3 elements. got {len(self)}")
        new = type(self)(
            expression=eval_set_expression(
                self.expression, symbol_table, "rank_variables", location
            ),
            operator=self.operator,
            value=self.value,
        )
        if len(new.expression) == 1 and "product" in new.operator:
            new.operator = new.operator.replace("product", "")
        return new

    def _constrained_to_one(self) -> bool:
        return self.value == 1 and self.operator in [
            "==",
            "<=",
            "product==",
            "product<=",
        ]

    def _split_expression(self) -> List[set[RankVariable]]:
        if "product" in self.operator:
            return [self.expression]
        return sorted(set((x,)) for x in self.expression)

    def _to_constraint_lambda(
        self,
        increasing_sizes: bool,
    ) -> Callable[[bool, np.ndarray], bool | np.ndarray]:
        # Equal operators can only evaluate when all sizes are known
        eq_op = lambda final: (
            np.equal
            if final
            else (np.less_equal if increasing_sizes else np.greater_equal)
        )

        # If we're increasing, we can evaluate leq immediately. If we're
        # decreasing, we can evaluate geq immediately. The other must wait
        # until all sizes are known.
        le_wrapper = lambda op: lambda final, sizes: (
            op(sizes) if final or increasing_sizes else True
        )
        ge_wrapper = lambda op: lambda final, sizes: (
            op(sizes) if final or not increasing_sizes else True
        )

        _all = lambda sizes: np.all(sizes, axis=1)
        _prod = lambda sizes: np.prod(sizes, axis=1)

        # fmt: off
        operator_to_wrapper = {
            "==":        lambda final, sizes: _all(eq_op(final)(sizes, self.value)),
            "product==": lambda final, sizes: eq_op(final)(_prod(sizes), self.value),
            "<=":        le_wrapper(lambda sizes: _all(sizes)  <= self.value),
            ">=":        ge_wrapper(lambda sizes: _all(sizes)  >= self.value),
            "<":         le_wrapper(lambda sizes: _all(sizes)  <  self.value),
            ">":         ge_wrapper(lambda sizes: _all(sizes)  >  self.value),
            "product<=": le_wrapper(lambda sizes: _prod(sizes) <= self.value),
            "product>=": ge_wrapper(lambda sizes: _prod(sizes) >= self.value),
            "product<":  le_wrapper(lambda sizes: _prod(sizes) <  self.value),
            "product>":  ge_wrapper(lambda sizes: _prod(sizes) >  self.value),
        }
        # fmt: on

        if self.operator in operator_to_wrapper:
            return operator_to_wrapper[self.operator]
        raise KeyError(
            f"Unknown operator: {self.operator}. Known operators: {list(operator_to_wrapper.keys())}"
        )


class Spatial(ParsableModel):
    """A one-dimensional spatial fanout in the architecture."""

    name: str
    """
    The name of the dimension over which this spatial fanout is occurring (e.g., X or Y).
    """

    fanout: ParsesTo[int]
    """ The size of this fanout. """

    may_reuse: str | InvertibleSet[TensorName] | set[TensorName] = "All"
    """ The tensors that can be reused spatially across instances of this fanout. This
    expression will be parsed for each mapping template. """

    loop_bounds: ParsableList[Comparison] = ParsableList()
    """ Bounds for loops over this dimension. This is a list of :class:`~.Comparison`
    objects, all of which must be satisfied by the loops to which this constraint
    applies.
    """

    min_usage: int | float | str = 0.0
    """ The minimum utilization of spatial instances, as a value from 0 to 1. A mapping
    is invalid if less than this porportion of this dimension's fanout is utilized.
    Mappers that support it (e.g., FFM) may, if no mappings satisfy this constraint,
    return the highest-utilization mappings.
    """

    reuse: str | InvertibleSet[TensorName] | set[TensorName] = "Nothing"
    """ A set of tensors or a set expression representing tensors that must be reused
    across spatial iterations. Spatial loops may only be placed that reuse ALL tensors
    given here.
    """

    usage_scale: ParsesTo[int | float | str] = 1
    """
    This factor scales the usage in this dimension. For example, if usage_scale is 2 and
    10/20 spatial instances are used, then the usage will be scaled to 20/20.
    """

    def _parse(self, symbol_table: dict[str, Any], location: str):
        return type(self)(
            name=self.name,
            fanout=self.fanout,
            may_reuse=set(
                eval_set_expression(
                    self.may_reuse,
                    symbol_table,
                    expected_space_name="tensors",
                    location=location + ".may_reuse",
                )
            ),
            loop_bounds=[
                x._parse(symbol_table, location + ".loop_bounds")
                for x in self.loop_bounds
            ],
            min_usage=parse_expression(
                self.min_usage,
                symbol_table,
                "min_usage",
                location + ".min_usage",
            ),
            reuse=eval_set_expression(
                self.reuse,
                symbol_table,
                "tensors",
                location + ".reuse",
            ),
        )


class LeafAttributes(ParsableModel):
    pass


class AttributesWithExtras(ParseExtras):
    pass


class AttributesWithEnergyLatency(AttributesWithExtras):
    energy: ParsesTo[int | float | None] = None
    energy_scale: ParsesTo[int | float] = 1
    latency_scale: ParsesTo[int | float] = 1


class ComponentAttributes(AttributesWithEnergyLatency):
    area: ParsesTo[int | float | None] = None
    """
    The area of a single instance of this component in m^2. If set, area calculations
    will use this value.
    """
    total_area: ParsesTo[int | float | None] = None
    """
    The total area of all instances of this component in m^2. Do not set this value. It
    is calculated when the architecture's area is calculated.
    """
    area_scale: ParsesTo[int | float] = 1
    """
    The scale factor for the area of this comxponent. This is used to scale the area of
    this component. For example, if the area is 1 m^2 and the scale factor is 2, then
    the area is 2 m^2.
    """

    leak_power: ParsesTo[int | float | None] = None
    """
    The leak power of a single instance of this component in W. If set, leak power
    calculations will use this value.
    """
    total_leak_power: ParsesTo[int | float | None] = None
    """
    The total leak power of all instances of this component in W. Do not set this value.
    It is calculated when the architecture's leak power is calculated.
    """
    leak_power_scale: ParsesTo[int | float] = 1
    """
    The scale factor for the leak power of this component. This is used to scale the
    leak power of this component. For example, if the leak power is 1 W and the scale
    factor is 2, then the leak power is 2 W.
    """

    energy_scale: ParsesTo[int | float] = 1
    """
    The scale factor for dynamic energy of this component. For each action, multiplies
    this action's energy. Multiplies the calculated energy of each action.
    """

    total_latency: str | int | float = (
        "sum(*action2latency.values()) / n_parallel_instances"
    )
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
    functions generally available in parsing. Note this expression is parsed after other
    component attributes are parsed.

    For example, the following expression calculates latency assuming that each read or
    write action takes 1ns: ``1e-9 * (read_actions + write_actions)``.
    """

    latency_scale: ParsesTo[int | float] = 1
    """
    The scale factor for the latency of this component. This is used to scale the
    latency of this component. For example, if the latency is 1 ns and the scale factor
    is 2, then the latency is 2 ns. Multiplies the calculated latency of each action.
    """

    n_parallel_instances: ParsesTo[int | float] = 1
    """
    The number of parallel instances of this component. Increasing parallel instances
    will proportionally increase area and leakage, while reducing latency (unless
    latency calculation is overridden).
    """


class FanoutAttributes(LeafAttributes):
    model_config = ConfigDict(extra="forbid")


class ActionArguments(AttributesWithEnergyLatency):
    """
    Arguments for an action of a component.
    """

    energy: ParsesTo[int | float | None] = None
    """
    Dynamic energy of this action. Per-action energy is multiplied by the component's
    attributes.energy_scale and the action's arguments.energy_scale.
    """
    energy_scale: ParsesTo[int | float] = 1
    """
    The scale factor for dynamic energy of this action. Multiplies this action's energy
    by this value.
    """
    latency: ParsesTo[int | float | None] = None
    """
    Latency of this action. Per-action latency is multiplied by the component's
    attributes.latency_scale and the action's arguments.latency_scale.
    """
    latency_scale: ParsesTo[int | float] = 1
    """
    The scale factor for dynamic latency of this action. Multiplies this action's
    latency by this value.
    """


class TensorHolderActionArguments(ActionArguments):
    bits_per_action: ParsesTo[int | float] = (
        "1 if attributes.bits_per_action is None else attributes.bits_per_action"
    )
    """ The number of bits accessed in this action. For example, setting bits_per_action
    to 16 means that each call to this action yields 16 bits. """


class Action(ParsableModel):
    name: str
    """ The name of this action. """

    arguments: ActionArguments = ActionArguments()
    """
    The arguments for this action. Passed to the component's model to calculate the
    energy and latency of the action.
    """


class TensorHolderAction(Action):
    arguments: TensorHolderActionArguments = TensorHolderActionArguments()
    """
    The arguments for this action. Passed to the component's model to calculate the
    energy and latency of the action.
    """


@_uninstantiable
class Leaf(ArchNode):
    """A leaf node in the architecture. This is an abstract class that represents any
    node that is not a `Branch`."""

    name: str
    """ The name of this `Leaf`. """

    attributes: LeafAttributes = LeafAttributes()
    """ The attributes of this `Leaf`. """

    spatial: ParsableList[Spatial] = ParsableList()
    """
    The spatial fanouts of this `Leaf`.

    Spatial fanouts describe the spatial organization of components in the architecture.
    A spatial fanout of size N for this node means that there are N instances of this
    node. Multiple spatial fanouts lead to a multi-dimensional fanout. Spatial
    constraints apply to the data exchange across these instances. Spatial fanouts
    specified at this level also apply to lower-level `Leaf` nodes in the architecture.
    """

    def _parse_expressions(self, *args, **kwargs):
        class PostCallLeaf(_PostCall):
            def __call__(self, field, value, parsed, symbol_table):
                if field == "attributes":
                    symbol_table.update(parsed.model_dump())
                return parsed

        parsed, symbol_table = super()._parse_expressions(
            *args, **kwargs, post_calls=(PostCallLeaf(),), order=("attributes",)
        )
        symbol_table[self.name] = self
        return parsed, symbol_table

    def get_fanout(self) -> int:
        """The spatial fanout of this node."""
        return int(math.prod(x.fanout for x in self.spatial))


@_uninstantiable
class Component(Leaf):
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

    actions: ParsableList[Action]
    """ The actions that this `Component` can perform. """

    attributes: ComponentAttributes = ComponentAttributes()
    """ The attributes of this `Component`. """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    enabled: str | bool = True
    """ Whether this component is enabled. If the expression resolves to False, then
    the component is disabled. This is parsed per-pmapping-template, so it is a function
    of the tensors in the current Einsum. For example, you may say `len(All) >= 3` and
    the component will only be enabled with Einsums with three or more tensors.
    """

    def _update_actions(self, new_actions: ParsableList[Action]):
        has_actions = set(x.name for x in self.actions)
        for action in new_actions:
            if action.name not in has_actions:
                self.actions.append(action)

    def get_component_class(self, trying_to_calculate: str = None) -> str:
        """Returns the class of this `Component`.

        Parameters
        ----------
        trying_to_parse : str, optional
            What was trying to be calculated using this component. If provided, the
            error message will be more specific.

        :raises ParseError: If the `component_class` is not set.
        """
        extra_info = ""
        if trying_to_calculate is not None:
            extra_info = f" Occurred while trying to calculate {trying_to_calculate}."

        if self.component_class is None:
            raise ParseError(
                f"component_class must be set to a valid string. "
                f"Got {self.component_class}. This occurred because the model tried to "
                "talk to hwcomponents, but was missing necessary attributes. If you do "
                "not want to use hwcomponents models, ensure that attributes.area and "
                "attributes.leak_power are set, as well as, for each action, "
                f"arguments.energy and arguments.latency are set.{extra_info}",
                source_field=f"{self.name}.component_class",
            )
        return self.component_class

    def populate_component_model(
        self: T,
        models: list[ComponentModel] | None = None,
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
        models : list[ComponentModel] | None
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
            self = self.model_copy()
            self.attributes = self.attributes.model_copy()
            self.actions = type(self.actions)([a.model_copy() for a in self.actions])
            for action in self.actions:
                action.arguments = action.arguments.model_copy()

        if self.component_model is None:
            if models is None:
                models = get_models()
            estimation = get_model(
                self.get_component_class(trying_to_calculate=trying_to_calculate),
                self.attributes.model_dump(),
                required_actions=list(x.name for x in self.actions),
                models=models,
                _return_estimation_object=True,
            )
            self.component_model = estimation.value
            self.component_modeling_log.extend(estimation.messages)
        return self

    def calculate_action_energy(
        self: T,
        models: list[ComponentModel] | None = None,
        in_place: bool = False,
    ) -> T:
        """
        Calculates energy for each action of this component. If energy is set in the
        arguments or attributes (with arguments taking precedence), that value will be
        used. Otherwise, the energy will be calculated using hwcomponents. Populates,
        for each action, the ``<action>.arguments.energy`` and field. Extends the
        ``component_modeling_log`` field with log messages.

        Uses the ``component_model`` attribute, or, if not set, the ``component_class``
        attribute to find the model and populate the ``component_model`` attribute.

        Note that these methods will be called by the Spec when calculating energy and
        area. If you call them yourself, note that string expressions may not be parsed
        because they need the Spec's global scope. If you are sure that all necessary
        values are present and not a result of an expression, you can call these
        directly. Otherwise, you can call the ``Spec.calculate_component_area_energy_latency_leak``
        and then grab components from the returned ``Spec``.

        Parameters
        ----------
        models : list[ComponentModel] | None
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
            self = self.model_copy()
            self.attributes = self.attributes.model_copy()
            self.actions = type(self.actions)([a.model_copy() for a in self.actions])
            for action in self.actions:
                action.arguments = action.arguments.model_copy()

        messages = self.component_modeling_log

        attributes = self.attributes
        for action in self.actions:
            messages.append(f"Calculating energy for {self.name} action {action.name}.")
            args = action.arguments
            if args.energy is not None:
                energy = args.energy
                messages.append(f"Setting {self.name} energy to {args.energy=}")
            else:
                self.populate_component_model(
                    models,
                    in_place=True,
                    trying_to_calculate=f"arguments.energy for action {action.name}",
                )
                energy = self.component_model.try_call_arbitrary_action(
                    action_name=action.name,
                    _return_estimation_object=True,
                    **{**attributes.model_dump(), **args.model_dump()},
                )
                messages.extend(energy.messages)
                energy = energy.value[0]
            if attributes.energy_scale != 1:
                energy *= attributes.energy_scale
                messages.append(
                    f"Scaling {self.name} energy by {attributes.energy_scale=}"
                )
            if args.energy_scale != 1:
                energy *= args.energy_scale
                messages.append(f"Scaling {self.name} energy by {args.energy_scale=}")
            action.arguments.energy = energy
        return self

    def calculate_leak_power(
        self: T,
        models: list[ComponentModel] | None = None,
        in_place: bool = False,
    ) -> T:
        """
        Calculates the leak power for this component. If leak power is set in the
        arguments or attributes (with arguments taking precedence), that value will be
        used. Otherwise, the leak power will be calculated using hwcomponents. Populates
        ``attributes.leak_power`` field. Extends the ``component_modeling_log`` field with log
        messages.

        Uses the ``component_model`` attribute, or, if not set, the ``component_class``
        attribute to find the model and populate the ``component_model`` attribute.

        Note that these methods will be called by the Spec when calculating energy and
        area. If you call them yourself, note that string expressions may not be parsed
        because they need the Spec's global scope. If you are sure that all necessary
        values are present and not a result of an expression, you can call these
        directly. Otherwise, you can call the ``Spec.calculate_component_area_energy_latency_leak``
        and then grab components from the returned ``Spec``.

        Parameters
        ----------
        models : list[ComponentModel] | None
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
            self = self.model_copy()
            self.attributes = self.attributes.model_copy()
            self.actions = type(self.actions)([a.model_copy() for a in self.actions])
            for action in self.actions:
                action.arguments = action.arguments.model_copy()

        attributes = self.attributes
        messages = self.component_modeling_log
        if attributes.leak_power is not None:
            leak_power = attributes.leak_power
            messages.append(
                f"Using predefined leak power value {attributes.leak_power=}"
            )
        else:
            self.populate_component_model(
                models,
                in_place=True,
                trying_to_calculate="attributes.leak_power",
            )
            leak_power = self.component_model.leak_power
        if attributes.leak_power_scale != 1:
            leak_power *= attributes.leak_power_scale
            messages.append(f"Scaling leak power by {attributes.leak_power_scale=}")
        if attributes.n_parallel_instances != 1:
            leak_power *= attributes.n_parallel_instances
            messages.append(f"Scaling leak power by {attributes.n_parallel_instances=}")
        self.attributes.leak_power = leak_power
        return self

    def calculate_area(
        self: T,
        models: list[ComponentModel] | None = None,
        in_place: bool = False,
    ) -> T:
        """
        Calculates the area for this component. If area is set in the attributes, that
        value will be used. Otherwise, the area will be calculated using the
        hwcomponents library. Populates ``attributes.area`` field. Extends the
        ``component_modeling_log`` field with log messages.

        Uses the ``component_model`` attribute, or, if not set, the ``component_class``
        attribute to find the model and populate the ``component_model`` attribute.

        Note that these methods will be called by the Spec when calculating
        energy and area. If you call them yourself, note that string expressions may not
        be parsed because they need the Spec's global scope. If you are sure
        that all necessary values are present and not a result of an expression, you can
        call these directly. Otherwise, you can call the
        ``Spec.calculate_component_area_energy_latency_leak`` and then grab components from
        the returned ``Spec``.

        Parameters
        ----------
        models : list[ComponentModel] | None
            The models to use for area calculation. If not provided, the models will be
            found with `hwcomponents.get_models()`.
        in_place : bool
            If True, the component will be modified in place. Otherwise, a copy will be
            returned.

        Returns
        -------
        T
            A copy of the component with the calculated area.
        """
        if not in_place:
            self = self.model_copy()
            self.attributes = self.attributes.model_copy()
            self.actions = type(self.actions)([a.model_copy() for a in self.actions])
            for action in self.actions:
                action.arguments = action.arguments.model_copy()

        attributes = self.attributes
        messages = self.component_modeling_log
        if attributes.area is not None:
            area = attributes.area
            messages.append(f"Using predefined area value {attributes.area=}")
        else:
            self.populate_component_model(
                models,
                in_place=True,
                trying_to_calculate="attributes.area",
            )
            area = self.component_model.area
        if attributes.area_scale != 1:
            area *= attributes.area_scale
            messages.append(f"Scaling area by {attributes.area_scale=}")
        if attributes.n_parallel_instances != 1:
            area *= attributes.n_parallel_instances
            messages.append(f"Scaling area by {attributes.n_parallel_instances=}")
        self.attributes.area = area
        return self

    def calculate_action_latency(
        self: T,
        models: list[ComponentModel] | None = None,
        in_place: bool = False,
    ) -> T:
        """
        Calculates the latency for each action by this component. Populates the
        ``<action>.arguments.latency`` field. Extends the ``component_modeling_log`` field with
        log messages.

        Parameters
        ----------
        models : list[ComponentModel] | None
            The models to use for latency calculation. If not provided, the models will be
            found with `hwcomponents.get_models()`.
        in_place : bool
            If True, the component will be modified in place. Otherwise, a copy will be
            returned.

        Returns
        -------
        T
            A copy of the component with the calculated latency for each action.
        """
        if not in_place:
            self = self.model_copy()
            self.attributes = self.attributes.model_copy()
            self.actions = type(self.actions)([a.model_copy() for a in self.actions])
            for action in self.actions:
                action.arguments = action.arguments.model_copy()

        messages = self.component_modeling_log

        attributes = self.attributes
        for action in self.actions:
            messages.append(
                f"Calculating latency for {self.name} action {action.name}."
            )
            args = action.arguments
            if args.latency is not None:
                latency = args.latency
                messages.append(f"Setting {self.name} latency to {args.latency=}")
            else:
                self.populate_component_model(
                    models,
                    in_place=True,
                    trying_to_calculate=f"arguments.latency for action {action.name}",
                )
                latency = self.component_model.try_call_arbitrary_action(
                    action_name=action.name,
                    _return_estimation_object=True,
                    **{**attributes.model_dump(), **args.model_dump()},
                )
                messages.extend(latency.messages)
                latency = latency.value[1]
            if attributes.latency_scale != 1:
                latency *= attributes.latency_scale
                messages.append(
                    f"Scaling {self.name} latency by {attributes.latency_scale=}"
                )
            if args.latency_scale != 1:
                latency *= args.latency_scale
                messages.append(f"Scaling {self.name} latency by {args.latency_scale=}")
            if attributes.n_parallel_instances != 1:
                latency /= attributes.n_parallel_instances
                messages.append(
                    f"Dividing {self.name} latency by {attributes.n_parallel_instances=}"
                )
            action.arguments.latency = latency
        return self

    def calculate_area_energy_latency_leak(
        self: T, models: list[ComponentModel] | None = None, in_place: bool = False
    ) -> T:
        """
        Calculates the area, energy, latency, and leak power for this component.
        Populates the ``attributes.area``, ``attributes.total_area``,
        ``attributes.leak_power``, ``attributes.total_leak_power``,
        ``attributes.total_latency``, and ``component_modeling_log`` fields of this
        component. Additionally, for each action, populates the
        ``<action>.arguments.area``, ``<action>.arguments.energy``,
        ``<action>.arguments.latency``, and ``<action>.arguments.leak_power`` fields.
        Extends the ``component_modeling_log`` field with log messages.

        Note that these methods will be called by the Spec when calculating energy and
        area. If you call them yourself, note that string expressions may not be parsed
        because they need the Spec's global scope. If you are sure that all necessary
        values are present and not a result of an expression, you can call these
        directly. Otherwise, you can call the ``Spec.calculate_component_area_energy_latency_leak``
        and then grab components from the returned ``Spec``.

        Parameters
        ----------
        models : list[ComponentModel] | None
            The models to use for energy calculation. If not provided, the models will
            be found with `hwcomponents.get_models()`.
        in_place : bool
            If True, the component will be modified in place. Otherwise, a copy will be
            returned.

        Returns
        -------
        T
            The component with the calculated energy, area, and leak power.
        """
        if not in_place:
            self = self.model_copy()
            self.attributes = self.attributes.model_copy()
            self.actions = type(self.actions)([a.model_copy() for a in self.actions])
            for action in self.actions:
                action.arguments = action.arguments.model_copy()
        self.calculate_area(models, in_place=True)
        self.calculate_action_energy(models, in_place=True)
        self.calculate_action_latency(models, in_place=True)
        self.calculate_leak_power(models, in_place=True)
        return self


class Container(Leaf):
    """A `Container` is an abstract node in the architecture that contains other nodes.
    For example, a P` may be a `Container` that contains `Memory`s and `Compute` units.
    """

    pass


MEMORY_ACTIONS = ParsableList[TensorHolderAction](
    [
        TensorHolderAction(name="read"),
        TensorHolderAction(name="write"),
    ]
)


PROCESSING_STAGE_ACTIONS = ParsableList[TensorHolderAction](
    [
        TensorHolderAction(name="read"),
    ]
)

COMPUTE_ACTIONS = ParsableList(
    [
        Action(name="compute"),
    ]
)


def _parse_tensor2bits(
    to_parse: dict[str, Any], location: str, symbol_table: dict[str, Any]
) -> dict[str, Any]:
    result = {}
    for key, value in to_parse.items():
        if isinstance(value, Number):
            result[key] = value
            continue
        result[key] = parse_expression(
            expression=value,
            symbol_table=symbol_table,
            attr_name=key,
            location=location,
        )
    return result


class TensorHolderAttributes(ComponentAttributes):
    """
    Attributes for a `TensorHolder`. `TensorHolder`s are components that hold tensors
    (usually `Memory`s). When specifying these attributes, it is recommended to
    underscore-prefix attribute names. See `TODO: UNDERSCORE_PREFIX_DISCUSSION`.
    """

    bits_per_value_scale: ParsesTo[dict | int | float] = {"All": 1}
    """
    A scaling factor for the bits per value of the tensors in this `TensorHolder`. If
    this is a dictionary, keys in the dictionary are parsed as expressions and may
    reference one or more tensors.
    """

    bits_per_action: ParsesTo[int | float | None] = None
    """
    The number of bits accessed in each of this component's actions. Overridden by
    bits_per_action in the action arguments. If set here, acts as a default value for
    the bits_per_action of all actions of this component.
    """

    def model_post_init(self, __context__=None) -> None:
        if not isinstance(self.bits_per_value_scale, dict):
            self.bits_per_value_scale = {"All": self.bits_per_value_scale}

    def _parse_expressions(self, *args, **kwargs):
        class MyPostCall(_PostCall):
            def __call__(self, field, value, parsed, symbol_table):
                if field == "bits_per_value_scale":
                    parsed = _parse_tensor2bits(
                        parsed,
                        location="bits_per_value_scale",
                        symbol_table=symbol_table,
                    )
                return parsed

        return super()._parse_expressions(*args, **kwargs, post_calls=(MyPostCall(),))


class MemoryAttributes(TensorHolderAttributes):
    """Attributes for a `Memory`."""

    size: ParsesTo[int | float]
    """ The size of this `Memory` in bits. """


class Tensors(ParsableModel):
    """
    Fields that control which tensor(s) are kept in a :py:class:`~.TensorHolder` and in
    what order their nodes may appear in the mapping.
    """

    keep: str | InvertibleSet[TensorName] | set[TensorName] = "<Defaults to Nothing>"
    """
    A set expression describing which tensors must be kept in this
    :class:`fastfusion.frontend.arch.TensorHolder`. If this is not defined, then all
    tensors must be kept.
    """

    may_keep: str | InvertibleSet[TensorName] | set[TensorName] = (
        "<Nothing if keep is defined, else All>"
    )
    """
    A set expression describing which tensors may optionally be kept in this
    :class:`fastfusion.frontend.arch.TensorHolder`. The mapper will explore both keeping
    and not keeping each of these tensors. If this is not defined, then all tensors may
    be kept.
    """

    tile_shape: ParsableList[Comparison] = []
    """
    The tile shape for each rank variable. This is given as a list of
    :class:`~.Comparison` objects, where each comparison must evaluate to True for a
    valid mapping.
    """

    no_refetch_from_above: str | InvertibleSet[TensorName] | set[TensorName] = "~All"
    """
    The tensors that are not allowed to be refetched from above. This is given as a set
    of :class:`~.TensorName` objects or a set expression that resolves to them. These
    tensors must be fetched at most one time from above memories, and may not be
    refetched across any temporal or spatial loop iterations. Tensors may be fetched in
    pieces (if they do not cause re-fetches of any piece).
    """

    tensor_order_options: ParsableList[
        ParsableList[str | InvertibleSet[TensorName] | set[TensorName]]
    ] = ParsableList()
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

    def _parse_tensor_order_options(
        self, symbol_table: dict[str, Any], location: str
    ) -> "Tensors":
        result = type(self)(
            tensor_order_options=[
                [
                    eval_set_expression(x, symbol_table, "tensors", location)
                    for x in order_choice
                ]
                for order_choice in self.tensor_order_options
            ],
        )
        # Assert that there are no intersecting sets
        for order in result.tensor_order_options:
            for i, s0 in enumerate(order):
                for j, s1 in enumerate(order):
                    if i == j:
                        continue
                    if s0 & s1:
                        raise ValueError(
                            f"Intersecting entries in dataflow constraint: {s0} and {s1}"
                        )
        return result

    def _parse_keep(self, symbol_table: dict[str, Any], location: str) -> "Tensors":
        keep, may_keep = self.keep, self.may_keep
        if may_keep == "<Nothing if keep is defined, else All>":
            may_keep = "All" if keep == "<Defaults to Nothing>" else "~All"
        if keep == "<Defaults to Nothing>":
            keep = "Nothing"

        may_keep_first = isinstance(keep, str) and re.findall(r"\bmay_keep\b", keep)
        keep_first = isinstance(may_keep, str) and re.findall(r"\bkeep\b", may_keep)
        if keep_first and may_keep_first:
            raise ValueError(
                f"Keep and may_keep reference each other: " f"{keep} and {may_keep}"
            )

        if may_keep_first:
            may_keep = eval_set_expression(may_keep, symbol_table, "tensors", location)
            symbol_table = copy.copy(symbol_table)
            symbol_table["may_keep"] = may_keep
            keep = eval_set_expression(keep, symbol_table, "tensors", location)
            return type(self)(keep=keep, may_keep=may_keep)
        else:
            keep = eval_set_expression(keep, symbol_table, "tensors", location)
            symbol_table = copy.copy(symbol_table)
            symbol_table["keep"] = keep
            may_keep = eval_set_expression(may_keep, symbol_table, "tensors", location)
            return type(self)(keep=keep, may_keep=may_keep)

    def _parse_non_keep(self, symbol_table: dict[str, Any], location: str) -> "Tensors":
        return type(self)(
            tile_shape=[x._parse(symbol_table, location) for x in self.tile_shape],
            no_refetch_from_above=eval_set_expression(
                self.no_refetch_from_above, symbol_table, "tensors", location
            ),
            force_memory_hierarchy_order=parse_expression(
                self.force_memory_hierarchy_order,
                symbol_table,
                "force_memory_hierarchy_order",
                location,
            ),
        )


@_uninstantiable
class TensorHolder(Component):
    """
    A `TensorHolder` is a component that holds tensors. These are usually `Memory`s,
    but can also be `ProcessingStage`s.
    """

    actions: ParsableList[TensorHolderAction] = MEMORY_ACTIONS
    """ The actions that this `TensorHolder` can perform. """

    attributes: TensorHolderAttributes = pydantic.Field(
        default_factory=TensorHolderAttributes
    )
    """ The `TensorHolderAttributes` that describe this `TensorHolder`. """

    tensors: Tensors = Tensors()
    """
    Fields that control which tensor(s) are kept in this `TensorHolder` and in what
    order their nodes may appear in the mapping.
    """

    def model_post_init(self, __context__=None) -> None:
        self._update_actions(MEMORY_ACTIONS)


class Fanout(Leaf):
    """
    Creates a spatial fanout, and doesn't do anything else.
    """

    attributes: FanoutAttributes = pydantic.Field(default_factory=FanoutAttributes)
    """ Fanout attributes. Zero energy, leak power, area, and latency. """


class Memory(TensorHolder):
    """A `Memory` is a `TensorHolder` that stores data over time, allowing for temporal
    reuse."""

    attributes: "MemoryAttributes" = pydantic.Field(default_factory=MemoryAttributes)
    """ The attributes of this `Memory`. """

    actions: ParsableList[TensorHolderAction] = MEMORY_ACTIONS
    """ The actions that this `Memory` can perform. """


class ProcessingStageAttributes(TensorHolderAttributes):
    """Attributes for a `ProcessingStage`."""

    direction: Literal["up", "down", "up_and_down"]
    """
    The direction in which data flows through this `ProcessingStage`. If "up", then data
    flows from below `TensorHolder`, through this `ProcessingStage` (plus paying
    associated costs), and then to the next `TensorHolder` above it. Other data
    movements are assumed to avoid this ProcessingStage.
    """


class ProcessingStage(TensorHolder):
    """A `ProcessingStage` is a `TensorHolder` that does not store data over time, and
    therefore does not allow for temporal reuse. Use this as a toll that charges reads
    and writes every time a piece of data moves through it.

    Every write to a `ProcessingStage` is immediately written to the next `Memory`
    (which may be above or below depending on where the write came from), and same for
    reads.

    The access counts of a `ProcessingStage` are only included in the "read" action.
    Each traversal through the `ProcessingStage` is counted as a read. Writes are always
    zero.
    """

    attributes: ProcessingStageAttributes = pydantic.Field(
        default_factory=ProcessingStageAttributes
    )
    """ The attributes of this `ProcessingStage`. """

    actions: ParsableList[TensorHolderAction] = PROCESSING_STAGE_ACTIONS
    """ The actions that this `ProcessingStage` can perform. """

    def model_post_init(self, __context__=None) -> None:
        self._update_actions(PROCESSING_STAGE_ACTIONS)


class ComputeAttributes(ComponentAttributes):
    """Attributes for a `Compute`."""

    pass


class Compute(Component):
    actions: ParsableList[Action] = COMPUTE_ACTIONS
    """ The actions that this `Compute` can perform. """

    attributes: ComputeAttributes = pydantic.Field(default_factory=ComputeAttributes)
    """ The attributes of this `Compute`. """

    def model_post_init(self, __context__=None) -> None:
        self._update_actions(COMPUTE_ACTIONS)


T = TypeVar("T")


@_uninstantiable
class Branch(ArchNode):
    # nodes: ArchNodes[_InferFromTag[Compute, Memory, "Hierarchical"]] = ArchNodes()
    nodes: ArchNodes[
        Annotated[
            Union[
                Annotated[Compute, Tag("Compute")],
                Annotated[Memory, Tag("Memory")],
                Annotated[ProcessingStage, Tag("ProcessingStage")],
                Annotated[Fanout, Tag("Fanout")],
                Annotated["Parallel", Tag("Parallel")],
                Annotated["Hierarchical", Tag("Hierarchical")],
            ],
            Discriminator(_get_tag),
        ]
    ] = ArchNodes()

    def get_nodes_of_type(self, types: Type[T] | tuple[Type[T], ...]) -> Iterator[T]:
        for node in self.nodes:
            if isinstance(node, types):
                yield node
            elif isinstance(node, Branch):
                yield from node.get_nodes_of_type(types)


class Parallel(Branch):
    def _flatten(
        self,
        attributes: dict,
        compute_node: str,
        fanout: int = 1,
        return_fanout: bool = False,
    ):
        nodes = []

        def _parse_node(node: Leaf, fanout: int):
            fanout *= node.get_fanout()
            node2 = node.model_copy()
            node2.attributes = type(node.attributes)(
                **{**attributes.model_dump(), **node.attributes.model_dump()}
            )
            nodes.append(node2)
            return fanout

        for node in self.nodes:
            if isinstance(node, Compute) and node.name == compute_node:
                fanout = _parse_node(node, fanout)
                break
            if isinstance(node, Branch):
                computes = node.get_nodes_of_type(Compute)
                if compute_node in [c.name for c in computes]:
                    new_nodes, new_fanout = node._flatten(
                        attributes, compute_node, fanout, return_fanout=True
                    )
                    nodes.extend(new_nodes)
                    fanout *= new_fanout
                    break
        else:
            raise ParseError(f"Compute node {compute_node} not found in parallel node")

        return nodes, fanout if return_fanout else nodes


class Hierarchical(Branch):
    def _flatten(
        self,
        attributes: dict,
        compute_node: str,
        fanout: int = 1,
        return_fanout: bool = False,
    ):
        nodes = []

        def _parse_node(node: Leaf, fanout: int):
            fanout *= node.get_fanout()
            node2 = node.model_copy()
            attrs = {**node.attributes.model_dump()}
            if isinstance(node.attributes, AttributesWithExtras):
                attrs = {**attributes.model_dump(), **attrs}
            node2.attributes = type(node.attributes)(**attrs)
            nodes.append(node2)
            return fanout

        for i, node in enumerate(self.nodes):
            try:
                if isinstance(node, (Hierarchical, Parallel)):
                    if isinstance(node, Parallel) and i < len(self.nodes) - 1:
                        raise ParseError(
                            f"Parallel node {node.name} must be the last node in a "
                            "hierarchical node"
                        )
                    new_nodes, new_fanout = node._flatten(
                        attributes, compute_node, fanout, return_fanout=True
                    )
                    nodes.extend(new_nodes)
                    fanout *= new_fanout
                    if any(
                        isinstance(n, Compute) and n.name == compute_node
                        for n in new_nodes
                    ):
                        break
                elif isinstance(node, Compute):
                    if node.name == compute_node:
                        fanout = _parse_node(node, fanout)
                        break
                elif isinstance(node, Leaf) and not isinstance(node, Container):
                    fanout = _parse_node(node, fanout)
                elif isinstance(node, Container):
                    fanout *= node.get_fanout()
                else:
                    raise TypeError(f"Can't flatten {node}")
            except ParseError as e:
                e.add_field(node)
                raise e

        if return_fanout:
            return nodes, fanout
        return nodes

    def render(self) -> str:
        graph = _pydot_graph()
        graph.add_node(pydot.Node("root", shape="box", label="TODO: Arch Render"))
        return _SVGJupyterRender(graph.create_svg(prog="dot").decode("utf-8"))

    def _repr_svg_(self) -> str:
        return self.render()


class _ConstraintLambda:
    def __init__(
        self,
        constraint: Comparison,
        target_mapping_nodes: list[Spatial],
        rank_variables: set[str],
    ):
        self.constraint = constraint
        self.constraint_lambda = (
            None if constraint is None else constraint._to_constraint_lambda(True)
        )
        self.target_mapping_nodes = target_mapping_nodes
        self.rank_variables = rank_variables
        self._target_node_indices = None
        self._target_loop_indices = None

    def __call__(self, rank_variables: set[RankVariable], sizes: np.ndarray) -> bool:
        final = self.rank_variables.issubset(rank_variables)
        return self.constraint_lambda(final, sizes)

    def _constrained_node_str(self) -> str:
        return f"constrains {self._target_node_indices}"

    def __bool__(self) -> bool:
        return bool(self.target_mapping_nodes)


class _TileShapeConstraintLambda(_ConstraintLambda):
    def pretty_str(self) -> str:
        return f"Tile shape {self.constraint.operator} {self.constraint.value} {self._constrained_node_str()}"


class _LoopBoundsConstraintLambda(_ConstraintLambda):
    def pretty_str(self) -> str:
        return f"Loop bounds {self.constraint.operator} {self.constraint.value} {self._constrained_node_str()}"


class _MinUtilizationConstraintLambda(_ConstraintLambda):
    def __init__(
        self,
        target_mapping_nodes: list[Spatial],
        rank_variables: set[str],
        min_usage: float,
    ):
        super().__init__(None, target_mapping_nodes, rank_variables)
        self.min_usage = min_usage

    def __call__(self, complete_indices: list[int], utilizations: np.ndarray) -> bool:
        # final = self.rank_variables.issubset(rank_variables)
        final = set(self._target_loop_indices).issubset(set(complete_indices))
        if not final:
            return np.ones(utilizations.shape[0], dtype=np.bool)

        # Some utilizations are already above the minimum. Return those.
        result = utilizations >= self.min_usage
        if np.sum(result) > 0:
            return result

        # Nobody is amove the minimum. Return the best we can do.
        max_utilization = np.max(utilizations, axis=0)
        return utilizations == max_utilization

    def pretty_str(self) -> str:
        return f"Min utilization {self.min_usage} {self._constrained_node_str()}"


class Arch(Hierarchical):
    # version: Annotated[str, assert_version] = __version__
    # """ The version of the architecture spec. """

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
            node.name: node.attributes.total_area
            for node in self.get_nodes_of_type(Component)
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
            node.name: node.attributes.total_leak_power
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

    def _parse_expressions(self, *args, **kwargs):
        symbol_table = kwargs["symbol_table"]
        for node in self.get_nodes_of_type(Leaf):
            symbol_table[node.name] = node
        return super()._parse_expressions(*args, **kwargs)

    def __getitem__(self, name: str) -> Leaf:
        return self.name2leaf(name)

    def model_post_init(self, __context__=None) -> None:
        # Make sure all leaf names are unique
        leaves = {}
        for l in self.get_nodes_of_type(Leaf):
            n = l.name
            leaves.setdefault(n, l)
            assert l is leaves[n], f"Duplicate name {n} found in architecture"


# We had to reference Hierarchical before it was defined
Branch.model_rebuild()
