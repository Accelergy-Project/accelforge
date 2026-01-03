from abc import ABC
import math
from numbers import Number
from typing import (
    Any,
    Iterator,
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
    EnergyAreaModel,
    get_models,
    get_model,
)

from fastfusion.util._basetypes import (
    ParsableModel,
    ParsableList,
    ParsesTo,
    _PostCall,
    _get_tag,
)
from fastfusion.util._parse_expressions import ParseError, parse_expression
from fastfusion.util._setexpressions import InvertibleSet, eval_set_expression
from fastfusion.frontend.renames import TensorName

from fastfusion._version import assert_version, __version__
from pydantic import Discriminator

from fastfusion.frontend.constraints import ConstraintGroup, MiscOnlyConstraints

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


class Spatial(ParsableModel):
    """A one-dimensional spatial fanout in the architecture."""

    name: str
    """
    The name of the dimension over which this spatial fanout is occurring (e.g., X or Y).
    """

    fanout: ParsesTo[int]
    """ The size of this fanout. """

    reuse: str | InvertibleSet[TensorName] | set[TensorName] = "All()"
    """ The tensors that are reused in this fanout. This expression will be parsed for
    each pmapping template. """

    def _parse(self, symbol_table: dict[str, Any], location: str):
        return type(self)(
            name=self.name,
            fanout=self.fanout,
            reuse=set(
                eval_set_expression(
                    self.reuse,
                    symbol_table,
                    expected_space_name="tensors",
                    location=location + ".reuse",
                )
            ),
        )


class LeafAttributes(ParsableModel):
    pass


class AttributesWithEnergy(ParsableModel):
    energy: ParsesTo[int | float | None] = None
    energy_scale: ParsesTo[int | float] = 1
    model_config = ConfigDict(extra="allow")


class ComponentAttributes(AttributesWithEnergy):
    latency: str | int | float = 0
    """
    An expression representing the latency of this component in seconds. This is used to
    calculate the latency of a given Einsum. Special variables available are `min`,
    `max`, and `X_actions`, where `X` is the name of an action that this component can
    perform. `X_actions` resolves to the number of times action `X` is performed. For
    example, `read_actions` is the number of times the read action is performed.

    For example, the following expression calculates latency assuming that each read or
    write action takes 1ns: ``1e-9 * (read_actions + write_actions)``.
    """
    area_scale: ParsesTo[int | float] = 1
    """
    The scale factor for the area of this comxponent. This is used to scale the area of
    this component. For example, if the area is 1 m^2 and the scale factor is 2, then
    the area is 2 m^2.
    """
    area: ParsesTo[int | float | None] = None
    """
    The area of a single instance of this component in m^2. If set, area calculations
    will use this value.
    """
    total_area: ParsesTo[int | float | None] = None
    """
    The total area of all instances of this component in m^2.
    """
    leak_power: ParsesTo[int | float | None] = None
    """
    The leak power of a single instance of this component in W. If set, leak power
    calculations will use this value.
    """
    total_leak_power: ParsesTo[int | float | None] = None
    """
    The total leak power of all instances of this component in W.
    """
    leak_power_scale: ParsesTo[int | float] = 1
    """
    The scale factor for the leak power of this component. This is used to scale the
    leak power of this component. For example, if the leak power is 1 W and the scale
    factor is 2, then the leak power is 2 W.
    """
    energy: ParsesTo[int | float | None] = None
    """
    Dynamic energy of all actions of this component. If set, energy calculations will
    use this value for all actions that are not overridden by the action's arguments.
    """
    energy_scale: ParsesTo[int | float] = 1
    """
    The scale factor for dynamic energy of this component. For each action, multiplies
    this action's energy.
    """


class ActionArguments(AttributesWithEnergy):
    """
    Arguments for an action of a component.
    """

    energy: ParsesTo[int | float | None] = None
    """
    Dynamic energy of this action. If set, this value will be used instead of the
    component's attributes.energy. Higher precedence than the component's
    attributes.energy. Per-action energy is multiplied by the component's
    attributes.energy_scale and the action's arguments.energy_scale.
    """
    energy_scale: ParsesTo[int | float] = 1
    """
    The scale factor for dynamic energy of this action. Multiplies this action's energy
    by this value.
    """


class Action(ParsableModel):
    name: str
    arguments: AttributesWithEnergy = AttributesWithEnergy()


class Leaf(ArchNode, ABC):
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

    constraints: ConstraintGroup = ConstraintGroup()
    """ Mapping constraints applied to this `Leaf`. """

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

    def _parse_constraints(self, outer_scope: dict[str, Any]) -> ConstraintGroup:
        self.constraints.name = self.name
        return self.constraints._parse(outer_scope, location=f"{self.name} constraints")


class Component(Leaf, ABC):
    """A component object in the architecture. This is overridden by different
    component types, such as `Memory` and `Compute`."""

    name: str
    """ The name of this `Component`. """

    component_class: Optional[str] = None
    """ The class of this `Component`. Used if an energy or area model needs to be
    called for this `Component`. """

    component_model: EnergyAreaModel | None = None
    """ The model to use for this `Component`. If not set, the model will be found with
    `hwcomponents.get_models()`. If set, the `component_class` will be ignored. """

    energy_area_log: list[str] = []
    """ A log of the energy and area calculations for this `Component`. """

    actions: ParsableList[Action]
    """ The actions that this `Component` can perform. """

    attributes: ComponentAttributes = ComponentAttributes()
    """ The attributes of this `Component`. """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _update_actions(self, new_actions: ParsableList[Action]):
        has_actions = set(x.name for x in self.actions)
        for action in new_actions:
            if action.name not in has_actions:
                self.actions.append(action)

    def get_component_class(self) -> str:
        """Returns the class of this `Component`.

        :raises ParseError: If the `component_class` is not set.
        """
        if self.component_class is None:
            raise ParseError(
                f"component_class must be set to a valid string. "
                f"Got {self.component_class}.",
                source_field=f"{self.name}.component_class",
            )
        return self.component_class

    def populate_component_model(
        self: T,
        models: list[EnergyAreaModel] | None = None,
        in_place: bool = False,
    ) -> T:
        """
        Populates the ``component_model`` attribute with the model for this component.
        Extends the ``energy_area_log`` field with log messages. Uses the
        ``component_class`` attribute to find the model and populate the
        ``component_model`` attribute. Uses the ``hwcomponents.get_model()`` function
        to find the model.

        Parameters
        ----------
        models : list[EnergyAreaModel] | None
            The models to use for energy calculation. If not provided, the models will
            be found with `hwcomponents.get_models()`.
        in_place : bool
            If True, the component will be modified in place. Otherwise, a copy will be
            returned.

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
                self.get_component_class(),
                self.attributes.model_dump(),
                required_actions=list(x.name for x in self.actions),
                models=models,
                return_estimation_object=True,
            )
            self.component_model = estimation.value
            self.energy_area_log.extend(estimation.messages)
        return self

    def calculate_energy(
        self: T,
        models: list[EnergyAreaModel] | None = None,
        in_place: bool = False,
    ) -> T:
        """
        Calculates the leak power for this component and the energy for each action of
        this component. If energy is set in the arguments or attributes (with arguments
        taking precedence), that value will be used. Otherwise, the energy will be
        calculated using the hwcomponents library. If the leak power is set in the
        attributes, that value will be used. Otherwise, the leak power will be
        calculated using the hwcomponents library. Populates ``attributes.leak_power``,
        and ``attributes.total_leak_power`` fields and, for each action, the
        ``<action>.arguments.energy`` and field. Extends the ``energy_area_log`` field
        with log messages.

        Uses the ``component_model`` attribute, or, if not set, the ``component_class``
        attribute to find the model and populate the ``component_model`` attribute.

        Note that these methods will be called by the Spec when calculating
        energy and area. If you call them yourself, note that string expressions may not
        be parsed because they need the Spec's global scope. If you are sure
        that all necessary values are present and not a result of an expression, you can
        call these directly. Otherwise, you can call the
        ``Spec.calculate_component_energy_area`` and then grab components from
        the returned ``Spec``.

        Parameters
        ----------
        models : list[EnergyAreaModel] | None
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

        messages = self.energy_area_log

        attributes = self.attributes
        for action in self.actions:
            messages.append(f"Calculating energy for {self.name} action {action.name}.")
            args = action.arguments
            if args.energy is not None:
                energy = args.energy
                messages.append(f"Setting {self.name} energy to {args.energy=}")
            elif attributes.energy is not None:
                energy = attributes.energy
                messages.append(f"Setting {self.name} energy to {attributes.energy=}")
            else:
                self.populate_component_model(models, in_place=True)
                energy = self.component_model.try_call_arbitrary_action(
                    action_name=action.name,
                    action_arguments={**attributes.model_dump(), **args.model_dump()},
                    return_estimation_object=True,
                )
                messages.extend(energy.messages)
                energy = energy.value
            if attributes.energy_scale != 1:
                energy *= attributes.energy_scale
                messages.append(
                    f"Scaling {self.name} energy by {attributes.energy_scale=}"
                )
            if args.energy_scale != 1:
                energy *= args.energy_scale
                messages.append(f"Scaling {self.name} energy by {args.energy_scale=}")
            action.arguments.energy = energy
        self._calculate_leak_power(models, in_place=True)
        return self

    def calculate_area(
        self: T,
        models: list[EnergyAreaModel] | None = None,
        in_place: bool = False,
    ) -> T:
        """
        Calculates the area for this component. If area is set in the attributes, that
        value will be used. Otherwise, the area will be calculated using the
        hwcomponents library. Populates ``attributes.area`` field. Extends the
        ``energy_area_log`` field with log messages.

        Uses the ``component_model`` attribute, or, if not set, the ``component_class``
        attribute to find the model and populate the ``component_model`` attribute.

        Note that these methods will be called by the Spec when calculating
        energy and area. If you call them yourself, note that string expressions may not
        be parsed because they need the Spec's global scope. If you are sure
        that all necessary values are present and not a result of an expression, you can
        call these directly. Otherwise, you can call the
        ``Spec.calculate_component_energy_area`` and then grab components from
        the returned ``Spec``.

        Parameters
        ----------
        models : list[EnergyAreaModel] | None
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
        messages = self.energy_area_log
        if attributes.area is not None:
            area = attributes.area
            messages.append(f"Using predefined area value {attributes.area=}")
        else:
            self.populate_component_model(models, in_place=True)
            area = self.component_model.area
        if attributes.area_scale != 1:
            area *= attributes.area_scale
            messages.append(f"Scaling area by {attributes.area_scale=}")
        self.attributes.area = area
        return self

    def _calculate_leak_power(
        self: T,
        models: list[EnergyAreaModel] | None = None,
        in_place: bool = False,
    ) -> T:
        """
        Calculates the leak power for this component. If leak power is set in the
        attributes, that value will be used. Otherwise, the leak power will be
        calculated using the hwcomponents library. Populates the attributes
        ``leak_power`` field. Extends the ``energy_area_log`` field with log messages.

        Uses the ``component_model`` attribute, or, if not set, the ``component_class``
        attribute to find the model and populate the ``component_model`` attribute.

        Note that these methods will be called by the Spec when calculating
        energy and area. If you call them yourself, note that string expressions may not
        be parsed because they need the Spec's global scope. If you are sure
        that all necessary values are present and not a result of an expression, you can
        call these directly. Otherwise, you can call the
        ``Spec.calculate_component_energy_area`` and then grab components from
        the returned ``Spec``.

        Parameters
        ----------
        models : list[EnergyAreaModel] | None
            The models to use for leak power calculation. If not provided, the models
            will be found with `hwcomponents.get_models()`.
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
        messages = self.energy_area_log
        if attributes.leak_power is not None:
            leak_power = attributes.leak_power
            messages.append(
                f"Using predefined leak power value {attributes.leak_power=}"
            )
        else:
            self.populate_component_model(models, in_place=True)
            leak_power = self.component_model.leak_power
        if attributes.leak_power_scale != 1:
            leak_power *= attributes.leak_power_scale
            messages.append(f"Scaling leak power by {attributes.leak_power_scale=}")
        self.attributes.leak_power = leak_power
        return self

    def calculate_energy_area(
        self: T, models: list[EnergyAreaModel] | None = None, in_place: bool = False
    ) -> T:
        """
        Calculates the energy, area, and leak power for this component. Populates the
        ``attributes.area``, ``attributes.total_area``, ``attributes.leak_power``,
        ``attributes.total_leak_power``, and ``energy_area_log`` fields of this
        component. Additionally, for each action, populates the
        ``<action>.arguments.energy`` field. Extends the ``energy_area_log`` field with
        log messages.

        Note that these methods will be called by the Spec when calculating
        energy and area. If you call them yourself, note that string expressions may not
        be parsed because they need the Spec's global scope. If you are sure
        that all necessary values are present and not a result of an expression, you can
        call these directly. Otherwise, you can call the
        ``Spec.calculate_component_energy_area`` and then grab components from
        the returned ``Spec``.

        Parameters
        ----------
        models : list[EnergyAreaModel] | None
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
        self.calculate_energy(models, in_place=True)
        self.calculate_area(models, in_place=True)
        self._calculate_leak_power(models, in_place=True)
        return self


class Container(Leaf, ABC):
    """A `Container` is an abstract node in the architecture that contains other nodes.
    For example, a P` may be a `Container` that contains `Memory`s and `Compute` units.
    """

    pass


class ArchMemoryActionArguments(AttributesWithEnergy):
    """Arguments for any `Memory` action."""

    bits_per_action: ParsesTo[int | float] = 1
    """ The number of bits accessed in this action. For example, setting bits_per_action
    to 16 means that each call to this action yields 16 bits. """


class ArchMemoryAction(Action):
    """An action that a `Memory` can perform."""

    arguments: ArchMemoryActionArguments = ArchMemoryActionArguments()
    """ The arguments for this `ArchMemoryAction`. """


MEMORY_ACTIONS = ParsableList(
    [
        ArchMemoryAction(name="read", arguments={"bits_per_action": 1}),
        ArchMemoryAction(name="write", arguments={"bits_per_action": 1}),
    ]
)


PROCESSING_STAGE_ACTIONS = ParsableList(
    [
        ArchMemoryAction(name="read", arguments={"bits_per_action": 1}),
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

    datawidth: ParsesTo[dict | int | float] = {}
    """
    Number of bits per value stored in this `TensorHolder`. If this is a dictionary,
    keys in the dictionary are parsed as expressions and may reference one or more
    `Tensor`s.
    """

    def model_post_init(self, __context__=None) -> None:
        if not isinstance(self.datawidth, dict):
            self.datawidth = {"All()": self.datawidth}

    def _parse_expressions(self, *args, **kwargs):
        class MyPostCall(_PostCall):
            def __call__(self, field, value, parsed, symbol_table):
                if field == "datawidth":
                    parsed = _parse_tensor2bits(
                        parsed, location="datawidth", symbol_table=symbol_table
                    )
                return parsed

        return super()._parse_expressions(*args, **kwargs, post_calls=(MyPostCall(),))


class MemoryAttributes(TensorHolderAttributes):
    """Attributes for a `Memory`."""

    size: ParsesTo[int | float]
    """ The size of this `Memory` in bits. """


class TensorHolder(Component):
    """
    A `TensorHolder` is a component that holds tensors. These are usually `Memory`s,
    but can also be `ProcessingStage`s.
    """

    actions: ParsableList[ArchMemoryAction] = MEMORY_ACTIONS
    """ The actions that this `TensorHolder` can perform. """

    attributes: TensorHolderAttributes = pydantic.Field(
        default_factory=TensorHolderAttributes
    )
    """ The `TensorHolderAttributes` that describe this `TensorHolder`. """

    def model_post_init(self, __context__=None) -> None:
        self._update_actions(MEMORY_ACTIONS)


class Memory(TensorHolder):
    """A `Memory` is a `TensorHolder` that stores data over time, allowing for temporal
    reuse."""

    attributes: "MemoryAttributes" = pydantic.Field(default_factory=MemoryAttributes)
    """ The attributes of this `Memory`. """


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

    actions: ParsableList[ArchMemoryAction] = PROCESSING_STAGE_ACTIONS
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

    constraints: MiscOnlyConstraints = MiscOnlyConstraints()
    """ Mapping constraints applied to this `Compute`. """

    def model_post_init(self, __context__=None) -> None:
        self._update_actions(COMPUTE_ACTIONS)


T = TypeVar("T")


class Branch(ArchNode, ABC):
    # nodes: ArchNodes[_InferFromTag[Compute, Memory, "Hierarchical"]] = ArchNodes()
    nodes: ArchNodes[
        Annotated[
            Union[
                Annotated[Compute, Tag("Compute")],
                Annotated[Memory, Tag("Memory")],
                Annotated[ProcessingStage, Tag("ProcessingStage")],
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
            node2.attributes = type(node.attributes)(
                **{**attributes.model_dump(), **node.attributes.model_dump()}
            )
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
            for node in self.get_nodes_of_type(Leaf)
        }
        for k, v in area.items():
            if v is None:
                raise ValueError(
                    f"Area of {k} is not set. Please call the Spec's "
                    "`calculate_component_energy_area` method before accessing this "
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
            for node in self.get_nodes_of_type(Leaf)
        }
        for k, v in leak_power.items():
            if v is None:
                raise ValueError(
                    f"Leak power of {k} is not set. Please call the Spec's "
                    "`calculate_component_energy_area` method before accessing this "
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
