"""
Defines Pydantic models to handle Binding Specifications that relate logical to
physical architectures.
"""

from abc import abstractmethod
from typing import Dict, Set, Tuple, TypeAlias

from pydantic import StrictFloat, model_validator
import islpy as isl

from accelforge.util._basetypes import EvalableDict, EvalableList, EvalableModel


class Domain(EvalableModel):
    """
    Represents an architecture dangling reference of the binding.
    """

    _prefix: str
    name: str
    dims: EvalableList[str]
    
    @property
    def isl_space(self) -> isl.Space:
        return isl.Space.create_from_names(
            isl.DEFAULT_CONTEXT, set=self.dims
        ).set_tuple_name(isl.dim_type.set, f"{self._prefix}_{self.name}_dims")

    @property
    def isl_universe(self) -> isl.Map:
        return isl.Map.universe(self.isl_space)


class LogicalDomain(Domain):
    """
    Represents the logical architecture domain space of logical dims × tensor ranks.
    """
    _prefix: str = 'l'
    pass


class PhysicalDomain(Domain):
    """
    Represents the logical architecture domain space of physical dims.
    The physical space is defined as the physical architecture dims.
    """
    _prefix: str = 'p'
    pass


class BindingNode(EvalableModel):
    """
    How a logical architecture is implemented on a particular physical architecture
    for a particular hardware level. Represents a injection relation between points
    in logical to physical space.

    The logical space is defined as logical architecture dims × tensor dims.
    The physical space is defined as the physical architecture dims.
    """

    logical: LogicalDomain
    """The logical domain of the components being bound."""
    physical: PhysicalDomain
    """The physical location of the components being bound."""
    relations: EvalableDict[str, str]
    """A relation between each tensor and its logical domain to physical domain."""

    @property
    def isl_relations(self) -> Dict[str, isl.Map]:
        """
        Converts the logical, physical, and binding relation strings into an
        isl.Map representing the bindings at this binding node.
        """

        def islify_relation(key: str) -> isl.Map:
            """Converts a relation at a given key into isl"""
            relation: str = self.relations[key]
            logical_space: isl.Space = self.logical.isl_space
            binding_space: isl.Space = logical_space.map_from_domain_and_range(
                range=self.physical.isl_space,
            )

            # Simple bodge to get the binding space into a real space
            binding_str: str = binding_space.to_str()
            binding_str: str = f"{binding_str[:-1]}: {relation} {binding_str[-1]}"

            binding: isl.Map = isl.Map.read_from_str(
                ctx=isl.DEFAULT_CONTEXT, str=binding_str
            )

            return binding

        isl_relations: Dict[str, isl.Map] = {
            key: islify_relation(key) for key in self.relations
        }

        return isl_relations


class Binding(EvalableModel):
    """
    A collection of binding nodes that fully specifies a relation between the
    logical and physical space.
    """

    version: StrictFloat
    """Version of the binding spec."""
    nodes: EvalableList[BindingNode]
    """Parts of the binding."""

'''
class NetworkOnChip(ParsableModel):
    """A model of a network-on-chip on the physical chip."""

    name: str
    """NoC name"""
    cost: isl.PwMultiAff
    """Cost of traversing the NoC"""
    domain: isl.Set
    """The defined points of the NoC"""

    @model_validator(mode="after")
    def validate(self):
        """Ensures the domain is not empty."""
        if self.domain.is_empty():
            raise ValueError("NoC domain cannot be empty:\n" f"Domain: {self.domain}")


class Placement(ParsableModel):
    """The location(s) of a hardware component physically on the chip."""

    noc: NetworkOnChip
    "The NoC objects are being placed on."
    placement: isl.Map
    "The placement of objects on the NoC."

    @model_validator(mode="after")
    def validate(self) -> "Placement":
        """
        Validates that the `placement` exists on the `noc`.
        """
        placement: isl.Map = self.placement
        if placement.is_empty():
            raise ValueError(f"Placement cannot be empty: {placement}")

        noc_domain: isl.Set = self.noc.domain
        placement_range: isl.Set = placement.range()
        if not placement_range.is_subset(noc_domain):
            raise ValueError(
                "Placement has areas off the NoC:\n"
                "--------------------------------\n"
                f"Placement Range: {placement_range}\n"
                f"NoC Domain: {noc_domain}"
            )

        return self


class PhysicalComponent(ParsableModel):
    """The description of the physical components."""

    name: str
    """Name of the physical component."""
    placement: Placement
    """Placement of the physical component on chip."""


class PhysicalComponents(ParsableModel):
    """List of componenents on the physical chip."""

    components: EvalableList[PhysicalComponent]
    """The components referred to."""


# For static analysis of types being used.
PhysicalStorage: TypeAlias = PhysicalComponent
PhysicalProcessingElement: TypeAlias = PhysicalComponent


# Ports connecting the different NoC Components.
class Port(ParsableModel):
    """Connection between two physical components of a chip and the cost to use."""

    parent: NetworkOnChip
    child: NetworkOnChip
    relation: isl.Map

    @model_validator(mode="after")
    def validate(self):
        # Ensures the parent and child NoCs are not the same.
        if self.parent.name == self.child.name:
            raise ValueError(
                "Parent should not be equal to the child.\n"
                f"Parent: {self.parent}\n"
                f"Child: {self.child}\n"
            )

        # Ensures port map is not null.
        if self.relation.is_empty():
            raise ValueError("Port cannot be empty\n" f"Port: {Port}")
        # Ensures the relation is contained in both NoCs.
        parent_domain: isl.Set = self.parent.domain
        child_domain: isl.Set = self.child.domain
        ports: isl.Map = self.relation.domain().unwrap()
        if not ports.range().is_subset(child_domain):
            raise ValueError(
                "Port map range is not a subset of the child domain.\n"
                f"Ports: {ports.range()}\n"
                f"Child: {self.child}"
            )
        if not ports.domain().is_subset(parent_domain):
            raise ValueError(
                "Port map domain is not a subset of the parent domain.\n"
                f"Ports: {ports.domain()}\n"
                f"Child: {self.parent}"
            )


class PhysicalSpec(ParsableModel):
    """Physical specification of an accelerator."""

    name: str
    nocs: Set[NetworkOnChip]
    ports: Tuple[Port]
    components: Set[PhysicalComponent]

    @model_validator(mode="after")
    def validate(self):
        # Checks that the networks are not empty.
        if not self.networks:
            raise ValueError(f"A physical chip {self.name} cannot have no NoCs.")
        # Checks that the components are not empty.
        if not self.components:
            raise ValueError(f"A physical chip {self.name} cannot have no components.")
        # Check that the components are on on-chip NoCs.
        for component in self.components:
            if component.placement.noc not in self.nocs:
                raise ValueError(
                    "Component placed on invalid NoC.\n"
                    f"Component: {component}\n"
                    f"NoCs: {self.nocs}"
                )
        # TODO: Check that all ports form a DAG between the networks and all
        # networks are connected.


class LogicalComponent(EvalableDict):
    """Logical components of an accelerator."""

    name: str
    indices: isl.Set


class LogicalSpec(ParsableModel):
    """Logical specification of an accelerator."""

    name: str
    components: Set[LogicalComponent]


class BindingSpec(ParsableModel):
    """The binding of physical to logical components."""

    name: str
    physical: PhysicalSpec
    logical: LogicalSpec
    bindings: Set[isl.Map]
'''
