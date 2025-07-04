"""
Defines Pydantic models to handle Binding Specifications that relate logical to
physical architectures.
"""

from abc import abstractmethod
from typing import Dict, Tuple

from pydantic import StrictFloat
import islpy as isl

from fastfusion.util.basetypes import ParsableDict, ParsableList, ParsableModel


class Domain(ParsableModel):
    """
    Represents an architecture dangling reference of the binding.
    """

    name: str

    @property
    @abstractmethod
    def isl_space(self) -> isl.Space:
        """Gets the domain as an isl.Space"""
        raise NotImplementedError(f"{type(self)} has not implemented isl_space")

    @property
    @abstractmethod
    def isl_universe(self) -> isl.Set:
        """Gets the domain as an isl.Set"""
        raise NotImplementedError(f"{type(self)} has not implemented isl_universe")


class LogicalDomain(Domain):
    """
    Represents the logical architecture domain space of logical dims × ranks.
    """

    ranks: Tuple[str] = ("c", "h", "w", "p", "q", "r", "s")
    l_dims: ParsableList[str]

    @property
    def isl_space(self) -> isl.Space:
        return isl.Space.create_from_names(
            isl.DEFAULT_CONTEXT, in_=self.ranks, out=self.l_dims
        ).set_tuple_name(isl.dim_type.out, f"l_{self.name}_dims")

    @property
    def isl_universe(self) -> isl.Map:
        return isl.Map.universe(self.isl_space)


class PhysicalDomain(Domain):
    """
    Represents the logical architecture domain space of physical dims.
    """

    p_dims: ParsableList[str]

    @property
    def isl_space(self) -> isl.Space:
        return isl.Space.create_from_names(
            isl.DEFAULT_CONTEXT, set=self.p_dims
        ).set_tuple_name(isl.dim_type.set, f"p_{self.name}_dims")

    @property
    def isl_universe(self) -> isl.Set:
        return isl.Set.universe(self.isl_space)


class BindingNode(ParsableModel):
    """
    How a logical architecture is implemented on a particular physical architecture
    for a particular hardware level. Represents a injection relation between points
    in logical to physical space.

    The logical space is defined as logical architecture dims × tensor dims.
    The physical space is defined as physical architecture dims × tensor dims.
    """

    logical: LogicalDomain
    physical: PhysicalDomain
    relations: ParsableDict[str, str]

    @property
    def isl_relations(self) -> Dict[str, isl.Map]:
        """
        Converts the logical, physical, and binding relation strings into an
        isl.Map representing the bindings at this binding node.
        """

        def islify_relation(key: str) -> isl.Map:
            """Converts a relation at a given key into isl"""
            relation: str = self.relations[key]
            logical_space: isl.Space = self.logical.isl_space.set_tuple_name(
                isl.dim_type.in_, f"{key}_ranks"
            )

            binding_space: isl.Space = logical_space.wrap().map_from_domain_and_range(
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


class Binding(ParsableModel):
    """
    A collection of binding nodes that fully specifies a relation between the
    logical and physical space.
    """

    version: StrictFloat
    nodes: ParsableList[BindingNode]


if __name__ == "__main__":
    # now loads YAML
    import yaml

    yaml_str: str = """
    binding:
        version: 0.4
        nodes:
        -   logical:
                name: PE
                l_dims: [i]
            physical: 
                name: PE
                p_dims: [x, y]
            relations:
                tensorA: i = x + y * 2 # This is a dimension-major compression into the logical. It is bijective.
                tensorB: i = x + y * 2 # This is a dimension-major compression into the logical. It is bijective.
        -   logical:
                name: Scratchpad
                l_dims: [x, y]
            physical:
                name: GLB
                p_dims: [gamma, omega]
            relations:
                tensorA: x = gamma and y = omega
                tensorB: x = omega and y = gamma
    """
    bind: Binding = Binding.model_validate(yaml.safe_load(yaml_str)["binding"])

    for node in bind.nodes:
        for bind in node.isl_relations.values():
            print(bind)
