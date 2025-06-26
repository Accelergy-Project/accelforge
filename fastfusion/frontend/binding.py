from typing import Tuple

from pydantic import StrictFloat

from fastfusion.util.basetypes import ParsableDict, ParsableList, ParsableModel
from fastfusion.frontend.type_wrappers.isl import ISLMap, ISLSpace

class Domain(ParsableModel):
    """
    Represents an architecture dangling reference of the binding.
    """
    name: str

class LogicalDomain(Domain):
    """
    Represents the logical architecture domain space of logical dims × ranks.
    """
    ranks: Tuple[str] = (
        'C', 'H', 'W', 'P', 'Q', 'R', 'S'
    )
    dims: Tuple[str]

class PhysicalDomain(Domain):
    """
    Represents the logical architecture domain space of physical dims.
    """
    p_dims: Tuple[str]

class BindingRelation(ParsableModel):
    """
    Represents a injection relation between points in logical to physical space.
    The logical space is defined as logical architecture dims × tensor dims.
    The physical space is defined as physical architecture dims × tensor dims.
    """
    relation: ParsableDict[str, str]

class BindingNode(ParsableModel):
    """
    How a logical architecture is implemented on a particular physical architecture
    for a particular hardware level.
    """
    logical: LogicalDomain
    physical: PhysicalDomain
    nodes: ParsableList[BindingRelation]

class Binding(ParsableModel):
    """
    A collection of binding nodes that fully specifies a relation between the
    logical and physical space.
    """
    version: StrictFloat
    nodes: ParsableList[BindingNode]

# now loads YAML
import yaml
yaml_str: str = """
binding:
  version: 0.4
  nodes:
  - logical:
      name: PE
      p_dims: [i]
    physical: 
      name: PE
      dims: [x, y]
    relation:
      tensorA: i = x + y * 2 # This is a dimension-major compression into the logical. It is bijective.
      tensorB: i = x + y * 2 # This is a dimension-major compression into the logical. It is bijective.
  # Bindings for a hypothetical scratchpad.
  - logical: scratchpad
    physical: scratchpad
    relation:
        tensorA: LogicalScratchpad[i] -> PhysicalScratchpad[x, y] : i = x # This is a dimension-based compression into the logical. It is not surjective.
"""

binding = Binding.model_validate(yaml.safe_load(yaml_str))
