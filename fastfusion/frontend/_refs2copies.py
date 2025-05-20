from abc import abstractmethod, ABC
import copy
import logging
from fastfusion.yamlparse.nodes import Node
from typing import Any, Optional

def refs2copies_fast(spec: "Specification", n: Any, seen_ids=None, visited=None, depth=0) -> Any:
    visited = [] if visited is None else visited
    seen_ids = set() if seen_ids is None else seen_ids
    
    visited.append(n)  # Avoid garbage collection
    if isinstance(n, Node):
        n.parent_node = None

    if id(n) in seen_ids:
        n = copy.deepcopy(n)
    seen_ids.add(id(n))

    if not isinstance(n, Node):
        return n

    if isinstance(n, Node):
        for i, x in n.items():
            n[i] = refs2copies_fast(spec, x, seen_ids, visited, depth + 1)
            if isinstance(n[i], Node):
                n[i].parent_node = n
                n[i].spec = spec
        n.parent_node = None
    return n
