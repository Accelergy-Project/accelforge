class FlattenedArch:
    """
    A flattened arch is an architecture spec that has been
    flattened into a hierarchy for the purpose of mapping
    a particular Einsum.

    Several steps (may not be exhaustive) are applied when
    an arch is flattened:
    - A compute unit is selected, and a flattened arch
      is a path from the root to that compute unit
    - Expressions have been evaluated in the context of an
      Einsum.
    - Non-hierarchical arch nodes (e.g., ones inside the
      `nodes` key of an `Array` has been inserted into
      a hierarchy.)

    This class should only be relevant to the model and
    mapper. That is, the user should generally not define
    a flattened arch directly. So, unlike other classes in
    `frontend`, this one is intentionally *not* an
    `EvalableModel`.
    """
    def __init__(self, nodes: list["Leaf"]):
        self.nodes = nodes

    def __getitem__(self, idx: int | str | slice):
        if isinstance(idx, (int, slice)):
            return self.nodes[idx]
        elif isinstance(idx, str):
            for node in self.nodes:
                if node.name == idx:
                    return node
            raise KeyError(f"arch node with name {idx} not found")
        raise ValueError(f"idx should be int or str, but instead {type(idx)}")

    def __iter__(self):
        for node in self.nodes:
            yield node

    def index(self, name: str):
        for i, node in enumerate(self.nodes):
            if node.name == name:
                return i
        raise ValueError(f"no node found with name {name}")

    def is_above(self, name_a: str, name_b: str):
        """
        Returns True if node with name_a is above node with name_b.
        Raises ValueError if either is not found.
        """
        idx_a = self.index(name_a)
        idx_b = self.index(name_b)
        return idx_a < idx_b
