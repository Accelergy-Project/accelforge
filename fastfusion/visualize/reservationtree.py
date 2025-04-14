from collections import defaultdict
import pydot
from typing import Any, Iterable
from pytimeloop.fastfusion.sim import Tiling, TensorStorage, Loop
from pytimeloop.fastfusion.util import expfmt
from pytimeloop.fastfusion.pareto import IN_PROGRESS_STATS, col2nameloop

PYDOT_NODE_DEFAULTS = {
    "shape": "box",
    "fontsize": "10",
    # "margin": "0.0",
    "margin": "0.1,0.0",
}


def get_einsum_key(name):
    return f"Einsum {name}"


class Node:
    def __init__(self, this_level: Iterable[Any] = (), children: Iterable["Node"] = ()):
        self.this_level = list(this_level)
        self.children = list(children)

    def access_level(self, index: int) -> "Node":
        if index == -1:
            return self
        if index == None:
            return self.children[-1].access_level(None) if self.children else self
        return self.children[-1].access_level(index - 1)

    def _to_yaml(self):
        this_level = [t.to_yaml() for t in self.this_level]
        if self.children:
            this_level.append(
                {
                    "type": "sequential",
                    "children": [c._to_yaml() for c in self.children],
                }
            )
        return this_level

    def to_yaml(self):
        return {"mapping": "fused", "nodes": self._to_yaml()}

    def to_pydot(self, graph, parent=None, invisible_root: bool = False):
        label_lines = []
        for t in self.this_level:
            label_lines.append(t.pydot_str() if hasattr(t, "pydot_str") else str(t))
        node_label = "\n".join(sorted(label_lines))
        if invisible_root:
            node = None
        else:
            node = pydot.Node(id(self), label=node_label, **PYDOT_NODE_DEFAULTS)
            graph.add_node(node)
        if parent:
            reservations = "\n".join(
                sorted(f"[{k}] {expfmt(v)}" for k, v in self.get_reservations().items())
            )
            graph.add_edge(pydot.Edge(parent, node, label=reservations))
        for child in self.children:
            child.to_pydot(graph, node, invisible_root=False)

    def add_stats(self, stats: dict[str, Any]):
        if self.children:
            self.children[-1].add_stats(stats)
        else:
            for k, v in stats.items():
                self.this_level.append(f"{k}: {expfmt(v)}")

    def get_reservations(self) -> dict[str, int]:
        reservations = defaultdict(lambda: 0)
        for c in self.children:
            for k, v in c.get_reservations().items():
                reservations[k] = max(reservations[k], v)
        for t in self.this_level:
            if isinstance(t, TensorStorage):
                reservations[str(t.memory_name)] += t.tile_size
        return dict(reservations)

    def get_all_storage(
        self, _entry: bool = True, start_at: int = 0
    ) -> list[TensorStorage]:
        if start_at <= 0:
            storage = set(t for t in self.this_level if isinstance(t, TensorStorage))
        else:
            storage = set()
        for c in self.children:
            storage.update(c.get_all_storage(_entry=False, start_at=start_at - 1))
        return sorted(storage) if _entry else storage

    def get_backing_stores(self):
        return TensorStorage.get_backing_stores(self.get_all_storage())

    def merge(self, other: "Node") -> "Node":
        new_this_level = self.this_level + other.this_level
        loops = [l for l in new_this_level if isinstance(l, Loop)]
        new_this_level = [t for t in new_this_level if not isinstance(t, Loop)]
        assert (
            len(loops) <= 2
        ), "Can only merge two nodes with two loops. Loops: " + str(loops)
        assert len(loops) > 0, "Can only merge two nodes with loops. Loops: " + str(
            loops
        )
        while len(loops) > 1:
            loops[0] = loops[0].merge_next(loops.pop(1))
        new_this_level.append(loops[0])
        return Node(new_this_level, self.children + other.children)

    def get_shared_tensors(
        self, other: "Node", start_at: int = 0
    ) -> set[TensorStorage]:
        return set(self.get_all_storage(start_at=start_at)) & set(
            other.get_all_storage(start_at=start_at)
        )

    def find_node_with(self, key: Any) -> "Node":
        if key in self.this_level:
            return self
        for c in self.children:
            found = c.find_node_with(key)
            if found:
                return found
        return None

    def validate_loops(self, einsum2ranks):
        """
        Validate the loops in the tree. This is a placeholder function and should be
        implemented with actual validation logic.
        """
        for c in self.children:
            c.validate_loops(einsum2ranks)
        for l in self.this_level:
            if not isinstance(l, Loop):
                continue
            for einsum, ranks in einsum2ranks.items():
                n_ranks_in_einsum = sum(1 for r in l.rank_names if r in ranks)
                if n_ranks_in_einsum > 1:
                    return False
        return True


def tilings2reservationtree(
    mappings: dict[str, Tiling],
    stats: dict[str, Any] = None,
    skip_backing_tensors_in_right_branch: Iterable[str] = (),
    still_live_tensors: set[str] = (),
    skip_merge: bool = False,
    per_component_energy: dict[dict[str, float]] = None,
    add_reservations: dict[str, Any] = None,
) -> Node:
    prev_tilings = []
    root = Node()
    einsum_ids = list(mappings.keys())
    if stats is not None:
        assert set(einsum_ids) == set(stats.keys())

    # If a tensor appears in non-back-to-back Einsums, then we need to store it for
    # all Einsums in between
    tensors_lifetimes = {e: [] for e in einsum_ids}
    all_tensors = set().union(*[set(t.storage) for t in mappings.values()])
    backers = TensorStorage.get_backing_stores(all_tensors)
    for t in backers:
        first_appearance = min(
            i for i, ts in enumerate(mappings.values()) if t in ts.storage
        )
        last_appearance = max(
            i for i, ts in enumerate(mappings.values()) if t in ts.storage
        )
        if t.tensor_name in still_live_tensors:
            last_appearance = len(einsum_ids) - 1
        for i, l in enumerate(tensors_lifetimes.values()):
            if first_appearance <= i <= last_appearance and t not in l:
                l.append(t)

    # Add each Einsum to the tree
    for i, einsum_id in enumerate(einsum_ids):
        tiling = mappings[einsum_id]
        index = -1
        n = root.access_level(index)
        loops = tiling.loops[index:] if index != -1 else tiling.loops
        for l in loops:
            n.children.append(Node())
            n = n.children[-1]
            n.this_level.append(l)

        # Add the tensors
        n.children.append(Node())  # Leaf node
        n.children[-1].this_level.append(get_einsum_key(einsum_id))
        if per_component_energy is not None:
            for k, v in per_component_energy[einsum_id].items():
                n.children[-1].this_level.append(f"{k} energy: {expfmt(v)}")
        id2tensor = defaultdict(set)
        for t in sorted(tiling.storage) + tensors_lifetimes[einsum_id]:
            id2tensor[t.tensor_name].add(t)
        id2tensor = {k: sorted(v) for k, v in id2tensor.items()}
        for tensor_name, storage in id2tensor.items():
            for tensor in storage:
                n = root.access_level(tensor.above_loop_index)
                # TODO if tensor not in n.this_level or tensor not in backers:
                if tensor not in n.this_level or tensor not in backers:
                    n.this_level.append(tensor)
        if stats is not None:
            root.add_stats(stats[einsum_id])
        prev_tilings.append(tiling)

    # Start at the root. Iterate through each leaf. Recursively:
    # - If two leaves have the same tensor and this tensor is a backing tensor,
    #   merge the two leaves. Remove the duplicate tensor from the right leaf.
    def merge_nodes(n: Node, level: int = 0):
        i = 0
        children = n.children
        while i < len(children) - 1:
            for j in range(len(children) - 1, i, -1):
                shared_tensors = children[i].get_shared_tensors(children[j], start_at=1)
                shared_tensors |= set(
                    c
                    for c in children[i].get_all_storage(start_at=1)
                    if c.tensor_name in still_live_tensors
                )
                if shared_tensors & set(backers):
                    while j != i:
                        # print(f'Level {level} merging {shared_tensors} between {i} and {j}')
                        children[i] = children[i].merge(children.pop(i + 1))
                        j -= 1
                    break
            else:
                i += 1

            this_level = children[i].this_level
            for t0 in range(len(this_level)):
                for t1 in range(len(this_level) - 1, t0, -1):
                    if this_level[t0] == this_level[t1] and this_level[t0] in backers:
                        this_level.pop(t1)

        for c in children:
            merge_nodes(c, level + 1)

        n.children = children

    if not skip_merge:
        merge_nodes(root)

    n = root
    skip_backing_tensors_in_right_branch = set(skip_backing_tensors_in_right_branch)
    while n is not None:
        i = 0
        while i < len(n.this_level):
            t = n.this_level[i]
            if t in backers and t.tensor_name in skip_backing_tensors_in_right_branch:
                n.this_level.pop(i)
            else:
                i += 1
        n = n.children[-1] if n.children else None

    if add_reservations is not None:
        for einsum, reservations in add_reservations.items():
            for resource, size in dict(reservations).items():
                nameloop = col2nameloop(resource)
                if nameloop is None:
                    continue
                name, _ = nameloop
                node = root.find_node_with(get_einsum_key(einsum))
                node.this_level.append(TensorStorage("Reservation", -1, name, size))

    return root


def tilings2svg(
    mappings: dict[str, Tiling],
    stats: dict[str, Any],
    per_component_energy: dict[dict[str, float]] = None,
):
    root = tilings2reservationtree(
        mappings, stats, per_component_energy=per_component_energy
    )
    graph = pydot.Dot(graph_type="digraph", ranksep="0.2", nodesep="0.2")
    root.to_pydot(graph)
    return graph.create_svg()


def tilings2yaml(mappings: dict[str, Tiling], stats: dict[str, Any]):
    root = tilings2reservationtree(mappings, stats)
    return root.to_yaml()
