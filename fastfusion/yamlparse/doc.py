"""Provides information on all node subtypes and their attributes."""

from .nodes import (
    Node,
    _subclassed,
    is_subclass,
    ListNode,
    default_unspecified_,
)
from typing import List, Union


def get_property_table(
    node: Union[Node, type] = None, col_len: int = 25, trim_cols: bool = False
) -> str:
    """
    Returns a table of all Node subclasses and their attributes.

    Args:
        node (Union[Node, type], optional): The Node subclass to generate the table for. If None, generates the table for all Node subclasses. Defaults to None.
        col_len (int, optional): The length of each column in the table. Defaults to 25.
        trim_cols (bool, optional): Whether to trim the columns to fit the specified length. Defaults to False.

    Returns:
        str: The generated table as a string.
    """
    result = []
    checker2str = [
        "key",
        "required_type",
        "default",
        "callfunc",
        "set_from",
    ]

    def formatter(x, capitalize: bool = False):
        if isinstance(x, (list, tuple)) and len(x) > 0:
            x = "/".join([str(getattr(y, "__name__", y)) for y in x])
        x = str(getattr(x, "__name__", x))
        if trim_cols:
            x = x[: col_len - 2]
        if capitalize:
            x = x.upper()
        if x == "":
            x = '""'
        return x.ljust(col_len)

    formatlist = lambda x, *a: ",".join([formatter(c, *a) for c in x])
    subclasses = [c for c in _subclassed] if node is None else [node]
    subclasses.sort(key=lambda x: x.__module__ + x.__name__)
    prev_module = None
    for subclass in subclasses:
        if subclass.__module__ != prev_module:
            result.append(
                "\n\n"
                + f"  {subclass.__module__}  ".center(col_len * len(checker2str), "=")
            )
            prev_module = subclass.__module__
        result.append(f"\n==== {subclass.__name__} ====")
        result.append("  " + formatlist(checker2str, True))
        checkers = subclass._get_type_specifiers(Node.get_global_spec())
        for k, c in checkers.items():
            r = [k] + [getattr(c, x, None) for x in checker2str[1:]]
            result.append("  " + formatlist(r))
    return "\n".join(result)


from typing import Union, List


def get_property_tree(
    node: Union[Node, type] = None, skip: Union[List[str], None] = None, n_levels=-1
) -> str:
    """
    Returns all node subtypes and their attributes in a tree format.

    Args:
        node (Union[Node, type], optional): The node or node type to generate the property tree for.
            If not provided, the Specification node will be used. Defaults to None.
        skip (Union[List[str], None], optional): A list of attribute names to skip in the property tree.
            Defaults to None.
        n_levels (int, optional): The number of levels to include in the property tree.
            A negative value indicates all levels. Defaults to -1.

    Returns:
        str: The property tree as a string.
    """
    if node is None:
        from ..v4 import Specification

        node = Specification
    start = "[KEY_OR_TAG]: [EXPECTED_TYPE] [REQUIRED or = DEFAULT_VALUE]\n"
    start += "├─ SUBNODES (If applicable)\n"
    start += "\n"
    start += node.__name__ + "\n"
    return start + _get_property_tree(node, Node.get_global_spec(), skip, n_levels)


def _get_property_tree(
    node: Union[Node, type],
    top: Union[Node, type],
    skip: Union[List[str], None] = None,
    n_levels=-1,
) -> str:
    """
    Returns all node subtypes and their attributes in a tree format.

    Args:
        node (Union[Node, type]): The node or type to generate the property tree for.
        top (Union[Node, type]): The top-level node or type.
        skip (Union[List[str], None], optional): List of attributes to skip. Defaults to None.
        n_levels (int, optional): Number of levels to include in the tree. Defaults to -1 (all levels).

    Returns:
        str: The property tree in a string format.
    """
    if n_levels == 0:
        return ""
    result = []
    skip = [] if skip is None else skip
    specifiers = []

    def replace_vpipes_with_str(s: str, replace_with: str, gap: int = 10) -> str:
        lines = s.split("\n")
        i = gap
        start_idx = gap
        while i < len(lines):
            l = lines[i]
            if l[0] == "\u2502":
                pipelen = i - start_idx
                if pipelen == len(replace_with):
                    for j in range(start_idx, i):
                        lines[j] = replace_with[j - start_idx] + lines[j][1:]
                    start_idx = i + gap
                    i += gap
            else:
                start_idx = i + 1
            i += 1
        return "\n".join(lines)

    for k, v in list(node._get_type_specifiers(top).items()):
        if is_subclass(node, ListNode):
            default = ""
        elif v.part_name_match and v.no_change_key:
            k = f"*{k}*"
            default = "Optional"
        elif v.default is default_unspecified_:
            default = "REQUIRED"
        else:
            default = f"= '{'None' if v.default is None else v.default}'"

        rt = v.required_type
        if not isinstance(rt, (list, tuple)):
            rt = [rt]
        if all([not is_subclass(x, Node) for x in rt]):
            rt = ("/".join([str(getattr(y, "__name__", y)) for y in rt]),)

        rt = list(rt)
        specs_strs = [r.__name__ if isinstance(r, type) else r for r in rt]

        for s, r in zip(specs_strs, rt):
            specifiers.append((k, r, default, s))

    for i, (k, v, default, s) in enumerate(specifiers):
        pipechar = f"\u251C" if i < len(specifiers) - 1 else "\u2514"
        vpipe = f"\u2502" if i < len(specifiers) - 1 else " "
        if v in skip:
            result.append(f"{pipechar}\u2500 '{k}': {s} {default} ...")
        else:
            result.append(f"{pipechar}\u2500 '{k}': {s} {default}")
            if is_subclass(v, Node):
                subtree = _get_property_tree(v, top, skip + [v], n_levels - 1)
                if subtree:
                    subtree = replace_vpipes_with_str(subtree, k)
                    result.append(f"{vpipe}  " + subtree.replace("\n", f"\n{vpipe}  "))
    return "\n".join(result)


def get_property_yaml(
    node: Node, top: "BaseSpecification" = None, skip: Union[List[str], None] = None
):
    """
    Returns all node subtypes and their attributes in a YAML format.

    Parameters:
        node (Node): The node for which to generate the YAML.
        top (BaseSpecification, optional): The top-level specification. Defaults to None.
        skip (Union[List[str], None], optional): List of attributes to skip. Defaults to None.

    Returns:
        str: The YAML representation of the node subtypes and their attributes.
    """
    if node is None:
        from ..v4 import Specification

        node = Specification

    result = _get_property_yaml(node, Node.get_global_spec(), skip)
    result = result.split("\n")
    sep = "<DOC_FUNCTION_SEPARATOR>"
    max_idx = max(s.index(sep) if sep in s else 0 for s in result)
    for i, r in enumerate(result):
        result[i] = r.replace(
            sep, " " * (1 + max_idx - r.index(sep) if sep in r else 0)
        )
    return "\n".join((s.rstrip()[2:] for s in result))


def _get_property_yaml(
    node: Node, top: "BaseSpecification" = None, skip: Union[List[str], None] = None
):
    """Returns a yaml representation of the specification."""
    top = top or node
    result = []
    skip = ["ignore"] if skip is None else skip
    specifiers = []
    for k, v in list(node._get_type_specifiers(top).items()):
        if is_subclass(node, ListNode):
            default = ""
        elif v.part_name_match and v.no_change_key:
            if not k:
                k = "<Any>"
            else:
                k = f"*{k}*"
            default = "Optional"
        elif v.default is default_unspecified_:
            default = "REQUIRED"
        else:
            default = f"'{'None' if v.default is None else v.default}'"

        rt = v.required_type
        if not isinstance(rt, (list, tuple)):
            rt = [rt]
        if all([not is_subclass(x, Node) for x in rt]):
            rt = ("/".join([str(getattr(y, "__name__", y)) for y in rt]),)

        rt = list(rt)
        specs_strs = [r.__name__ if isinstance(r, type) else r for r in rt]

        for s, r in zip(specs_strs, rt):
            specifiers.append([k, r, default, s])

    def keyformat(k, subclass_node):
        if is_subclass(node, ListNode):
            if k:
                return f"- {k} "
            return "-"
        else:
            if subclass_node:
                return f"{k or '<Any>'}:<DOC_FUNCTION_SEPARATOR>"
            return f"{k or '<Any>'}:<DOC_FUNCTION_SEPARATOR>"

    for i, (k, v, default, s) in enumerate(specifiers):
        if "ignore" in k:
            continue
        subclass_node = is_subclass(v, Node)

        k = keyformat(k, subclass_node)
        r = f"< Type = {s}"
        if not is_subclass(node, ListNode):
            if default == "REQUIRED":
                r += ", REQUIRED"
            else:
                r += f",  Default = {default}"

        s = "" if is_subclass(node, ListNode) else "  "

        subtree = ""
        if v in skip:
            subtree = f"  {k.rstrip()} < Type {s} already shown above > "
        else:
            if is_subclass(v, Node):
                subtree = _get_property_yaml(v, top, skip + [v])
                if subtree:
                    subtree = s + subtree.replace("\n", f"\n{s}")
                    st_split = subtree.split("\n")
                    if len(st_split) > 5 and st_split[-1].strip():
                        subtree = subtree + "\n"

        result.append(f"{s}{k} {'# ' if subtree else ''}{r} >")
        if subtree:
            result.append(subtree)

    sep = "<DOC_FUNCTION_SEPARATOR>"
    max_idx = max([0] + [s.index(sep) if sep in s else 0 for s in result])
    for i, r in enumerate(result):
        result[i] = r.replace(
            sep, " " * (1 + max_idx - r.index(sep) if sep in r else 0)
        )

    return "\n".join(result)
