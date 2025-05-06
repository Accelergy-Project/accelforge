"""Node classes for parsing and processing specification trees."""

from abc import ABC
import copy
import glob
import inspect
import logging
import os
import threading
from pathlib import Path
from typing import (
    Callable,
    Any,
    Dict,
    Optional,
    Set,
    TextIO,
    TypeVar,
    Union,
    List,
    Tuple,
    Iterable,
    Type,
)
from . import yaml
from .parse_expressions import parse_expression, is_quoted_string


class ParseError(Exception):
    """Exception for nodes."""


class Unspecified:
    """Class to represent an unspecified value."""

    def __str__(self):
        return "REQUIRED"

    def __repr__(self):
        return "REQUIRED"


default_unspecified_ = Unspecified()


T = TypeVar("T", bound="Node")


def is_subclass(x: Any, of: Any) -> bool:
    return inspect.isclass(x) and issubclass(x, of)


class TypeSpecifier:
    """
    Represents a type specifier for a node in the TimeloopFE library.

    Attributes:
        name (str): The name of the type specifier.
        required_type (Type): The required type for the node.
        default (Any): The default value for the type specifier.
        callfunc (Union[Callable, None]): The function to call for casting the value.
        should_have_been_removed_by (Type): The type that should have removed or transformed the node.
        part_name_match (bool): Flag indicating if the name should be partially matched.
        no_change_key (bool): Flag indicating if the key should not be changed.

    Methods:
        removed_by_str(self): Get the string representation of the type that should have removed or transformed the node.
        cast_check_type(self, value: Any, node: "Node", key: str) -> Any: Check and cast the value to the required type.
        cast(self, value: Any, __node_skip_parse: bool = False) -> Any: Cast the value to the required type.
        check_type(self, value: Any, node: "Node", key: str): Check if the value matches the required type.
    """

    def __init__(
        self,
        name: str,
        required_type: Type,
        default: Any = default_unspecified_,
        callfunc: Union[Callable, None] = None,
        should_have_been_removed_by: Type = None,
        part_name_match: bool = False,
        no_change_key: bool = False,
    ):
        self.name: str = name
        self.required_type: Type = required_type
        self.default: Any = default
        self.callfunc: Callable = callfunc
        # Check if required type is a class and inherit from Node
        if self.callfunc is None and is_subclass(required_type, Node):
            self.callfunc = required_type
            rt = required_type

            def callfunc(x, __node_skip_parse=False):
                if isinstance(x, rt) or x is None:
                    return x
                return rt(x, __node_skip_parse=__node_skip_parse)

            callfunc.__name__ = required_type.__name__
            self.callfunc = callfunc
            # self.callfunc = lambda x: x if isinstance(x, rt) else rt(x)
        self.should_have_been_removed_by: Type = should_have_been_removed_by
        self.set_from: Any = self.should_have_been_removed_by
        self.part_name_match: bool = part_name_match
        self.no_change_key: bool = no_change_key

    def removed_by_str(self):
        if self.should_have_been_removed_by is not None:
            rmclass = getattr(
                self.should_have_been_removed_by,
                "__class__",
                self.should_have_been_removed_by,
            )
            rmname = getattr(
                rmclass,
                "__name__",
                rmclass,
            )
            return (
                f"This should have been removed or transformed by "
                f"{rmname}, but was not. "
                f"Did something go wrong in the parsing?"
            )
        return ""

    def cast_check_type(self, value: Any, node: "Node", key: str) -> Any:
        if value is default_unspecified_:
            assert self.default is not default_unspecified_, (
                "Can not call cast_check_type() with default_unspecified_"
                "if default value is also default_unspecified_."
            )
            value = copy.deepcopy(self.default)

        try:
            casted = self.cast(value)
        except Exception as exc:
            callname = (
                self.callfunc.__name__
                if isinstance(self.callfunc, type)
                else (
                    self.callfunc.__name__
                    if isinstance(self.callfunc, Callable)
                    else str(self.callfunc)
                )
            )
            casted_successfully = False
            try:  # If we can cast without parsing lower-level nodes, then
                # the exception came from below and we should re-raise it.
                value = self.cast(value, True)
                casted_successfully = True
            except Exception:
                pass  # This level is the problem. Re-raise ParseError
                # with more info.

            last_non_node_exception = getattr(exc, "_last_non_node_exception", None)

            if not isinstance(exc, ParseError):
                last_non_node_exception = exc
            if casted_successfully:
                raise exc

            estr = ""
            if last_non_node_exception is not None:
                estr = f"\n\n{last_non_node_exception}"

            new_exc = ParseError(
                f'Error calling cast function "{callname}" '
                f'for value "{value}" in {node.get_name()}[{key}]. '
                f"{self.removed_by_str()}{estr}. {exc.__class__.__name__}: {exc}"
            )
            new_exc._last_non_node_exception = last_non_node_exception
            raise new_exc from exc

        # self.check_type(casted, node, key)
        return casted

    def cast(self, value: Any, __node_skip_parse: bool = False) -> Any:
        tag = Node.get_tag(value)
        if self.callfunc is not None:
            if __node_skip_parse:
                value = self.callfunc(value, __node_skip_parse=__node_skip_parse)
            else:
                value = self.callfunc(value)
        try:
            value.tag = tag
        except AttributeError:
            pass
        return value

    def check_type(self, value: Any, node: "Node", key: str):
        if value == self.default or self.required_type is None:
            return
        t = (
            self.required_type
            if isinstance(self.required_type, tuple)
            else (self.required_type,)
        )

        for s in t:
            if isinstance(s, str) and str(value) == s:
                return True
            if isinstance(s, type) and isinstance(value, s):
                return True
            if s is None and value is None:
                return True
        raise TypeError(
            f"Expected one of {self.required_type}, got {type(value)} "
            f'with value "{value}" in {node.get_name()}[{key}]. '
            f"{self.removed_by_str()}"
        )


def isempty(x: Iterable) -> bool:
    if x is None:
        return True
    if isinstance(x, (Node)):
        return x.isempty()
    try:
        return len(x) == 0
    except TypeError:
        return False


_local = threading.local()


class GrabParentAddMe:
    def __init__(self, add_elem: "Node"):
        setattr(_local, "parent_stack", [])
        self.stack: List[Node] = _local.parent_stack
        self.add_elem = add_elem

    def __enter__(self):
        rval = self.stack[-1] if self.stack else None
        self.stack.append(self.add_elem)
        return rval

    def __exit__(self, *args):
        self.stack.pop()


_subclassed: Set[Type] = set()
_thread_local = threading.local()
lock = threading.Lock()
_thread_local.top_spec = None


class Node(ABC):
    """
    Base class for all nodes in the hierarchy.

    Attributes:
        parent_node (Node): The parent node of the current node.
        spec (Specification): The global specification object.
        _init_args (Tuple): The arguments and keyword arguments used to initialize the node.
        __currently_parsing_index (Union[int, str]): The index or key currently being parsed.
        logger (Logger): The logger object for the node's class.
        _default_parse (bool): Flag indicating whether the node should be parsed using default rules.

    Methods:
        get_specifiers_from_processors(cls, spec): Get the specifiers from processors.
        reset_specifiers_from_processors(cls, processor): Reset the specifiers from processors.
        declare_attrs(cls, *args, **kwargs): Initialize the attributes of this node.
        reset_processor_elems(cls, processor): Reset the processor elements.
        recognize_all(cls, recognize_all): Set whether all attributes under this node should be recognized.
        _get_type_specifiers(cls, spec): Get the type specifiers for this node.
        _get_all_recognized(self): Check if all attributes under this node are recognized.
        _get_tag(x): Get the tag of a node.
        get_global_spec(): Get the global specification object.
        set_global_spec(spec): Set the global specification object.
        get_tag(self): Get the tag of this node.
        _get_index2checker(self, key2elem): Get the index-to-checker mapping.
        items(self): Get an iterable of (key, value) or (index, value) pairs.
        combine_index(self, key, value): Combine the value at the given key with the given value.
        _parse_elem(self, key, check, value_override): Parse an element of the node.
    """

    _needs_declare_attrs = set()

    def __init__(self, *args, **kwargs):
        self.parent_node: Node = None
        self.spec: "Specification" = Node.get_global_spec()
        # Keep in memory such that the ID is not reused.
        self._init_args: Tuple = (args, kwargs)
        self.__currently_parsing_index: Union[int, str] = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self._default_parse = False
        self.from_data = None

    @classmethod
    def get_specifiers_from_processors(cls, spec: "BaseSpecification"):
        """Get the specifiers that have been set from processors."""
        if spec is None or not hasattr(spec, "_processor_attributes"):
            result = {}
        else:
            result = spec._processor_attributes.setdefault(cls.unique_class_name(), {})
        return result

    @classmethod
    def reset_specifiers_from_processors(cls, processor: Optional[Type] = None):
        """Reset the specifiers that have been set from processors."""
        spec = Node.get_global_spec()
        if spec is None or not hasattr(spec, "_processor_attributes"):
            return {}

        d = spec._processor_attributes.setdefault(cls.unique_class_name(), {})
        for k, v in list(d.items()):
            if (
                processor is None
                or type(v.set_from) == processor
                or v.set_from is processor
            ):
                del d[k]

    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        """Initialize the attributes of this node."""
        setattr(cls, "_param_type_specifiers", {})
        # cls.reset_specifiers_from_processors()
        setattr(cls, "Node_all_recognized", False)
        cls.add_attr(
            "ignore",
            required_type=None,
            default=None,
            part_name_match=True,
            no_change_key=True,
        )
        _subclassed.add(cls)

    @classmethod
    def reset_processor_elems(cls, processor: Optional[Type] = None):
        for s in list(_subclassed):
            s.reset_specifiers_from_processors(processor=processor)

    @classmethod
    def recognize_all(cls, recognize_all: bool = True):
        """
        Set whether all attributes under this node should be recognized.

        Attributes:
            recognize_all (bool): Flag indicating whether all attributes under this node should be recognized.
        """
        if cls is Node or cls is DictNode or cls is ListNode:
            raise TypeError(
                f'Called recognize_all() on class "{cls}".'
                f"Call this method on a subclass of Node or DictNode."
            )
        setattr(cls, "Node_all_recognized", recognize_all)

    @classmethod
    def _get_type_specifiers(
        cls, spec: "BaseSpecification"
    ) -> Dict[str, TypeSpecifier]:
        """
        Get the type specifiers for this node.

        Attributes:
            spec (Specification): The global specification object.

        Returns:
            Dict[str, TypeSpecifier]: The type specifiers for this node.
        """
        classname = cls.__name__
        if not hasattr(cls, "_param_type_specifiers"):
            raise AttributeError(
                f"Class {classname} inherits from DictNode but does not have "
                f"a _param_type_specifiers attribute. Was __init__() "
                f"called before {classname}.declare_attrs()?"
            )
        rval = {}
        for c in cls.mro()[::-1]:  # Iterate through superclasses
            rval.update(getattr(c, "_param_type_specifiers", {}))
            if hasattr(c, "get_specifiers_from_processors"):
                rval.update(c.get_specifiers_from_processors(spec))
        return rval

    def _get_all_recognized(self):
        def recognized(x):
            return getattr(x, "Node_all_recognized", 0)

        return any([recognized(c) for c in self.__class__.mro()])

    @staticmethod
    def _get_tag(x) -> str:
        tag = getattr(x, "tag", None)
        if isinstance(tag, str):
            return tag
        if tag is None:
            return ""
        if not hasattr(tag, "value"):
            raise AttributeError(
                f"Tag '{tag}' in {x}.tag is not a string "
                f"and has no 'value' attribute."
            )
        if tag.value is None:
            return ""
        return tag.value

    @staticmethod
    def get_global_spec() -> "BaseSpecification":
        """Get the global specification object."""
        return _thread_local.top_spec

    @staticmethod
    def set_global_spec(spec: "BaseSpecification"):
        """Set the global specification object."""
        _thread_local.top_spec = spec

    def get_tag(self) -> str:
        """Get the tag of this node."""
        return Node._get_tag(self)

    def _get_index2checker(
        self, key2elem: Optional[List[Tuple[str, Any]]] = None
    ) -> Dict[Union[str, int], TypeSpecifier]:
        specifiers = self._get_type_specifiers(self.spec)
        if key2elem is not None:
            index2checker = {}
            for i, (k, v) in enumerate(key2elem):
                index2checker[i] = specifiers.get(k, None)
                if index2checker[i] is None:
                    for s in specifiers.values():
                        if s.part_name_match and s.name in k:
                            index2checker[i] = s
                            break
            return index2checker
        if isinstance(self, DictNode):
            index2checker = {k: specifiers.get(k, None) for k in self.keys()}
            for k, v in index2checker.items():
                if v is None:
                    for s in specifiers.values():
                        if s.part_name_match and s.name in k:
                            index2checker[k] = s
                            break
            return index2checker
        if isinstance(self, ListNode):
            checks = {}
            for k, v in enumerate(self):
                check = specifiers.get(Node._get_tag(v), None)
                if check is None:
                    check = specifiers.get("!" + v.__class__.__name__, None)
                checks[k] = check
            return checks
        raise TypeError(
            f"Called _get_index2checker on {self.__class__}"
            f"which is not a DictNode or ListNode."
        )

    def items(self) -> Iterable[Tuple[Union[str, int], Any]]:
        """Get iterable of (key, value) or (index, value) pairs."""
        if isinstance(self, dict):
            return super().items()  # type: ignore
        return enumerate(self)  # type: ignore

    def combine_index(self, key: Union[str, int], value: T) -> T:
        """Combine the value at the given key with the given value.

        If there is no value at the given key, sets the value at the given key.
        If there is, attempts to combine the two values.

        Args:
            key: The key to combine.
            value: The value to combine.

        Returns:
            The combined value.
        """
        # Done to get rid of type warnings
        s: DictNode = self  # type: ignore
        if key in s:
            s[key] = Node.try_combine(s[key], value, self, str(key))
        else:
            s[key] = value
        if isinstance(s[key], Node):
            s[key].parent_node = self
        return s[key]

    def _parse_elem(
        self,
        key: Union[str, int],
        check: TypeSpecifier,
        value_override: Any = None,
    ):
        if value_override is not None:
            v = value_override
        elif isinstance(self, DictNode) and check is not None:
            v = self.pop(key)  # Remove so we can add back with checker name
        else:
            v = self[key]

        if v is default_unspecified_:
            if check.default is default_unspecified_:
                rq, found = [key], []
                for k, v in self._get_index2checker().items():
                    if (
                        (self[k] is default_unspecified_)
                        and (v is not None)
                        and v.default is default_unspecified_
                    ):
                        rq.append(k)
                    elif self[k] is not default_unspecified_:
                        found.append(k)
                rq = "Required keys not found: " + ", ".join(rq)
                found = "Found keys: " + ", ".join(found)
                raise KeyError(
                    f'Missing required key "{key}" in {self}. {rq}. {found}.'
                )

        self.__currently_parsing_index = key
        for reserved in ["tag", "parent_node"]:
            if key == reserved:
                tagstr = (
                    (
                        f"If you'd like to set the tag of "
                        f"this node in Python, set it through the tag "
                        f"attribute. Example: mynode.tag = '!MyTag'. Error in "
                        f"{self.get_name()} {self}"
                    )
                    if reserved == "tag"
                    else ""
                )
                raise ParseError(
                    f"The key {reserved} is reserved for use by the YAML "
                    f"parser. Please use a different key. Found in "
                    f"{self.get_name()}. {tagstr}"
                )
        tag = Node._get_tag(v)
        if check is not None:
            v = check.cast_check_type(v, self, key)

        if isinstance(v, Node):
            v.tag = tag
        # Check for alt name
        if isinstance(self, DictNode) and check is not None:
            newkey = key if check.no_change_key else check.name
            try:
                self.combine_index(newkey, v)
            except Exception as exc:
                raise ValueError(
                    f'Could not combine values in indices "{key}" and '
                    f'"{newkey}" in {self.get_name()}. {exc}'  # type: ignore
                ) from exc
        else:
            self[key] = v
        self.__currently_parsing_index = None

    def _parse_elems(self):
        with GrabParentAddMe(self) as parent:
            self.parent_node = parent
            self.spec = parent.spec if parent is not None else Node.get_global_spec()
            for k, check in self._get_index2checker().items():
                self._parse_elem(k, check)
            # self._check_unrecognized(ignore_should_have_been_removed_by=1)

    def _parse_extra_elems(self, key2elem: List[Tuple[str, Any]]):
        with GrabParentAddMe(self) as parent:
            self.parent_node = parent
            self.spec = parent.spec if parent is not None else Node.get_global_spec()
            checkers = self._get_index2checker(key2elem)
            for (_, check), (k, v) in zip(checkers.items(), key2elem):
                try:
                    self._parse_elem(k, check, v)
                except Exception as exc:
                    raise ParseError(
                        f"Failed to combine duplicate attribute in key [{k}]. "
                        f"There are two ways to fix this: "
                        f"1. Remove the duplicate key '{k}'. "
                        f"2. Ensure that the duplicate values are "
                        f"identical or able to be combined. "
                    ) from exc
            # self._check_unrecognized(ignore_should_have_been_removed_by=1)

    def get_name(self, seen: Union[Set, None] = None) -> str:
        """Get the name of this node."""
        if seen is None:
            seen = set()
        if id(self) in seen:
            return f"{self.__class__.__name__}"
        seen.add(id(self))
        namekey = ""
        if isinstance(self, dict) and "name" in self:
            namekey = f'({self["name"]})'
        parentname = ""
        if self.parent_node is not None:
            parent = self.parent_node
            idx = ""
            if isinstance(parent, ListNode) and self in parent:
                idx = parent.index(self)
            elif isinstance(parent, DictNode) and self in parent.values():
                idx = next(k for k, v in parent.items() if v == self)
            elif parent.__currently_parsing_index is not None:
                idx = parent.__currently_parsing_index
            parentname = f"{self.parent_node.get_name(seen)}[{idx}]."
        return f"{parentname}{self.__class__.__name__}{namekey}"

    def check_unrecognized(
        self,
        ignore_empty: bool = False,
        ignore_should_have_been_removed_by=False,
    ):
        """Check for unrecognized keys in this node and all subnodes.
        Also checks for correct types.

        Args:
            ignore_empty (bool): Flag indicating whether to ignore empty nodes.
            ignore_should_have_been_removed_by (bool): Flag indicating whether to ignore nodes that should have been removed by a processor.

        Raises:
            ParseError: If an unrecognized key is found.
        """
        self.recursive_apply(
            lambda x: x._check_unrecognized(
                ignore_empty, ignore_should_have_been_removed_by
            ),
            self_first=True,
        )

    def recursive_apply(
        self, func: callable, self_first: bool = False, applied_to: set = None
    ) -> Any:
        """Apply a function to this node and all subnodes.

        Args:
            func (callable): The function to apply.
            self_first (bool): Flag indicating whether to apply the function to this node before applying it to subnodes.
            applied_to (set): A set of ids of nodes that have already been visited. Prevents infinite recursion.

        Returns:
            The return value of the function applied to this node.
        """
        if applied_to is None:
            applied_to = set()
        if id(self) in applied_to:
            return self
        applied_to.add(id(self))
        if self_first:
            rval = func(self)
        for _, v in self.items():
            if isinstance(v, Node):
                v.recursive_apply(func, self_first, applied_to)
        if self_first:
            return rval
        return func(self)

    def clean_empties(self):
        """Remove empty nodes from this node and all subnodes."""

        def clean(x):
            items = list((x.items()) if isinstance(x, dict) else enumerate(x))
            for k, v in items[::-1]:
                if v is None or isempty(v):
                    del x[k]

        self.recursive_apply(clean)
        clean(self)

    def isempty(self) -> bool:
        """Return True if this node is empty. Good to override."""
        if isinstance(self, (DictNode, ListNode)):
            return len(self) == 0
        return False

    def isempty_recursive(self) -> bool:
        """Return True if this node or all subnodes are empty."""

        empties = set()

        def emptycheck(x):
            allempty = True
            for _, v in x.items():
                if id(v) in empties or isempty(v):
                    empties.add(id(x))
                elif isinstance(self, DictNode) and v is None:
                    empties.add(id(x))
                else:
                    allempty = False
                    break
            if allempty:
                return True
            raise StopIteration

        try:
            self.recursive_apply(emptycheck, self_first=False)
        except StopIteration:
            return False
        return True
    
    def notempty(self) -> bool:
        """Return True if this node is not empty."""
        return not self.isempty()
    
    def notempty_recursive(self) -> bool:
        """Return True if this node or all subnodes are not empty."""
        return not self.isempty_recursive()

    @classmethod
    def add_attr(
        cls,
        key_or_tag: str,
        required_type: Optional[
            Union[type, Tuple[type, ...], Tuple[None, ...], Tuple[str, ...], None]
        ] = None,
        default: Any = default_unspecified_,
        callfunc: Optional[Callable] = None,
        part_name_match: Optional[bool] = None,
        no_change_key: Optional[bool] = None,
        _processor_responsible_for_removing: Any = None,
        _add_checker_to: Optional[Dict[str, TypeSpecifier]] = None,
    ):
        """Initialize a type specifier for this class.

        Args:
            key_or_tag: The key/tag or tag to use for this type specifier.
            required_type: The type of value that this type specifier will be
            default: The default value to use if the key/tag is not found.
            callfunc: A function to call on the value before returning it.
            part_name_match: If True, the key/tag will match if it is a substring of the actual key/tag.
            no_change_key: If True, a parsed key will not be changed when a partial name match is found. Otherwise, the parsed key will be changed to the actual key.
            _processor_responsible_for_removing: The processor that will be responsible for removing this key from the containing node, if any.
            _add_checker_to: The dictionary to add the checker to. If None, add the checker to the class's type specifiers.

        Raises:
            AttributeError: If the class does not have a _param_type_specifiers attribute.
        """
        if not hasattr(cls, "_param_type_specifiers"):
            raise AttributeError(
                f"Class {cls.__name__} must call super.declare_attrs() before "
                f"super.add_attr()."
            )

        checker = TypeSpecifier(
            name=key_or_tag,
            required_type=required_type,
            default=default,
            callfunc=callfunc,
            should_have_been_removed_by=_processor_responsible_for_removing,
            part_name_match=part_name_match,
            no_change_key=no_change_key,
        )

        add_checker_to = (
            _add_checker_to
            if _add_checker_to is not None
            else cls._param_type_specifiers
        )
        add_checker_to[key_or_tag] = checker

        def assert_key(self):
            if key_or_tag not in self:
                raise KeyError(f"Key '{key_or_tag}' not found in {self.get_name()}.")

        if is_subclass(cls, dict):

            def getter(self):
                assert_key(self)
                return self[key_or_tag]

            def setter(self, value):
                self[key_or_tag] = value

            def deleter(self):
                assert_key(self)
                del self[key_or_tag]

            prop = property(getter, setter, deleter)
            setattr(cls, key_or_tag, prop)

        return checker

    def _check_unrecognized(
        self, ignore_empty=False, ignore_should_have_been_removed_by=False
    ):
        if self._get_all_recognized():
            return
        if isinstance(self, ListNode) and not self._get_type_specifiers(self.spec):
            return

        classname = self.__class__.__name__
        if isinstance(self, DictNode):
            name = f"dict {classname} {self.get_name()}"
            keytag = "key"
        else:
            name = f"list {classname} {self.get_name()}"
            keytag = "tag"

        recognized = self._get_type_specifiers(self.spec)
        checks = self._get_index2checker()
        rkeys = list(recognized.keys())

        if len(rkeys) == 1 and rkeys[0] == "ignore":
            return

        for k, v in checks.items():
            if ignore_empty and (self[k] is None or isempty(self[k])):
                continue

            if v is None and isinstance(self, ListNode):
                v = recognized.get("!" + self[k].__class__.__name__, None)

            if v is None:
                has_tag = hasattr(self[k], "tag")
                idxstr = f" index {k}" if keytag == "tag" else ""
                t = Node._get_tag(self[k]) if keytag == "tag" else k
                tag_clarif = "(no .tag in Python object, no !TAG in YAML)"
                if has_tag or keytag != "tag":
                    tag_str = f"'{t}'"
                else:
                    tag_str = f"'{t}' {tag_clarif}"
                raise ParseError(
                    f"Unrecognized {keytag} {tag_str} in {name}{idxstr}.  "
                    f"Recognized {keytag}s: {list(recognized.keys())}."
                )
            v.check_type(self[k], self, k)
            if (
                v.should_have_been_removed_by is not None
                and not ignore_should_have_been_removed_by
            ):
                from .processor import ProcessorError

                key = Node._get_tag(self[k]) if keytag == "tag" else k
                s = f'Found {keytag} "{key}" in {name}[{k}].'
                raise ProcessorError(f"{s} {v.removed_by_str()}")

    def get_nodes_of_type(self, node_type: Type[T]) -> List[T]:
        """Return a list of all subnodes of a given type.

        Args:
            node_type: The type of node to search for.

        Returns:
            A list of all subnodes of the given type.
        """
        found = []
        found_ids = set()

        def search(x):
            for c in [v for _, v in x.items()]:
                if isinstance(c, node_type) and id(c) not in found_ids:
                    found_ids.add(id(c))
                    found.append(c)

        self.recursive_apply(search, self_first=True)
        self.logger.debug("Found %d nodes of type %s.", len(found), node_type)
        return found

    def get_setter_lambda(self, keytag: Union[str, int]) -> Callable:
        """Get a function that can be used to set a value in this node. The setter takes one argument, the value to set.

        Args:
            keytag: The key or tag to set.

        Returns:
            A function that can be used to set a value in this node.
        """

        def setval(x):
            if isinstance(x, Node) and isinstance(self, Node):
                x.parent_node = self
            self[keytag] = x  # type: ignore

        return setval

    def get_combiner_lambda(self, keytag: Union[str, int]) -> Callable:
        """Get a function that can be used to combine a value to this node. The combiner takes one argument, the value to combine.

        Args:
            keytag: The key or tag to combine.

        Returns:
            A function that can be used to combine a value to this node.
        """
        return lambda x: self.combine_index(keytag, x)

    def get_setters_for_keytag(
        self, keytag: str, recursive: bool = True
    ) -> List[Tuple[Any, Callable]]:
        """Get a list of tuples of the form (value, setter) for all keys/tags in this node that match the given key/tag.
        A setter is a function that can be used to set a value in this node.

        Args:
            keytag: The key or tag to search for.
            recursive: If True, search recursively.
        """
        rval = []

        def search(node: Node):
            for i, x in node.items():
                if i == keytag or Node._get_tag(x) == keytag:
                    rval.append((x, node.get_setter_lambda(i)))

        search(self)
        if recursive:
            self.recursive_apply(search)
        self.logger.debug("Found %d nodes for keytag %s.", len(rval), keytag)
        return rval

    def get_combiners_for_keytag(
        self, keytag: str, recursive: bool = True
    ) -> List[Tuple[Any, Callable]]:
        """Get a list of tuples of the form (value, combiner) for all keys/tags in this node that match the given key/tag.

        A combiner is a function that can be used to combine a value to this node.

        Args:
            keytag: The key or tag to search for.
            recursive: If True, search recursively.
        """
        rval = []

        def search(node: Node):
            for i, x in node.items():
                if i == keytag or Node._get_tag(x) == keytag:
                    rval.append((x, node.get_setter_lambda(i)))

        search(self)
        if recursive:
            self.recursive_apply(search)
        self.logger.debug("Found %d nodes for keytag %s.", len(rval), keytag)
        return rval

    def get_setters_for_type(
        self, t: Type, recursive: bool = True
    ) -> List[Tuple[Any, Callable]]:
        """Get a list of tuples of the form (value, setter) for all keys/tags in this node that match the given type.

        A setter is a function that can be used to set a value in this node.

        Args:
            t: The type to search for.
            recursive: If True, search recursively.

        Returns:
            A list of tuples of the form (value, setter) for all keys/tags in this node that match the given type.
        """
        rval = []

        def search(node: Node):
            for i, x in node.items():
                if isinstance(x, t):
                    rval.append((x, node.get_setter_lambda(i)))

        search(self)
        if recursive:
            self.recursive_apply(search)
        self.logger.debug("Found %d nodes for type %s.", len(rval), t)
        return rval

    def get_combiners_for_type(
        self, t: Type, recursive: bool = True
    ) -> List[Tuple[Any, Callable]]:
        """Get a list of tuples of the form (value, combiner) for all keys/tags in this node that match the given type.

        A combiner is a function that can be used to combine a value in this node.

        Args:
            t: The type to search for.
            recursive: If True, search recursively.

        Returns:
            A list of tuples of the form (value, combiner) for all keys/tags in this node that match the given type.
        """
        rval = []

        def search(node: Node):
            for i, x in node.items():
                if isinstance(x, t):
                    rval.append((x, node.get_setter_lambda(i)))

        search(self)
        if recursive:
            self.recursive_apply(search)
        self.logger.debug("Found %d nodes for type %s.", len(rval), t)
        return rval

    def __str__(self):
        """Return the name of this node."""
        as_str = ""
        if isinstance(self, dict):
            as_str = str(dict(self))
        if isinstance(self, list):
            as_str = str(list(self))
        as_str = as_str if len(as_str) < 50 else as_str[:50] + "..."

        return self.get_name() + "=" + as_str

    def __format__(self, format_spec):
        """Formats the name of this node."""
        return str(self).__format__(format_spec)

    @staticmethod
    def try_combine(
        a: Any,
        b: Any,
        innonde: Union["Node", None] = None,
        index: Union[int, str, None] = None,
    ) -> Any:
        """Try to combine two values.

        Args:
            a: The first value.
            b: The second value.
            innonde: The node that contains the values. For error messages.
            index: The index of the values in the node. For error messages.

        Returns:
            The combined value.
        """

        set_on_fail = isinstance(index, str) and "ignore" in index

        if a is None or isempty(a) or a is default_unspecified_:
            return b
        if b is None or isempty(b) or b is default_unspecified_:
            return a
        if a == b:
            return a

        contextstr = ""
        if innonde is not None:
            contextstr = "" if innonde is None else f" In {innonde.get_name()}"
            if index is not None:
                contextstr += f"[{index}]"
        if a.__class__ != b.__class__:
            if set_on_fail:
                return b
            raise TypeError(
                f"Can not combine different classes {a.__class__.__name__} "
                f"and {b.__class__.__name__}.{contextstr}"
            )
        if isinstance(f := getattr(a, "combine", None), Callable):
            return f(b)
        if set_on_fail:
            return b
        raise AttributeError(
            f"Can not combine {a} and {b}. {a.__class__.__name__} does not "
            f"have a combine() method.{contextstr}"
        )

    def is_defined_non_default_non_empty(self, key: str) -> bool:
        """Returns True if the given key is defined in this node and is not the default value and is not empty."""
        idx2checker = self._get_index2checker()
        if key not in idx2checker:
            return False
        try:
            v = self[key]  # type: ignore
        except (KeyError, IndexError):
            return False
        if v is None or Node.isempty(v) or v == idx2checker[key].default:
            return False
        return True

    # pylint: disable=useless-super-delegation
    def __getitem__(self, key: Union[str, int]) -> Any:
        """Get the value at the given key or index."""
        # pylint: disable=no-member
        return super().__getitem__(key)  # type: ignore

    # pylint: disable=useless-super-delegation
    def __setitem__(self, key: Union[str, int], value: Any):
        """Set the value at the given key or index."""
        # pylint: disable=no-member
        super().__setitem__(key, value)  # type: ignore

    def parse_expressions(
        self,
        symbol_table: Optional[Dict[str, Any]] = None,
        parsed_ids: Optional[set] = None,
        callfunc: Optional[Callable] = None,
    ):
        """Parse expressions in this node and all subnodes.
        Args:
            symbol_table: A dictionary mapping variable names to values.
            parsed_ids: A set of IDs of nodes that have already been parsed.
            callfunc: A function to call on each node after parsing.
        """
        if symbol_table is None:
            symbol_table = {}
        parsed_ids = parsed_ids or set()

        n_symbol_table = symbol_table.copy()
        n_symbol_table["parent_node"] = self
        index2checker = self._get_index2checker()
        for i, x in self.items():
            checker = index2checker[i]
            was_str = isinstance(x, str)
            if isinstance(x, Node) and id(x) not in parsed_ids:
                x.parse_expressions(n_symbol_table, parsed_ids)
            elif isinstance(x, str) and id(x) not in parsed_ids:
                self._parse_expression(i, n_symbol_table, index2checker.get(i, None))
            if checker:
                if was_str:
                    try:
                        self[i] = checker.cast_check_type(self[i], self, i)
                    except Exception as exc:
                        raise TypeError(
                            f'Could not parse expression "{self[i]}" in '
                            f'"{self.get_name()}[{i}]". {exc}'
                        ) from exc
                else:
                    checker.check_type(self[i], self, i)
            parsed_ids.add(id(self[i]))
            if callfunc is not None:
                self[i] = callfunc(self[i], n_symbol_table)
            if isinstance(self, DictNode):
                n_symbol_table[i] = self[i]
        return n_symbol_table

    def _parse_expression(
        self,
        index: Union[str, int],
        symbol_table: Dict[str, Any],
        checker: Optional[TypeSpecifier] = None,
    ):
        # Do we have to parse?
        # DO NOT PARSE if quote-enclosed string
        # Parse if:
        #    Parenthes-enclosed string
        #    OR it's currently failing type check

        v = self[index]

        if not isinstance(v, str) or is_quoted_string(v):
            # print(f'Was not a string: {v}')
            return

        parse = self._default_parse
        parse = parse or (v.startswith("(") and v.endswith(")"))
        try:
            if checker is not None:
                checker.check_type(v, self, index)
        except Exception:
            parse = True

        if parse:
            v = parse_expression(v, symbol_table, f"{self.get_name()}[{index}]")
            if checker is not None:
                v = checker.cast_check_type(v, self, index)
            self[index] = v

    @classmethod
    def unique_class_name(cls):
        """Return a unique name for this class."""
        return ".".join(c.__name__ for c in cls.mro()[::-1])

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Node._needs_declare_attrs.add(cls)

    def clear_upward_references(self):
        """Clear all references to global specs and parent nodes."""
        self.parent_node = None
        self.spec = None

        if isinstance(self, DictNode):
            # Recursively clear references for all child nodes
            for _, v in self.items():
                if isinstance(v, Node):
                    v.clear_upward_references()
        if isinstance(self, ListNode):
            # Recursively clear references for all child nodes
            for v in self:
                if isinstance(v, Node):
                    v.clear_upward_references()

    def claim_child_nodes(self):
        """Claim all child nodes."""
        if isinstance(self, DictNode):
            for _, v in self.items():
                if isinstance(v, Node):
                    v.claim_child_nodes()
                    v.parent_node = self
        if isinstance(self, ListNode):
            for v in self:
                if isinstance(v, Node):
                    v.claim_child_nodes()
                    v.parent_node = self

    def __deepcopy__(self, memo):
        """Create a deep copy of this node."""
        # Create a new instance of the same class
        if isinstance(self, DictNode):
            for k, v in self.items():
                if isinstance(v, Node):
                    v.parent_node = None
                    v.spec = None
            new_instance = self.__class__(
                **{k: copy.deepcopy(v) for k, v in self.items()}
            )
            for k, v in self.items():
                if isinstance(v, Node):
                    v.parent_node = self
                    v.spec = self.spec
            for k, v in new_instance.items():
                if isinstance(v, Node):
                    v.parent_node = new_instance
                    v.spec = self.spec

        elif isinstance(self, ListNode):
            for i, v in enumerate(self):
                if isinstance(v, Node):
                    v.parent_node = None
                    v.spec = None
            new_instance = self.__class__([copy.deepcopy(v) for v in self])
            for i, v in enumerate(self):
                if isinstance(v, Node):
                    v.parent_node = new_instance
                    v.spec = self.spec
            for i, v in enumerate(new_instance):
                if isinstance(v, Node):
                    v.parent_node = new_instance
                    v.spec = self.spec
        return new_instance

    def to_yaml(self, f: Union[str, Path, TextIO]):
        yaml.write_yaml_file(f, self)


class ListNode(Node, list):
    """A node that is a list of other nodes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        __node_skip_parse = kwargs.pop("__node_skip_parse", False)
        myname = self.__class__.__name__
        for a in args:
            if isinstance(a, set):
                a = list(a)
            if isinstance(a, list):
                self.extend(a)
            else:
                raise TypeError(f"ListNode {myname} got a non-list: {a}")
        if kwargs:
            raise TypeError(f"ListNode {myname} got keyword args: {kwargs}")
        self.from_data = list(self)
        if not __node_skip_parse:
            self._parse_elems()

    def __getitem__(self, key: Union[str, int]) -> Any:
        """
        Get the value at the given key or index. If the index is an int, it is
        used as an index into the list. If the index is a str, iterates through
        elements of the list. If elements in the lists are also lists, recurse.
        If elements in the list are dicts, searches for a "name" key. If found
        and the name matches the index, returns the element. If any dicts without
        a "name" key are found, raises a KeyError.
        """

        if isinstance(key, int):
            return super().__getitem__(key)  # type: ignore
        elif isinstance(key, str):
            for elem in self:
                if isinstance(elem, dict):
                    if "name" not in elem:
                        raise KeyError(
                            f"Dict {elem} has no 'name' key. If we are indexing into"
                            f"a list of dicts, the dicts must have a 'name' key."
                            f"Fail in {self.get_name()}[{key}]"
                        )
                    if elem["name"] == key:
                        return elem
                elif isinstance(elem, ListNode):
                    return elem[key]
        else:
            raise TypeError(
                f"ListNode {self.get_name()} got a non-str non-int index: {key}"
            )

        found = []
        for elem in self:
            if isinstance(elem, dict) and "name" in elem:
                found.append(elem["name"])
            elif isinstance(elem, ListNode):
                found.extend(elem.get_names())
        raise KeyError(
            f"No element with name \"{key}\" found in {self.get_name()}. "
            f"Candidates are {', '.join(found)}"
        )


class CombinableListNode(ListNode):
    """A list node that can be combined with others by extending."""

    def combine(self, other: "CombinableListNode") -> "CombinableListNode":
        """Extends this list with the contents of another list."""
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"Can not combine different classes {self.__class__.__name__} "
                f"and {other.__class__.__name__}."
            )
        self.extend(other)
        return self


class FlatteningListNode(ListNode):
    """A list node that flattens lists of lists."""

    def _flatten(self):
        while any(isinstance(x, list) for x in self):
            for i, x in enumerate(self):
                if isinstance(x, list):
                    self.extend(self.pop(i))

    def combine(self, other: "FlatteningListNode") -> "FlatteningListNode":
        self.extend(other)
        self._flatten()
        return self

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flatten()


class DictNode(Node, dict):
    """A node that is a dictionary of other nodes."""

    def __init__(self, *args, __node_skip_parse=False, **kwargs):
        __node_skip_parse = kwargs.pop("__node_skip_parse", False)
        super().__init__(*args, **kwargs)

        self.update(**kwargs)
        to_update = [a for a in args]
        while to_update:
            t = to_update.pop(0)
            if isinstance(t, dict):
                self._update_combine_pre_parse(t)
            elif isinstance(t, list):
                to_update.extend(t)
            else:
                raise TypeError(
                    f"DictNode {self.__class__.__name__} got a {type(t)} argument."
                    f"Expected a dict or list of dicts."
                )

        self.update(kwargs)
        for k, v in self._get_type_specifiers(self.spec).items():
            if k not in self and not v.no_change_key:
                self[k] = default_unspecified_
        if not __node_skip_parse:
            self._parse_elems()

    def _update_combine_pre_parse(self, other: dict):
        for k, v in other.items():
            try:
                self.combine_index(k, v)
            except Exception as exc:
                raise ValueError(
                    f"Re-specification of key {k} in {self.get_name()}. "
                    f"First value: {self[k]}. Second value: {v}."
                ) from exc

    @classmethod
    def require_one_of(cls, *args):
        """Require that at least one of the given keys is present."""
        _require_one_of = getattr(cls, "_require_one_of", [])
        if not _require_one_of:
            cls._require_one_of = _require_one_of
        cls._require_one_of.append(args)

    @classmethod
    def require_all_or_none_of(cls, *args):
        """Require that all or none of the given keys are present."""
        _require_all_or_none_of = getattr(cls, "_require_all_or_none_of", [])
        if not _require_all_or_none_of:
            cls._require_all_or_none_of = _require_all_or_none_of
        cls._require_all_or_none_of.append(args)

    def combine(self, other: "DictNode") -> "DictNode":
        """Combines this dictionary with another dictionary.
        If a key is present in both dictionaries, the values are combined.
        Otherwise, the key is taken from whichever dictionary has it.
        """
        keys = list(self.keys()) + [k for k in other.keys() if k not in self]
        # Make sure the classes are the same
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"Can not combine different classes {self.__class__.__name__} "
                f"and {other.__class__.__name__}."
            )
        for k in keys:
            mine, others = self.get(k, None), other.get(k, None)
            if mine is None:
                self[k] = others
            elif others is None:
                other[k] = mine
            elif isempty(mine):
                self[k] = others
            elif isempty(others):
                self[k] = mine
            elif mine == others:
                pass
            else:
                self[k] = Node.try_combine(mine, others)
        return self

    @classmethod
    def from_yaml_files(*args, **kwargs):
        return DictNode.from_yaml(*args, **kwargs)

    @classmethod
    def from_yaml(
        cls: type[T],
        *files: Union[str, List[str], Path, list[Path]],
        jinja_parse_data: Dict[str, Any] = None,
        **kwargs,
    ) -> T:
        """
        Loads a dictionary from one more more yaml files.

        Each yaml file should contain a dictionary. Dictionaries are combined in the order they are given.

        Keyword arguments are also added to the dictionary.

        Args:
            files: A list of yaml files to load.
            jinja_parse_data: A dictionary of data to use when parsing
            kwargs: Extra keyword arguments to add to the dictionary.

        Returns:
            A DictNode containing the combined dictionaries.
        """

        """Loads a dictionary from a list of yaml files. Each yaml file
        should contain a dictionary. Dictionaries are in the given order.
        Keyword arguments are also added to the dictionary.
        !@param files A list of yaml files to load.
        !@param jinja_parse_data A dictionary of data to use when parsing
        !@param kwargs Extra keyword arguments to add to the dictionary.
        """
        allfiles = []
        jinja_parse_data = jinja_parse_data or {}
        for f in files:
            if isinstance(f, (list, tuple)):
                if isinstance(f[0], Path):
                    f = list(map(str, f))
                allfiles.extend(f)
            else:
                if isinstance(f, Path):
                    f = str(f)
                allfiles.append(f)
        files = allfiles
        rval = {}
        key2file = {}
        extra_elems = []
        to_parse = []
        for f in files:
            logging.info("Loading yaml file %s", f)
            globbed = [x for x in glob.glob(f) if os.path.isfile(x)]
            if not globbed:
                raise FileNotFoundError(f"Could not find file {f}")
            for g in globbed:
                if any(os.path.samefile(g, x) for x in to_parse):
                    logging.info('Ignoring duplicate file "%s" in yaml load', g)
                else:
                    to_parse.append(g)

        for f in to_parse:
            if not (
                f.endswith(".yaml") or f.endswith(".jinja") or f.endswith(".jinja2")
            ):
                logging.warning(
                    f"File {f} does not end with .yaml, .jinja, or .jinja2. Skipping."
                )
            logging.info("Loading yaml file %s", f)
            loaded = yaml.load_yaml(f, data=jinja_parse_data)
            if not isinstance(loaded, dict):
                raise TypeError(
                    f"Expected a dictionary from file {f}, got {type(loaded)}"
                )
            for k, v in loaded.items():
                if k in rval:
                    logging.info("Found extra top-key %s in %s", k, f)
                    extra_elems.append((k, v))
                else:
                    logging.info("Found top key %s in %s", k, f)
                    key2file[k] = f
                    rval[k] = v

        c = cls(**rval, **kwargs)
        logging.info(
            "Parsing extra attributes %s", ", ".join([x[0] for x in extra_elems])
        )
        c._parse_extra_elems(extra_elems)
        return c

    def _check_alias(self, key) -> None:
        if not isinstance(key, str):
            return
        aliases_with = None
        if "_" in key and key.replace("_", "-") in self:
            aliases_with = key.replace("_", "-")
        if "-" in key and key.replace("-", "_") in self:
            aliases_with = key.replace("-", "_")
        if "-" in key:
            for k in self._get_index2checker().keys():
                if k == key.replace("-", "_"):
                    aliases_with = k
                    break

        if aliases_with is not None:
            raise KeyError(
                f'Key "{key}" is an alias for "{aliases_with}" in {self}. '
                f"Use the alias instead."
            )

    def __getitem__(self, __key: Any) -> Any:
        self._check_alias(__key)
        return super().__getitem__(__key)

    def __setitem__(self, __key: Any, __value: Any) -> None:
        self._check_alias(__key)
        super().__setitem__(__key, __value)

    def get(self, __key: Any, __default: Any = None) -> Any:
        """
        Gets a key from the dictionary.
        """
        self._check_alias(__key)
        return super().get(__key, __default)

    def setdefault(self, __key: Any, __default: Any = None) -> Any:
        """
        Sets the default value for a key.
        """
        self._check_alias(__key)
        return super().setdefault(__key, __default)

    def pop(self, __key: Any, __default: Any = None) -> Any:
        """
        Pops a key from the dictionary.
        """
        self._check_alias(__key)
        return super().pop(__key, __default)

    def check_unrecognized(self, *args, **kwargs) -> None:
        """
        Check for unrecognized keys in this node and all subnodes.
        """
        super().check_unrecognized(*args, **kwargs)
        checkers = self._get_index2checker()

        def check(keys: list, expected: Union[tuple, str], countstr: str):
            found = []
            for k in keys:
                v, checker = self.get(k, None), checkers.get(k, None)
                if v is not None and checker is not None and v != checker.default:
                    found.append(k)
            countmatch = len(found) == expected
            if isinstance(expected, tuple):
                countmatch = len(found) in expected
            if not countmatch:
                raise KeyError(
                    f"Expected {countstr} of {keys} in {self}, "
                    f"found {len(found)}. Values: "
                    f'{", ".join([f"{k}: {self[k]}" for k in found])}'
                )

        for required_one in getattr(self, "_require_one_of", []):
            check(required_one, (1,), "exactly one")
        for required_all in getattr(self, "_require_all_or_none_of", []):
            check(required_all, (0, len(required_all)), "all or none")

    def __getattr__(self, name):
        """Index into the attributes or the contents of this node."""
        if name in self:
            return self[name]
        try:
            return super().__getattr__(name)
        except AttributeError:
            raise AttributeError(
                f"Could not find attribute {name} in {self.__class__}. "
                f"Available keys: {list(self.keys())}. Available attributes: "
                f"{list(super().__dir__())}"
            ) from None

    def __setattr__(self, name, value):
        if name in self:
            self[name] = value
        else:
            try:
                super().__setattr__(name, value)
            except AttributeError:
                raise AttributeError(
                    f"Could not find attribute {name} in {self}."
                ) from None
