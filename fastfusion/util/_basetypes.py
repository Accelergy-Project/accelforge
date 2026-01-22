import copy
import glob
import logging
import os
from pathlib import Path
import re
from pydantic import BaseModel, ConfigDict, Tag, ValidationError
from pydantic.main import IncEx
from pydantic_core.core_schema import (
    CoreSchema,
    chain_schema,
    list_schema,
    union_schema,
    no_info_plain_validator_function,
    str_schema,
    dict_schema,
    tagged_union_schema,
)
from typing import (
    Iterator,
    List,
    Mapping,
    TypeVar,
    Generic,
    Any,
    Callable,
    TypeVarTuple,
    Dict,
    Optional,
    Type,
    TypeAlias,
    Union,
    get_args,
    get_origin,
    TYPE_CHECKING,
    Self,
)

from fastfusion.util import _yaml
from fastfusion.util._parse_expressions import (
    parse_expression,
    ParseError,
    LiteralString,
    is_literal_string,
)

# Import will be resolved at runtime to avoid circular dependency
TYPE_CHECKING_RUNTIME = False
if TYPE_CHECKING or TYPE_CHECKING_RUNTIME:
    from fastfusion.util._setexpressions import InvertibleSet, eval_set_expression

T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)
K = TypeVar("K")
V = TypeVar("V")
PM = TypeVar("PM", bound="ParsableModel")
PL = TypeVar("PL", bound="ParsableList[Any]")

Ts = TypeVarTuple("Ts")


def _get_tag(value: Any) -> str:
    if not isinstance(value, dict):
        return value.__class__.__name__
    tag = None

    def try_get_tag(attr: str) -> str:
        if hasattr(value, attr) and getattr(value, attr) is not None:
            return getattr(value, attr)
        return None

    def try_index(attr: str) -> str:
        try:
            return value[attr]
        except:
            return None

    tag = None
    for attr in ("type", "_type", "_yaml_tag"):
        if tag := try_get_tag(attr):
            break
        if tag := try_index(attr):
            break
    if tag is None:
        raise ValueError(
            f"No tag found for {value}. Either set the type field " "or use a YAML tag."
        )
    tag = str(tag)
    if tag.startswith("!"):
        tag = tag[1:]
    return tag


def _uninstantiable(cls):
    prev_init = cls.__init__

    def _get_all_subclasses(cls):
        subclasses = set()
        for subclass in cls.__subclasses__():
            subclasses.add(subclass.__name__)
            subclasses.update(_get_all_subclasses(subclass))
        return subclasses

    def __init__(self, *args, **kwargs):
        if self.__class__ is cls:
            subclasses = _get_all_subclasses(cls)
            raise ValueError(
                f"{cls} can not be instantiated directly. Use a subclass. "
                f"Supported subclasses are:\n\t" + "\n\t".join(sorted(subclasses))
            )
        return prev_init(self, *args, **kwargs)

    cls.__init__ = __init__
    return cls


class _InferFromTag(Generic[*Ts]):
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: Callable
    ) -> CoreSchema:
        type_args = get_args(source_type)
        if not type_args:
            raise TypeError(
                f"_InferFromTag must be used with a type parameter, e.g. _InferFromTag[int]"
            )

        # type_args contains all the possible types: (Compute, Memory, "Hierarchical")
        target_types = []
        for arg in type_args:
            if isinstance(arg, str):
                # Handle string type names - we'll need to resolve them later
                target_types.append(arg)
            elif isinstance(arg, type):
                target_types.append(arg)
            else:
                target_types.append(arg)

        # Create tag to class mapping
        tag2class = {}
        for target_type in target_types:
            if isinstance(target_type, str):
                # For string types, use the string as both key and placeholder
                tag2class[target_type] = target_type
            elif hasattr(target_type, "__name__"):
                tag2class[target_type.__name__] = target_type
            else:
                # Fallback for other types
                tag2class[str(target_type)] = target_type

        def validate(value: Any) -> T:
            if hasattr(value, "_yaml_tag"):
                tag = value._yaml_tag
            elif hasattr(value, "_type"):
                tag = value._type
            else:
                for to_try in ("_yaml_tag", "_type", "type"):
                    try:
                        tag = value[to_try]
                        break
                    except:
                        pass
                else:
                    raise ValueError(
                        f"No tag found for {value}. Either set the type field "
                        "or use a YAML tag."
                    )
            tag = str(tag)
            if tag.startswith("!"):
                tag = tag[1:]
            value._type = tag

            print(f"Tag found! {tag}")
            if tag in tag2class:
                return tag2class[tag](**value)
            else:
                raise ValueError(
                    f"Unknown tag: {tag}. Supported tags are: {sorted(tag2class.keys())}"
                )

        # target_schema = handler.generate_schema(target_types)
        schemas = []
        for t in target_types:
            schemas.append(handler.generate_schema(t))
        target_schema = union_schema(schemas)
        # return chain_schema([
        #     no_info_plain_validator_function(validate),
        #     target_schema
        # ])
        return chain_schema(
            [
                no_info_plain_validator_function(validate),
                tagged_union_schema(tag2class, discriminator="_type"),
            ]
        )


class NoParse(Generic[T]):
    """A type skips parsing of the specified object."""

    _class_name: str = "NoParse"

    def __init__(self, value: T):
        self._value = value
        self._type = T

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: Callable
    ) -> CoreSchema:
        # Get the type parameter T from ParsesTo[T]
        type_args = get_args(source_type)
        if not type_args:
            raise TypeError(
                f"{cls._class_name} must be used with a type parameter, "
                f"e.g. {cls._class_name}[int]"
            )
        target_type = type_args[0]

        # Get the schema for the target type
        target_schema = handler(target_type)

        def validate_raw_string(value):
            if isinstance(value, str) and is_literal_string(value):
                return LiteralString(value)
            # raise ValueError("Not a raw string")

        # Create a union schema that either validates as raw string or normal validation
        return target_schema

class ParsesTo(Generic[T]):
    """A type that parses to the specified type T.

    Example:
        class Example(ParsableModel):
            a: ParsesTo[int]  # Will parse string expressions to integers
            b: ParsesTo[str]  # Will parse string expressions to strings
            c: str  # Regular string, no parsing
    """

    _class_name: str = "ParsesTo"

    def __init__(self, value: str):
        self._value = value
        self._is_literal_string = is_literal_string(value)
        self._type = T

        assert self._type != str, (
            f"{self._class_name}[str] is not allowed. Use str directly instead."
            f"If something should just be a string, no expressions are allowed. "
            f"This is so the users don't have to quote-wrap all strings."
        )

    def __str__(self) -> str:
        return str(self._value)

    def __repr__(self) -> str:
        return f"{self._class_name}({repr(self._value)})"

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: Callable
    ) -> CoreSchema:
        # Get the type parameter T from ParsesTo[T]
        type_args = get_args(source_type)
        if not type_args:
            raise TypeError(
                f"{cls._class_name} must be used with a type parameter, "
                f"e.g. {cls._class_name}[int]"
            )
        target_type = type_args[0]

        # Get the schema for the target type
        target_schema = handler(target_type)

        def validate_raw_string(value):
            if isinstance(value, str) and is_literal_string(value):
                return LiteralString(value)
            # raise ValueError("Not a raw string")

        # Create a union schema that either validates as raw string or normal validation
        return union_schema(
            [
                # First option: validate as raw string
                chain_schema(
                    [
                        no_info_plain_validator_function(validate_raw_string),
                        str_schema(),
                        # target_schema
                    ]
                ),
                # Second option: normal validation (string then target type)
                chain_schema(
                    [
                        str_schema(),
                        # target_schema
                    ]
                ),
                # Third option: direct target type validation
                target_schema,
            ]
        )


class TryParseTo(ParsesTo, Generic[T]):
    """
    A type that tries to parse to the specified type T. If the parsing fails, the value
    is returned as a string.
    """

    _class_name: str = "TryParseTo"

    def __init__(self, value: str):
        super().__init__(value)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: Callable
    ) -> CoreSchema:
        # Get the type parameter T from ParsesTo[T]
        type_args = get_args(source_type)
        if not type_args:
            raise TypeError(
                f"{cls._class_name} must be used with a type parameter, "
                f"e.g. {cls._class_name}[int]"
            )
        target_type = type_args[0]

        # Get the schema for the target type
        target_schema = handler(target_type)

        def validate_raw_string(value):
            if isinstance(value, str) and is_literal_string(value):
                return LiteralString(value)
            # raise ValueError("Not a raw string")

        # Create a union schema that either validates as raw string or normal validation
        return union_schema(
            [
                # First option: validate as raw string
                chain_schema(
                    [
                        no_info_plain_validator_function(validate_raw_string),
                        str_schema(),
                        # target_schema
                    ]
                ),
                # Second option: normal validation (string then target type)
                chain_schema(
                    [
                        str_schema(),
                        # target_schema
                    ]
                ),
                # Third option: direct target type validation
                target_schema,
                # Fourth option: return the value as a string
                str_schema(),
            ]
        )


if TYPE_CHECKING:
    try:
        from typing_extensions import TypeAliasType

        _T_alias = TypeVar("_T_alias")
        ParsesTo = TypeAliasType("ParsesTo", _T_alias, type_params=(_T_alias,))
        TryParseTo = TypeAliasType("TryParseTo", _T_alias, type_params=(_T_alias,))
    except Exception:
        # Best-effort fallback for type checkers that don't support TypeAliasType
        pass


class _PostCall(Generic[T]):
    def __call__(self, field: str, value: T, symbol_table: dict[str, Any]) -> T:
        return value


@_uninstantiable
class Parsable(Generic[M]):
    """An abstract base class for parsing. Parsables support the `_parse_expressions`
    method, which is used to parse the object from a string.
    """

    def _parse_expressions(
        self, symbol_table: dict[str, Any] = None, **kwargs
    ) -> tuple[M, dict[str, Any]]:
        raise NotImplementedError("Subclasses must implement this method")

    def get_fields(self) -> list[str]:
        raise NotImplementedError("Subclasses must implement this method")

    def get_validator(self, field: str) -> type:
        raise NotImplementedError("Subclasses must implement this method")

    def _parse_expressions_final(
        self,
        symbol_table: dict[str, Any],
        order: tuple[str, ...],
        post_calls: tuple[_PostCall[T], ...],
        use_setattr: bool = True,
        already_parsed: dict[str, Any] | None = None,
        **kwargs,
    ) -> tuple["Parsable", dict[str, Any]]:
        self._parsed = True

        if already_parsed is None:
            already_parsed = {}

        fields = [f for f in self.get_fields() if f not in already_parsed]

        field_order = _get_parsable_field_order(
            order,
            [
                (
                    f,
                    getattr(self, f) if use_setattr else self[f],
                    self.get_validator(f),
                )
                for f in fields
            ],
        )
        prev_symbol_table = symbol_table.copy()
        # for k, v in symbol_table.items():
        #     if isinstance(k, str) and k.startswith("global_") and v is None:
        #         raise ParseError(
        #             f"Global variable {k} is required. Please set it in "
        #             f"either the attributes or an outer scope. Try setting it with "
        #             f"Spec.variables.{k} = [value]."
        #         )

        for field, value in already_parsed.items():
            symbol_table[field] = value
            if use_setattr:
                setattr(self, field, value)
            else:
                self[field] = value
            symbol_table[field] = value

        for field in field_order:
            value = getattr(self, field) if use_setattr else self[field]
            validator = self.get_validator(field)
            parsed = _parse_field(field, value, validator, symbol_table, self, **kwargs)
            for post_call in post_calls:
                parsed = post_call(field, value, parsed, symbol_table)
            if use_setattr:
                setattr(self, field, parsed)
            else:
                self[field] = parsed
            symbol_table[field] = parsed

        for k, v in prev_symbol_table.items():
            if (
                isinstance(k, str)
                and k.startswith("global_")
                and symbol_table.get(k, None) != v
            ):
                raise ParseError(
                    f"Global variable {k} is already set to {v} in the outer scope. "
                    f"It cannot be changed to {symbol_table[k]}."
                )

        return self, symbol_table


class _FromYAMLAble:
    @classmethod
    def from_yaml(
        cls: type[T],
        *files: str | list[str] | Path | list[Path],
        jinja_parse_data: dict[str, Any] | None = None,
        top_key: str | None = None,
        **kwargs,
    ) -> T:
        """
        Loads a dictionary from one more more yaml files.

        Each yaml file should contain a dictionary. Dictionaries are combined in the
        order they are given.

        Keyword arguments are also added to the dictionary.

        Args:
            files:
                A list of yaml files to load.
            jinja_parse_data: Optional[Dict[str, Any]]
                A dictionary of Jinja2 data to use when parsing the yaml files.
            top_key: Optional[str]
                The top key to use when parsing the yaml files.
            kwargs: Extra keyword arguments to be passed to the constructor.

        Returns:
            A dict containing the combined dictionaries.
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
            loaded = _yaml.load_yaml(f, data=jinja_parse_data)
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

        if top_key is not None:
            if top_key not in rval:
                raise KeyError(f"Top key {top_key} not found in {files}")
            rval = rval[top_key]

        c = None
        try:
            c = cls(**rval, **kwargs)
        except Exception as e:
            pass
        if c is None and len(rval) == 1:
            logging.warning(
                f"Trying to parse a single element dictionary as a {cls.__name__}. "
            )
            try:
                rval_first = list(rval.values())[0]
                if not isinstance(rval_first, dict):
                    raise TypeError(
                        f"Expected a dictionary as the top-level element in {files}, "
                        f"got {type(rval_first)}."
                    )
                c = cls(**rval_first, **kwargs)
            except Exception as e:
                logging.warning(
                    f"Error parsing {files} with top key {top_key}. " f"Error: {e}"
                )
        if c is None:
            c = cls(**rval, **kwargs)

        if extra_elems:
            logging.info(
                "Parsing extra attributes %s", ", ".join([x[0] for x in extra_elems])
            )
        c._yaml_source = ",".join(files)
        return c


def _parse_field(
        field,
        value,
        validator,
        symbol_table,
        parent,
        must_parse_try_parse_to: bool = False,
        must_copy: bool = True,
        **kwargs
):
    from fastfusion.util._setexpressions import InvertibleSet, eval_set_expression

    try:
        # Get the origin type (ParsesTo or TryParseTo) and its arguments
        origin = get_origin(validator)
        if origin is ParsesTo or origin is TryParseTo:
            try:
                target_type = get_args(validator)[0]
                parsed = value
                if isinstance(target_type, tuple) and any(issubclass(t, InvertibleSet) for t in target_type):
                    raise NotImplementedError(
                        f"InvertibleSet must be used directly, not as a part of a "
                        f"union, else this function must be updated."
                    )

                # Check if validator is for InvertibleSet
                if isinstance(target_type, type) and issubclass(target_type, InvertibleSet):
                    # Get the target type from the validator

                    # If the given type is a set, replace it with a string that'll parse
                    if isinstance(value, set):
                        value = " | ".join(str(v) for v in value)

                    type_args = target_type.__pydantic_generic_metadata__['args']
                    assert len(type_args) == 1, "Expected exactly one type argument"
                    expected_element_type = type_args[0]

                    try:
                        # eval_set_expression does the type checking for us
                        return eval_set_expression(
                            value,
                            symbol_table,
                            expected_space=expected_element_type,
                            location=field,
                        )
                    except ParseError as e:
                        if origin is TryParseTo and not must_parse_try_parse_to:
                            return LiteralString(value)
                        raise
                elif is_literal_string(value):
                    parsed = LiteralString(value)
                else:
                    parsed = parse_expression(value, symbol_table)

                if must_copy and id(parsed) == id(value):
                    parsed = copy.deepcopy(parsed)

                # Get the target type from the validator
                target_any = (
                    target_type is Any
                    or isinstance(target_type, tuple)
                    and Any in target_type
                )
                if not target_any and not isinstance(parsed, target_type):
                    raise ParseError(
                        f'{value} parsed to "{parsed}" with type {type(parsed).__name__}.'
                        f" Expected {target_type}.",
                    )
            except ParseError as e:
                if origin is TryParseTo and not must_parse_try_parse_to:
                    return LiteralString(value)
                raise
        else:
            parsed = value

        if isinstance(parsed, Parsable) and origin is not NoParse:
            parsed, _ = parsed._parse_expressions(
                symbol_table=symbol_table,
                must_copy=must_copy,
                must_parse_try_parse_to=must_parse_try_parse_to,
                **kwargs
            )
            return parsed
        elif isinstance(parsed, str):
            return LiteralString(parsed)
        else:
            return parsed
    except ParseError as e:
        try:
            e.add_field(parent[field].name)
        except:
            e.add_field(field)
        raise e


# python_name_regex = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b')


def _get_parsable_field_order(
    order: tuple[str, ...], field_value_validator_triples: list[tuple[str, Any, type]]
) -> list[str]:
    order = list(order)
    to_sort = []
    for field, value, validator in field_value_validator_triples:
        if field in order:
            continue
        if get_origin(validator) is not ParsesTo:
            order.append(field)
            continue
        if not isinstance(value, str) or is_literal_string(value):
            order.append(field)
            continue
        to_sort.append((field, value))

    dependencies = {field: set() for field, _ in to_sort}
    for field, value in to_sort:
        for other_field, other_value in to_sort:
            if field != other_field:
                if re.findall(r"\b" + re.escape(field) + r"\b", other_value):
                    dependencies[other_field].add(field)

    while to_sort:
        for field, value in to_sort:
            if all(dep in order for dep in dependencies[field]):
                order.append(field)
                to_sort.remove((field, value))
                break
        else:
            raise ParseError(
                f"Circular dependency detected in expressions. "
                f"Fields: {', '.join(t[0] for t in to_sort)}"
            )

    return order


class _ModelWithUnderscoreFields(BaseModel, _FromYAMLAble, Mapping):
    def __init__(self, **kwargs):
        new_kwargs = {}
        for field, value in kwargs.items():
            if (
                field.startswith("_")
                and field not in self.__class__.model_fields
                and field[1:] in self.__class__.model_fields
            ):
                new_kwargs[field[1:]] = value
            else:
                new_kwargs[field] = value
        super().__init__(**new_kwargs)

    # Exclude is supported OK, but makes the docs a lot longer because it's in so many
    # objects and has a very long type.
    def to_yaml(
        self, f: str | None = None
    ) -> str:  # , exclude: IncEx | None = None) -> str:
        """
        Dump the model to a YAML string.

        Parameters
        ----------
        f: str | None
            The file to write the YAML to. If not given, then returns as a string.
        exclude: IncEx | None
            The fields to exclude from the YAML.

        Returns
        -------
        str
            The YAML string.
        """
        dump = self.model_dump()  # exclude=exclude)

        def _to_str(value: Any):
            if isinstance(value, list):
                return [_to_str(x) for x in value]
            elif isinstance(value, dict):
                return {_to_str(k): _to_str(v) for k, v in value.items()}
            elif isinstance(value, str):
                return str(value)
            return value

        if f is not None:
            _yaml.write_yaml_file(f, _to_str(dump))
        return _yaml.to_yaml_string(_to_str(dump))

    def all_fields_default(self):
        for field in self.__class__.model_fields:
            default = self.__class__.model_fields[field].default
            if getattr(self, field) != default:
                return False
        return True

    def model_dump_non_none(self, **kwargs):
        return {k: v for k, v in self.model_dump(**kwargs).items() if v is not None}

    def __contains__(self, key: str) -> bool:
        try:
            self[key]
            return True
        except KeyError:
            return False

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            pass
        raise KeyError(f"Key {key} not found in {self.__class__.__name__}")

    def __setitem__(self, key: str, value: Any):
        setattr(self, key, value)

    def __delitem__(self, key: str):
        delattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.get_fields())

    def __len__(self) -> int:
        return len(self.get_fields())


@_uninstantiable
class ParsableModel(_ModelWithUnderscoreFields, Parsable["ParsableModel"]):
    """A model that will parse any fields that are given to it. When parsing, submodels
    will also be parsed if they support it. Parsing will parse any fields that are given
    as strings and do not match the expected type.

    Underscore-prefixed fields will have type checking applied strictly, meaning that
    even if this class or a subclass allows extra fields, underscore-prefixed fields
    must be recognized directly. The leading underscore is removed before setting any
    variable names.
    """

    model_config = ConfigDict(extra="forbid")
    # type: Optional[str] = None

    def __init__(self, **kwargs):
        required_type = kwargs.pop("type", None)
        super().__init__(**kwargs)
        if required_type is not None:
            if not isinstance(self, required_type):
                raise TypeError(
                    f"type field {required_type} does not match"
                    f"{self.__class__.__name__}"
                )

    def get_validator(self, field: str) -> Type:
        if field in self.__class__.model_fields:
            return self.__class__.model_fields[field].annotation
        return ParsesTo[Any]

    def get_fields(self) -> list[str]:
        fields = set(self.__class__.model_fields.keys())
        if getattr(self, "__pydantic_extra__", None) is not None:
            fields.update(self.__pydantic_extra__.keys())
        return sorted(fields)

    def _parse_expressions(
        self,
        symbol_table: dict[str, Any] = None,
        order: tuple[str, ...] = (),
        post_calls: tuple[_PostCall[T], ...] = (),
        already_parsed: dict[str, Any] | None = None,
        **kwargs,
    ) -> tuple[Self, dict[str, Any]]:
        new = self.model_copy()
        symbol_table = symbol_table.copy() if symbol_table is not None else {}
        kwargs = dict(kwargs)
        return new._parse_expressions_final(
            symbol_table,
            order,
            post_calls,
            use_setattr=True,
            already_parsed=already_parsed,
            **kwargs,
        )

class NonParsableModel(_ModelWithUnderscoreFields):
    """A model that will not parse any fields."""

    model_config = ConfigDict(extra="forbid")
    type: Optional[str] = None

    def get_validator(self, field: str) -> Type:
        return Any


class ParsableList(list[T], Parsable["ParsableList[T]"], Generic[T]):
    """
    A list that can be parsed from a string. ParsableList[T] means that a given string
    can be parsed, yielding a list of objects of type T.
    """

    def get_validator(self, field: str) -> Type:
        return T

    def _parse_expressions(
        self,
        symbol_table: dict[str, Any] = None,
        order: tuple[str, ...] = (),
        post_calls: tuple[_PostCall[T], ...] = (),
        already_parsed: dict[str, Any] | None = None,
        **kwargs,
    ) -> tuple["ParsableList[T]", dict[str, Any]]:
        new = ParsableList[T](x for x in self)
        symbol_table = symbol_table.copy() if symbol_table is not None else {}
        order = order + tuple(x for x in range(len(new)) if x not in order)
        return new._parse_expressions_final(
            symbol_table,
            order,
            post_calls,
            use_setattr=False,
            already_parsed=already_parsed,
            **kwargs,
        )

    def get_fields(self) -> list[str]:
        return sorted(range(len(self)))

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: Callable
    ) -> CoreSchema:
        # Get the type parameter T from ParsableList[T]
        type_args = get_args(source_type)
        if not type_args:
            raise TypeError(
                f"ParsableList must be used with a type parameter, e.g. ParsableList[int]"
            )
        item_type = type_args[0]

        # Get the schema for the item type
        item_schema = handler(item_type)

        # Create a schema that validates lists of the item type
        return chain_schema(
            [
                list_schema(item_schema),
                no_info_plain_validator_function(lambda x: cls(x)),
            ]
        )

    def __getitem__(self, key: str | int | slice) -> T:
        if isinstance(key, int):
            return super().__getitem__(key)  # type: ignore

        elif isinstance(key, slice):
            return ParsableList[T](super().__getitem__(key))

        elif isinstance(key, str):
            found = None
            for elem in self:
                name = None
                if isinstance(elem, dict):
                    name = elem.get("name", None)
                elif hasattr(elem, "name"):
                    name = elem.name
                if name is not None and name == key:
                    if found is not None:
                        raise ValueError(f'Multiple elements with name "{key}" found.')
                    found = elem
            if found is not None:
                return found

        fields = self.get_fields()
        fields += [
            (
                x.name
                if hasattr(x, "name")
                else x.get("name", None) if isinstance(x, dict) else None
            )
            for x in self
        ]
        fields = sorted(str(x) for x in fields if x is not None)
        raise KeyError(
            f'No element with name "{key}" found. Available names: {', '.join(fields)}'
        )

    def __contains__(self, item: Any) -> bool:
        try:
            self[item]
            return True
        except KeyError:
            return super().__contains__(item)

    def __copy__(self) -> Self:
        return type(self)(x for x in self)


class ParsableDict(
    dict[K, V], Parsable["ParsableDict[K, V]"], Generic[K, V], _FromYAMLAble
):
    """A dictionary that can be parsed from a string. ParsableDict[K, V] means that a
    given string can be parsed, yielding a dictionary with keys of type K and values of
    type V.
    """

    def get_validator(self, field: str) -> type:
        return V

    def get_fields(self) -> list[str]:
        return sorted(self.keys())

    def _parse_expressions(
        self,
        symbol_table: dict[str, Any] = None,
        order: tuple[str, ...] = (),
        post_calls: tuple[_PostCall[V], ...] = (),
        already_parsed: dict[str, Any] | None = None,
        **kwargs,
    ) -> tuple["ParsableDict[K, V]", dict[str, Any]]:
        new = ParsableDict[K, V](self)
        symbol_table = symbol_table.copy() if symbol_table is not None else {}
        return new._parse_expressions_final(
            symbol_table,
            order,
            post_calls,
            use_setattr=False,
            already_parsed=already_parsed,
            **kwargs,
        )

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: Callable
    ) -> CoreSchema:
        # Get the type parameters K and V from ParsableDict[K, V]
        type_args = get_args(source_type)
        if len(type_args) != 2:
            raise TypeError(
                f"ParsableDict must be used with two type parameters, e.g. ParsableDict[str, int]"
            )
        key_type, value_type = type_args

        # Get the schemas for the key and value types
        key_schema = handler.generate_schema(key_type)
        value_schema = handler.generate_schema(value_type)

        # Create a schema that validates dictionaries with the specified key and value types
        return chain_schema(
            [
                dict_schema(key_schema, value_schema),
                no_info_plain_validator_function(lambda x: cls(x)),
            ]
        )

    def __copy__(self) -> Self:
        return type(self)({k: v for k, v in self.items()})

class ParseExtras(ParsableModel):
    """
    A model that will parse any extra fields that are given to it.

    Underscore-prefixed fields will have type checking applied strictly, meaning that
    even if this class or a subclass allows extra fields, underscore-prefixed fields
    must be recognized directly. The leading underscore is removed before setting any
    variable names.
    """

    model_config = ConfigDict(extra="allow")

    def get_validator(self, field: str) -> type:
        if field not in self.__class__.model_fields:
            return ParsesTo[Any]
        return self.__class__.model_fields[field].annotation

    def __init__(self, **kwargs):
        new_kwargs = {}
        for field, value in kwargs.items():
            if field.startswith("_"):
                field = field[1:]
                if field not in self.__class__.model_fields:
                    raise ValueError(
                        f"Field {field} is not a known field for "
                        f"{self.__class__.__name__}. Known fields are: "
                        f"{', '.join(sorted(self.__class__.model_fields.keys()))}"
                    )
            new_kwargs[field] = value
        super().__init__(**new_kwargs)
