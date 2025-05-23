from abc import ABC
import copy
import glob
import logging
import os
from pathlib import Path
import re
from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic_core.core_schema import CoreSchema, chain_schema, list_schema, union_schema, no_info_plain_validator_function, str_schema, dict_schema
from ruamel.yaml.scalarstring import DoubleQuotedScalarString, SingleQuotedScalarString
from typing import Iterator, List, TypeVar, Generic, Any, Callable, Union, Dict, Optional, Type, TypeAlias, get_args, get_origin

from fastfusion.util import yaml
from fastfusion.util.parse_expressions import parse_expression, ParseError
from fastfusion.util import yaml

T = TypeVar('T')
M = TypeVar('M', bound=BaseModel)
K = TypeVar('K')
V = TypeVar('V')

class RawString(str):
    pass

def is_raw_string(value: Any) -> bool:
    return isinstance(value, (DoubleQuotedScalarString, SingleQuotedScalarString, RawString))

class ParsesTo(Generic[T]):
    """A type that parses to the specified type T.
    
    Example:
        class Example(ParsableModel):
            a: ParsesTo[int]  # Will parse string expressions to integers
            b: ParsesTo[str]  # Will parse string expressions to strings
            c: str  # Regular string, no parsing
    """
    def __init__(self, value: str):
        self._value = value
        self._is_raw_string = is_raw_string(value)
        self._type = T
        
        assert self._type != str, (
            f"ParsesTo[str] is not allowed. Use str directly instead."
            f"If something should just be a string, no expressions are allowed. "
            f"This is so the users don't have to quote-wrap all strings."
        )

    def __str__(self) -> str:
        return str(self._value)

    def __repr__(self) -> str:
        return f"ParsesTo({repr(self._value)})"

    def check_type(self, value: Any) -> None:
        if not isinstance(value, self._type):
            raise ParseError(
                f"Incorrect type. Expected {self._type.__name__}, got {type(value).__name__}",
            )

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Callable) -> CoreSchema:
        # Get the type parameter T from ParsesTo[T]
        type_args = get_args(source_type)
        if not type_args:
            raise TypeError(f"ParsesTo must be used with a type parameter, e.g. ParsesTo[int]")
        target_type = type_args[0]

        # Get the schema for the target type
        target_schema = handler(target_type)

        # Create a schema that validates either strings or instances of the target type
        return union_schema([
            # First try to validate as a string
            str_schema(),
            # Then try to validate as an instance of the target type
            target_schema
        ])

class PostCall(Generic[T]):
    def __call__(self, field: str, value: T, symbol_table: dict[str, Any]) -> T:
        return value

class Parsable(ABC, Generic[M]):
    def parse_expressions(self, symbol_table: dict[str, Any] = None, **kwargs) -> tuple[M, dict[str, Any]]:
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_fields(self) -> list[str]:
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_validator(self, field: str) -> type:
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_instances_of_type(self, type: Type[T]) -> Iterator[T]:
        if isinstance(self, type):
            yield self
        elif isinstance(self, list):
            for item in self:
                if isinstance(item, Parsable):
                    yield from item.get_instances_of_type(type)
                elif isinstance(item, type):
                    yield item
        elif isinstance(self, dict):
            for item in self.values():
                if isinstance(item, Parsable):
                    yield from item.get_instances_of_type(type)
                elif isinstance(item, type):
                    yield item


    def _parse_expressions(self, symbol_table: dict[str, Any], order: tuple[str, ...], post_calls: tuple[PostCall[T], ...], use_setattr: bool = True, **kwargs) -> tuple["Parsable", dict[str, Any]]:
        field_order = get_parsable_field_order(
            order,
            [(field, getattr(self, field) if use_setattr else self[field], self.get_validator(field))
                for field in self.get_fields()]
        )
        for field in field_order:
            value = getattr(self, field) if use_setattr else self[field]
            validator = self.get_validator(field)
            parsed = parse_field(field, value, validator, symbol_table, **kwargs)
            for post_call in post_calls:
                parsed = post_call(field, value, parsed, symbol_table)
            if use_setattr:
                setattr(self, field, parsed)
            else:
                self[field] = parsed
            symbol_table[field] = parsed
        return self, symbol_table


class FromYAMLAble:
    @classmethod
    def from_yaml(
        cls: type[T],
        *files: Union[str, List[str], Path, list[Path]],
        jinja_parse_data: Dict[str, Any] = None,
        top_key: Optional[str] = None,
        **kwargs,
    ) -> T:
        """
        Loads a dictionary from one more more yaml files.

        Each yaml file should contain a dictionary. Dictionaries are combined in
        the order they are given.

        Keyword arguments are also added to the dictionary.

        Args:
            files: A list of yaml files to load. jinja_parse_data: A dictionary
            of data to use when parsing kwargs: Extra keyword arguments to add
            to the dictionary.

        Returns:
            A dict containing the combined dictionaries.
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
                    f"Error parsing {files} with top key {top_key}. "
                    f"Error: {e}"
                )
        if c is None:
            c = cls(**rval, **kwargs)

        if extra_elems:
            logging.info(
                "Parsing extra attributes %s", ", ".join([x[0] for x in extra_elems])
            )
        return c

def parse_field(field, value, validator, symbol_table, **kwargs):
    try:
        # Get the origin type (ParsesTo) and its arguments
        origin = get_origin(validator)
        if origin is ParsesTo:
            if value == "REQUIRED":
                if field in symbol_table:
                    return copy.deepcopy(symbol_table[field])
                else:
                    raise ParseError(
                        f"{field} is required. Please set it in "
                        f"either the attributes or an outer scope."
                    )
            elif is_raw_string(value):
                return value
            else:
                parsed = copy.deepcopy(parse_expression(value, symbol_table))
                # Get the target type from the validator
                target_type = get_args(validator)[0]
                target_any = target_type is Any or isinstance(target_type, tuple) and Any in target_type
                if not target_any and not isinstance(parsed, target_type):
                    raise ParseError(
                        f"{value} parsed to {parsed} with type {type(parsed).__name__}. "
                        f"Expected {target_type.__name__}.",
                    )
                return parsed
        elif isinstance(value, Parsable):
            parsed, _ = value.parse_expressions(symbol_table, **kwargs)
            return parsed
        else:
            return value
    except ParseError as e:
        e.add_field(field)
        raise e

# python_name_regex = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b')

def get_parsable_field_order(order: tuple[str, ...], field_value_validator_triples: list[tuple[str, Any, type]]) -> list[str]:
    order = list(order)
    to_sort = []
    for field, value, validator in field_value_validator_triples:
        if field in order:
            continue
        if get_origin(validator) is not ParsesTo:
            order.append(field)
            continue
        if not isinstance(value, str) or is_raw_string(value):
            order.append(field)
            continue
        to_sort.append((field, value))
        
    dependencies = {field: set() for field, _ in to_sort}
    for field, value in to_sort:
        for other_field, other_value in to_sort:
            if field != other_field:
                if re.findall(r'\b' + re.escape(field) + r'\b', other_value):
                    dependencies[field].add(other_field)

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

class ParsableModel(BaseModel, Parsable['ParsableModel'], FromYAMLAble):
    model_config = ConfigDict(extra="forbid")
    
    def get_validator(self, field: str) -> type:
        if field in self.__class__.model_fields:
            return self.__class__.model_fields[field].annotation
        return ParsesTo[Any]

    def get_fields(self) -> list[str]:
        fields = set(self.__class__.model_fields.keys())
        if getattr(self, '__pydantic_extra__', None) is not None:
            fields.update(self.__pydantic_extra__.keys())
        return sorted(fields)


    def __init__(self, **data):
        try:
            super().__init__(**data)
        except ValidationError as e:
            for error in e.errors():
                if error["type"] == "extra_forbidden":
                    error["msg"] += f". Allowed fields are: {sorted(self.__class__.model_fields)}"
            raise e

    def parse_expressions(self, symbol_table: dict[str, Any] = None, order: tuple[str, ...] = (), post_calls: tuple[PostCall[T], ...] = (), **kwargs) -> tuple['ParsableModel', dict[str, Any]]:
        new = self.model_copy()
        symbol_table = symbol_table.copy() if symbol_table is not None else {}
        return new._parse_expressions(symbol_table, order, post_calls, use_setattr=True, **kwargs)

    def to_yaml(self, f: str):
        yaml.write_yaml_file(f, self.model_dump())
        
    def all_fields_default(self):
        for field in self.__class__.model_fields:
            default = self.__class__.model_fields[field].default
            if getattr(self, field) != default:
                return False
        return True

class NonParsableModel(BaseModel, FromYAMLAble):
    model_config = ConfigDict(extra="forbid")
    
    def get_validator(self, field: str) -> type:
        return Any

class ParsableList(list[T], Parsable['ParsableList[T]'], Generic[T]):
    def get_validator(self, field: str) -> type:
        return T

    def parse_expressions(self, symbol_table: dict[str, Any] = None, order: tuple[str, ...] = (), post_calls: tuple[PostCall[T], ...] = (), **kwargs) -> tuple['ParsableModel', dict[str, Any]]:
        new = ParsableList[T](x for x in self)
        symbol_table = symbol_table.copy() if symbol_table is not None else {}
        order = order + tuple(x for x in range(len(new)) if x not in order)
        return new._parse_expressions(symbol_table, order, post_calls, use_setattr=False, **kwargs)
    
    def get_fields(self) -> list[str]:
        return sorted(range(len(self)))

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Callable) -> CoreSchema:
        # Get the type parameter T from ParsableList[T]
        type_args = get_args(source_type)
        if not type_args:
            raise TypeError(f"ParsableList must be used with a type parameter, e.g. ParsableList[int]")
        item_type = type_args[0]

        # Get the schema for the item type
        item_schema = handler(item_type)

        # Create a schema that validates lists of the item type
        return chain_schema([
            list_schema(item_schema),
            no_info_plain_validator_function(lambda x: cls(x))
        ])

    def __getitem__(self, key: Union[str, int]) -> Any:
        if isinstance(key, int):
            return super().__getitem__(key)  # type: ignore

        elif isinstance(key, str):
            for elem in self:
                name = None
                if isinstance(elem, dict):
                    name = elem.get("name", None)
                elif hasattr(elem, "name"):
                    name = elem.name
                if name is not None and name == key:
                    return elem
        raise KeyError(f"No element with name \"{key}\" found.")
    
    def __contains__(self, item: Any) -> bool:
        try:
            self[item]
            return True
        except KeyError:
            return super().__contains__(item)

class ParsableDict(dict[K, V], Parsable['ParsableDict[K, V]'], Generic[K, V], FromYAMLAble):
    
    def get_validator(self, field: str) -> type:
        return V
    
    def get_fields(self) -> list[str]:
        return sorted(self.keys())
    
    
    def parse_expressions(self, symbol_table: dict[str, Any] = None, order: tuple[str, ...] = (), post_calls: tuple[PostCall[V], ...] = (), **kwargs) -> tuple['ParsableDict[K, V]', dict[str, Any]]:
        new = ParsableDict[K, V](self)
        symbol_table = symbol_table.copy() if symbol_table is not None else {}
        return new._parse_expressions(symbol_table, order, post_calls, use_setattr=False, **kwargs)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Callable) -> CoreSchema:
        # Get the type parameters K and V from ParsableDict[K, V]
        type_args = get_args(source_type)
        if len(type_args) != 2:
            raise TypeError(f"ParsableDict must be used with two type parameters, e.g. ParsableDict[str, int]")
        key_type, value_type = type_args

        # Get the schemas for the key and value types
        key_schema = handler.generate_schema(key_type)
        value_schema = handler.generate_schema(value_type)

        # Create a schema that validates dictionaries with the specified key and value types
        return chain_schema([
            dict_schema(key_schema, value_schema),
            no_info_plain_validator_function(lambda x: cls(x))
        ])

class ParseExtras:
    def get_validator(self, field: str) -> type:
        if field not in self.__class__.model_fields:
            return ParsesTo[Any]
        return self.__class__.model_fields[field].annotation
