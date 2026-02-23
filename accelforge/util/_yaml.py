import copy
import functools
import logging
import os
import glob
import re
import io
from typing import Callable, List, Dict, Any, OrderedDict, Tuple
import ruamel.yaml
import warnings
from ruamel.yaml.error import ReusedAnchorWarning
from jinja2 import StrictUndefined, Environment, FileSystemLoader
import threading
import time


PARSING_LOCK = threading.Lock()
THREAD_ID = 0

SCRIPTS_FROM = []
EXTRA_PLUG_IN_PATHS = []


class LockAcquirer:
    def __init__(self):
        self.has_lock = False

    def __enter__(self):
        while True:
            if PARSING_LOCK.acquire(blocking=False):
                global THREAD_ID
                THREAD_ID = threading.get_ident()
                self.has_lock = True
                break
            if THREAD_ID == threading.get_ident():
                break
            time.sleep(0.01)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.has_lock:
            PARSING_LOCK.release()


def recursive_mutator_stop(func):
    return func
    cache = set()

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        assert not kwargs, (
            f"Recursive mutator stop only works with non-keyword arguments. "
            f"Args were {args} and kwargs were {kwargs}."
        )
        k = id(args[0])
        if k in cache:
            return args[0]
        cache.add(k)
        try:
            result = func(*args, **kwargs)
        finally:
            cache.remove(k)
        return result

    return wrapper


def recursive_mutator_eq_stop(func):
    return func
    cache = set()

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if args[0] in cache:
            failstr = (
                f"Infinite recursion detected: {func.__name__}("
                f"{args[0]}) called from within itself. Does a YAML file "
                f"include itself?"
            )
            raise RuntimeError(failstr)
        cache.add(args[0])
        try:
            result = func(*args, **kwargs)
        finally:
            cache.remove(args[0])
        return result

    return wrapper


class MultiIncludeWrapper:
    def __init__(self, contents: List):
        self.contents = contents


def append_path(p: str, cur_path: str, include_dirs: List[str]):
    new_paths = find_paths(p, cur_path, include_dirs)
    include_dirs += new_paths
    logging.info(f"YAML Adding {new_paths} to include paths")
    return ""


def find_paths(p: str, cur_path: str, include_dirs: List[str]):
    if isinstance(p, list):
        paths = [find_paths(x, cur_path, include_dirs) for x in p]
    else:
        searched = []
        paths = []
        prepend = (
            [""] if os.path.isabs(p) else [os.path.dirname(cur_path)] + include_dirs
        )

        for d in prepend:
            s = os.path.abspath(os.path.realpath(os.path.join(d, p)))
            globbed_paths = glob.glob(s)
            if globbed_paths:
                paths.extend(globbed_paths)
            searched.append(s)
        if not paths:
            raise FileNotFoundError(
                f"Could not find file {p} in any of the following paths:"
                + "\n  "
                + "\n  ".join(searched)
            )

    unique_paths = []
    uniques = set()
    while paths:
        p = os.path.realpath(os.path.abspath(paths.pop(0)))
        if p not in uniques:
            unique_paths.append(p)
            uniques.add(p)
    return unique_paths


def find_path(p: str, cur_path: str, include_dirs: List[str]):
    prepend = [""] if os.path.isabs(p) else [os.path.dirname(cur_path)] + include_dirs
    searched = []
    for d in prepend:
        s = os.path.abspath(os.path.realpath(os.path.join(d, p)))
        if os.path.exists(s):
            return s
        searched.append(s)
    raise FileNotFoundError(
        f"Could not find file {p} in any of the following paths:"
        + "\n  "
        + "\n  ".join(searched)
    )


@recursive_mutator_eq_stop
def load_file_and_includes(
    path: str,
    data: Dict[str, Any] = None,
    include_dirs: List[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Load a YAML file and recursively load any included YAML files
    :param path: string that specifies the path of the YAML file to be loaded
    :param data: dictionary that contains the data to be rendered
    :param include_dirs: list of directories to search for included files
    :return: string that contains the loaded YAML content
    """
    path = os.path.abspath(os.path.realpath(path))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find file {path}")

    data = data or {}
    data = {k: v for k, v in data.items()}
    include_dirs = include_dirs or []
    include_dirs = [d for d in include_dirs]

    include_counter = 0

    def include(p, single, indices: str = "", **kwargs):
        # If the path is a relative path, make it relative to the current file
        to_include = []
        indices = indices.lstrip(".")
        nonlocal include_counter
        include_name = (
            os.path.basename(path).rsplit(".", 1)[0] + "_" + str(include_counter)
        )

        include_name = re.sub(r"\W+", "", include_name)
        for np in find_paths(p, path, include_dirs):
            logging.info(
                f"YAML Adding {np} to document with !include{'_all' if not single else ''}"
            )
            to_include.append(load_yaml(np, {**data, **kwargs}, include_dirs))

        if single:
            if len(to_include) > 1:
                raise RuntimeError(
                    f"More than one file found for {path}: {find_paths(path)}."
                    f"To include multiple files, use include_all()."
                )

        data[include_name] = (
            to_include[0] if len(to_include) == 1 else MultiIncludeWrapper(to_include)
        )
        v = f"!include_loaded {include_name}"
        if indices:
            v += "." + indices
        include_counter += 1
        return v

    def include_single(p, indices: str = "", **kwargs):
        return include(p, True, indices, **kwargs)

    def include_all(p, indices: str = "", **kwargs):
        return include(p, False, indices, **kwargs)

    def include_text(p, **kwargs):
        found = []
        for np in find_paths(p, path, include_dirs):
            found.append(
                load_file_and_includes(np, {**data, **kwargs}, include_dirs)[0]
            )
            logging.info(f"YAML Adding {np} to document with !include_text")
        return "\n".join(found)

    # Add include_as to the template environment
    env = Environment(
        loader=FileSystemLoader(os.path.dirname(path)), undefined=StrictUndefined
    )

    def setenv(key, value):
        key, value = str(key), str(value)
        os.environ[key] = value
        return "{{ setenv('" + key + "', '" + value + "') }}}}"

    def path_exists(p):
        try:
            find_path(p, path, include_dirs)
            return True
        except FileNotFoundError:
            return False

    env.globals["cwd"] = lambda: os.path.dirname(path)
    env.globals["include"] = include_single
    env.globals["include_all"] = include_all
    env.globals["include_text"] = include_text
    env.globals["find_path"] = lambda x: find_path(x, path, include_dirs)
    env.globals["find_paths"] = lambda x: find_paths(x, path, include_dirs)
    env.globals["path_exists"] = path_exists
    env.globals["setenv"] = setenv

    env.globals["add_to_path"] = lambda p: append_path(p, path, include_dirs)

    path_file = path[len(os.path.dirname(path)) + 1 :]
    template = env.get_template(path_file)
    string = template.render(data, undefined=StrictUndefined)
    return string, data, include_dirs


@recursive_mutator_stop
def merge_check(x: dict[str, Any] | list[Any] | Any) -> None:
    if isinstance(x, list):
        for i, v in enumerate(x):
            x[i] = merge_check(v)
    elif isinstance(x, dict):
        found_merge = False
        for k, v in list(x.items()):
            x[k] = merge_check(v)
            if str(k) == "<<<" or str(k) == "<<":
                assert not found_merge, (
                    f'Cannot have multiple "<<<" or "<<" keys in a dict. '
                    f"Keys were {list(x.keys())}"
                )
                found_merge = True
                x = merge(x, copy.deepcopy(x.pop(k)), str(k) == "<<<")
    return x


ERRCOUNT = 0


def load_yaml(
    path: str,
    data: Dict[str, Any] = None,
    include_dirs: List[str] = None,
) -> Dict[str, Any]:
    """
    Load YAML content from a file or string
    :param path: string that specifies the path of the YAML file to be loaded
    :param data: dictionary that contains the data to be rendered
    :param include_dirs: list of directories to search for included files
    :return: evaluated YAML content or YAML object
    """
    with LockAcquirer():
        data = {k: v for k, v in data.items()} if data is not None else {}
        path = os.path.abspath(os.path.realpath(path))
        try:
            evaluated, data, include_dirs = load_file_and_includes(
                path, data, include_dirs
            )
        except Exception as e:
            e.add_note(f"Error loading YAML file {path}")
            raise
        try:
            result = merge_check(get_yaml(path, data).load(evaluated))
            return result
        except Exception as e:
            global ERRCOUNT
            failpath = f"/tmp/yamleval_error{ERRCOUNT}.yaml"
            ERRCOUNT += 1
            with open(failpath, "w") as f:
                f.write(evaluated)
            e.add_note(
                f"Error parsing YAML file {path}. Offending file written to "
                f"{failpath}"
            )
            raise


@recursive_mutator_stop
def merge(
    merge_into: dict, tomerge: dict | list | tuple, recursive: bool = True
) -> dict:
    if isinstance(tomerge, (list, tuple)):
        combined = dict()
        for m in tomerge:
            combined = merge(combined, m, recursive)
        tomerge = combined
    if not isinstance(tomerge, dict):
        raise ValueError(
            f'Expected a dict under the "<<<" or "<<" keys, but ' f"got {tomerge}"
        )
    if not isinstance(merge_into, dict):
        raise ValueError(
            f'Expected to merge into a dict with the "<<<" key, '
            f"but got {merge_into}"
        )

    for k, v in tomerge.items():
        if k not in merge_into:
            merge_into[k] = v
        elif not recursive:
            continue
        elif isinstance(merge_into[k], (NoMergeListWrapper, NoMergeDictWrapper)):
            continue
        elif isinstance(merge_into[k], dict) and isinstance(v, dict):
            merge_into[k] = merge(merge_into[k], v, recursive)
        elif isinstance(merge_into[k], list) and isinstance(v, list):
            merge_into[k] = merge_into[k] + v
    return merge_into


def represent_none(self, data: None) -> str:
    """
    Represent None as 'null' in YAML
    :param self: YAML representer object
    :param data: None object to be represented
    :return: 'null' string
    """
    return self.represent_scalar("tag:yaml.org,2002:null", "null")


def ordereddict_to_dict(self, dictionary: OrderedDict) -> Dict[str, Any]:
    """
    Change an OrderedDict to a dictionary in YAML
    :param self: YAML representer object
    :param dictionary: OrderedDict object to be represented
    :return: dictionary object
    """
    d = {}
    for key in dictionary.keys():
        d[key] = dictionary[key]
    return self.represent_dict(dictionary)


@recursive_mutator_stop
def recursive_unorder_dict(to_unorder: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(to_unorder, dict):
        return {k: recursive_unorder_dict(v) for k, v in to_unorder.items()}
    elif isinstance(to_unorder, list):
        return [recursive_unorder_dict(v) for v in to_unorder]
    return to_unorder


@recursive_mutator_stop
def callables2strings(to_convert: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(to_convert, dict):
        to_convert = {k: callables2strings(v) for k, v in to_convert.items()}
    elif isinstance(to_convert, list):
        to_convert = [callables2strings(v) for v in to_convert]
    elif isinstance(to_convert, Callable):
        to_convert = str(getattr(to_convert, "_original_expression", to_convert))
    return to_convert


def write_yaml_file(filepath: str, content: Dict[str, Any]) -> None:
    """
    Write YAML content to a file
    :param filepath: string that specifies the destination file path
    :param content: YAML string that needs to be written to the destination file
    :return: None
    """
    if os.path.exists(filepath):
        os.remove(filepath)
    if os.path.dirname(filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    out_file = open(filepath, "a")
    out_file.write(to_yaml_string(content))


def to_yaml_string(content: Dict[str, Any]) -> str:
    """
    Convert YAML content to a string
    :param content: YAML content to be converted to a string
    :return: string representation of the YAML content
    """
    with LockAcquirer():
        dumpstream = io.StringIO()
        get_base_yaml().dump(
            callables2strings(recursive_unorder_dict(content)), stream=dumpstream
        )
        return dumpstream.getvalue()


def get_base_yaml() -> ruamel.yaml.YAML:
    yaml = ruamel.yaml.YAML(typ="rt")
    # yaml.default_flow_style = None
    yaml.indent(mapping=4, sequence=4, offset=2)
    yaml.preserve_quotes = True

    def recursive_mutator_stop(func):
        cache = set()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            assert not kwargs, (
                f"Recursive mutator stop only works with non-keyword "
                f"arguments. Args were {args} and kwargs were {kwargs}."
            )
            k = id(args[0])
            if k in cache:
                return args[0]
            cache.add(k)
            try:
                result = func(*args, **kwargs)
            finally:
                cache.remove(k)
            return result

        return wrapper

    yaml.representer.add_representer(type(None), recursive_mutator_stop(represent_none))
    yaml.representer.add_representer(
        OrderedDict, recursive_mutator_stop(ordereddict_to_dict)
    )

    return yaml


def get_yaml(path: str, data: Dict[str, Any] = None) -> ruamel.yaml.YAML:
    """Get a YAML object with the right settings"""
    yaml = get_base_yaml()
    ymf = YAMLFileLoader(path, data)
    # yaml.default_flow_style = None

    warnings.simplefilter("ignore", ReusedAnchorWarning)
    yaml.constructor.add_constructor("!include_loaded", ymf.include_loaded)
    yaml.constructor.add_constructor("!include", ymf.include)
    yaml.constructor.add_constructor("!includedir", ymf.includedir)
    yaml.constructor.add_constructor("!nomerge", ymf.nomerge)

    return yaml


class NoMergeListWrapper(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class NoMergeDictWrapper(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class YAMLFileLoader:
    def __init__(self, path: str, data: Dict[str, Any] = None) -> None:
        self.path = path
        self.data = data or {}
        self.include_counter = 0
        self.loading_from_dir = os.path.abspath(os.path.dirname(path))
        self.include_data = data or {}
        self.env = Environment(
            loader=FileSystemLoader(os.path.dirname(path)), undefined=StrictUndefined
        )

    def nomerge(self, constructor, node):
        # print(f"Got node {node}")
        # Pop the tag
        # node.tag = None
        # print(f"Got node {node}")
        # # value = constructor.construct_object(node, deep=True)
        # print(f"COnstructed object {value}")
        if isinstance(node, ruamel.yaml.nodes.SequenceNode):
            return NoMergeListWrapper(constructor.construct_sequence(node, deep=True))
        if isinstance(node, ruamel.yaml.nodes.MappingNode):
            return NoMergeDictWrapper(
                ruamel.yaml.SafeConstructor.construct_mapping(
                    constructor, node, deep=True
                )
            )
        raise ValueError(f"!nomerge tag must be applied to a list or dict, not {node}")

    def include_loaded(
        self,
        constructor: ruamel.yaml.constructor.Constructor,
        node: ruamel.yaml.nodes.ScalarNode,
    ) -> dict[str, Any] | None:
        """
        Constructor that parses the !include_loaded relative_file_path and loads the file
        from relative_file_path
        :param self: YAML constructor object
        :param node: YAML node object
        :return: evaluated YAML content
        """
        x = constructor.construct_scalar(node)
        found = self.include_data
        for k in str(x).split("."):
            try:
                if isinstance(found, MultiIncludeWrapper):
                    for i, f in enumerate(found.contents):
                        found.contents[i] = f[k]
                else:
                    found = found[k]
            except (KeyError, TypeError) as e:
                if isinstance(found, MultiIncludeWrapper):
                    e.add_note(
                        f"Could not parse !include_loaded {x}: {k} not found " f"in {f}"
                    )
                    raise
                e.add_note(
                    f"Could not parse !include_loaded {x}: {k} not found "
                    f"in {list(found.keys())}"
                )
                raise
        return found.contents if isinstance(found, MultiIncludeWrapper) else found

    def include(
        self,
        constructor: ruamel.yaml.constructor.Constructor,
        node: ruamel.yaml.nodes.ScalarNode,
    ) -> dict[str, Any] | None:
        """
        Constructor that parses !include relative_file_path and loads the file
        from relative_file_path
        :param self: YAML constructor object
        :param node: YAML node object
        :return: evaluated YAML content
        """
        filepath = constructor.construct_scalar(node)
        if filepath[-1] == ",":
            filepath = filepath[:-1]
        load_from = self.loading_from_dir
        return load_yaml(os.path.join(load_from, filepath), self.include_data)

    def includedir(
        self,
        constructor: ruamel.yaml.constructor.Constructor,
        node: ruamel.yaml.nodes.ScalarNode,
    ) -> list[dict[str, Any]]:
        """
        Constructor that parses the !includedir relative_file_path and loads the
        file from relative_file_path
        :param self: YAML constructor object
        :param node: YAML node object
        :return: list of evaluated YAML contents
        """
        filepath = constructor.construct_scalar(node)
        if filepath[-1] == ",":
            filepath = filepath[:-1]
        dirname = os.path.join(self.loading_from_dir, filepath)
        yamllist = []
        for filename in glob.glob(dirname + "/*.yaml"):
            yamllist.append(load_yaml(filename, self.include_data))
        return yamllist


yaml = get_base_yaml()
