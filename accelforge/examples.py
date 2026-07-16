import os
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def _dir_or_only_yaml(path: Path):
    """
    If a path points to a directory that has no subdirectories and only one
    non-underscore-prefixed YAML file, returns that file.
    """
    if any(child.is_dir() for child in path.iterdir()):
        return Directory(path)
    yamls = sorted(path.glob("*.yaml"))
    non_underscored = [y for y in yamls if not y.name.startswith("_")]
    if len(yamls) == 1:
        return yamls[0]
    if len(non_underscored) == 1:
        return non_underscored[0]
    return Directory(path)


class Directory:
    def __init__(self, path):
        self.path: Path = path

    def __repr__(self):
        return f"Directory({self.path})"

    def __getattr__(self, name: Path):
        target_stem: Path = self.path / name
        if target_stem.is_dir():
            return _dir_or_only_yaml(target_stem)

        target_yaml = target_stem.with_suffix(".yaml")
        if target_yaml.is_file():
            return target_yaml

        name_replaced = name.replace(".", "_")
        name_yaml_replaced = target_yaml.stem.replace(".", "_")
        for item in os.listdir(self.path):
            stem = Path(item).stem
            if stem.replace(".", "_") in [name_replaced, name_yaml_replaced]:
                path = self.path / item
                return _dir_or_only_yaml(path) if path.is_dir() else path

        raise ValueError(f"Not found: {target_stem} or {target_yaml}")

    def iter(self):
        if not self.path.is_dir():
            return
        for path in self.path.iterdir():
            yield path.stem


examples = Directory(EXAMPLES_DIR)
"""
Convenient variable for getting path to examples directory.

For example:
```
path_to_gh100_yaml: pathlib.Path = examples.arches.gh100
```
"""
