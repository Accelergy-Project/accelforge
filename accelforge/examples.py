from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


class Directory:
    def __init__(self, path):
        self.path: Path = path

    def __repr__(self):
        return f"Directory({self.path})"

    def __getattr__(self, name: Path):
        target_stem: Path = self.path / name
        if target_stem.is_dir():
            return Directory(self.path / name)

        target_yaml = target_stem.with_suffix(".yaml")
        if target_yaml.is_file():
            return target_yaml

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
