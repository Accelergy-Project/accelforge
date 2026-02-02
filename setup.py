"""Setup.py to ensure accelforge._version_scheme is importable during build."""

import sys
import os
from pathlib import Path

current_dir = Path(__file__).parent.absolute()
parent_dir = current_dir.parent

for path in [str(current_dir), str(parent_dir)]:
    if path not in sys.path:
        sys.path.insert(0, path)

pythonpath = os.environ.get("PYTHONPATH", "")
if pythonpath:
    for path in pythonpath.split(os.pathsep):
        if path and path not in sys.path:
            sys.path.insert(0, path)

import importlib.util
_version_scheme_path = current_dir / "accelforge" / "_version_scheme.py"
spec = importlib.util.spec_from_file_location("_version_scheme", _version_scheme_path)
_version_scheme = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_version_scheme)
post_version = _version_scheme.post_version
no_local = _version_scheme.no_local

from setuptools import setup

setup(
    use_scm_version={
        "version_scheme": post_version,
        "local_scheme": no_local,
        "write_to": "accelforge/_version.py",
        "fallback_version": "1.0",
    }
)
