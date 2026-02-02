from __future__ import annotations

from accelforge.frontend import arch
from accelforge.frontend import config
from accelforge.frontend import mapping
from accelforge.frontend import renames
from accelforge.frontend import spec
from accelforge.frontend import variables
from accelforge.frontend import workload
from accelforge.frontend.spec import Spec, Spec
from accelforge.mapper.FFM import Metrics
from accelforge.util import set_n_parallel_jobs
from accelforge.util import LiteralString
import accelforge.mapper as mapper
from accelforge.examples import examples
from accelforge.util.exceptions import EvaluationError

from accelforge.frontend.variables import Variables
from accelforge.frontend.arch import Arch
from accelforge.frontend.config import Config
from accelforge.frontend.mapping import Mapping
from accelforge.frontend.renames import Renames
from accelforge.frontend.spec import Spec
from accelforge.frontend.workload import Workload

__all__ = [
    # Submodules
    "arch",
    "config",
    "mapping",
    "mapper",
    "renames",
    "spec",
    "variables",
    "workload",
    # Main classes
    "Arch",
    "Config",
    "Mapping",
    "Metrics",
    "Renames",
    "Spec",
    "Variables",
    "Workload",
    # Utilities
    "LiteralString",
    "set_n_parallel_jobs",
    "examples",
    "EvaluationError",
]
