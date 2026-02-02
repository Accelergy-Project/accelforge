"""Timeloop Spec. Each piece below (minus processors) corresponds to a top key in the Timeloop spec."""

from .spec import *
import accelforge.frontend.arch as arch
import accelforge.frontend.config as config
import accelforge.frontend.mapping as mapping
import accelforge.frontend.renames as renames
import accelforge.frontend.spec as spec
import accelforge.frontend.variables as variables
import accelforge.frontend.workload as workload

__all__ = [
    "arch",
    "config",
    "mapping",
    "renames",
    "spec",
    "variables",
    "workload",
    "Spec",
]
