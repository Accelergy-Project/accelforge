from .common import *
from . import v5 as v5

for d in Node._needs_declare_attrs:
    if hasattr(d, 'declare_attrs'):
        d.declare_attrs(d)
