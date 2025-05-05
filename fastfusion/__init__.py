from fastfusion.yamlparse import Node
from fastfusion.frontend.specification import Specification

for d in Node._needs_declare_attrs:
    if hasattr(d, "declare_attrs"):
        d.declare_attrs(d)
