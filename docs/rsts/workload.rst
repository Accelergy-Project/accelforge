.. _specifying-workload:

Specifying the Workload
=======================

This document shows how to convert a cascade of Einsums into an FFM workload specification.
If you are not familiar with the Einsum notation, please refer to a (TODO) before proceeding.

The FFM workload has the following structure
::

  workload:
    version: <version-string>
    shape: <shape-map>
    einsums: <einsums-list>

The current ``<version-string>`` is ``"0.5"``.

The map ``<shape-map>`` has rank variables as keys and a constraint expression as values.
For example, one can constrain the rank variable ``m`` by specifying a map entry
``m: 0 <= m < 4``.

The <einsum-list> is a list of Einsums in the workload. Each Einsum has the following structure
::

  name: <einsum-name-string>
  tensor_accesses:
  - name: <tensor-name-string>
    projection: <projection-list-or-map>
    output: <whether-the-tensor-is-output>
  - name: <other-tensor-name-string>
    ...
  ...

The projection can be a list of rank variable names if each rank is indexed by a rank variable
with a name that is the lowercase of the rank name (*e.g.*, rank `M` is indexed by `m`).
*I.e.*, a list can be used when the rank name can be uniquely inferred from the rank variable name.

If any of the rank does not meet the criteria above, then a map must be specified in which the keys
are rank names and the values are the index expressions (*e.g.*, ``H: p+r``).

.. _workload-rename:

Renaming Tensors and Rank Variables
-----------------------------------