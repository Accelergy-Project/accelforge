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

It is often conventient to be able to assign a generic name that resolves to tensors of an Einsum.
For example, architecture constraints often use tensor names. However, the tensors in an Einsum changes
between Einsums in the cascade. Thus, it is useful to define a name `output` and use it to write
architecture constraints, and resolve it for specific Einsums, such as `output = Z` in `Z = AB`.

This feature is supported in FFM using the renaming feature. Renames are defined by specifying a
`renames` key as follows
::

  renames:
    version: <version-string>
    einsums: <list-of-renames-per-einsum>

The ``<list-of-renames-per-einsum>`` contains items with the following from
::

  name: <einsum-name-or-default>
  tensor_accesses:
  - name: <generic-tensor-name>
    source: <specific-tensor-name-or-set-expression>
  ...
  rank_variables:
  - name: <generic-rank-name>
    source: <specific-rank-name-or-set-expression>

The <einsum-name-or-default> can be an Einsum name or the keyword ``default``, which means that the
renames will be used by default when an Einsum does not have a specific renames specified.