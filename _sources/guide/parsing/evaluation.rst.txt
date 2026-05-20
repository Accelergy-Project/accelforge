Expression Evaluation
=====================

Objects can include expressions that are evaluated when the
:py:class:`~accelforge.frontend.spec.Spec` is evaluated. Evaluation occurs when the
:py:func:`~accelforge.frontend.spec.Spec` is going to be used to model the energy, area,
or latency of an accelerator, such as when the
:py:func:`~accelforge.frontend.spec.Spec.calculate_component_area_energy_latency_leak`
method is called.

To-be-evaluated expressions can include Python code, and supported
operations include many standard library functions (*e.g.,* ``range``, ``min``) and
functions from the ``math`` standard library (*e.g.,* ``log2``, ``ceil``).

The scope available for evaluation includes the following in order of increasing
precedence:

- Variables defined in a top-level :py:class:`~accelforge.frontend.variables.Variables`
  object.
- Variables defined in outer-level YAML objects. Dictionary keys can be referenced by
  names, and list entries by index. The dot syntax can be used to access dictionaries;
  for example, ``x.y.z`` is equivalent to ``outer_scope["x"]["y"]["z"]``.
- Variables defined in the current YAML object. Dictionary keys may reference each other
  as long as references are not cyclic.

The following is an example of valid evaluated data:

.. code-block:: yaml

  variables:
    a: 123
    b: a + 5
    c: min(b, 3)
    d: sum(y for y in range(1, 10))

  # In some later scope
  ... outer_scope:
    x: 123
    y: a + x # Reference top-level variables
    inner_scope:
        a: 3 # Override outer scope
        b: outer_scope.x
        # Statements can be out-of-order if not cyclic referencing
        firt_item: second_item
        second_item: 3

Additionally, values can be set directly in Python code. For example:

.. code-block:: python

  from accelforge.frontend.arch import ComponentAttributes
  attributes = ComponentAttributes(
    value1=123,
    value2="value1 + 5"
    # ... other attributes
  )


Included Functions
------------------

The following are available functions. In addition to the below, Python keywords that
are available witout import (*e.g.,* ``min``) are also available

- ``ceil``: :py:func:`math.ceil`
- ``comb``: :py:func:`math.comb`
- ``copysign``: :py:func:`math.copysign`
- ``fabs``: :py:func:`math.fabs`
- ``factorial``: :py:func:`math.factorial`
- ``floor``: :py:func:`math.floor`
- ``fmod``: :py:func:`math.fmod`
- ``frexp``: :py:func:`math.frexp`
- ``fsum``: :py:func:`math.fsum`
- ``gcd``: :py:func:`math.gcd`
- ``isclose``: :py:func:`math.isclose`
- ``isfinite``: :py:func:`math.isfinite`
- ``isinf``: :py:func:`math.isinf`
- ``isnan``: :py:func:`math.isnan`
- ``isqrt``: :py:func:`math.isqrt`
- ``ldexp``: :py:func:`math.ldexp`
- ``modf``: :py:func:`math.modf`
- ``perm``: :py:func:`math.perm`
- ``prod``: :py:func:`math.prod`
- ``remainder``: :py:func:`math.remainder`
- ``trunc``: :py:func:`math.trunc`
- ``exp``: :py:func:`math.exp`
- ``expm1``: :py:func:`math.expm1`
- ``log``: :py:func:`math.log`
- ``log1p``: :py:func:`math.log1p`
- ``log2``: :py:func:`math.log2`
- ``log10``: :py:func:`math.log10`
- ``pow``: :py:func:`math.pow`
- ``sqrt``: :py:func:`math.sqrt`
- ``acos``: :py:func:`math.acos`
- ``asin``: :py:func:`math.asin`
- ``atan``: :py:func:`math.atan`
- ``atan2``: :py:func:`math.atan2`
- ``cos``: :py:func:`math.cos`
- ``dist``: :py:func:`math.dist`
- ``hypot``: :py:func:`math.hypot`
- ``sin``: :py:func:`math.sin`
- ``tan``: :py:func:`math.tan`
- ``degrees``: :py:func:`math.degrees`
- ``radians``: :py:func:`math.radians`
- ``acosh``: :py:func:`math.acosh`
- ``asinh``: :py:func:`math.asinh`
- ``atanh``: :py:func:`math.atanh`
- ``cosh``: :py:func:`math.cosh`
- ``sinh``: :py:func:`math.sinh`
- ``tanh``: :py:func:`math.tanh`
- ``erf``: :py:func:`math.erf`
- ``erfc``: :py:func:`math.erfc`
- ``gamma``: :py:func:`math.gamma`
- ``lgamma``: :py:func:`math.lgamma`
- ``pi``: :py:func:`math.pi`
- ``e``: :py:func:`math.e`
- ``tau``: :py:func:`math.tau`
- ``inf``: :py:func:`math.inf`
- ``nan``: :py:func:`math.nan`
- ``abs``: :py:func:`abs`
- ``round``: :py:func:`round`
- ``pow``: :py:func:`pow`
- ``sum``: :py:func:`sum`
- ``range``: :py:func:`range`
- ``len``: :py:func:`len`
- ``min``: :py:func:`min`
- ``max``: :py:func:`max`
- ``float``: :py:func:`float`
- ``int``: :py:func:`int`
- ``str``: :py:func:`str`
- ``bool``: :py:func:`bool`
- ``list``: :py:func:`list`
- ``tuple``: :py:func:`tuple`
- ``enumerate``: :py:func:`enumerate`
- ``getcwd``: :py:func:`os.getcwd`
- ``map``: :py:func:`map`

Additional funcitons can be added to the scope by definining the
:py:attr:`~accelforge.frontend.spec.Spec.expression_custom_functions` attribute and
listing either functionss or paths to Python files that contain functions.

.. _set-expressions:

Set Expressions
===============

In the architecture, set expressions may reference the tensors and rank variables of the
specific Einsum being executed. Set expressions are evaluated for each Einsum +
Flattened-Architecture (:ref:`flattening`) combination.

As an example of a set expression, we can describe all tensors that are not
intermediates using the following:

.. code-block:: yaml

    ~Intermediates

Set expressions can use the full Python syntax, including the following:

- ``&``: Intersection
- ``|``: Union
- ``~``: Complement
- ``-``: Difference

You may also use Pythonic language with set expressions in some locations. For example,
we may want to use input tensors if and only if there are three or fewer total tensors:

.. code-block:: yaml

    Inputs if len(All) > 3 else All

Set expressions are evaluated for every Einsum + Flattened-Architecture (:ref:`flattening`)
combination. The following set expressions are supported:

- ``All``: All tensors used in the current Einsum.
- ``Inputs``: Tensors input to the current Einsum.
- ``Intermediates``: Tensors produced by one Einsum and consumed by another.
- ``Nothing``: The empty set.
- ``Outputs``: Tensors output from the current Einsum.
- ``Persistent``: Tensors that must remain in backing storage for the full duration of
  the workload's execution.
- ``Shared``: Tensors that are shared between multiple Einsums.
- ``Tensors``: Alias for ``All``.

Additionally, the following special variables are available:

- ``Above``: The set of all tensors that are stored in all memory objects above the
  current memory object in the hierarchy. Includes the same caveats as the previous
  ``MemoryObject``.
- ``<Any Tensor Name>``: Resolves to the tensor with the given name. If the tensor is
  not used in the current Einsum, then it resolves to the empty set.
.. - ``Einsum``: The name of the currently-processed Einsum. May be used in expressions
..   such as ``Inputs if Einsum == "Conv" else All``.
- ``MemoryObject.Tensors``: The set of all tensors that are stored in the memory object.
  Architectures are evaluated from the top down, so this will only be available
  ``MemoryObject`` has been evaluated. Lower-level memory objects may reference upper-level
  memory objects, but not vice versa. Additionally, this may not be used for energy and
  area calculations.

All tensor expressions can be converted into relevant rank variables by accessing
``.rank_variables``, which will return the set of all rank variables that index into the
tensor. If multiple tensors are referenced, then the union of all indexing rank
variables is returned. For example, ``MemoryObject.tensors.rank_variables`` will return
the set of all rank variables that index into any of the tensors stored in
``MemoryObject``.

Every tensor expression also has a ``bits_per_value`` attribute that returns the number
of bits per value for the tensor. This can only be called on size-one sets of tensors,
or else an error will be raised.

Additional keys can be defined following the renaming section of the :ref:`Workload and
Renames Specification <renaming-tensors-rank-variables>`.
