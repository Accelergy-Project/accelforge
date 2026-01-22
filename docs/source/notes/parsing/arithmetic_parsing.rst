Arithmetic and Parsing
======================

Objects can include expressions that are parsed when the
:py:class:`~fastfusion.frontend.spec.Spec` is parsed. Parsing occurs when the
:py:func:`~fastfusion.frontend.spec.Spec` is going to be used to model the energy, area,
or latency of an accelerator, such as when the
:py:func:`~fastfusion.frontend.spec.Spec.calculate_component_area_energy_latency_leak` method is
called.

To-be-parsed expressions can include Python code, and supported
operations include many standard library functions (*e.g.,* ``range``, ``min``) and
functions from the ``math`` standard library (*e.g.,* ``log2``, ``ceil``).

The scope available for parsing includes the following in order of increasing
precedence:

- Variables defined in a top-level :py:class:`~fastfusion.frontend.variables.Variables`
  object.
- Variables defined in outer-level YAML objects. Dictionary keys can be referenced by
  names, and list entries by index. The dot syntax can be used to access dictionaries;
  for example, ``x.y.z`` is equivalent to ``outer_scope["x"]["y"]["z"]``.
- Variables defined in the current YAML object. Dictionary keys may reference each other
  as long as references are not cyclic.

The following is an example of valid parsed data:

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

  from fastfusion.frontend.arch import ComponentAttributes
  attributes = ComponentAttributes(
    value1=123,
    value2="value1 + 5"
    # ... other attributes
  )


Supported Arithmetic Operations
-------------------------------

The following are available expressions. In addition to the below, Python keywords that
are available witout import (*e.g.,* ``min``) are also available

- ``ceil``: :py:func:`math.ceil`
- ``comb``: `math.comb`
- ``copysign``: `math.copysign`
- ``fabs``: :py:func:`math.fabs`
- ``factorial``: :py:func:`math.factorial`
- ``floor``: :py:func:`math.floor`
- ``fmod``: :py:func:`math.fmod`
- ``frexp``: :py:func:`math.frexp`
- ``fsum``: :py:func:`math.fsum`
- ``gcd``: :py:func:`math.gcd`
- ``isclose``: `math.isclose`
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
